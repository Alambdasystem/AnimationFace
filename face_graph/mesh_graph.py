"""
Face mesh graph built from facial landmarks.

Builds a Delaunay-triangulated mesh whose nodes are the 72 canonical face
landmark positions.  Key mouth/jaw nodes are designated as *animation
control points* (handles); surrounding nodes act as *anchors* that pin the
rest of the mesh in place during deformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial import Delaunay

from .landmark_detector import FaceLandmarks
from .canonical_model import (
    CONTROL_POINT_NAMES,
    ANIMATED_CP_INDICES,
    ANCHOR_INDICES,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FaceMeshGraph:
    """Triangulated mesh locked to face landmark positions.

    Attributes:
        vertices:          (N, 2) landmark positions in pixels (float64).
        triangles:         (M, 3) triangle vertex indices (Delaunay).
        control_names:     Ordered list of the 11 animation handle names.
        control_indices:   (11,) indices into *vertices* for each handle.
        anchor_indices:    Indices into *vertices* for fixed anchor points.
        face_scale:        Reference eye-width in pixels (used for normalisation).
    """

    vertices: np.ndarray           # (N, 2)
    triangles: np.ndarray          # (M, 3)
    control_names: List[str]
    control_indices: np.ndarray    # (11,)
    anchor_indices: np.ndarray
    face_scale: float

    # ------------------------------------------------------------------
    @property
    def control_positions(self) -> np.ndarray:
        """(11, 2) pixel positions of animation handles."""
        return self.vertices[self.control_indices]

    @property
    def anchor_positions(self) -> np.ndarray:
        """(A, 2) pixel positions of anchor points."""
        return self.vertices[self.anchor_indices]

    def get_handle(self, name: str) -> np.ndarray:
        """Return the (x, y) pixel position of a named control point."""
        idx = self.control_names.index(name)
        return self.vertices[self.control_indices[idx]]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_face_graph(landmarks: FaceLandmarks) -> FaceMeshGraph:
    """Construct a :class:`FaceMeshGraph` from detected face landmarks.

    Args:
        landmarks: 72-point :class:`FaceLandmarks` (pixel coordinates).

    Returns:
        A :class:`FaceMeshGraph` ready for deformation and rendering.
    """
    vertices = landmarks.xy   # (72, 2)

    # Delaunay triangulation of all 72 points
    tri = Delaunay(vertices)
    triangles = tri.simplices.copy()

    # Face scale: inter-eye distance (left-eye to right-eye centre)
    # Left eye centre ≈ mean of indices 35-40; right eye ≈ mean of 41-46
    left_eye_cx  = vertices[35:41, 0].mean()
    right_eye_cx = vertices[41:47, 0].mean()
    face_scale = max(abs(right_eye_cx - left_eye_cx), 1.0)

    return FaceMeshGraph(
        vertices=vertices.copy(),
        triangles=triangles,
        control_names=list(CONTROL_POINT_NAMES),
        control_indices=ANIMATED_CP_INDICES.copy(),
        anchor_indices=ANCHOR_INDICES.copy(),
        face_scale=face_scale,
    )
