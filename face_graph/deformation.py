"""
RBF-based mesh deformation engine.

Given displacements for the 11 mouth/jaw control points, the deformer uses
Thin-Plate Spline (TPS) Radial Basis Function interpolation to propagate the
deformation smoothly across all 72 mesh vertices.

Fixed anchor points (nose, eyes, forehead, cheeks) receive zero displacement,
which naturally confines the deformation to the mouth / jaw region.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from .mesh_graph import FaceMeshGraph


class MeshDeformer:
    """Deform a face mesh via TPS-RBF interpolation.

    Usage::

        deformer = MeshDeformer(graph)
        new_verts = deformer.deform(displacement_array)
    """

    def __init__(self, graph: FaceMeshGraph, smoothing: float = 0.0):
        """
        Args:
            graph:     The :class:`FaceMeshGraph` to deform.
            smoothing: RBF smoothing parameter (0 = exact interpolation).
        """
        self._graph = graph
        self._base = graph.vertices.copy()   # (72, 2) reference positions
        self._smoothing = smoothing

    # ------------------------------------------------------------------
    def deform(self, control_displacements: np.ndarray) -> np.ndarray:
        """Apply control-point displacements and propagate across the mesh.

        Args:
            control_displacements: (11, 2) displacement vectors (pixels)
                                   for each of the 11 animation handles,
                                   in the same order as
                                   ``graph.control_names``.

        Returns:
            deformed_vertices: (72, 2) new vertex positions (pixels).
        """
        ctrl_pos   = self._base[self._graph.control_indices]   # (11, 2)
        anchor_pos = self._base[self._graph.anchor_indices]    # ( A, 2)

        n_anchors = len(anchor_pos)
        anchor_zeros = np.zeros((n_anchors, 2), dtype=np.float64)

        all_src  = np.vstack([ctrl_pos,  anchor_pos])          # (11+A, 2)
        all_disp = np.vstack([control_displacements, anchor_zeros])

        if np.allclose(all_disp, 0.0, atol=1e-8):
            return self._base.copy()

        try:
            rbf = RBFInterpolator(
                all_src, all_disp,
                kernel="thin_plate_spline",
                degree=1,
                smoothing=self._smoothing,
            )
            full_disp = rbf(self._base)          # (72, 2)
            deformed  = self._base + full_disp
        except Exception:
            # Graceful fallback: only move the control points directly
            deformed = self._base.copy()
            deformed[self._graph.control_indices] += control_displacements

        return deformed

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Return the base (undeformed) vertex positions."""
        return self._base.copy()

    # ------------------------------------------------------------------
    def update_base(self, new_vertices: np.ndarray) -> None:
        """Update the reference mesh (e.g. when tracking across frames)."""
        self._base = new_vertices.copy()
