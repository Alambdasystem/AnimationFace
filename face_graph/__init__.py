"""
AnimationFace face_graph package.

Provides face landmark detection, mesh graph construction,
and RBF-based mesh deformation.
"""

from .canonical_model import CANONICAL_POINTS, CONTROL_POINT_INDICES, ANCHOR_POINT_INDICES
from .landmark_detector import FaceLandmarkDetector, FaceLandmarks, SyntheticFaceGenerator
from .mesh_graph import FaceMeshGraph, build_face_graph
from .deformation import MeshDeformer

__all__ = [
    "CANONICAL_POINTS",
    "CONTROL_POINT_INDICES",
    "ANCHOR_POINT_INDICES",
    "FaceLandmarkDetector",
    "FaceLandmarks",
    "SyntheticFaceGenerator",
    "FaceMeshGraph",
    "build_face_graph",
    "MeshDeformer",
]
