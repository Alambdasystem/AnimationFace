"""Tests for face_graph package: canonical model, landmark detection, mesh, deformation."""

import numpy as np
import pytest

from face_graph.canonical_model import (
    CANONICAL_POINTS,
    CONTROL_POINT_INDICES,
    ANCHOR_POINT_INDICES,
    CONTROL_POINT_NAMES,
    ANIMATED_CP_INDICES,
    ANCHOR_INDICES,
)
from face_graph.landmark_detector import (
    FaceLandmarks,
    FaceLandmarkDetector,
    SyntheticFaceGenerator,
    _place_canonical,
)
from face_graph.mesh_graph import build_face_graph, FaceMeshGraph
from face_graph.deformation import MeshDeformer


# ── canonical_model ────────────────────────────────────────────────────────────

class TestCanonicalModel:
    def test_shape(self):
        assert CANONICAL_POINTS.shape == (72, 2)

    def test_range(self):
        assert CANONICAL_POINTS.min() >= 0.0
        assert CANONICAL_POINTS.max() <= 1.0

    def test_control_point_count(self):
        assert len(CONTROL_POINT_INDICES) == 11

    def test_control_indices_in_range(self):
        for name, idx in CONTROL_POINT_INDICES.items():
            assert 0 <= idx < 72, f"{name}: index {idx} out of range"

    def test_anchor_indices_in_range(self):
        for name, idx in ANCHOR_POINT_INDICES.items():
            assert 0 <= idx < 72, f"{name}: index {idx} out of range"

    def test_animated_cp_indices_array(self):
        assert len(ANIMATED_CP_INDICES) == 11
        assert ANIMATED_CP_INDICES.dtype in (np.int32, np.int64)

    def test_control_names_match_dict(self):
        assert CONTROL_POINT_NAMES == list(CONTROL_POINT_INDICES.keys())


# ── SyntheticFaceGenerator ────────────────────────────────────────────────────

class TestSyntheticFaceGenerator:
    def setup_method(self):
        self.gen = SyntheticFaceGenerator(width=320, height=320)

    def test_output_shape(self):
        img, lm = self.gen.generate(mouth_open=0.0)
        assert img.shape == (320, 320, 3)
        assert img.dtype == np.uint8

    def test_landmark_count(self):
        _, lm = self.gen.generate()
        assert lm.points.shape == (72, 2)

    def test_landmarks_in_image(self):
        img, lm = self.gen.generate()
        h, w = img.shape[:2]
        assert lm.points[:, 0].max() <= w
        assert lm.points[:, 1].max() <= h
        assert lm.points.min() >= 0

    def test_mouth_open_shifts_lower_lip(self):
        _, lm0 = self.gen.generate(mouth_open=0.0)
        _, lm1 = self.gen.generate(mouth_open=1.0)
        # Lower lip inner centre (index 70) should be lower when open
        assert lm1.points[70, 1] > lm0.points[70, 1]

    def test_image_width_height_fields(self):
        _, lm = self.gen.generate()
        assert lm.image_width == 320
        assert lm.image_height == 320


# ── FaceLandmarkDetector ──────────────────────────────────────────────────────

class TestFaceLandmarkDetector:
    def test_detect_on_synthetic_face(self):
        """Detector should fall back and return 72 landmarks even for a synthetic face."""
        gen = SyntheticFaceGenerator(400, 400)
        img, _ = gen.generate()
        det = FaceLandmarkDetector()
        lm = det.detect(img)
        # May not find face precisely, but must return landmarks
        assert lm is not None
        assert lm.points.shape == (72, 2)

    def test_place_canonical(self):
        bbox = (50, 30, 200, 250)
        lm = _place_canonical(bbox, 400, 400)
        assert lm.points.shape == (72, 2)
        x0, y0, w, h = bbox
        # All x within [x0, x0+w], y within [y0, y0+h]
        assert lm.points[:, 0].min() >= x0 - 1
        assert lm.points[:, 0].max() <= x0 + w + 1
        assert lm.points[:, 1].min() >= y0 - 1
        assert lm.points[:, 1].max() <= y0 + h + 1


# ── FaceMeshGraph ─────────────────────────────────────────────────────────────

class TestFaceMeshGraph:
    def setup_method(self):
        gen = SyntheticFaceGenerator(400, 400)
        _, lm = gen.generate()
        self.graph = build_face_graph(lm)

    def test_vertices_shape(self):
        assert self.graph.vertices.shape == (72, 2)

    def test_triangles_shape(self):
        # Delaunay of 72 points → many triangles
        assert self.graph.triangles.ndim == 2
        assert self.graph.triangles.shape[1] == 3
        assert len(self.graph.triangles) > 50

    def test_control_indices(self):
        assert len(self.graph.control_indices) == 11

    def test_anchor_indices(self):
        assert len(self.graph.anchor_indices) == 6

    def test_face_scale_positive(self):
        assert self.graph.face_scale > 0

    def test_control_positions_shape(self):
        cp = self.graph.control_positions
        assert cp.shape == (11, 2)

    def test_get_handle(self):
        pt = self.graph.get_handle("left_corner")
        assert pt.shape == (2,)

    def test_control_names_correct(self):
        assert self.graph.control_names == list(CONTROL_POINT_INDICES.keys())


# ── MeshDeformer ──────────────────────────────────────────────────────────────

class TestMeshDeformer:
    def setup_method(self):
        gen = SyntheticFaceGenerator(400, 400)
        _, lm = gen.generate()
        self.graph   = build_face_graph(lm)
        self.deformer = MeshDeformer(self.graph)

    def test_zero_displacement_returns_base(self):
        zeros = np.zeros((11, 2))
        result = self.deformer.deform(zeros)
        np.testing.assert_allclose(result, self.graph.vertices, atol=1e-6)

    def test_output_shape(self):
        disp = np.zeros((11, 2))
        result = self.deformer.deform(disp)
        assert result.shape == (72, 2)

    def test_deformation_moves_control_points(self):
        disp = np.zeros((11, 2))
        disp[4, 1] = 20.0   # move ll_inner_center downward
        result = self.deformer.deform(disp)
        # The ll_inner_center vertex should have moved down
        orig_y = self.graph.vertices[self.graph.control_indices[4], 1]
        new_y  = result[self.graph.control_indices[4], 1]
        assert new_y > orig_y + 5.0

    def test_anchor_points_barely_move(self):
        """Anchor points should not move significantly when mouth opens."""
        disp = np.zeros((11, 2))
        disp[4, 1] = 20.0   # move ll_inner_center
        result = self.deformer.deform(disp)
        # Nose tip (anchor index in anchor_indices[0]) should barely move
        nose_idx = self.graph.anchor_indices[0]
        nose_orig = self.graph.vertices[nose_idx]
        nose_new  = result[nose_idx]
        dist = np.linalg.norm(nose_new - nose_orig)
        assert dist < 10.0   # less than 10 pixels

    def test_reset_returns_base(self):
        base = self.deformer.reset()
        np.testing.assert_array_equal(base, self.graph.vertices)

    def test_update_base(self):
        new_verts = self.graph.vertices + 5.0
        self.deformer.update_base(new_verts)
        np.testing.assert_array_equal(self.deformer._base, new_verts)
