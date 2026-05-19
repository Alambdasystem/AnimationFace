"""Tests for the animation package: viseme controller, speech animator."""

import numpy as np
import pytest

from animation.viseme_controller import (
    VisemeModel,
    VISEME_NAMES,
    PHONEME_TO_VISEME,
)
from animation.speech_animator import (
    SpeechAnimator,
    text_to_phonemes,
    _g2p_word,
)
from face_graph.landmark_detector import SyntheticFaceGenerator
from face_graph.mesh_graph import build_face_graph
from face_graph.deformation import MeshDeformer


# ── PHONEME_TO_VISEME coverage ────────────────────────────────────────────────

class TestPhonemeToViseme:
    def test_all_basic_phonemes_present(self):
        basics = ["SIL", "M", "B", "P", "F", "V", "TH", "DH",
                  "D", "T", "N", "L", "K", "G", "NG",
                  "S", "Z", "R", "ER",
                  "AA", "AE", "IY", "IH", "EH",
                  "OW", "AO", "UW", "UH", "AH", "W", "Y", "HH"]
        for ph in basics:
            assert ph in PHONEME_TO_VISEME, f"Missing phoneme: {ph}"

    def test_stress_variants_present(self):
        for base in ["AH", "IY", "OW", "AA"]:
            for s in ("0", "1", "2"):
                assert base + s in PHONEME_TO_VISEME

    def test_viseme_indices_in_range(self):
        n = len(VISEME_NAMES)
        for ph, vi in PHONEME_TO_VISEME.items():
            assert 0 <= vi < n, f"{ph} → viseme {vi} out of range [0, {n})"


# ── VisemeModel ───────────────────────────────────────────────────────────────

class TestVisemeModel:
    def setup_method(self):
        self.model = VisemeModel(face_scale=150.0)

    def test_predict_returns_correct_shape(self):
        for ph in ["SIL", "AH", "M", "S", "R"]:
            d = self.model.predict(ph)
            assert d.shape == (11, 2), f"predict('{ph}') shape mismatch"

    def test_predict_rest_near_zero(self):
        d = self.model.predict("SIL")
        assert np.allclose(d, 0, atol=2.0), "REST displacement should be near zero"

    def test_aa_jaw_drops(self):
        """AA (wide-open) should have a large downward jaw displacement."""
        d = self.model.predict("AA")
        jaw_dy = d[10, 1]   # jaw_center dy
        assert jaw_dy > 5.0, f"AA jaw_dy={jaw_dy:.2f} too small"

    def test_ee_corners_stretch(self):
        """EE should pull mouth corners apart (positive dx on right, negative on left)."""
        d = self.model.predict("IY")
        left_dx  = d[8, 0]   # left_corner dx
        right_dx = d[9, 0]   # right_corner dx
        assert left_dx  < 0, f"EE left_corner dx={left_dx:.2f} should be negative"
        assert right_dx > 0, f"EE right_corner dx={right_dx:.2f} should be positive"

    def test_oo_corners_round(self):
        """OO should pull corners inward (right_corner dx < 0, left_corner dx > 0)."""
        d = self.model.predict("UW")
        left_dx  = d[8, 0]
        right_dx = d[9, 0]
        assert left_dx  > 0, f"OO left_corner dx={left_dx:.2f} should be positive"
        assert right_dx < 0, f"OO right_corner dx={right_dx:.2f} should be negative"

    def test_blend_midpoint(self):
        d_rest = self.model.predict("SIL")
        d_aa   = self.model.predict("AA")
        d_mid  = self.model.blend("SIL", "AA", 0.5)
        # Mid-blend should be between rest and AA
        assert d_mid[10, 1] > d_rest[10, 1]
        assert d_mid[10, 1] < d_aa[10, 1] + 1.0

    def test_blend_t0_equals_phoneme1(self):
        d1   = self.model.predict("M")
        d_b0 = self.model.blend("M", "AA", 0.0)
        np.testing.assert_allclose(d_b0, d1, atol=1e-6)

    def test_blend_t1_equals_phoneme2(self):
        d2   = self.model.predict("AA")
        d_b1 = self.model.blend("M", "AA", 1.0)
        np.testing.assert_allclose(d_b1, d2, atol=1e-6)

    def test_unknown_phoneme_returns_rest(self):
        d = self.model.predict("XYZ_UNKNOWN")
        d_rest = self.model.predict("SIL")
        np.testing.assert_allclose(d, d_rest, atol=1e-6)

    def test_viseme_displacement_by_name(self):
        d = self.model.viseme_displacement("AA")
        assert d.shape == (11, 2)
        assert d[10, 1] > 5.0   # jaw drops for AA

    def test_all_visemes_differ(self):
        """All 13 viseme outputs should not all be identical."""
        disps = [self.model.viseme_displacement(vn) for vn in VISEME_NAMES]
        # At least some pairs should differ
        diffs = [np.max(np.abs(disps[i] - disps[j]))
                 for i in range(len(disps)) for j in range(i+1, len(disps))]
        assert max(diffs) > 1.0, "All viseme displacements are identical – model broken"

    def test_set_face_scale_rescales(self):
        m1 = VisemeModel(face_scale=100.0)
        m2 = VisemeModel(face_scale=200.0)
        d1 = m1.predict("AA")
        d2 = m2.predict("AA")
        # Larger scale → larger absolute displacements
        assert np.max(np.abs(d2)) > np.max(np.abs(d1))


# ── text_to_phonemes ──────────────────────────────────────────────────────────

class TestTextToPhonemes:
    def test_hello_world(self):
        phs = text_to_phonemes("hello world")
        assert "HH" in phs
        assert "W" in phs

    def test_output_is_list_of_strings(self):
        phs = text_to_phonemes("hi there")
        assert isinstance(phs, list)
        assert all(isinstance(p, str) for p in phs)

    def test_sil_between_words(self):
        phs = text_to_phonemes("hello world")
        assert "SIL" in phs

    def test_empty_string(self):
        phs = text_to_phonemes("")
        # Should return at least one SIL
        assert phs == ["SIL"]

    def test_all_phonemes_in_phoneme_map(self):
        from animation.viseme_controller import PHONEME_TO_VISEME
        phs = text_to_phonemes("the quick brown fox jumps over the lazy dog")
        for ph in phs:
            # Each phoneme must be mappable to a viseme (unknown → 0)
            vi = PHONEME_TO_VISEME.get(ph.upper(), None)
            assert vi is not None, f"Unmappable phoneme: {ph}"

    def test_g2p_fallback(self):
        phs = _g2p_word("blah")
        assert isinstance(phs, list)
        assert len(phs) > 0

    def test_uppercase_input(self):
        phs1 = text_to_phonemes("Hello")
        phs2 = text_to_phonemes("hello")
        assert phs1 == phs2


# ── SpeechAnimator ────────────────────────────────────────────────────────────

def _make_animator(fps=12, phoneme_ms=80):
    gen = SyntheticFaceGenerator(320, 320)
    _, lm = gen.generate()
    graph    = build_face_graph(lm)
    deformer = MeshDeformer(graph)
    model    = VisemeModel(face_scale=graph.face_scale)
    return SpeechAnimator(graph, model, deformer,
                          fps=fps, phoneme_ms=phoneme_ms), graph


class TestSpeechAnimator:
    def setup_method(self):
        self.animator, self.graph = _make_animator()

    def test_animate_text_returns_frames(self):
        frames, phonemes = self.animator.animate_text("hello")
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_frames_have_correct_vertex_shape(self):
        frames, _ = self.animator.animate_text("hi")
        for f in frames:
            assert f.shape == (72, 2)

    def test_animate_empty_text(self):
        frames, phonemes = self.animator.animate_text("")
        assert len(frames) >= 1

    def test_animate_phonemes_directly(self):
        phs = ["SIL", "HH", "EH", "L", "OW", "SIL"]
        frames = self.animator.animate_phonemes(phs)
        assert len(frames) > len(phs)   # at least one frame per phoneme

    def test_frames_differ_during_animation(self):
        """Not all frames should be identical (mouth must move)."""
        frames, _ = self.animator.animate_text("hello world")
        if len(frames) < 2:
            pytest.skip("Too few frames")
        diffs = [np.max(np.abs(frames[i] - frames[i-1]))
                 for i in range(1, min(20, len(frames)))]
        assert max(diffs) > 0.1

    def test_animate_visemes(self):
        from animation.viseme_controller import VISEME_NAMES
        frames = self.animator.animate_visemes(VISEME_NAMES)
        assert len(frames) > len(VISEME_NAMES)
