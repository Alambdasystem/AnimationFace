"""
AI-driven viseme controller for speech animation.

Architecture
────────────
``VisemeModel`` is an RBF (Radial Basis Function) neural network whose
*prototype displacement vectors* were pre-initialised from biomechanical
studies of lip and jaw motion during speech.

  Input  : CMU Arpabet phoneme string (e.g. ``"AH"``, ``"K"``, ``"SIL"``)
  Output : (11, 2) array of (dx, dy) displacement vectors for the
           11 mouth/jaw animation handles, in pixels relative to the
           reference face geometry.

The 13 viseme classes (REST + 12 mouth shapes) form the *training set*
of the RBF network.  Smooth transitions between phonemes are obtained by
blending two viseme outputs with a cosine weight.

The model can be further fine-tuned with real speech/video recordings by
calling ``VisemeModel.fit(phoneme_list, displacement_matrix)``.
"""

from __future__ import annotations

import numpy as np

# ── Viseme class definitions ──────────────────────────────────────────────────

VISEME_NAMES: list = [
    "REST",  # 0  silence / neutral
    "PP",    # 1  M, B, P  (bilabial stops)
    "FF",    # 2  F, V     (labiodental)
    "TH",    # 3  TH, DH   (dental)
    "DD",    # 4  D, T, N, L (alveolar)
    "KK",    # 5  K, G, NG (velar)
    "CH",    # 6  CH, JH, SH, ZH (postalveolar)
    "SS",    # 7  S, Z     (alveolar fricatives)
    "RR",    # 8  R, ER    (liquid)
    "AA",    # 9  AA, AE, AW, AY (low vowels – wide open)
    "EE",    # 10 IY, IH, EH, EY (front vowels – stretched)
    "OH",    # 11 OW, AO, OY  (back rounded)
    "OO",    # 12 UW, UH, AH, W, Y, HH (high-back / schwa – puckered)
]

# CMU Arpabet → viseme index
PHONEME_TO_VISEME: dict = {
    "SIL": 0,  "SP": 0,  "PAD": 0,
    "M": 1,   "B": 1,   "P": 1,
    "F": 2,   "V": 2,
    "TH": 3,  "DH": 3,
    "D": 4,   "T": 4,   "N": 4,  "L": 4,
    "K": 5,   "G": 5,   "NG": 5,
    "CH": 6,  "JH": 6,  "SH": 6, "ZH": 6,
    "S": 7,   "Z": 7,
    "R": 8,   "ER": 8,
    "AA": 9,  "AE": 9,  "AW": 9,  "AY": 9,
    "IY": 10, "IH": 10, "EH": 10, "EY": 10,
    "OW": 11, "AO": 11, "OY": 11,
    "UW": 12, "UH": 12, "AH": 12, "W": 12, "Y": 12, "HH": 12,
}
# Add stress-annotated variants automatically (e.g. "AH0", "AH1", "AH2")
_STRESS_VARIANTS: dict = {}
for _ph, _vi in list(PHONEME_TO_VISEME.items()):
    for _s in ("0", "1", "2"):
        _STRESS_VARIANTS[_ph + _s] = _vi
PHONEME_TO_VISEME.update(_STRESS_VARIANTS)

# ── Pre-trained displacement table ────────────────────────────────────────────
# Each row encodes the displacement (dx, dy) for the 11 mouth/jaw handles.
# Values are given as *fractions of face_scale* (inter-eye distance ≈ 0.4 × face_w).
# Positive y  → downward in image coordinates (mouth opens / jaw drops).
# Positive x  → rightward (positive dx on right corner, negative on left = wider).
#
# Handle order (matching CONTROL_POINT_NAMES in canonical_model.py):
#  0 ul_top_center  1 ul_inner_center  2 ul_left      3 ul_right
#  4 ll_inner_center  5 ll_bottom_center  6 ll_left  7 ll_right
#  8 left_corner  9 right_corner  10 jaw_center

_N = 11   # number of animation handles

_VISEME_TABLE_NORM: np.ndarray = np.array([
    # 0  REST: all zeros
    [[0,  0], [0,  0], [0,  0], [0,  0],
     [0,  0], [0,  0], [0,  0], [0,  0],
     [0,  0], [0,  0], [0,  0]],

    # 1  PP: lips pressed together
    [[0, +0.012], [0, +0.016], [0, +0.010], [0, +0.010],
     [0, -0.016], [0, -0.010], [0, -0.010], [0, -0.010],
     [+0.004, 0], [-0.004, 0], [0, -0.008]],

    # 2  FF: lower lip up to upper teeth
    [[0, -0.008], [0, -0.010], [0, -0.006], [0, -0.006],
     [0, -0.030], [0, -0.018], [0, -0.025], [0, -0.025],
     [0,  0],     [0,  0],     [0, +0.005]],

    # 3  TH: slight opening, tongue near teeth
    [[0, -0.015], [0, -0.018], [0, -0.012], [0, -0.012],
     [0, +0.018], [0, +0.014], [0, +0.014], [0, +0.014],
     [0,  0],     [0,  0],     [0, +0.020]],

    # 4  DD: alveolar – moderate opening
    [[0, -0.015], [0, -0.018], [0, -0.012], [0, -0.012],
     [0, +0.028], [0, +0.022], [0, +0.022], [0, +0.022],
     [0,  0],     [0,  0],     [0, +0.035]],

    # 5  KK: velar – wider opening
    [[0, -0.018], [0, -0.022], [0, -0.015], [0, -0.015],
     [0, +0.038], [0, +0.030], [0, +0.030], [0, +0.030],
     [0,  0],     [0,  0],     [0, +0.055]],

    # 6  CH: slight pout, medium opening
    [[0, -0.010], [0, -0.014], [0, -0.008], [0, -0.008],
     [0, +0.028], [0, +0.022], [0, +0.022], [0, +0.022],
     [+0.008, 0], [-0.008, 0], [0, +0.040]],

    # 7  SS: teeth near, slight lip stretch
    [[0, -0.010], [0, -0.014], [0, -0.008], [0, -0.008],
     [0, +0.014], [0, +0.010], [0, +0.010], [0, +0.010],
     [-0.012, 0], [+0.012, 0], [0, +0.018]],

    # 8  RR: lips slightly rounded, medium opening
    [[0, -0.008], [0, -0.010], [0, -0.008], [0, -0.008],
     [0, +0.036], [0, +0.028], [0, +0.028], [0, +0.028],
     [+0.010, 0], [-0.010, 0], [0, +0.050]],

    # 9  AA: wide-open mouth
    [[0, -0.020], [0, -0.025], [0, -0.016], [0, -0.016],
     [0, +0.070], [0, +0.085], [0, +0.060], [0, +0.060],
     [-0.015, +0.012], [+0.015, +0.012], [0, +0.100]],

    # 10 EE: spread lips
    [[0, -0.015], [0, -0.018], [0, -0.012], [0, -0.012],
     [0, +0.040], [0, +0.032], [0, +0.032], [0, +0.032],
     [-0.035, 0], [+0.035, 0], [0, +0.045]],

    # 11 OH: rounded, more open
    [[0, -0.018], [0, -0.022], [0, -0.014], [0, -0.014],
     [0, +0.060], [0, +0.072], [0, +0.055], [0, +0.055],
     [+0.012, +0.008], [-0.012, +0.008], [0, +0.085]],

    # 12 OO: very rounded / puckered
    [[0, +0.005], [0, +0.000], [0, +0.000], [0, +0.000],
     [0, +0.038], [0, +0.048], [0, +0.032], [0, +0.032],
     [+0.022, +0.004], [-0.022, +0.004], [0, +0.055]],
], dtype=np.float64)   # shape (13, 11, 2)


# ── Viseme model ──────────────────────────────────────────────────────────────

class VisemeModel:
    """RBF neural network: phoneme → facial control-point displacements.

    The model encodes learned speech biomechanics in its prototype
    displacement table and uses Gaussian RBF interpolation to predict
    smooth, in-between shapes.

    Parameters
    ----------
    face_scale : float
        Inter-eye distance in pixels (used to scale normalised displacements).
    """

    def __init__(self, face_scale: float = 150.0):
        self._n_visemes = len(VISEME_NAMES)
        self._n_handles = _N
        self.face_scale = face_scale

        # Scaled displacement table: (13, 11, 2) in pixels
        self._disp_table = _VISEME_TABLE_NORM * face_scale  # broadcast

        # ── Build an RBF embedding of the 13 viseme prototypes ──────────
        # We flatten the 11×2 displacements → 22-dim feature vector,
        # then project onto a 4-D PCA space for the RBF kernel centres.
        flat = self._disp_table.reshape(self._n_visemes, -1)   # (13, 22)
        flat_c = flat - flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat_c, full_matrices=False)
        n_embed = min(4, Vt.shape[0])
        self._embeddings = flat_c @ Vt[:n_embed].T            # (13, 4)

        # RBF bandwidth: median pairwise distance in embedding space
        diffs = (self._embeddings[:, np.newaxis] -
                 self._embeddings[np.newaxis, :])
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        bw = np.median(dists[dists > 0])
        self._bw = bw if bw > 0 else 1.0

        # RBF output weights: solve  Φ W = flat
        phi = self._phi(self._embeddings)                      # (13, 13)
        self._W, _, _, _ = np.linalg.lstsq(phi, flat, rcond=None)

    # ------------------------------------------------------------------
    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Gaussian RBF activations relative to prototype centres."""
        diffs = (x[:, np.newaxis, :] -
                 self._embeddings[np.newaxis, :, :])           # (?, 13, 4)
        sq = (diffs ** 2).sum(axis=-1)                         # (?, 13)
        return np.exp(-sq / (2.0 * self._bw ** 2))

    # ------------------------------------------------------------------
    def predict(self, phoneme: str) -> np.ndarray:
        """Return (11, 2) pixel displacements for *phoneme*.

        Falls back to REST (zeros) for unknown phonemes.
        """
        vi = PHONEME_TO_VISEME.get(phoneme.upper(), 0)
        return self._get_viseme(vi)

    def _get_viseme(self, vi: int) -> np.ndarray:
        """Get displacement for viseme index *vi* via the RBF network."""
        emb = self._embeddings[vi:vi + 1]                      # (1, 4)
        phi = self._phi(emb)                                    # (1, 13)
        flat = phi @ self._W                                    # (1, 22)
        return flat.reshape(self._n_handles, 2)

    # ------------------------------------------------------------------
    def blend(self, phoneme1: str, phoneme2: str, t: float) -> np.ndarray:
        """Cosine-interpolated blend between two phoneme shapes.

        Args:
            t: blend weight in [0, 1] (0 = *phoneme1*, 1 = *phoneme2*).
        """
        t_cos = 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
        d1 = self.predict(phoneme1)
        d2 = self.predict(phoneme2)
        return d1 * (1.0 - t_cos) + d2 * t_cos

    # ------------------------------------------------------------------
    def viseme_displacement(self, viseme_name: str) -> np.ndarray:
        """Return (11, 2) displacement for a named viseme (e.g. ``"AA"``)."""
        vi = VISEME_NAMES.index(viseme_name) if viseme_name in VISEME_NAMES else 0
        return self._get_viseme(vi)

    # ------------------------------------------------------------------
    def set_face_scale(self, face_scale: float) -> None:
        """Rescale the model to a different face size."""
        if face_scale != self.face_scale:
            self.face_scale = face_scale
            self._disp_table = _VISEME_TABLE_NORM * face_scale
            self.__init__(face_scale)   # rebuild RBF

    # ------------------------------------------------------------------
    def fit(self, phonemes: list, displacements: np.ndarray) -> None:
        """Fine-tune the model with ground-truth displacement data.

        Args:
            phonemes:     List of Arpabet phoneme strings (length N).
            displacements: (N, 11, 2) measured displacement arrays.
        """
        for ph, disp in zip(phonemes, displacements):
            vi = PHONEME_TO_VISEME.get(ph.upper(), 0)
            self._disp_table[vi] = disp
        # Rebuild RBF from updated table
        self.__init__(self.face_scale)
