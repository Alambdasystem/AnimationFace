"""
Canonical 72-point parametric face landmark model.

Defines landmark positions as fractions of the face bounding box (x, y) in
[0, 1] × [0, 1], with (0, 0) at the top-left corner of the bounding box.

Layout:
  - Indices  0-25 : Head outline (26 pts, clockwise from forehead)
  - Index   26    : Forehead interior
  - Indices 27-30 : Left eyebrow  (4 pts)
  - Indices 31-34 : Right eyebrow (4 pts)
  - Indices 35-40 : Left eye      (6 pts)
  - Indices 41-46 : Right eye     (6 pts)
  - Indices 47-51 : Nose          (5 pts)
  - Indices 52-53 : Cheek anchors (2 pts)
  - Indices 54-65 : Mouth outer   (12 pts)
  - Indices 66-71 : Mouth inner   (6 pts)
  Total: 72 points
"""

import numpy as np

# fmt: off
CANONICAL_POINTS: np.ndarray = np.array([
    # ── HEAD OUTLINE (0-25): clockwise from forehead top-centre ──────────
    (0.50, 0.02),  # 0  forehead top-centre
    (0.60, 0.03),  # 1
    (0.70, 0.06),  # 2
    (0.80, 0.11),  # 3
    (0.88, 0.18),  # 4
    (0.93, 0.27),  # 5
    (0.96, 0.37),  # 6
    (0.96, 0.48),  # 7
    (0.93, 0.59),  # 8
    (0.88, 0.70),  # 9
    (0.80, 0.79),  # 10
    (0.70, 0.87),  # 11
    (0.60, 0.91),  # 12
    (0.50, 0.94),  # 13  chin centre  ← jaw animation handle
    (0.40, 0.91),  # 14
    (0.30, 0.87),  # 15
    (0.20, 0.79),  # 16
    (0.12, 0.70),  # 17
    (0.07, 0.59),  # 18
    (0.04, 0.48),  # 19
    (0.04, 0.37),  # 20
    (0.07, 0.27),  # 21
    (0.12, 0.18),  # 22
    (0.20, 0.11),  # 23
    (0.30, 0.06),  # 24
    (0.40, 0.03),  # 25
    # ── FOREHEAD INTERIOR (26) ────────────────────────────────────────────
    (0.50, 0.14),  # 26  forehead mid  ← anchor
    # ── LEFT EYEBROW (27-30) ──────────────────────────────────────────────
    (0.22, 0.30),  # 27
    (0.28, 0.27),  # 28
    (0.34, 0.27),  # 29
    (0.40, 0.29),  # 30
    # ── RIGHT EYEBROW (31-34) ─────────────────────────────────────────────
    (0.60, 0.29),  # 31
    (0.66, 0.27),  # 32
    (0.72, 0.27),  # 33
    (0.78, 0.30),  # 34
    # ── LEFT EYE (35-40) ──────────────────────────────────────────────────
    (0.23, 0.38),  # 35  left corner
    (0.27, 0.35),  # 36  upper-left
    (0.33, 0.35),  # 37  upper-right  ← eye anchor
    (0.38, 0.38),  # 38  right corner
    (0.33, 0.41),  # 39  lower-right
    (0.27, 0.41),  # 40  lower-left
    # ── RIGHT EYE (41-46) ─────────────────────────────────────────────────
    (0.62, 0.38),  # 41  left corner
    (0.67, 0.35),  # 42  upper-left
    (0.73, 0.35),  # 43  upper-right  ← eye anchor
    (0.77, 0.38),  # 44  right corner
    (0.73, 0.41),  # 45  lower-right
    (0.67, 0.41),  # 46  lower-left
    # ── NOSE (47-51) ──────────────────────────────────────────────────────
    (0.50, 0.42),  # 47  nose bridge top
    (0.50, 0.52),  # 48  nose bridge mid
    (0.50, 0.61),  # 49  nose tip  ← anchor
    (0.43, 0.63),  # 50  left nostril
    (0.57, 0.63),  # 51  right nostril
    # ── CHEEK ANCHORS (52-53) ─────────────────────────────────────────────
    (0.18, 0.58),  # 52  left cheek  ← anchor
    (0.82, 0.58),  # 53  right cheek ← anchor
    # ── MOUTH OUTER (54-65) ───────────────────────────────────────────────
    (0.37, 0.75),  # 54  left corner        ← animation handle
    (0.41, 0.71),  # 55  upper-left
    (0.46, 0.69),  # 56  upper centre-left  ← animation handle (ul_left)
    (0.50, 0.68),  # 57  upper centre-top   ← animation handle (ul_top)
    (0.54, 0.69),  # 58  upper centre-right ← animation handle (ul_right)
    (0.59, 0.71),  # 59  upper-right
    (0.63, 0.75),  # 60  right corner       ← animation handle
    (0.59, 0.80),  # 61  lower-right
    (0.54, 0.83),  # 62  lower centre-right ← animation handle (ll_right)
    (0.50, 0.84),  # 63  lower centre-bot   ← animation handle (ll_bot)
    (0.46, 0.83),  # 64  lower centre-left  ← animation handle (ll_left)
    (0.41, 0.80),  # 65  lower-left
    # ── MOUTH INNER (66-71) ───────────────────────────────────────────────
    (0.42, 0.75),  # 66  inner left corner
    (0.50, 0.73),  # 67  inner upper centre ← animation handle (ul_inner)
    (0.58, 0.75),  # 68  inner right corner
    (0.55, 0.78),  # 69  inner lower-right
    (0.50, 0.79),  # 70  inner lower centre ← animation handle (ll_inner)
    (0.45, 0.78),  # 71  inner lower-left
], dtype=np.float64)
# fmt: on

# ── Animation control-point indices (11 handles) ─────────────────────────────
CONTROL_POINT_INDICES: dict = {
    "ul_top_center":   57,   # upper lip top centre
    "ul_inner_center": 67,   # upper lip inner centre
    "ul_left":         56,   # upper lip centre-left
    "ul_right":        58,   # upper lip centre-right
    "ll_inner_center": 70,   # lower lip inner centre
    "ll_bottom_center": 63,  # lower lip bottom centre
    "ll_left":         64,   # lower lip centre-left
    "ll_right":        62,   # lower lip centre-right
    "left_corner":     54,   # left mouth corner
    "right_corner":    60,   # right mouth corner
    "jaw_center":      13,   # chin / jaw centre
}

# ── Anchor points (fixed reference points for RBF deformation) ───────────────
ANCHOR_POINT_INDICES: dict = {
    "nose_tip":     49,
    "left_eye":     37,
    "right_eye":    43,
    "forehead":     26,
    "left_cheek":   52,
    "right_cheek":  53,
}

# Ordered name list (matches CONTROL_POINT_INDICES insertion order)
CONTROL_POINT_NAMES: list = list(CONTROL_POINT_INDICES.keys())
ANIMATED_CP_INDICES: np.ndarray = np.array(
    [CONTROL_POINT_INDICES[n] for n in CONTROL_POINT_NAMES], dtype=np.int32
)
ANCHOR_INDICES: np.ndarray = np.array(
    list(ANCHOR_POINT_INDICES.values()), dtype=np.int32
)
