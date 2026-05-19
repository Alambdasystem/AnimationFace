"""
Face landmark detection and synthetic face generation.

Two detection backends are available:
1. Haar cascade detector (built-in to OpenCV, always available):
   Detects a face bounding box, then places canonical 72-point landmarks
   relative to that box.

2. MediaPipe FaceLandmarker (optional, higher accuracy):
   Requires a local copy of the `face_landmarker.task` model file.
   Download instructions: see README.md.

Use ``SyntheticFaceGenerator`` to create a cartoon face with perfect
landmark ground-truth (ideal for testing / demos without a camera).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .canonical_model import CANONICAL_POINTS

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmarks:
    """72 (x, y) landmark positions in pixel coordinates."""

    points: np.ndarray      # shape (72, 2)  float64, pixel coords
    image_width: int
    image_height: int

    @property
    def xy(self) -> np.ndarray:
        return self.points.copy()

    def get_point(self, idx: int) -> np.ndarray:
        return self.points[idx]


# ---------------------------------------------------------------------------
# Helper: place canonical landmarks inside a bounding box
# ---------------------------------------------------------------------------

def _place_canonical(bbox: Tuple[int, int, int, int],
                     img_w: int, img_h: int) -> FaceLandmarks:
    """Map canonical [0,1] landmark positions into pixel space."""
    x0, y0, w, h = bbox
    pts = CANONICAL_POINTS.copy()
    pts[:, 0] = x0 + pts[:, 0] * w
    pts[:, 1] = y0 + pts[:, 1] * h
    return FaceLandmarks(points=pts, image_width=img_w, image_height=img_h)


# ---------------------------------------------------------------------------
# Haar cascade detector (no external model download required)
# ---------------------------------------------------------------------------

class FaceLandmarkDetector:
    """Detect face landmarks using OpenCV Haar cascade + canonical model.

    The cascade locates the face bounding box; the canonical 72-point model
    is then scaled into that box.  For production-quality detection, replace
    with a MediaPipe FaceLandmarker (see README.md).
    """

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray,
               scale_factor: float = 1.1,
               min_neighbors: int = 3) -> Optional[FaceLandmarks]:
        """Detect the first face in *image* and return 72 landmarks.

        Args:
            image: BGR or grayscale uint8 numpy array.
            scale_factor: Haar detection scale factor.
            min_neighbors: Haar minimum neighbour count.

        Returns:
            ``FaceLandmarks`` if a face is found, ``None`` otherwise.
        """
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3 else image)
        h_img, w_img = gray.shape[:2]

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if not isinstance(detections, np.ndarray) or len(detections) == 0:
            # Fall back: treat whole image as the face bbox
            margin_x = int(w_img * 0.1)
            margin_y = int(h_img * 0.05)
            bbox = (margin_x, margin_y,
                    w_img - 2 * margin_x, h_img - 2 * margin_y)
        else:
            # Pick the largest detected face
            areas = detections[:, 2] * detections[:, 3]
            bbox = tuple(detections[np.argmax(areas)])

        return _place_canonical(bbox, w_img, h_img)


# ---------------------------------------------------------------------------
# Optional: MediaPipe FaceLandmarker wrapper
# ---------------------------------------------------------------------------

def try_load_mediapipe_detector(
        model_path: str = "face_landmarker.task") -> Optional[object]:
    """Try to load a MediaPipe FaceLandmarker.

    Returns a detector with the same ``detect(image) -> FaceLandmarks``
    interface, or ``None`` if MediaPipe is unavailable / model missing.
    """
    if not os.path.isfile(model_path):
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return _MediaPipeLandmarker(landmarker)
    except Exception:
        return None


class _MediaPipeLandmarker:
    """Thin wrapper around MediaPipe FaceLandmarker (478-point mesh)."""

    # Mapping from MediaPipe 478-pt indices to our 72 canonical slots.
    # Where no exact correspondence exists, the nearest anatomical point is used.
    _MP_IDX: list = [
        # head outline (26) – approximate jaw + outline points
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        # forehead (1)
        10,
        # left eyebrow (4)
        55, 65, 52, 53,
        # right eyebrow (4)
        285, 295, 282, 283,
        # left eye (6)
        33, 160, 158, 133, 153, 144,
        # right eye (6)
        362, 385, 387, 263, 373, 380,
        # nose (5)
        168, 6, 4, 98, 327,
        # cheeks (2)
        234, 454,
        # mouth outer (12)
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 146,
        # mouth inner (6)
        78, 13, 308, 317, 14, 87,
    ]

    def __init__(self, landmarker):
        self._lm = landmarker

    def detect(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        import mediapipe as mp

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._lm.detect(mp_image)

        if not result.face_landmarks:
            return None

        all_lm = result.face_landmarks[0]
        pts = np.array(
            [[all_lm[i].x * w, all_lm[i].y * h] for i in self._MP_IDX],
            dtype=np.float64,
        )
        return FaceLandmarks(points=pts, image_width=w, image_height=h)


# ---------------------------------------------------------------------------
# Synthetic face generator
# ---------------------------------------------------------------------------

class SyntheticFaceGenerator:
    """Draw a cartoon face and return it with perfect canonical landmarks.

    Useful for demos and unit tests that do not require a camera or a photo.
    """

    # Colours (BGR)
    _SKIN    = (180, 210, 230)
    _SKIN_D  = (140, 170, 190)   # darker skin for shading
    _WHITE   = (245, 245, 245)
    _IRIS    = ( 90, 120,  60)
    _PUPIL   = ( 20,  20,  20)
    _BROW    = ( 50,  60,  80)
    _LIP_O   = (100,  80, 160)   # outer lip (dark)
    _LIP_I   = (130, 110, 195)   # inner lip (lighter)
    _TEETH   = (240, 240, 230)
    _TONGUE  = (100, 130, 200)
    _BG      = (200, 215, 235)

    def __init__(self, width: int = 400, height: int = 400):
        self.width = width
        self.height = height

    def generate(self, mouth_open: float = 0.0) -> Tuple[np.ndarray, FaceLandmarks]:
        """Generate a cartoon face image.

        Args:
            mouth_open: value in [0, 1] controlling how wide the mouth is open
                        (0 = closed, 1 = fully open).

        Returns:
            (image_bgr, landmarks)  where image_bgr is uint8 (H, W, 3) and
            landmarks has 72 points in pixel coordinates.
        """
        img = np.full((self.height, self.width, 3), self._BG, dtype=np.uint8)
        w, h = self.width, self.height

        # The face occupies the central 80 % of the image
        face_x = int(w * 0.10)
        face_y = int(h * 0.05)
        face_w = int(w * 0.80)
        face_h = int(h * 0.90)

        cx, cy = face_x + face_w // 2, face_y + face_h // 2

        # ── Head ─────────────────────────────────────────────────────────
        cv2.ellipse(img, (cx, cy), (face_w // 2, face_h // 2),
                    0, 0, 360, self._SKIN, -1)

        # ── Eyebrows ─────────────────────────────────────────────────────
        for side in (-1, 1):
            ex = cx + side * int(face_w * 0.22)
            ey = cy - int(face_h * 0.23)
            bw = int(face_w * 0.14)
            cv2.ellipse(img, (ex, ey), (bw, int(face_h * 0.025)),
                        0, 180, 360, self._BROW, int(face_h * 0.018))

        # ── Eyes ─────────────────────────────────────────────────────────
        eye_rx = int(face_w * 0.09)
        eye_ry = int(face_h * 0.04)
        for side in (-1, 1):
            ex = cx + side * int(face_w * 0.22)
            ey = cy - int(face_h * 0.12)
            cv2.ellipse(img, (ex, ey), (eye_rx, eye_ry),
                        0, 0, 360, self._WHITE, -1)
            cv2.ellipse(img, (ex, ey), (int(eye_rx * 0.65), eye_ry),
                        0, 0, 360, self._IRIS, -1)
            cv2.circle(img, (ex, ey), int(eye_rx * 0.30), self._PUPIL, -1)
            cv2.ellipse(img, (ex, ey), (eye_rx, eye_ry),
                        0, 0, 360, self._SKIN_D, 1)

        # ── Nose ─────────────────────────────────────────────────────────
        nx, ny = cx, cy + int(face_h * 0.11)
        nose_h = int(face_h * 0.09)
        cv2.line(img, (nx, cy - int(face_h * 0.04)),
                 (nx, ny + nose_h // 2), self._SKIN_D, 2)
        nar = int(face_w * 0.06)
        cv2.ellipse(img, (nx - nar, ny + nose_h // 2),
                    (nar, int(nose_h * 0.30)), 0, 0, 180, self._SKIN_D, 2)
        cv2.ellipse(img, (nx + nar, ny + nose_h // 2),
                    (nar, int(nose_h * 0.30)), 0, 0, 180, self._SKIN_D, 2)

        # ── Mouth ────────────────────────────────────────────────────────
        mw = int(face_w * 0.26)      # half-width of mouth
        my = cy + int(face_h * 0.24) # vertical centre of mouth
        open_px = int(mouth_open * face_h * 0.12)  # jaw-drop in pixels

        # Outer lip shape
        lip_pts_upper = np.array([
            [cx - mw, my],
            [cx - int(mw * 0.54), my - int(face_h * 0.035)],
            [cx - int(mw * 0.10), my - int(face_h * 0.050)],
            [cx,                  my - int(face_h * 0.055)],
            [cx + int(mw * 0.10), my - int(face_h * 0.050)],
            [cx + int(mw * 0.54), my - int(face_h * 0.035)],
            [cx + mw, my],
        ], dtype=np.int32)

        lip_pts_lower = np.array([
            [cx + mw, my + open_px],
            [cx + int(mw * 0.54), my + int(face_h * 0.04) + open_px],
            [cx + int(mw * 0.10), my + int(face_h * 0.07) + open_px],
            [cx,                  my + int(face_h * 0.08) + open_px],
            [cx - int(mw * 0.10), my + int(face_h * 0.07) + open_px],
            [cx - int(mw * 0.54), my + int(face_h * 0.04) + open_px],
            [cx - mw, my + open_px],
        ], dtype=np.int32)

        # Teeth / tongue (visible when open)
        if open_px > 2:
            teeth_pts = np.vstack([lip_pts_upper[1:-1],
                                   lip_pts_lower[-2:0:-1]])
            cv2.fillPoly(img, [teeth_pts], self._TEETH)
            if open_px > 8:
                tongue_pts = np.array([
                    [cx - int(mw * 0.3), my + open_px // 2],
                    [cx,                  my + open_px],
                    [cx + int(mw * 0.3), my + open_px // 2],
                ], dtype=np.int32)
                cv2.fillPoly(img, [tongue_pts], self._TONGUE)

        # Fill lip polygon
        mouth_pts = np.vstack([lip_pts_upper, lip_pts_lower[::-1]])
        cv2.fillPoly(img, [mouth_pts], self._LIP_O)

        # Upper-lip highlight
        cv2.polylines(img, [lip_pts_upper], False, self._LIP_I, 2, cv2.LINE_AA)
        cv2.polylines(img, [lip_pts_lower], False, self._LIP_O, 2, cv2.LINE_AA)

        # ── Build landmarks from canonical model ──────────────────────────
        lm = _place_canonical((face_x, face_y, face_w, face_h),
                              w, h)
        # Adjust the mouth landmarks to match the drawn mouth_open offset
        # Indices 54-65 outer mouth, 66-71 inner mouth, 13 = jaw
        #  (y displacement for lower-lip and jaw)
        for idx in [60, 61, 62, 63, 64, 65, 69, 70, 71]:
            lm.points[idx, 1] += open_px
        lm.points[13, 1] += open_px * 0.4   # jaw drops a bit too

        return img, lm
