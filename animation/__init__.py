"""
AnimationFace animation package.

Provides viseme-based AI animation control, text-to-phoneme conversion,
speech-driven animation frame generation, and face rendering.
"""

from .viseme_controller import VisemeModel, VISEME_NAMES, PHONEME_TO_VISEME
from .speech_animator import SpeechAnimator, text_to_phonemes
from .renderer import FaceRenderer

__all__ = [
    "VisemeModel",
    "VISEME_NAMES",
    "PHONEME_TO_VISEME",
    "SpeechAnimator",
    "text_to_phonemes",
    "FaceRenderer",
]
