"""
Text-to-phoneme conversion and speech animation frame generation.

``text_to_phonemes`` converts English text into a flat list of CMU Arpabet
phonemes.  The conversion tries the following chain:

  1. CMU Pronouncing Dictionary look-up via the ``pronouncing`` library
     (if installed: ``pip install pronouncing``).
  2. Internal mini-dictionary of the ~300 most common English words.
  3. Grapheme-to-phoneme fallback rules (letter-by-letter English phonics).

``SpeechAnimator`` drives a :class:`~face_graph.deformation.MeshDeformer`
through a sequence of phonemes and returns per-frame deformed vertex arrays
suitable for rendering.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

import numpy as np

from face_graph.deformation import MeshDeformer
from face_graph.mesh_graph import FaceMeshGraph
from animation.viseme_controller import VisemeModel

# ── Minimal CMU word dictionary (most common ~300 English words) ──────────────
# Format: lower-case word → space-separated Arpabet phonemes (stress-stripped).
_MINI_DICT: dict = {
    "a":       "AH",
    "about":   "AH B AW T",
    "all":     "AO L",
    "also":    "AO L S OW",
    "am":      "AE M",
    "an":      "AE N",
    "and":     "AE N D",
    "any":     "EH N IY",
    "are":     "AA R",
    "as":      "AE Z",
    "at":      "AE T",
    "back":    "B AE K",
    "be":      "B IY",
    "because": "B IH K AO Z",
    "been":    "B IH N",
    "big":     "B IH G",
    "but":     "B AH T",
    "by":      "B AY",
    "can":     "K AE N",
    "come":    "K AH M",
    "could":   "K UH D",
    "day":     "D EY",
    "did":     "D IH D",
    "do":      "D UW",
    "down":    "D AW N",
    "each":    "IY CH",
    "even":    "IY V AH N",
    "face":    "F EY S",
    "find":    "F AY N D",
    "first":   "F ER S T",
    "for":     "F AO R",
    "from":    "F R AH M",
    "get":     "G EH T",
    "give":    "G IH V",
    "go":      "G OW",
    "good":    "G UH D",
    "great":   "G R EY T",
    "had":     "HH AE D",
    "has":     "HH AE Z",
    "have":    "HH AE V",
    "he":      "HH IY",
    "hello":   "HH EH L OW",
    "her":     "HH ER",
    "here":    "HH IH R",
    "him":     "HH IH M",
    "his":     "HH IH Z",
    "how":     "HH AW",
    "i":       "AY",
    "if":      "IH F",
    "in":      "IH N",
    "into":    "IH N T UW",
    "is":      "IH Z",
    "it":      "IH T",
    "its":     "IH T S",
    "just":    "JH AH S T",
    "know":    "N OW",
    "large":   "L AA R JH",
    "last":    "L AE S T",
    "leave":   "L IY V",
    "life":    "L AY F",
    "like":    "L AY K",
    "little":  "L IH T AH L",
    "long":    "L AO NG",
    "look":    "L UH K",
    "made":    "M EY D",
    "make":    "M EY K",
    "man":     "M AE N",
    "many":    "M EH N IY",
    "may":     "M EY",
    "me":      "M IY",
    "might":   "M AY T",
    "more":    "M AO R",
    "most":    "M OW S T",
    "much":    "M AH CH",
    "my":      "M AY",
    "name":    "N EY M",
    "new":     "N UW",
    "no":      "N OW",
    "not":     "N AA T",
    "now":     "N AW",
    "of":      "AH V",
    "on":      "AO N",
    "one":     "W AH N",
    "only":    "OW N L IY",
    "or":      "AO R",
    "other":   "AH DH ER",
    "our":     "AW R",
    "out":     "AW T",
    "over":    "OW V ER",
    "own":     "OW N",
    "part":    "P AA R T",
    "people":  "P IY P AH L",
    "place":   "P L EY S",
    "put":     "P UH T",
    "right":   "R AY T",
    "said":    "S EH D",
    "same":    "S EY M",
    "say":     "S EY",
    "see":     "S IY",
    "seem":    "S IY M",
    "she":     "SH IY",
    "should":  "SH UH D",
    "since":   "S IH N S",
    "small":   "S M AO L",
    "so":      "S OW",
    "some":    "S AH M",
    "still":   "S T IH L",
    "such":    "S AH CH",
    "take":    "T EY K",
    "than":    "DH AE N",
    "that":    "DH AE T",
    "the":     "DH AH",
    "their":   "DH EH R",
    "them":    "DH EH M",
    "then":    "DH EH N",
    "there":   "DH EH R",
    "they":    "DH EY",
    "think":   "TH IH NG K",
    "this":    "DH IH S",
    "through": "TH R UW",
    "time":    "T AY M",
    "to":      "T UW",
    "too":     "T UW",
    "turn":    "T ER N",
    "two":     "T UW",
    "under":   "AH N D ER",
    "up":      "AH P",
    "us":      "AH S",
    "use":     "Y UW Z",
    "very":    "V EH R IY",
    "was":     "W AH Z",
    "water":   "W AO T ER",
    "way":     "W EY",
    "we":      "W IY",
    "well":    "W EH L",
    "went":    "W EH N T",
    "were":    "W ER",
    "what":    "W AH T",
    "when":    "W EH N",
    "where":   "W EH R",
    "which":   "W IH CH",
    "while":   "W AY L",
    "who":     "HH UW",
    "why":     "W AY",
    "will":    "W IH L",
    "with":    "W IH DH",
    "word":    "W ER D",
    "work":    "W ER K",
    "world":   "W ER L D",
    "would":   "W UH D",
    "write":   "R AY T",
    "year":    "Y IH R",
    "you":     "Y UW",
    "your":    "Y AO R",
}


# ── Grapheme-to-phoneme fallback rules ────────────────────────────────────────
# Applied left-to-right on individual characters when word not in dict.

# Multi-character digraph → list of phonemes (after splitting on space).
_DIGRAPH: dict = {
    "th": ["TH"], "sh": ["SH"], "ch": ["CH"], "ph": ["F"],
    "wh": ["W"],  "ck": ["K"],  "ng": ["NG"], "qu": ["K", "W"],
    "gh": [],     "kn": ["N"],  "wr": ["R"],
}

_VOWEL_MAP: dict = {
    "a": "AE", "e": "EH", "i": "IH", "o": "AO", "u": "AH",
}

# All values are single Arpabet tokens; "x" maps to two separate entries.
_CONS_MAP: dict = {
    "b": ["B"],  "c": ["K"],  "d": ["D"],  "f": ["F"],  "g": ["G"],
    "h": ["HH"], "j": ["JH"], "k": ["K"],  "l": ["L"],  "m": ["M"],
    "n": ["N"],  "p": ["P"],  "q": ["K"],  "r": ["R"],  "s": ["S"],
    "t": ["T"],  "v": ["V"],  "w": ["W"],  "x": ["K", "S"], "y": ["Y"],
    "z": ["Z"],
}


def _g2p_word(word: str) -> List[str]:
    """Very simple grapheme-to-phoneme for unknown English words."""
    word = word.lower()
    phonemes: List[str] = []
    i = 0
    while i < len(word):
        # Try digraph first
        if i + 1 < len(word) and word[i:i+2] in _DIGRAPH:
            phonemes.extend(_DIGRAPH[word[i:i+2]])
            i += 2
            continue
        ch = word[i]
        if ch in _VOWEL_MAP:
            phonemes.append(_VOWEL_MAP[ch])
        elif ch in _CONS_MAP:
            phonemes.extend(_CONS_MAP[ch])
        # else: skip punctuation / unknown characters
        i += 1
    return phonemes if phonemes else ["SIL"]


def _try_pronouncing(word: str) -> Optional[List[str]]:
    """Try the ``pronouncing`` library (returns None if unavailable)."""
    try:
        import pronouncing  # type: ignore
        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            # Strip stress digits (0, 1, 2) from each token
            return [re.sub(r"\d", "", p) for p in phones[0].split()]
    except ImportError:
        pass
    return None


def text_to_phonemes(text: str) -> List[str]:
    """Convert English text to a flat list of Arpabet phonemes.

    Conversion priority:
      1. ``pronouncing`` library CMU dict (if installed).
      2. Internal mini-dictionary.
      3. Grapheme-to-phoneme fallback rules.

    Silence tokens ``"SIL"`` are inserted between words.
    """
    # Normalise: lowercase, strip accents, keep only letters/spaces
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    words = re.findall(r"[a-z]+", text)

    phonemes: List[str] = ["SIL"]
    for word in words:
        phs = _try_pronouncing(word)
        if phs is None:
            entry = _MINI_DICT.get(word)
            phs = entry.split() if entry else _g2p_word(word)
        phonemes.extend(phs)
        phonemes.append("SIL")   # brief pause between words

    return phonemes


# ── Speech animator ───────────────────────────────────────────────────────────

class SpeechAnimator:
    """Generate animated face-mesh vertex sequences driven by phonemes.

    Each phoneme is held for ``phoneme_frames`` frames at 24 fps.
    Transitions between consecutive phonemes are smoothed with a cosine
    cross-fade of ``transition_frames`` frames.

    Parameters
    ----------
    graph   : FaceMeshGraph
    model   : VisemeModel
    deformer: MeshDeformer
    fps     : int   frames per second (default 24)
    phoneme_ms : int  milliseconds each phoneme is displayed (default 100)
    transition_ms : int  milliseconds for the cross-fade (default 50)
    """

    def __init__(
        self,
        graph:    FaceMeshGraph,
        model:    VisemeModel,
        deformer: MeshDeformer,
        fps: int = 24,
        phoneme_ms: int = 100,
        transition_ms: int = 50,
    ):
        self._graph    = graph
        self._model    = model
        self._deformer = deformer
        self._fps      = fps
        self._p_frames = max(1, round(phoneme_ms  * fps / 1000))
        self._t_frames = max(1, round(transition_ms * fps / 1000))

    # ------------------------------------------------------------------
    def animate_phonemes(self, phonemes: List[str]) -> List[np.ndarray]:
        """Generate deformed vertex arrays for a phoneme sequence.

        Returns a list of (72, 2) vertex arrays (one per animation frame).
        """
        if not phonemes:
            return [self._deformer.reset()]

        frames: List[np.ndarray] = []
        prev_ph = "SIL"

        for ph in phonemes:
            hold = self._p_frames
            trans = self._t_frames

            # Transition from previous phoneme
            for t_idx in range(trans):
                t = (t_idx + 1) / (trans + 1)
                disp = self._model.blend(prev_ph, ph, t)
                frames.append(self._deformer.deform(disp))

            # Hold current phoneme
            disp_full = self._model.predict(ph)
            for _ in range(hold):
                frames.append(self._deformer.deform(disp_full))

            prev_ph = ph

        # Return to REST
        for t_idx in range(self._t_frames * 2):
            t = (t_idx + 1) / (self._t_frames * 2 + 1)
            disp = self._model.blend(prev_ph, "SIL", t)
            frames.append(self._deformer.deform(disp))

        frames.append(self._deformer.reset())
        return frames

    # ------------------------------------------------------------------
    def animate_text(self, text: str) -> Tuple[List[np.ndarray], List[str]]:
        """Convert text to phonemes and return animated frames.

        Returns:
            (frames, phoneme_list)
        """
        phonemes = text_to_phonemes(text)
        return self.animate_phonemes(phonemes), phonemes

    # ------------------------------------------------------------------
    def animate_visemes(self, viseme_names: List[str]) -> List[np.ndarray]:
        """Animate through a list of viseme names (e.g. for a demo)."""
        phoneme_map = {v: v for v in viseme_names}   # use viseme name as "phoneme"
        # Temporarily add viseme names as identity phoneme entries
        from animation.viseme_controller import VISEME_NAMES, PHONEME_TO_VISEME
        frames: List[np.ndarray] = []
        prev_ph = "SIL"
        for vname in viseme_names:
            vi = VISEME_NAMES.index(vname) if vname in VISEME_NAMES else 0
            for t_idx in range(self._t_frames):
                t = (t_idx + 1) / (self._t_frames + 1)
                d_prev = self._model.predict(prev_ph)
                d_cur  = self._model._get_viseme(vi)
                t_cos  = 0.5 * (1.0 - np.cos(np.pi * t))
                disp = d_prev * (1.0 - t_cos) + d_cur * t_cos
                frames.append(self._deformer.deform(disp))
            disp_full = self._model._get_viseme(vi)
            for _ in range(self._p_frames):
                frames.append(self._deformer.deform(disp_full))
            prev_ph = "SIL"   # reset after each viseme
        frames.append(self._deformer.reset())
        return frames
