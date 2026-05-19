# AnimationFace

**AI-driven Face Graph Speech Animation in Python**

AnimationFace creates a deformable mesh graph that is locked to a face image and uses a pre-trained AI model to determine which graph nodes to pull — and by how much — so the face moves naturally when speaking.

---

## Overview

```
Input image / synthetic face
        │
        ▼
┌───────────────────────┐
│  FaceLandmarkDetector │  ← AI (Haar cascade + canonical parametric model;
│  72 landmark points   │     optional MediaPipe upgrade)
└──────────┬────────────┘
           │  72 (x, y) pixel positions
           ▼
┌───────────────────────┐
│     FaceMeshGraph     │  ← Delaunay triangulation: 72 vertices, ~116 triangles
│  11 animation handles │     11 mouth/jaw control points, 6 fixed anchors
│   6 anchor points     │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐       ┌─────────────────────────┐
│   SpeechAnimator      │ ◄──── │     text_to_phonemes     │
│   per-frame driver    │       │  (CMU dict / G2P rules)  │
└──────────┬────────────┘       └─────────────────────────┘
           │ phoneme stream
           ▼
┌───────────────────────┐
│     VisemeModel (AI)  │  ← RBF neural network pre-trained with anatomical
│  phoneme → (11,2)     │    speech biomechanics data (13 viseme prototypes)
│  displacements        │    Cosine-interpolated smooth blending
└──────────┬────────────┘
           │ control-point displacement vectors
           ▼
┌───────────────────────┐
│     MeshDeformer      │  ← Thin-Plate Spline RBF interpolation
│  propagates deform.   │    (deforms all 72 vertices from 11 handles + anchors)
└──────────┬────────────┘
           │ deformed vertices per frame
           ▼
┌───────────────────────┐
│     FaceRenderer      │  ← Triangle mesh pixel warping + mesh overlay
│  GIF / MP4 / PNG      │    (affine warp per triangle, Delaunay tessellation)
└───────────────────────┘
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Optional – high-accuracy MediaPipe landmark detection:**
```bash
# 1. Install MediaPipe
pip install mediapipe

# 2. Download the face landmarker model
#    (replace VERSION with the latest from https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
wget -O face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

---

## Quick Start

```bash
# Demo with synthetic cartoon face, says "Hello world"
python main.py --demo

# Animate your own face image
python main.py --image face.jpg --text "Hello world" --output speech.gif

# Show the face graph mesh only (no animation)
python main.py --image face.jpg --show-mesh --output mesh.png

# Cycle through all 13 viseme mouth shapes
python main.py --demo --visemes --output visemes.gif

# Use MediaPipe for accurate landmark detection
python main.py --image face.jpg --text "Hi there" --mediapipe face_landmarker.task
```

---

## Module Reference

### `face_graph/`

| Module | Description |
|---|---|
| `canonical_model.py` | 72-point parametric face landmark model (positions as fractions of face bounding box) |
| `landmark_detector.py` | `FaceLandmarkDetector` (Haar cascade), `SyntheticFaceGenerator` (cartoon face), optional MediaPipe wrapper |
| `mesh_graph.py` | `FaceMeshGraph` — Delaunay triangulation with 11 animation handles and 6 anchors |
| `deformation.py` | `MeshDeformer` — Thin-Plate Spline RBF mesh deformation |

### `animation/`

| Module | Description |
|---|---|
| `viseme_controller.py` | `VisemeModel` — RBF neural network mapping CMU Arpabet phonemes → 11 control-point displacement vectors |
| `speech_animator.py` | `SpeechAnimator` — drives the deformer through a phoneme sequence; `text_to_phonemes` G2P conversion |
| `renderer.py` | `FaceRenderer` — mesh overlay drawing, triangle pixel warping, GIF/MP4/PNG export |

### `main.py`

CLI entry point. Run `python main.py --help` for all options.

---

## AI Model Details

`VisemeModel` implements a **Radial Basis Function (RBF) neural network** whose prototype displacement vectors were initialised from biomechanical studies of lip and jaw motion during speech:

- **Input:** CMU Arpabet phoneme string (e.g. `"AH"`, `"K"`, `"SIL"`)
- **Output:** `(11, 2)` displacement array for the 11 mouth/jaw animation handles
- **Architecture:** PCA embedding of 13 viseme prototypes → Gaussian RBF activations → linear output layer
- **Smooth transitions:** cosine-weighted blend between consecutive phonemes
- **Fine-tuning:** `model.fit(phonemes, displacements)` accepts ground-truth data

The 13 viseme classes cover the full English phoneme inventory:

| Index | Viseme | Phonemes |
|---|---|---|
| 0 | REST | silence |
| 1 | PP | M, B, P |
| 2 | FF | F, V |
| 3 | TH | TH, DH |
| 4 | DD | D, T, N, L |
| 5 | KK | K, G, NG |
| 6 | CH | CH, JH, SH, ZH |
| 7 | SS | S, Z |
| 8 | RR | R, ER |
| 9 | AA | AA, AE, AW, AY |
| 10 | EE | IY, IH, EH, EY |
| 11 | OH | OW, AO, OY |
| 12 | OO | UW, UH, AH, W, Y, HH |

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

56 tests cover canonical model integrity, landmark detection, mesh construction, deformation physics, the AI viseme model, phoneme conversion, and the speech animator.

---

## Extending the System

**Use a real camera:**
```python
import cv2
from face_graph import FaceLandmarkDetector, build_face_graph
from face_graph import MeshDeformer
from animation import VisemeModel, SpeechAnimator, FaceRenderer

detector = FaceLandmarkDetector()
cap = cv2.VideoCapture(0)
_, frame = cap.read()
lm = detector.detect(frame)
graph    = build_face_graph(lm)
deformer = MeshDeformer(graph)
model    = VisemeModel(graph.face_scale)
animator = SpeechAnimator(graph, model, deformer)
renderer = FaceRenderer(graph)

frames, _ = animator.animate_text("Hello world")
renderer.save_gif(frame, frames, "output.gif")
```

**Fine-tune the AI model** with your own recordings:
```python
# Collect (phoneme, displacement) pairs from motion-capture data
model.fit(["AH", "IY", "UW"], displacements_array)
```
