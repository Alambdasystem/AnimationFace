#!/usr/bin/env python3
"""
AnimationFace — AI-driven face graph animation.

Usage examples
--------------

Demo (synthetic cartoon face, says "Hello World"):
    python main.py --demo

Animate a real face image:
    python main.py --image face.jpg --text "Hello World"

Show mesh only (no animation):
    python main.py --image face.jpg --show-mesh --output mesh.png

Cycle through all 13 viseme shapes:
    python main.py --demo --visemes --output visemes.gif

Pass a MediaPipe model for accurate landmark detection:
    python main.py --image face.jpg --text "Hello" --mediapipe face_landmarker.task
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from face_graph.canonical_model import CONTROL_POINT_NAMES
from face_graph.landmark_detector import (
    FaceLandmarkDetector,
    SyntheticFaceGenerator,
    try_load_mediapipe_detector,
)
from face_graph.mesh_graph import build_face_graph
from face_graph.deformation import MeshDeformer
from animation.viseme_controller import VisemeModel, VISEME_NAMES
from animation.speech_animator import SpeechAnimator, text_to_phonemes
from animation.renderer import FaceRenderer


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AnimationFace — AI face graph speech animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image",      metavar="PATH",
                   help="Input face image (JPG/PNG).")
    p.add_argument("--text",       metavar="TEXT", default="Hello world",
                   help="Text to animate (default: 'Hello world').")
    p.add_argument("--output",     metavar="PATH", default="output.gif",
                   help="Output file path (GIF or PNG, default: output.gif).")
    p.add_argument("--demo",       action="store_true",
                   help="Run demo with a synthetic cartoon face.")
    p.add_argument("--show-mesh",  action="store_true",
                   help="Save a static mesh image instead of animating.")
    p.add_argument("--visemes",    action="store_true",
                   help="Cycle through all 13 viseme shapes.")
    p.add_argument("--fps",        type=int, default=24,
                   help="Animation frame rate (default: 24).")
    p.add_argument("--phoneme-ms", type=int, default=120,
                   help="Milliseconds per phoneme (default: 120).")
    p.add_argument("--mediapipe",  metavar="MODEL_PATH", default=None,
                   help="Path to face_landmarker.task (enables MediaPipe).")
    p.add_argument("--no-label",   action="store_true",
                   help="Suppress control-point labels in mesh image.")
    return p.parse_args()


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def load_or_generate_face(args: argparse.Namespace):
    """Return (image_bgr, face_landmarks, mouth_open)."""
    gen = SyntheticFaceGenerator(width=480, height=480)

    if args.demo or args.image is None:
        print("[info] Generating synthetic cartoon face …")
        image, landmarks = gen.generate(mouth_open=0.0)
        return image, landmarks

    if not os.path.isfile(args.image):
        print(f"[error] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[error] Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Try MediaPipe first
    if args.mediapipe:
        mp_det = try_load_mediapipe_detector(args.mediapipe)
        if mp_det is not None:
            print("[info] Using MediaPipe FaceLandmarker …")
            lm = mp_det.detect(image)
            if lm is not None:
                return image, lm
            print("[warn] MediaPipe found no face; falling back to Haar cascade.")

    # Haar cascade fallback
    print("[info] Detecting face with Haar cascade …")
    detector = FaceLandmarkDetector()
    lm = detector.detect(image)
    if lm is None:
        print("[warn] No face detected; using whole image as face region.")
        lm = detector.detect(image, min_neighbors=1)
    return image, lm


def build_pipeline(image: np.ndarray, landmarks, args: argparse.Namespace):
    """Instantiate the full face-graph animation pipeline."""
    graph    = build_face_graph(landmarks)
    deformer = MeshDeformer(graph)
    model    = VisemeModel(face_scale=graph.face_scale)
    animator = SpeechAnimator(
        graph, model, deformer,
        fps=args.fps,
        phoneme_ms=args.phoneme_ms,
    )
    renderer = FaceRenderer(graph)
    return graph, deformer, model, animator, renderer


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  AnimationFace — AI Face Graph Speech Animator")
    print("=" * 60)

    # 1. Load / generate face + detect landmarks
    image, landmarks = load_or_generate_face(args)
    print(f"[info] Image size : {image.shape[1]}×{image.shape[0]} px")
    print(f"[info] Landmarks  : {len(landmarks.points)} points detected")

    # 2. Build pipeline
    graph, deformer, model, animator, renderer = build_pipeline(image, landmarks, args)
    print(f"[info] Face scale : {graph.face_scale:.1f} px (inter-eye distance)")
    print(f"[info] Mesh       : {len(graph.vertices)} vertices, "
          f"{len(graph.triangles)} triangles")
    print(f"[info] Handles    : {', '.join(graph.control_names)}")

    output = args.output

    # 3a. Static mesh image
    if args.show_mesh:
        out_path = output if output.endswith(".png") else os.path.splitext(output)[0] + ".png"
        renderer.save_static(image, output_path=out_path)
        print(f"[done] Static mesh saved → {out_path}")
        return

    # 3b. Viseme showcase
    if args.visemes:
        print("[info] Animating through all 13 viseme shapes …")
        frames = animator.animate_visemes(VISEME_NAMES)
        print(f"[info] Generated {len(frames)} frames")
        if output.endswith(".gif"):
            renderer.save_gif(image, frames, output, fps=args.fps, scale=0.8)
        else:
            renderer.save_mp4(image, frames, output, fps=args.fps)
        print(f"[done] Viseme animation saved → {output}")
        return

    # 3c. Text-driven speech animation
    text = args.text
    print(f"[info] Animating text: '{text}'")
    phonemes = text_to_phonemes(text)
    print(f"[info] Phonemes   : {' '.join(phonemes)}")
    frames, _ = animator.animate_text(text)
    print(f"[info] Generated  : {len(frames)} frames @ {args.fps} fps "
          f"({len(frames)/args.fps:.1f} s)")

    if output.endswith(".gif"):
        renderer.save_gif(image, frames, output, fps=args.fps, scale=0.8)
    elif output.endswith(".png"):
        # Save first deformed frame as PNG alongside static mesh
        renderer.save_static(image, frames[len(frames)//2], output_path=output)
    else:
        renderer.save_mp4(image, frames, output, fps=args.fps)

    # Always also save a static mesh PNG for quick inspection
    mesh_png = os.path.splitext(output)[0] + "_mesh.png"
    renderer.save_static(image, output_path=mesh_png)

    print(f"[done] Animation saved  → {output}")
    print(f"[done] Static mesh PNG  → {mesh_png}")


if __name__ == "__main__":
    main()
