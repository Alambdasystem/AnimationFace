"""
Face mesh renderer.

Draws the Delaunay mesh overlay on a face image, highlighting:
  - Mesh triangles         (semi-transparent green lines)
  - Animation handles      (cyan circles)
  - Anchor points          (yellow circles)
  - Displacement arrows    (magenta arrows, optional)

Can save a static PNG or an animated GIF / MP4.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image

from face_graph.mesh_graph import FaceMeshGraph


# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV calls, RGB for matplotlib)
# ---------------------------------------------------------------------------
_MESH_BGR    = (0, 200, 80)      # green
_CTRL_BGR    = (255, 200, 0)     # cyan (control handles)
_ANCH_BGR    = (0, 200, 255)     # yellow (anchors)
_ARROW_BGR   = (180, 0, 220)     # magenta (displacement)
_MESH_RGB    = (0.10, 0.78, 0.31)
_CTRL_RGB    = (1.00, 0.78, 0.00)
_ANCH_RGB    = (0.00, 0.78, 1.00)


def warp_face(
    src_image: np.ndarray,
    src_vertices: np.ndarray,
    dst_vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Warp a face image by triangle mesh morphing.

    For each triangle in the mesh, compute the affine transform from the
    source (rest) position to the destination (deformed) position and copy
    the warped pixels into the output image.

    Args:
        src_image:    BGR uint8 source (rest) face image.
        src_vertices: (N, 2) source vertex positions in pixels.
        dst_vertices: (N, 2) destination (deformed) vertex positions.
        triangles:    (M, 3) triangle vertex indices.

    Returns:
        Warped BGR uint8 image the same size as *src_image*.
    """
    h, w = src_image.shape[:2]
    output = src_image.copy()   # start with original; untouched areas show through

    for tri_idx in triangles:
        src_tri = src_vertices[tri_idx].astype(np.float32)
        dst_tri = dst_vertices[tri_idx].astype(np.float32)

        # Bounding rect of source and destination triangles
        r_src = cv2.boundingRect(src_tri.reshape(1, 3, 2).astype(np.int32))
        r_dst = cv2.boundingRect(dst_tri.reshape(1, 3, 2).astype(np.int32))
        rx_s, ry_s, rw_s, rh_s = r_src
        rx_d, ry_d, rw_d, rh_d = r_dst
        if rw_s <= 0 or rh_s <= 0 or rw_d <= 0 or rh_d <= 0:
            continue

        src_crop_local = src_tri - np.array([rx_s, ry_s], dtype=np.float32)
        dst_crop_local = dst_tri - np.array([rx_d, ry_d], dtype=np.float32)

        # Affine transform: src local → dst local
        M = cv2.getAffineTransform(src_crop_local, dst_crop_local)

        # Clamp to image bounds
        rx_s = max(0, rx_s); ry_s = max(0, ry_s)
        rw_s = min(rw_s, w - rx_s); rh_s = min(rh_s, h - ry_s)
        rx_d = max(0, rx_d); ry_d = max(0, ry_d)
        rw_d = min(rw_d, w - rx_d); rh_d = min(rh_d, h - ry_d)
        if rw_s <= 0 or rh_s <= 0 or rw_d <= 0 or rh_d <= 0:
            continue

        src_patch = src_image[ry_s:ry_s+rh_s, rx_s:rx_s+rw_s]
        try:
            warped = cv2.warpAffine(
                src_patch, M, (rw_d, rh_d),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        except cv2.error:
            continue

        # Mask using the destination triangle polygon
        mask = np.zeros((rh_d, rw_d), dtype=np.uint8)
        cv2.fillConvexPoly(mask,
                           dst_crop_local.reshape(1, 3, 2).astype(np.int32),
                           255)
        alpha_m = mask[:rh_d, :rw_d, np.newaxis].astype(np.float32) / 255.0
        roi = output[ry_d:ry_d+rh_d, rx_d:rx_d+rw_d]
        roi[:] = (warped[:rh_d, :rw_d] * alpha_m
                  + roi * (1.0 - alpha_m)).astype(np.uint8)

    return output


class FaceRenderer:
    """Render face mesh overlays and export animations."""

    def __init__(
        self,
        graph: FaceMeshGraph,
        show_anchors: bool = True,
        show_handles: bool = True,
        mesh_alpha: float = 0.55,
        handle_radius: int = 4,
        warp_image: bool = True,
    ):
        """
        Args:
            warp_image: If True, warp the face pixels with the mesh deformation
                        (stores the first image passed to ``draw_mesh`` as the
                        reference and uses it for subsequent warped frames).
        """
        self._graph   = graph
        self._show_an = show_anchors
        self._show_h  = show_handles
        self._alpha   = mesh_alpha
        self._hrad    = handle_radius
        self._warp    = warp_image
        self._base_image: Optional[np.ndarray] = None

    def set_base(self, image: np.ndarray) -> None:
        """Explicitly set the reference (rest-pose) face image for warping."""
        self._base_image = image.copy()

    # ------------------------------------------------------------------
    # Static frame rendering
    # ------------------------------------------------------------------

    def draw_mesh(
        self,
        image: np.ndarray,
        vertices: Optional[np.ndarray] = None,
        displacements: Optional[np.ndarray] = None,
        show_arrows: bool = False,
    ) -> np.ndarray:
        """Draw the face mesh overlay onto *image* (BGR uint8).

        When ``warp_image=True`` and *vertices* differ from the base,
        the face image pixels are triangle-warped to follow the mesh
        deformation before the overlay is drawn.

        Args:
            image:         Background face image (BGR uint8).
            vertices:      (72, 2) vertex positions; uses graph base if None.
            displacements: (11, 2) control-point displacements for arrows.
            show_arrows:   Draw displacement arrows on handles.

        Returns:
            Annotated BGR image (copy).
        """
        verts = vertices if vertices is not None else self._graph.vertices

        # Optionally warp the background image to follow the mesh deformation
        if self._warp and vertices is not None:
            if self._base_image is None:
                self._base_image = image.copy()
            warped = warp_face(
                self._base_image,
                self._graph.vertices,
                verts,
                self._graph.triangles,
            )
            # Blend warped with original to soften boundary artefacts
            out = cv2.addWeighted(warped, 0.85, image, 0.15, 0)
        else:
            out = image.copy()
            if self._base_image is None:
                self._base_image = image.copy()

        overlay = out.copy()

        # Triangulation edges
        for tri in self._graph.triangles:
            pts = verts[tri].astype(np.int32)
            cv2.polylines(overlay, [pts], True, _MESH_BGR, 1, cv2.LINE_AA)

        # Anchor points
        if self._show_an:
            for idx in self._graph.anchor_indices:
                pt = tuple(verts[idx].astype(np.int32))
                cv2.circle(out, pt, self._hrad, _ANCH_BGR, -1, cv2.LINE_AA)

        # Control-point handles
        if self._show_h:
            ctrl_verts = verts[self._graph.control_indices]
            for i, pt in enumerate(ctrl_verts.astype(np.int32)):
                cv2.circle(out, tuple(pt), self._hrad + 1, _CTRL_BGR, -1, cv2.LINE_AA)

        # Displacement arrows
        if show_arrows and displacements is not None:
            base_ctrl = self._graph.vertices[self._graph.control_indices]
            for base, disp in zip(base_ctrl, displacements):
                tip = base + disp
                cv2.arrowedLine(out,
                                tuple(base.astype(np.int32)),
                                tuple(tip.astype(np.int32)),
                                _ARROW_BGR, 2, cv2.LINE_AA, tipLength=0.25)

        # Blend overlay
        cv2.addWeighted(overlay, self._alpha, out, 1.0 - self._alpha, 0, out)
        return out

    # ------------------------------------------------------------------
    def draw_mesh_matplotlib(
        self,
        image: np.ndarray,
        vertices: Optional[np.ndarray] = None,
        title: str = "Face Mesh Graph",
        figsize: Tuple[int, int] = (6, 6),
    ) -> plt.Figure:
        """Return a matplotlib Figure with a high-quality mesh overlay."""
        verts = vertices if vertices is not None else self._graph.vertices
        h, w  = image.shape[:2]

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)   # y-axis flipped to match image convention
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=8)

        # Triangulation
        tri = mtri.Triangulation(
            verts[:, 0], verts[:, 1],
            triangles=self._graph.triangles,
        )
        ax.triplot(tri, color=_MESH_RGB, lw=0.7, alpha=0.75)

        # Anchor points
        if self._show_an:
            ap = verts[self._graph.anchor_indices]
            ax.scatter(ap[:, 0], ap[:, 1], c=[_ANCH_RGB],
                       s=30, zorder=4, label="Anchor")

        # Control-point handles
        if self._show_h:
            cp = verts[self._graph.control_indices]
            ax.scatter(cp[:, 0], cp[:, 1], c=[_CTRL_RGB],
                       s=50, zorder=5, marker="D", label="Handle")
            # Label handles
            for name, idx in zip(self._graph.control_names,
                                  self._graph.control_indices):
                pt = verts[idx]
                ax.annotate(
                    name.replace("_", "\n"),
                    xy=(pt[0], pt[1]),
                    xytext=(pt[0] + 6, pt[1] - 6),
                    fontsize=5,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="black", alpha=0.5),
                )

        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Animation export
    # ------------------------------------------------------------------

    def save_gif(
        self,
        image: np.ndarray,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 24,
        scale: float = 1.0,
    ) -> None:
        """Export animation frames as an animated GIF.

        Args:
            image:       Background face image (BGR).
            frames:      List of (72, 2) vertex arrays.
            output_path: File path (e.g. ``"output.gif"``).
            fps:         Frames per second.
            scale:       Resize scale factor (< 1 for smaller GIFs).
        """
        pil_frames: List[Image.Image] = []
        for verts in frames:
            frame = self.draw_mesh(image, verts)
            if scale != 1.0:
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        delay_ms = max(20, int(1000 / fps))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=delay_ms,
            loop=0,
        )
        print(f"[renderer] Saved {len(frames)}-frame GIF → {output_path}")

    # ------------------------------------------------------------------
    def save_mp4(
        self,
        image: np.ndarray,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 24,
    ) -> None:
        """Export animation frames as an MP4 video (requires OpenCV FFMPEG)."""
        h, w = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for verts in frames:
            frame = self.draw_mesh(image, verts)
            writer.write(frame)
        writer.release()
        print(f"[renderer] Saved {len(frames)}-frame MP4 → {output_path}")

    # ------------------------------------------------------------------
    def save_static(
        self,
        image: np.ndarray,
        vertices: Optional[np.ndarray] = None,
        output_path: str = "face_mesh.png",
        use_matplotlib: bool = True,
    ) -> None:
        """Save a single annotated frame as PNG.

        Args:
            use_matplotlib: If True, use the high-quality matplotlib renderer;
                            otherwise use the OpenCV renderer.
        """
        if use_matplotlib:
            fig = self.draw_mesh_matplotlib(
                image, vertices,
                title="AnimationFace — Face Graph Grid",
            )
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            out = self.draw_mesh(image, vertices)
            cv2.imwrite(output_path, out)
        print(f"[renderer] Saved static mesh → {output_path}")

    # ------------------------------------------------------------------
    def matplotlib_animation(
        self,
        image: np.ndarray,
        frames: List[np.ndarray],
        output_path: Optional[str] = None,
        fps: int = 24,
        figsize: Tuple[int, int] = (6, 6),
    ) -> FuncAnimation:
        """Create a matplotlib FuncAnimation from vertex frames.

        Args:
            output_path: If given, save to this path (GIF or MP4).

        Returns:
            The ``FuncAnimation`` object.
        """
        h, w = image.shape[:2]
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        ax.axis("off")

        bg_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_artist  = ax.imshow(bg_rgb, extent=[0, w, h, 0])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        tri_obj = mtri.Triangulation(
            frames[0][:, 0], frames[0][:, 1],
            triangles=self._graph.triangles,
        )
        triplot_artists = ax.triplot(tri_obj, color=_MESH_RGB, lw=0.7, alpha=0.8)
        scat_h = ax.scatter(
            frames[0][self._graph.control_indices, 0],
            frames[0][self._graph.control_indices, 1],
            c=[_CTRL_RGB], s=40, zorder=5, marker="D",
        )

        def _update(frame_idx: int):
            verts = frames[frame_idx]
            # Update triangulation lines
            for artist in triplot_artists:
                artist.set_data(verts[:, 0], verts[:, 1])
            scat_h.set_offsets(verts[self._graph.control_indices])
            return triplot_artists + [scat_h]

        interval_ms = int(1000 / fps)
        anim = FuncAnimation(
            fig, _update,
            frames=len(frames),
            interval=interval_ms,
            blit=True,
        )
        if output_path:
            if output_path.endswith(".gif"):
                anim.save(output_path, writer="pillow", fps=fps)
            else:
                anim.save(output_path, writer="ffmpeg", fps=fps)
            print(f"[renderer] Saved matplotlib animation → {output_path}")
        plt.close(fig)
        return anim
