#!/usr/bin/env python3
"""Figures for the runway-screening paper: three detection examples + the
system architecture. Outputs PNGs under reports/paper_figures/."""

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("reports/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
INK = "#1f2a33"

GREEN = ((40, 30, 25), (95, 255, 255))
WHITE = ((0, 0, 225), (180, 28, 255))
YELLOW = ((18, 60, 120), (38, 255, 255))


def hsv_mask(im, lo, hi):
    m = cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), lo, hi)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def panel(im, w=720):
    h = int(im.shape[0] * w / im.shape[1])
    im = cv2.resize(im, (w, h))
    bar = np.full((30, w, 3), 28, np.uint8)
    return np.vstack([bar, im]), bar  # caption bar is the top strip


def label(tile, text):
    cv2.putText(tile, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def side_by_side(left, right, name):
    h = max(left.shape[0], right.shape[0])
    def pad(t):
        return cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gap = np.full((h, 8, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / name), np.hstack([pad(left), gap, pad(right)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def crack_figure():
    src = cv2.imread("frames/frame_000004.jpg")
    ov = cv2.imread("reports/runway_defects_full/overlays/frame_000004.jpg")
    a, _ = panel(src); a = label(a, "(a) source frame")
    b, _ = panel(ov);  b = label(b, "(b) crack segmentation (red = crack)")
    side_by_side(a, b, "fig_crack.png")


def vegetation_figure():
    src = cv2.imread("frames/frame_000189.jpg")
    m = hsv_mask(src, *GREEN)
    ov = src.copy(); ov[m] = (0, 230, 0)
    ov = cv2.addWeighted(src, 0.45, ov, 0.55, 0)
    a, _ = panel(src); a = label(a, "(a) source frame")
    b, _ = panel(ov);  b = label(b, "(b) vegetation (green = grass in joints)")
    side_by_side(a, b, "fig_vegetation.png")


def marking_figure():
    src = cv2.imread("frames/frame_000188.jpg")
    m = hsv_mask(src, *WHITE) | hsv_mask(src, *YELLOW)
    ov = src.copy(); ov[m] = (255, 0, 255)
    ov = cv2.addWeighted(src, 0.45, ov, 0.55, 0)
    a, _ = panel(src); a = label(a, "(a) source frame")
    b, _ = panel(ov);  b = label(b, "(b) painted markings (magenta = bright paint)")
    side_by_side(a, b, "fig_marking.png")


def architecture_figure():
    fig = plt.figure(figsize=(11.5, 6.2)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.2",
                     lw=1.2, edgecolor="#34495e", facecolor=fc))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, sub, ha="center", va="top",
                fontsize=7.6, color="#3b4a55")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=13, lw=1.4, color="#7f8c8d"))

    ax.text(50, 54.5, "Figure 1.  Runway surface-assessment architecture",
            ha="center", fontsize=11, fontweight="bold", color=INK)

    # acquisition
    box(2, 23, 17, 11, "Low-altitude\nUAV survey",
        "single pass\n5 fps · ~1.25 mm/px\n274 frames", "#fdebd0")
    # three detection tracks
    box(27, 40, 26, 12, "Crack segmentation",
        "crack-specialised CNN\n→ instance masks\n→ skeleton, length, type", "#f5b7b1")
    box(27, 22.5, 26, 12, "Vegetation",
        "HSV colour separation\n(green vs grey asphalt)\n→ coverage", "#abebc6")
    box(27, 5, 26, 12, "Painted markings",
        "HSV white/yellow\nthreshold\n→ marking extent", "#fbe7a2")
    arrow(19, 28.5, 27, 46); arrow(19, 28.5, 27, 28.5); arrow(19, 28.5, 27, 11)

    # spatial referencing
    box(60, 31.5, 17, 12, "Spatial\nreferencing",
        "frame-to-frame\nregistration →\nchainage (m)", "#aed6f1")
    arrow(53, 46, 60, 40); arrow(53, 28.5, 60, 36); arrow(53, 11, 60, 33)

    # aggregation + reporting
    box(82, 28, 16, 13, "Condition\nsynthesis",
        "density · hot zones\ndistress inventory\nASTM D5340 frame", "#d7dbdd")
    arrow(77, 37, 82, 35)
    box(82, 9, 16, 11, "Inspection\nreport", "located, prioritised\nfindings", "#e8daef")
    arrow(90, 28, 90, 20)

    fig.savefig(OUT / "fig_architecture.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    crack_figure()
    vegetation_figure()
    marking_figure()
    architecture_figure()
    print("figures ->", sorted(p.name for p in OUT.glob("*.png")))
