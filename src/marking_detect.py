#!/usr/bin/env python3
"""Line-based painted-marking detector.

Colour alone confuses sunlit grass with faded paint. This isolates markings by
what actually makes them markings: they sit on the grey asphalt, they are
brighter than the asphalt around them, and they form long thin lines rather
than blobs. Run directly to write a quick visual check to /tmp.
"""

import cv2
import numpy as np


def detect_markings(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # bright paint: white (low saturation AND much brighter than grey asphalt)
    # OR warm yellow (saturated hue, which grey asphalt never is)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 45, 255))
    yellow = cv2.inRange(hsv, (14, 28, 135), (42, 255, 255))
    paint = cv2.bitwise_or(white, yellow)

    # Dried grass and faded cream paint share almost the same hue, so a plain
    # colour subtraction would also erase the paint. Instead exclude only the
    # large green REGIONS (the verge) — grass is a broad blob, paint is a thin
    # line on asphalt away from it.
    green = cv2.inRange(hsv, (26, 18, 25), (95, 255, 255))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    verge = np.zeros_like(green)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 15000:
            verge[lab == i] = 255
    verge = cv2.dilate(verge, np.ones((21, 21), np.uint8))
    paint = cv2.bitwise_and(paint, cv2.bitwise_not(verge))

    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # bridge dash gaps so a broken line reads as one component
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # markings are long AND have real width; grooves are hairline-thin, grass is
    # blobby — keep long, elongated, non-hairline components
    out = np.zeros_like(paint)
    cnts, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        (w, h) = cv2.minAreaRect(c)[1]
        lo, hi = min(w, h), max(w, h)
        if lo < 1:
            continue
        if hi / lo >= 3.0 and hi >= 55 and lo >= 5:
            cv2.drawContours(out, [c], -1, 1, -1)
    return out > 0


def _test():
    tiles = []
    for f in ["frame_000188.jpg", "frame_000006.jpg", "frame_000001.jpg"]:
        im = cv2.imread(f"frames/{f}")
        m = detect_markings(im)
        ov = im.copy(); ov[m] = (255, 0, 255)
        ov = cv2.addWeighted(im, 0.5, ov, 0.5, 0)
        s = 380 / im.shape[1]
        ov = cv2.resize(ov, (380, int(im.shape[0] * s)))
        bar = np.full((22, ov.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f"{f}  {int(m.sum())}px", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, ov]))
        print(f, "marking px:", int(m.sum()), flush=True)
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 5, cv2.BORDER_CONSTANT,
             value=(255, 255, 255)) for t in tiles]
    cv2.imwrite("/tmp/markings_v2.jpg", np.hstack(tiles))
    print("wrote /tmp/markings_v2.jpg", flush=True)


if __name__ == "__main__":
    _test()
