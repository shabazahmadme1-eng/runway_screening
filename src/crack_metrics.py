#!/usr/bin/env python3
"""Engineering crack metrics from SAM2 masks: width, length, type, severity, PCI.

For each crack detection: prompt SAM2 with the box, skeletonise the mask, and
derive real-world measurements using the ground sample distance (GSD):
- length  = skeleton path length x GSD  (metres)
- width   = 2 x distance-transform at skeleton x GSD  (mm; mean & max)
- type    = longitudinal / transverse / diagonal (orientation vs runway axis)
            or alligator (interconnected network -> FOD risk)
- severity= ASTM D5340/D6433 width bands: low <3.2 mm, med 3.2-6.4, high >6.4

Rolls up an INDICATIVE surface condition score (0-100, PCI-style — not a
certified ASTM D5340 PCI, which needs the standard deduct-value curves) plus
a severity-weighted treatment summary.

Outputs crack_metrics.csv (per crack) and crack_metrics_summary.json.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from skimage.morphology import skeletonize
from ultralytics import SAM

# ASTM width severity thresholds (mm)
SEV_LOW, SEV_HIGH = 3.2, 6.4   # low <3.2, medium 3.2-6.4, high >6.4
SEV_WEIGHT = {"low": 2.0, "medium": 5.0, "high": 10.0}
PCI_A = 0.10  # deduct curve steepness per unit weighted-density %


def pci_class(p):
    return ("Good" if p >= 86 else "Satisfactory" if p >= 71 else
            "Fair" if p >= 56 else "Poor" if p >= 41 else
            "Very Poor" if p >= 26 else "Serious" if p >= 11 else "Failed")


def treatment(sev, alligator):
    if alligator or sev == "high":
        return ("Localised full-depth patch / structural investigation"
                + ("; alligator pattern = FOD risk, prioritise" if alligator else ""))
    if sev == "medium":
        return "Rout and seal to prevent water ingress"
    return "Crack sealing / routine monitoring"


def analyse_mask(mask_roi, gsd_mm):
    """mask_roi: bool ROI. -> dict of metrics or None if too small."""
    if mask_roi.sum() < 30:
        return None
    skel = skeletonize(mask_roi)
    n_skel = int(skel.sum())
    if n_skel < 3:
        return None
    edt = distance_transform_edt(mask_roi)
    widths_px = 2.0 * edt[skel]
    mean_w_mm = float(widths_px.mean()) * gsd_mm
    max_w_mm = float(widths_px.max()) * gsd_mm
    length_m = n_skel * gsd_mm / 1000.0

    # branch points (network detection) via 8-neighbour count
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = convolve(skel.astype(np.uint8), k, mode="constant")
    n_branch = int(((nb >= 3) & skel).sum())

    # orientation via PCA on skeleton coordinates
    ys, xs = np.nonzero(skel)
    angle_from_vert = 90.0
    elong = 1.0
    if len(xs) >= 5:
        pts = np.column_stack([ys - ys.mean(), xs - xs.mean()]).astype(float)
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, -1]  # principal (row, col)
        angle_from_vert = math.degrees(math.atan2(abs(v[1]), abs(v[0])))
        if evals[0] > 1e-6:
            elong = float((evals[-1] / max(evals[0], 1e-6)) ** 0.5)
    return dict(length_m=length_m, mean_w_mm=mean_w_mm, max_w_mm=max_w_mm,
                n_branch=n_branch, angle_from_vert=angle_from_vert, elong=elong,
                area_px=int(mask_roi.sum()))


def classify_type(m, runway_axis):
    if m["n_branch"] >= 5 and m["elong"] < 3.5:
        return "alligator"
    a = m["angle_from_vert"] if runway_axis == "vertical" else 90 - m["angle_from_vert"]
    if a < 35:
        return "longitudinal"
    if a > 55:
        return "transverse"
    return "diagonal"


def severity(max_w_mm):
    return "high" if max_w_mm > SEV_HIGH else "medium" if max_w_mm >= SEV_LOW else "low"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections", type=Path,
                    default=Path("reports/runway_defects_full/detections.csv"))
    ap.add_argument("--frames", type=Path, default=Path("frames"))
    ap.add_argument("--out", type=Path, default=Path("reports/runway_defects_full"))
    ap.add_argument("--model", default="sam2_t.pt")
    ap.add_argument("--gsd-mm", type=float, default=1.25,
                    help="ground sample distance, mm per pixel")
    ap.add_argument("--runway-axis", choices=["vertical", "horizontal"],
                    default="vertical", help="runway long-axis in the frames")
    ap.add_argument("--max-crack-width-mm", type=float, default=50.0,
                    help="masks wider than this are wide linear features "
                         "(pavement edges / joints / region fills), not cracks; "
                         "excluded from crack severity/width stats")
    args = ap.parse_args()

    boxes = defaultdict(list)
    for r in csv.DictReader(open(args.detections)):
        if r["class"] == "crack":
            boxes[r["frame"]].append([float(r["x1"]), float(r["y1"]),
                                      float(r["x2"]), float(r["y2"])])
    frames = sorted(boxes)
    sam = SAM(args.model)

    per_crack = []
    tot_surface = 0.0
    tot_weighted_area = 0.0
    cid = 0
    for i, fname in enumerate(frames, 1):
        im = cv2.imread(str(args.frames / fname))
        if im is None:
            continue
        H, W = im.shape[:2]
        tot_surface += H * W
        bb = boxes[fname]
        res = sam(str(args.frames / fname), bboxes=bb, verbose=False)[0]
        if res.masks is None:
            continue
        masks = res.masks.data.cpu().numpy()
        for j, (x1, y1, x2, y2) in enumerate(bb):
            if j >= len(masks):
                break
            mk = masks[j] > 0
            if mk.shape != (H, W):
                mk = cv2.resize(mk.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            pad = 12
            rx1, ry1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
            rx2, ry2 = min(W, int(x2) + pad), min(H, int(y2) + pad)
            roi = mk[ry1:ry2, rx1:rx2]
            m = analyse_mask(roi, args.gsd_mm)
            if m is None:
                continue
            ctype = classify_type(m, args.runway_axis)
            sev = severity(m["max_w_mm"])
            allig = ctype == "alligator"
            if allig and sev == "low":
                sev = "medium"  # interconnected network is at least medium
            fod = sev == "high" or (allig and sev != "low")
            # a real crack is thin; wider masks are edges/joints/region fills
            plausible = m["mean_w_mm"] <= args.max_crack_width_mm
            cid += 1
            per_crack.append(dict(
                crack_id=cid, frame=fname, plausible_crack=plausible,
                type=ctype, severity=sev, fod_risk=fod and plausible,
                length_m=round(m["length_m"], 3),
                mean_width_mm=round(m["mean_w_mm"], 2),
                max_width_mm=round(m["max_w_mm"], 2),
                cx=int((x1 + x2) / 2), cy=int((y1 + y2) / 2),
                area_px=m["area_px"]))
            if plausible:
                tot_weighted_area += SEV_WEIGHT[sev] * m["area_px"]
        if i % 25 == 0 or i == len(frames):
            print(f"  [{i}/{len(frames)}] {cid} cracks measured", flush=True)

    # ----- roll-ups (over plausible cracks only; wide features reported apart)
    cracks = [c for c in per_crack if c["plausible_crack"]]
    wide_features = len(per_crack) - len(cracks)
    weighted_density = 100 * tot_weighted_area / tot_surface if tot_surface else 0
    deduct = 100 * (1 - math.exp(-PCI_A * weighted_density))
    pci = round(100 - deduct)
    by_sev = Counter(c["severity"] for c in cracks)
    by_type = Counter(c["type"] for c in cracks)
    total_len = sum(c["length_m"] for c in cracks)
    max_w = max((c["max_width_mm"] for c in cracks), default=0)
    fod_n = sum(1 for c in cracks if c["fod_risk"])
    all_w = sorted(c["mean_width_mm"] for c in per_crack)

    def pct(p):
        return all_w[min(len(all_w) - 1, int(p * len(all_w)))] if all_w else 0

    fields = ["crack_id", "frame", "plausible_crack", "type", "severity",
              "fod_risk", "length_m", "mean_width_mm", "max_width_mm",
              "cx", "cy", "area_px"]
    with open(args.out / "crack_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(per_crack)

    summary = dict(
        n_detections=len(per_crack), n_cracks=len(cracks),
        wide_features_excluded=wide_features, gsd_mm=args.gsd_mm,
        max_crack_width_mm_cap=args.max_crack_width_mm,
        total_crack_length_m=round(total_len, 1),
        max_crack_width_mm=round(max_w, 2),
        width_mm_p50=pct(0.5), width_mm_p90=pct(0.9),
        severity_counts=dict(by_sev), type_counts=dict(by_type),
        fod_risk_cracks=fod_n,
        condition_score_indicative=pci, condition_class=pci_class(pci),
        weighted_density_pct=round(weighted_density, 3),
        treatments={
            "high/alligator": treatment("high", True),
            "medium": treatment("medium", False),
            "low": treatment("low", False)})
    (args.out / "crack_metrics_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Crack engineering metrics (GSD {:.2f} mm/px) ===".format(args.gsd_mm))
    print(f"detections: {len(per_crack)} | plausible cracks: {len(cracks)} | "
          f"wide features excluded: {wide_features}")
    print(f"mean-width mm  p50={pct(0.5):.1f}  p90={pct(0.9):.1f}  "
          f"(cap {args.max_crack_width_mm:.0f})")
    print(f"plausible cracks: total length {total_len:.1f} m | max width {max_w:.1f} mm")
    print(f"severity: {dict(by_sev)}")
    print(f"type:     {dict(by_type)}")
    print(f"FOD-risk cracks: {fod_n}")
    print(f"INDICATIVE condition score: {pci}/100 ({pci_class(pci)})  "
          f"[not a certified ASTM D5340 PCI]")
    print(f"-> {args.out}/crack_metrics.csv, crack_metrics_summary.json")


if __name__ == "__main__":
    main()
