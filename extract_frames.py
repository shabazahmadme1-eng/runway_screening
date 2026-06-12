import os
import cv2

VIDEO = r"C:\Users\shaba\runway_screening\source\DJI_20240131092024_0006_D.MP4"
OUTDIR = r"C:\Users\shaba\runway_screening\frames"
TARGET_FPS = 5.0
JPEG_QUALITY = 90

os.makedirs(OUTDIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("ERROR: could not open video")

native_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / native_fps if native_fps > 0 else 0

print(f"native_fps={native_fps:.6f} total_frames={total_frames} "
      f"resolution={width}x{height} duration={duration:.3f}s")

# Emulate ffmpeg's fps filter: emit the first source frame whose timestamp
# reaches each 5 FPS grid point.
saved = 0
src_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = src_idx / native_fps
    if t >= saved / TARGET_FPS:
        saved += 1
        path = os.path.join(OUTDIR, f"frame_{saved:06d}.jpg")
        if not cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise SystemExit(f"ERROR: failed to write {path}")
    src_idx += 1

cap.release()
print(f"saved={saved} frames read={src_idx}")

manifest = os.path.join(OUTDIR, "manifest.txt")
with open(manifest, "w") as f:
    f.write("Runway screening training frames\n")
    f.write("================================\n")
    f.write(f"Source video      : {os.path.basename(VIDEO)}\n")
    f.write(f"Video duration    : {duration:.3f} s\n")
    f.write(f"Video native FPS  : {native_fps:.3f}\n")
    f.write(f"Resolution        : {width}x{height} (original, no downscale)\n")
    f.write(f"Extraction FPS    : {TARGET_FPS:g}\n")
    f.write(f"JPEG quality      : {JPEG_QUALITY}\n")
    f.write(f"Frame count       : {saved}\n")
    f.write("Naming            : frame_000001.jpg .. "
            f"frame_{saved:06d}.jpg\n")
print(f"manifest written to {manifest}")
