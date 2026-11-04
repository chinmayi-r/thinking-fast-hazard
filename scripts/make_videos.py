"""Turn HAZARD's recorded frames (submit(record_with_agents=True) -> <run>/images/..., TDW ImageCapture) into mp4s
with ffmpeg (C:\\ffmpeg\\bin\\ffmpeg.exe), one per recorded run, into results/videos/. Frames are found recursively,
sorted by frame number, copied to a numbered temp sequence (Windows ffmpeg has no glob input), encoded at 30 fps.
"""
import glob
import re
import os
import shutil
import subprocess
import tempfile

ROOT = r"G:\iw-thinking-fast"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
OUT = os.path.join(ROOT, "results", "videos")
os.makedirs(OUT, exist_ok=True)

runs = sorted(d for d in glob.glob(os.path.join(ROOT, "results", "fire_*_ep1_*")) if os.path.isdir(os.path.join(d, "images")))
if not runs:
    print("no recorded runs (no <run>/images folder) found")
for run in runs:
    # HAZARD saves frames as img_<comm_counter>.jpg with no zero padding (fireagent_controller.py:473), so a plain
    # name sort puts img_1000 right after img_100 and interleaves early and late frames (found 2026-10-28 when Tom
    # saw the agent "appear in the fire"). Sort by the integer in the name instead.
    def frame_key(path):
        m = re.search(r"(\d+)", os.path.basename(path))
        return (int(m.group(1)) if m else -1, path)
    frames = sorted((p for p in glob.glob(os.path.join(run, "images", "**", "*"), recursive=True)
                     if p.lower().endswith((".jpg", ".png"))), key=frame_key)
    name = os.path.basename(run)
    if len(frames) < 10:
        print(f"{name}: only {len(frames)} frames, skipped")
        continue
    ext = os.path.splitext(frames[0])[1].lower()
    tmp = tempfile.mkdtemp(prefix="vid_", dir=os.path.join(ROOT, ".cache", "tmp"))
    for i, f in enumerate(frames):
        shutil.copyfile(f, os.path.join(tmp, f"f_{i:06d}{ext}"))
    out = os.path.join(OUT, f"{name}.mp4")
    cmd = [FFMPEG, "-y", "-framerate", "30", "-i", os.path.join(tmp, f"f_%06d{ext}"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    size = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"{name}: {len(frames)} frames -> {out} ({size:.1f} MB) rc={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-800:])
