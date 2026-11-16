"""Render HAZARD's explored map with A* cost colouring for chosen decision frames.
Height is reconstructed from height_map_<f>.png and sem_map_<f>.png (visualize_obs, env_actions.py:40-83):
  sem_map grey = 200 - 200*min(h/2,1) for explored cells with h>0.3  -> h = (200-grey)/100 when 0.3<h<2
  height_map  = h * int(255/hmax), cast to uint8  -> k = px/h on cells where h is known from sem_map
Weights follow get_astar_weight, policy/astar.py:36-58 exactly.
"""
import os, sys, math, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = r"G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\0"
OUTDIR = r"G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1"
OFF = 256; G = 0.1
TARGETS = {  # runtime id: (name, x, z, value) from scene log.txt add_object order (+1), see ANALYSIS.md
    56: ("bag_3 (14901620) value 5", -1.427, -2.454, 5),
    57: ("b04_comb (8265913) value 1", -0.677, -2.204, 1),
    62: ("small_purse (3819180) value 5", 0.323, 1.546, 5),
    64: ("b03_hair_comb_2010 (8657745) value 1", 0.823, 2.796, 1),
    66: ("b04_comb (15599511) value 1", -1.427, 3.296, 1),
}
AGENT_SPAWN = (-0.25, 0.75)
FIRES = [(0.5, 2.25), (-0.25, 2.5), (0.5, 1.5)]
ROOM = (-1.576 - 0.3, 1.856 + 0.3, -3.204 - 0.3, 3.296 + 0.3)  # xmin,xmax,zmin,zmax: object bbox padded 0.3 m

def real_to_grid(x, z):  # utils/model.py:224-228
    return int(x / G + OFF + 0.5), int(z / G + OFF + 0.5)

def astar_weight(explored, height):  # policy/astar.py:36-58
    w, h = explored.shape
    weight = np.ones((w, h))
    weight[explored == 0] = 30
    weight[height > 0.5] += np.exp(height[height > 0.5] * 3) * 7
    weight[height > 0.1] += 10
    conv = np.zeros((w, h))
    for i in range(-2, 3):
        for j in range(-2, 3):
            conv[max(0, i): min(w, w + i), max(0, j): min(h, h + j)] += weight[max(0, -i): min(w, w - i), max(0, -j): min(h, h - j)] * 1.0 / (abs(i) + abs(j) + 1)
    return conv.astype(np.float32)

def load(f):
    sem = np.array(Image.open(os.path.join(D, f"sem_map_{f}.png")).convert("RGB")).astype(float)
    exp = (np.array(Image.open(os.path.join(D, f"explore_{f}.png")).convert("L")) > 127).astype(np.int64)
    hpx = np.array(Image.open(os.path.join(D, f"height_map_{f}.png")).convert("L")).astype(float)
    red = (sem[:, :, 0] == 255) & (sem[:, :, 1] == 0) & (sem[:, :, 2] == 0)
    grey = sem[:, :, 1].copy()  # G channel; red block excluded below
    # explored & h>0.3 cells have grey in (0,200]; h = (200-grey)/100 valid while h<2 (grey>0)
    known = (exp == 1) & (~red) & (grey < 200) & (grey > 0)
    h_known = (200 - grey[known]) / 100.0
    px_known = hpx[known]
    ok = (h_known > 0.35) & (h_known < 1.9) & (px_known > 0) & (px_known < 255)
    k = np.median(px_known[ok] / h_known[ok]) if ok.sum() else float("nan")
    height = hpx / k
    # saturated pixels: at least 255/k
    sat = hpx >= 255
    # cells the sem_map says are >=2 m (grey==0 and explored): keep hpx/k (already large)
    return sem, exp, hpx, height, k, red, sat

def render(f, title, path_note=""):
    sem, exp, hpx, height, k, red, sat = load(f)
    W = astar_weight(exp, height)
    # colour: log10 weight, 0 (=1) .. 4
    L = np.log10(np.maximum(W, 1.0))
    L = np.clip(L / 4.0, 0, 1)
    img = np.zeros(L.shape + (3,), dtype=np.uint8)
    # unexplored (explored==0): light grey; explored: viridis-ish from green (cheap) to red (expensive)
    cheap = np.array([40, 160, 60]); mid = np.array([240, 200, 40]); exp_ = np.array([180, 30, 30])
    t = L[..., None]
    col = np.where(t < 0.5, cheap + (mid - cheap) * (t / 0.5), mid + (exp_ - mid) * ((t - 0.5) / 0.5))
    img[:] = col.astype(np.uint8)
    img[exp == 0] = [225, 225, 225]
    # very high weight (wall-like) in explored: dark
    img[(exp == 1) & (W > 5000)] = [20, 20, 20]
    scale = 6
    # orient: image x = world x (grid row), image y = world z upwards (grid col, flipped)
    img = np.flipud(img.transpose(1, 0, 2))
    im = Image.fromarray(np.ascontiguousarray(img)).resize((512 * scale, 512 * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    def P(x, z):
        gx, gz = real_to_grid(x, z)
        return gx * scale + scale // 2, (511 - gz) * scale + scale // 2
    # room bbox
    x0, x1, z0, z1 = ROOM
    dr.rectangle([P(x0, z1), P(x1, z0)], outline=(0, 90, 200), width=2)
    for fx, fz in FIRES:
        cx, cy = P(fx, fz); dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(255, 120, 0), width=2)
    for tid, (name, tx, tz, v) in TARGETS.items():
        cx, cy = P(tx, tz)
        dr.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], outline=(0, 0, 255), width=2)
        dr.text((cx + 7, cy - 6), f"{tid}", fill=(0, 0, 255))
    rr = np.argwhere(red)
    path_info = ""
    if len(rr):
        import pyastar2d
        o = (int(round(rr[:, 0].mean())), int(round(rr[:, 1].mean())))
        tx, tz = TARGETS[56][1], TARGETS[56][2]
        dgoal = real_to_grid(tx, tz)
        path = pyastar2d.astar_path(W, o, dgoal, allow_diagonal=False)
        if path is not None:
            cost = float(sum(W[p[0], p[1]] for p in path[1:]))
            outside = sum(1 for p in path if not (ROOM[0] <= (p[0]-OFF)*G <= ROOM[1] and ROOM[2] <= (p[1]-OFF)*G <= ROOM[3]))
            path_info = f"A* path robot->56 on reconstructed map: {len(path)} cells, cost {cost:.0f}, {outside} cells outside room bbox"
            pts = [(p[0] * scale + scale // 2, (511 - p[1]) * scale + scale // 2) for p in path]
            dr.line(pts, fill=(0, 220, 255), width=3)
        else:
            path_info = "A* found no path"
        print("  ", path_info)
    if len(rr):
        cx, cy = rr[:, 0].mean() * scale + scale // 2, (511 - rr[:, 1].mean()) * scale + scale // 2
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(255, 0, 0))
        dr.text((cx + 9, cy - 6), "robot", fill=(255, 0, 0))
    sx, sy = P(*AGENT_SPAWN); dr.text((sx + 4, sy), "spawn", fill=(90, 90, 90))
    dr.text((10, 10), title, fill=(0, 0, 0))
    dr.text((10, 26), "colour: log10 A* weight (astar.py:36-58): green=1 (explored free), yellow~30 (unexplored), red/black=obstacle; light grey=unexplored; blue box=room bbox (objects+0.3m); blue squares=targets (runtime id); orange=initial fires", fill=(0, 0, 0))
    dr.text((10, 42), f"height scale k=int(255/hmax)={k:.0f} px/m -> hmax~{255/k:.2f} m; saturated cells={int(sat.sum())}; axes: image x = world x (right), image y = world z (up); grid offset 256, 0.1 m/cell", fill=(0, 0, 0))
    dr.text((10, 58), "cyan line = A* path (pyastar2d, 4-connected as in astar.py:32) from robot to bag 56 on the reconstructed weights; " + path_info, fill=(0, 0, 0))
    # crop to region of interest: world x in [-4,5], z in [-5,6]
    (cx0, cy0), (cx1, cy1) = P(-4.0, -5.0), P(5.5, 6.5)
    im = im.crop((min(cx0, cx1), min(cy0, cy1), max(cx0, cx1), max(cy0, cy1)))
    out = os.path.join(OUTDIR, f"astar_cost_frame{f}.png")
    im.save(out)
    print("saved", out, "k", k, "hmax", 255 / k, "explored", int(exp.sum()), "robot", rr.mean(0) if len(rr) else None)
    # weight stats along a straight line from robot to target 56, and on robot cell
    if len(rr):
        r0, r1 = int(rr[:, 0].mean()), int(rr[:, 1].mean())
        print("  weight at robot cell", W[r0, r1], "explored there", exp[r0, r1], "height there", height[r0, r1])
    return W, exp, height

if __name__ == "__main__":
    for f, t in [(703, "Decision 11 (frame 703): walk_to 56 begins; this walk leaves the house"),
                 (809, "Decision 12 (frame 809): robot already outside north wall"),
                 (1458, "Decision 18 (frame 1458): last decision; robot outside east wall")]:
        render(f, t)
