"""A43 verification: read the arrays saved by --save_maps and answer (1) what fraction of explored cells carry wall-height
values and where, (2) which image rows and world heights the points behind those values come from, re-projecting the saved
depth with HAZARD's own PointCloud.get (utils/model.py:140-197), (3) what the mapper's height layer would be with a
navigation-stack clearance cut (only points below H_CLEAR count), and what HAZARD's planner weight does in each case.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 scripts/hazard_fixes/analyze_maps.py results/a43_maps_check/mm_kitchen_3a-1/maps
"""
import glob
import os
import sys

import numpy as np
import torch

from policy.astar import get_astar_weight  # HAZARD
from utils.model import PointCloud  # HAZARD's depth -> world points

H_CLEAR = 1.8  # the body's clearance: points above it are overhead, not obstacles (Navigation2 max_obstacle_height is 2.0)


def frame_points(z):
    """World points (3, H, W) from the saved depth and camera matrix, exactly as the mapper computes them."""
    depth = torch.from_numpy(np.asarray(z["depth"], dtype=np.float32))
    if depth.ndim == 3:
        depth = depth[0]
    cam = torch.from_numpy(np.asarray(z["camera_matrix"], dtype=np.float32))
    pts = PointCloud.get(depth=depth, camera_matrix=cam, vfov=120.0)
    return pts[0].numpy()  # (3, H, W): x, y (up), z


def cell_of(x, zc, offset, grid):
    return (int(x / grid + offset[0] + 0.5), int(zc / grid + offset[1] + 0.5))


def analyze(path):
    z = np.load(path, allow_pickle=True)
    ex, hg = z["explored"], z["height"]
    agent = np.argwhere(z["goal_map"] == -2)
    agent = tuple(agent[0]) if len(agent) else None
    seen = ex == 1
    hv = hg[seen]
    out = [f"== {os.path.basename(path)}: explored {int(seen.sum())} cells, agent cell {agent}, "
           f"agent y {float(z['agent_position'][1]):.2f} m" if "agent_position" in z else f"== {os.path.basename(path)}"]
    out.append(f"   explored cells by height: 0: {np.mean(hv <= 0.0):.2f} | 0-0.1: {np.mean((hv > 0) & (hv <= 0.1)):.2f} | "
               f"0.1-0.5: {np.mean((hv > 0.1) & (hv <= 0.5)):.2f} | 0.5-1.8: {np.mean((hv > 0.5) & (hv <= 1.8)):.2f} | "
               f">1.8: {np.mean(hv > 1.8):.2f} | max {hv.max():.2f}")
    if agent is not None:
        rr, cc = np.indices(hg.shape)
        d = np.hypot(rr - agent[0], cc - agent[1])
        prof = []
        for k in range(0, 30, 3):
            m = seen & (d >= k) & (d < k + 3)
            prof.append(f"{k / 10:.1f}m:{np.mean(hg[m] > 1.8):.2f}" if m.sum() else f"{k / 10:.1f}m:-")
        out.append("   fraction > 1.8 m by distance from the robot: " + " ".join(prof))
    if "depth" in z and "camera_matrix" in z and "map_offset" in z:
        pts = frame_points(z)
        X, Y, Z = pts[0], pts[1], pts[2]
        H, W = Y.shape
        offset, grid = z["map_offset"], float(z["grid_size"][0])
        rows = np.repeat(np.arange(H)[:, None], W, axis=1)
        keep = Y < 2.5  # the mapper's cut (model.py:279)
        tall = keep & (Y > H_CLEAR)
        out.append(f"   this frame's points: {int(keep.sum())} kept (Y < 2.5); of them {np.mean(Y[keep] > H_CLEAR):.2f} above {H_CLEAR} m; "
                   f"Y of those: median {np.median(Y[tall]):.2f}, 90th pct {np.percentile(Y[tall], 90):.2f}, max {Y[tall].max():.2f}; "
                   f"their image rows: median {np.median(rows[tall]):.0f} of {H} (0 = top)")
        # cells whose mapper height comes from a point above H_CLEAR: rebuild this frame's per-cell max with and without the cut
        xi = np.clip((X / grid + offset[0] + 0.5).astype(int), 0, hg.shape[0] - 1)
        zi = np.clip((Z / grid + offset[1] + 0.5).astype(int), 0, hg.shape[1] - 1)
        flat = xi * hg.shape[1] + zi
        maxY = np.full(hg.size, -1.0)
        np.maximum.at(maxY, flat[keep], Y[keep])
        maxY_clear = np.full(hg.size, -1.0)
        np.maximum.at(maxY_clear, flat[keep & (Y <= H_CLEAR)], Y[keep & (Y <= H_CLEAR)])
        touched = maxY >= 0
        n_touch = int(touched.sum())
        over = touched & (maxY > H_CLEAR)
        floor_under = over & (maxY_clear <= 0.1)  # cells that are plain floor once overhead points are dropped
        out.append(f"   cells touched this frame: {n_touch}; with max point > {H_CLEAR} m: {int(over.sum())} ({over.sum() / max(1, n_touch):.2f}); "
                   f"of those, floor (<= 0.1 m) without the overhead points: {int(floor_under.sum())} ({floor_under.sum() / max(1, over.sum()):.2f})")
        # planner weights: explored cells this frame, mapper height vs clearance-cut height
        hm = np.where(touched, np.maximum(maxY, 0), 0).reshape(hg.shape)
        hc = np.where(touched, np.maximum(maxY_clear, 0), 0).reshape(hg.shape)
        exf = touched.reshape(hg.shape).astype(np.int64)
        w_m = get_astar_weight({"explored": exf, "height": hm.astype(np.float32)}, (0, 0), (1, 1))
        w_c = get_astar_weight({"explored": exf, "height": hc.astype(np.float32)}, (0, 0), (1, 1))
        wm, wc = w_m[exf == 1], w_c[exf == 1]
        out.append(f"   planner weight on this frame's cells: mapper median {np.median(wm):.0f}, share >= 1000: {np.mean(wm >= 1000):.2f}; "
                   f"with the {H_CLEAR} m cut median {np.median(wc):.0f}, share >= 1000: {np.mean(wc >= 1000):.2f} (unexplored floor costs 30 x smoothing)")
    return "\n".join(out)


if __name__ == "__main__":
    folder = sys.argv[1]
    files = sorted(glob.glob(os.path.join(folder, "*.npz")), key=lambda p: int(p.split("_dec")[1].split("_")[0]))
    for f in files:
        print(analyze(f))
