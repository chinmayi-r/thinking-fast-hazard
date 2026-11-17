"""The robot's own map as a picture (A41 follow-up, CR Nov 17: "keep a square of what we have explored"). Drawn from the
same arrays the program API and the honest text use (sem_map explored/height/id, goal_map), nothing from the scene file:
unexplored floor dark, explored floor light, low obstacles (0.1-0.5 m) tan, tall obstacles (> 0.5 m) near-black,
seen objects grey, targets red with their id, the robot green with a heading tick, planned paths blue, a 1 m bar.
Image x = map column (world z), image y = map row (world x), so the body's heading 90 degrees points right.
For the human page (beside the camera frame) and, later, for a model that takes images (Jev, the VLM agent)."""
import numpy as np

CELL_M = 0.1

COL_UNSEEN = (44, 46, 54)
COL_FLOOR = (226, 224, 214)
COL_LOW = (196, 160, 110)
COL_TALL = (28, 22, 20)
COL_OBJ = (120, 130, 150)
COL_TARGET = (214, 48, 40)
COL_ROBOT = (40, 170, 90)
COL_PATH = (50, 110, 230)
COL_PATH_UNSEEN = (240, 150, 30)
COL_PATH_BLOCKED = (230, 40, 200)
COL_TEXT = (20, 20, 20)


def crop_box(explored, agent=None, margin_cells=8):
    """Rows/cols spanned by the explored cells (plus the robot and a margin); the whole map when nothing is explored."""
    h, w = explored.shape
    pts = np.argwhere(explored == 1)
    if len(pts) == 0:
        return 0, h, 0, w
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    if agent is not None:
        r0, c0 = min(r0, agent[0]), min(c0, agent[1])
        r1, c1 = max(r1, agent[0] + 1), max(c1, agent[1] + 1)
    r0, c0 = max(0, int(r0) - margin_cells), max(0, int(c0) - margin_cells)
    r1, c1 = min(h, int(r1) + margin_cells), min(w, int(c1) + margin_cells)
    return r0, r1, c0, c1


def render_map(state, obs=None, out_path=None, px=384, paths=None, frame=None, path_marks=None):
    """Draw the map; save to out_path when given. Returns (PIL image, info dict with the crop box and scale).
    paths: {id: corner points} drawn blue; path_marks: {id: [((r, c), kind)]} drawn on top, unseen stretches orange
    and stretches through a known obstacle magenta (honest_text.classify_path)."""
    from PIL import Image, ImageDraw, ImageFont

    sem = state["sem_map"]
    explored = np.asarray(sem["explored"])
    height = np.asarray(sem["height"]) if sem.get("height") is not None else np.zeros_like(explored, dtype=float)
    ids = np.asarray(sem["id"]) if sem.get("id") is not None else np.zeros_like(explored)
    agent = obs["agent_cell"] if obs else None
    r0, r1, c0, c1 = crop_box(explored, agent)
    hh, ww = r1 - r0, c1 - c0
    scale = max(1.0, min(px / max(1, hh), px / max(1, ww)))

    # base colours per cell
    img = np.empty((hh, ww, 3), dtype=np.uint8)
    img[:] = COL_UNSEEN
    ex = explored[r0:r1, c0:c1] == 1
    img[ex] = COL_FLOOR
    hgt = height[r0:r1, c0:c1]
    img[ex & (hgt > 0.1)] = COL_LOW
    img[ex & (hgt > 0.5)] = COL_TALL
    img[ex & (ids[r0:r1, c0:c1] != 0)] = COL_OBJ
    im = Image.fromarray(img).resize((max(1, int(ww * scale)), max(1, int(hh * scale))), Image.NEAREST)
    draw = ImageDraw.Draw(im)

    def to_xy(cell):
        return ((cell[1] - c0 + 0.5) * scale, (cell[0] - r0 + 0.5) * scale)

    try:
        font = ImageFont.load_default(size=max(10, int(scale * 2.2)))
    except TypeError:  # older Pillow: no size argument
        font = ImageFont.load_default()

    if paths:
        for pts in paths.values():
            if pts and len(pts) > 1:
                draw.line([to_xy(p) for p in pts], fill=COL_PATH, width=max(1, int(scale * 0.5)))
    if path_marks:
        rr = max(1.5, scale * 0.45)
        for marks in path_marks.values():
            for cell, kind in marks:
                if kind in ("known", "goal", "start"):  # goal/start footprints are not obstacles (classify_path)
                    continue
                x, y = to_xy(cell)
                draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=COL_PATH_UNSEEN if kind == "unseen" else COL_PATH_BLOCKED)
    if obs:
        rad = max(3.0, scale * 1.6)
        for o in obs["objects"]:
            if not o.get("is_target") or o.get("cell") is None:
                continue
            x, y = to_xy(o["cell"])
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=COL_TARGET, outline=(255, 255, 255))
            draw.text((x + rad + 1, y - rad), str(o["id"]), fill=COL_TEXT, font=font)
        if agent is not None:
            x, y = to_xy(agent)
            rr = max(3.0, scale * 2.0)
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=COL_ROBOT, outline=(255, 255, 255))
            if obs.get("heading_deg") is not None:
                th = np.radians(obs["heading_deg"])
                # heading: angle from +row toward +col; image x is col, image y is row
                draw.line([(x, y), (x + np.sin(th) * rr * 2.5, y + np.cos(th) * rr * 2.5)], fill=COL_ROBOT, width=max(2, int(scale * 0.6)))
    # 1 m scale bar and the frame number
    bar = 10 * scale
    draw.line([(6, im.height - 8), (6 + bar, im.height - 8)], fill=COL_TEXT, width=3)
    draw.text((8, im.height - 8 - max(12, int(scale * 2.4))), "1 m", fill=COL_TEXT, font=font)
    if frame is not None:
        draw.text((6, 4), f"frame {frame}", fill=COL_TEXT, font=font)
    if out_path:
        im.save(out_path, format="PNG")
    return im, {"crop": (int(r0), int(r1), int(c0), int(c1)), "scale": float(scale), "size": im.size}
