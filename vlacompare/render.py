"""Render per-model prediction overlays and tile them into one comparison video.

Runs in the lightweight project env (numpy, opencv, pillow). Reads only the JSONL
result files written by the runners, so it never touches a model.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlacompare.common import load_results, probe_video

PALETTE = [(0, 200, 255), (255, 120, 60), (120, 230, 90), (230, 90, 220), (255, 220, 60)]
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


@dataclass
class Camera:
    """Uncalibrated pinhole guess for a dashcam; only used for drawing."""

    hfov_deg: float = 65.0
    height_m: float = 1.35       # camera above ground
    forward_m: float = 1.5       # camera ahead of the rear axle (trajectory origin)
    horizon: float = 0.5         # horizon row as a fraction of image height

    def project(self, xy: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        f = (w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        z = xy[:, 0] - self.forward_m
        ok = z > 1.0
        u = w / 2 + f * (-xy[:, 1]) / np.where(ok, z, 1)
        v = h * self.horizon + f * self.height_m / np.where(ok, z, 1)
        return np.stack([u, v], -1), ok


# ----------------------------------------------------------------- model track


@dataclass
class Track:
    name: str
    color: tuple[int, int, int]
    preds: list[dict]
    header: dict

    def active(self, t: float, realtime: bool) -> dict | None:
        """Latest prediction visible at video time ``t``."""
        best, best_ready = None, -1.0
        for p in self.preds:                      # sorted by query time
            ready = p["t"] + (p["latency_s"] if realtime else 0.0)
            if ready <= t + 1e-6 and ready >= best_ready:
                best, best_ready = p, ready
        return best

    @property
    def mean_latency(self) -> float:
        lat = [p["latency_s"] for p in self.preds]
        return float(np.mean(lat)) if lat else float("nan")


def describe_maneuver(xy: np.ndarray, dt: float) -> str:
    """Coarse meta-action derived from the trajectory (for models that give none)."""
    if len(xy) < 20:
        return "n/a"
    k = int(round(1.0 / dt))
    v_start = np.linalg.norm(xy[k] - xy[0]) / (k * dt) if k < len(xy) else 0.0
    v_end = np.linalg.norm(xy[-1] - xy[-1 - k]) / (k * dt)
    if v_end < 0.5 and v_start < 1.0:
        speed = "STOPPED"
    elif v_end < 0.5:
        speed = "STOP"
    elif v_end < v_start - 1.5:
        speed = "SLOW DOWN"
    elif v_end > v_start + 1.5:
        speed = "SPEED UP"
    else:
        speed = "KEEP SPEED"
    lat = xy[-1, 1]
    heading = math.degrees(math.atan2(xy[-1, 1] - xy[-1 - k, 1], xy[-1, 0] - xy[-1 - k, 0] + 1e-6))
    if abs(heading) > 30:
        turn = "TURN LEFT" if heading > 0 else "TURN RIGHT"
    elif abs(lat) > 2.0:
        turn = "VEER LEFT" if lat > 0 else "VEER RIGHT"
    else:
        turn = "STRAIGHT"
    return f"{speed} · {turn} · {v_start*3.6:.0f}→{v_end*3.6:.0f} km/h"


# ------------------------------------------------------------------- drawing


def draw_trajectory(img: np.ndarray, xy: np.ndarray, color, cam: Camera) -> None:
    h, w = img.shape[:2]
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return
    # Ribbon one car-width wide: left/right edges offset perpendicular to the path.
    d = np.gradient(xy, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    left, right = xy + 0.9 * n, xy - 0.9 * n
    (pl, okl), (pr, okr), (pc, okc) = (cam.project(a, w, h) for a in (left, right, xy))
    ok = okl & okr & okc
    if ok.sum() < 2:
        return
    poly = np.concatenate([pl[ok], pr[ok][::-1]]).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
    cv2.polylines(img, [pc[ok].astype(np.int32)], False, color, 3, cv2.LINE_AA)
    for i in range(9, len(xy), 10):     # a dot every second
        if ok[i]:
            cv2.circle(img, tuple(pc[i].astype(int)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pc[i].astype(int)), 5, color, 2, cv2.LINE_AA)


def draw_bev(img: np.ndarray, tracks_xy: list[tuple[np.ndarray, tuple]], box: tuple) -> None:
    """Top-down inset: ego at bottom centre, x forward up, ±12 m wide, 0..60 m."""
    x0, y0, bw, bh = box
    sub = img[y0:y0 + bh, x0:x0 + bw]
    sub[:] = (sub * 0.35).astype(np.uint8)
    rng_x, rng_y = 60.0, 12.0
    s = min(bh / rng_x, bw / (2 * rng_y))

    def px(p):
        return np.stack([bw / 2 - p[:, 1] * s, bh - 8 - p[:, 0] * s], -1).astype(np.int32)

    for m in range(10, int(rng_x) + 1, 10):
        yy = int(bh - 8 - m * s)
        cv2.line(sub, (0, yy), (bw, yy), (90, 90, 90), 1)
        cv2.putText(sub, f"{m}m", (3, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (160, 160, 160), 1, cv2.LINE_AA)
    cv2.rectangle(sub, (bw // 2 - 4, bh - 8), (bw // 2 + 4, bh - 1), (255, 255, 255), -1)
    for xy, color in tracks_xy:
        pts = px(np.asarray(xy, float))
        cv2.polylines(sub, [pts], False, color, 2, cv2.LINE_AA)
    cv2.rectangle(sub, (0, 0), (bw - 1, bh - 1), (200, 200, 200), 1)


def _text_box(draw: ImageDraw.ImageDraw, xy, lines, font, fill=(255, 255, 255),
              bg=(0, 0, 0, 170), pad=6, line_gap=3) -> int:
    x, y = xy
    if not lines:
        return y
    heights = [font.getbbox(line)[3] for line in lines]
    width = max(draw.textlength(line, font=font) for line in lines)
    total = sum(heights) + line_gap * (len(lines) - 1)
    draw.rectangle([x, y, x + width + 2 * pad, y + total + 2 * pad], fill=bg)
    cy = y + pad
    for line, hgt in zip(lines, heights):
        draw.text((x + pad, cy), line, font=font, fill=fill)
        cy += hgt + line_gap
    return y + total + 2 * pad


def render_model_tile(frame: np.ndarray, track: Track, t: float, realtime: bool,
                      cam: Camera, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    pred = track.active(t, realtime)
    if pred is not None:
        xy = np.asarray(pred["traj_xy"], float)
        draw_trajectory(img, xy, track.color, cam)
        bw = int(w * 0.17)
        draw_bev(img, [(xy, track.color)], (w - bw - 8, 70, bw, int(bw * 1.5)))

    pil = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_title, f_body, f_small = _font(max(14, h // 26), True), _font(max(11, h // 38)), \
        _font(max(10, h // 46))

    lat = track.mean_latency
    fps_txt = f"{1/lat:.3f}" if lat and lat > 0 else "n/a"
    d.rectangle([0, 0, w, h // 9], fill=(0, 0, 0, 190))
    d.rectangle([0, 0, 8, h // 9], fill=track.color + (255,))
    d.text((16, 6), track.name, font=f_title, fill=(255, 255, 255))
    dev = (pred or {}).get("device", "cpu")
    dtype = (pred or {}).get("extra", {}).get("dtype", "")
    d.text((16, 8 + f_title.size + 2),
           f"mean {lat:.1f} s/query · {fps_txt} FPS · {dev} {dtype}",
           font=f_small, fill=(210, 210, 210))

    if pred is None:
        _text_box(d, (16, h // 9 + 10), ["waiting for first prediction…"], f_body)
    else:
        age = t - pred["t"]
        info = [describe_maneuver(np.asarray(pred["traj_xy"], float), pred["traj_dt"]),
                f"prediction from t={pred['t']:.1f}s (age {age:.1f}s, took "
                f"{pred['latency_s']:.1f}s)"]
        if pred.get("meta_action"):
            info.insert(0, pred["meta_action"])
        _text_box(d, (16, h // 9 + 10), info, f_body)
        text = (pred.get("text") or "").strip().replace("\n", " ")
        if text == "<NOMINAL>":   # Alpamayo's adaptive mode: routine scene, no CoC trace
            text = "[nominal scene: reasoning skipped by the model]"
        if text:
            chars = max(20, int(w / (f_body.size * 0.55)))
            lines = textwrap.wrap(text, chars)
            if len(lines) > 4:
                lines = lines[:4]
                lines[-1] = lines[-1][: chars - 1] + "…"
            foot = h - (f_small.size + 10)
            box_h = len(lines) * (f_body.size + 4) + 14
            _text_box(d, (10, foot - box_h - 4), lines, f_body)
        synth = "synth: " + "; ".join(pred.get("synthesized_inputs", []))
        d.text((10, h - f_small.size - 6), synth, font=f_small, fill=(170, 170, 170))
    pil = Image.alpha_composite(pil, layer).convert("RGB")
    return np.asarray(pil)


def render_input_tile(frame, t, name, size):
    w, h = size
    img = Image.fromarray(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(max(14, h // 26), True)
    d.rectangle([0, 0, w, h // 11], fill=(0, 0, 0, 190))
    d.text((16, 8), f"Input · {name} · t={t:5.2f}s", font=f, fill=(255, 255, 255))
    return np.asarray(img)


def render_empty_tile(size, label="free slot"):
    w, h = size
    img = Image.new("RGB", (w, h), (28, 28, 32))
    d = ImageDraw.Draw(img)
    f = _font(max(14, h // 20), True)
    tw = d.textlength(label, font=f)
    d.text(((w - tw) / 2, h / 2 - f.size / 2), label, font=f, fill=(110, 110, 120))
    return np.asarray(img)


# ------------------------------------------------------------------- driver


def render(video: str, results: list[str], output: str, realtime: bool = False,
           include_input: bool = True, tile_w: int = 960, cols: int = 2,
           cam: Camera | None = None, max_seconds: float | None = None) -> Path:
    cam = cam or Camera()
    info = probe_video(video)
    tile_h = int(round(tile_w * info["height"] / info["width"] / 2) * 2)
    tracks = []
    for i, path in enumerate(results):
        header, preds = load_results(path)
        tracks.append(Track(header.get("model", Path(path).stem), PALETTE[i % len(PALETTE)],
                            preds, header))
    n_tiles = len(tracks) + int(include_input)
    rows = max(1, math.ceil(n_tiles / cols))
    out_w, out_h = tile_w * cols, tile_h * rows

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", f"{info['fps']:.5f}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         str(out)],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(video))
    name = Path(video).name
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / info["fps"]
            if max_seconds and t > max_seconds:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tiles = [render_model_tile(frame, tr, t, realtime, cam, (tile_w, tile_h))
                     for tr in tracks]
            if include_input:
                tiles.insert(0, render_input_tile(frame, t, name, (tile_w, tile_h)))
            while len(tiles) < rows * cols:
                tiles.append(render_empty_tile((tile_w, tile_h)))
            grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
            enc.stdin.write(np.ascontiguousarray(grid).tobytes())
            i += 1
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--results", nargs="+", required=True, help="runner JSONL files")
    p.add_argument("--output", required=True)
    p.add_argument("--realtime", action="store_true",
                   help="show each output only after its measured latency has elapsed")
    p.add_argument("--no-input-tile", action="store_true")
    p.add_argument("--tile-width", type=int, default=960)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--hfov", type=float, default=65.0)
    p.add_argument("--cam-height", type=float, default=1.35)
    p.add_argument("--horizon", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float)
    a = p.parse_args()
    out = render(a.video, a.results, a.output, a.realtime, not a.no_input_tile,
                 a.tile_width, a.cols, Camera(a.hfov, a.cam_height, 1.5, a.horizon),
                 a.max_seconds)
    print(out)


if __name__ == "__main__":
    main()
