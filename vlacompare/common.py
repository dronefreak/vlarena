"""Dependency-light helpers shared by every model runner.

Runners execute inside each model's own Python environment, so this module only
uses the standard library, numpy and an ``ffmpeg`` binary on PATH.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- video


def probe_video(path: str | Path) -> dict:
    """Return fps, frame count, width, height and duration of a video."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(s["nb_read_packets"])
    return {"fps": fps, "num_frames": n, "width": int(s["width"]),
            "height": int(s["height"]), "duration": n / fps}


def read_frames(path: str | Path, indices: set[int] | None = None,
                size: tuple[int, int] | None = None) -> dict[int, np.ndarray]:
    """Decode RGB uint8 frames [H, W, 3]; keep only ``indices`` (all if None)."""
    info = probe_video(path)
    w, h = size or (info["width"], info["height"])
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if size:
        cmd += ["-vf", f"scale={w}:{h}:flags=bicubic"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    # stderr muted: we deliberately close the pipe early, which ffmpeg reports as an error.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    frames, idx = {}, 0
    last = max(indices) if indices else None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if indices is None or idx in indices:
                frames[idx] = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            idx += 1
            if last is not None and idx > last:
                break
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return frames


def time_to_index(t: float, fps: float, num_frames: int) -> int:
    return int(min(max(round(t * fps), 0), num_frames - 1))


def query_times(duration: float, history_s: float, stride_s: float,
                max_queries: int | None = None) -> list[float]:
    """Query timestamps that leave ``history_s`` of past video available."""
    times, t = [], history_s
    while t <= duration - 1e-3:
        times.append(round(t, 3))
        t += stride_s
    return times[:max_queries] if max_queries else times


# ---------------------------------------------------------------- synthetic inputs


def center_crop_zoom(img: np.ndarray, zoom: float = 2.0) -> np.ndarray:
    """Fake a telephoto view: central 1/zoom crop (caller resizes as needed)."""
    h, w = img.shape[:2]
    ch, cw = int(h / zoom), int(w / zoom)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])


def constant_speed_history(speed_mps: float, times_s: np.ndarray) -> np.ndarray:
    """Straight-line ego history in the current ego frame (x fwd, y left).

    ``times_s`` are non-positive offsets from now; returns [N, 3] (x, y, heading).
    """
    t = np.asarray(times_s, dtype=np.float64)
    return np.stack([speed_mps * t, np.zeros_like(t), np.zeros_like(t)], axis=-1)


# --------------------------------------------------------------------- predictions


@dataclass
class Prediction:
    """One model output at one query time. Trajectory is ego-frame (x fwd, y left)."""

    model: str
    t: float                      # query time in the video, seconds
    frame_index: int
    latency_s: float              # wall-clock for this query (preprocess + inference)
    traj_xy: list[list[float]] = field(default_factory=list)   # [[x, y], ...]
    traj_dt: float = 0.1
    text: str | None = None       # reasoning / explanation
    meta_action: str | None = None
    device: str = "cpu"
    synthesized_inputs: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class ResultWriter:
    """Append-only JSONL writer that lets runs resume after interruption."""

    def __init__(self, path: str | Path, header: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[float] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line from a crash
                if rec.get("kind") == "prediction":
                    self.done.add(round(rec["t"], 3))
        elif header is not None:
            self._write({"kind": "header", **header})

    def _write(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write(self, pred: Prediction) -> None:
        self._write({"kind": "prediction", **asdict(pred)})
        self.done.add(round(pred.t, 3))


def load_results(path: str | Path) -> tuple[dict, list[dict]]:
    header, preds = {}, []
    for line in Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "header":
            header = rec
        elif rec.get("kind") == "prediction":
            preds.append(rec)
    preds.sort(key=lambda r: r["t"])
    return header, preds


# ------------------------------------------------------------------------ resources


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return float("nan")


def configure_cpu(threads: int) -> None:
    """Hide GPUs and cap threads so a concurrent GPU training job is untouched."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        os.nice(19)
    except OSError:
        pass


def configure_device(device: str, threads: int) -> str:
    """Call before importing torch. 'cpu' hides all GPUs; 'cuda' uses the GPU(s)
    visible through CUDA_VISIBLE_DEVICES (compare.py pins one GPU per model)."""
    if device == "cpu":
        configure_cpu(threads)
    elif device != "cuda":
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    return device


def add_runtime_args(parser) -> None:
    """Flags shared by every runner. Defaults are the upstream (original) settings."""
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                        help="bfloat16 is the upstream default for every model")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (cpu device)")
    parser.add_argument("--stride", type=float, default=1.0, help="seconds between queries")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--speed", type=float, default=8.0,
                        help="assumed ego speed (m/s) for the synthesized ego state")
    parser.add_argument("--seed", type=int, default=0)


def sync(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
