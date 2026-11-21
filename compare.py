"""One video in -> every model run -> one tiled comparison video.

    # GPU server: one model per GPU, run concurrently (4th model queues for a free GPU)
    .venv/bin/python compare.py --video clip.mp4 --gpus 0,1,2

    # no --gpus: CPU only, one model at a time, each under a hard RAM cap
    .venv/bin/python compare.py --video clip.mp4

Each model runs in its own Python env via its runner script with the upstream
(original) inference settings, and writes results/<clip>/<model>.jsonl (resumable).
Rendering only reads those files, so `--render-only` re-renders in seconds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vlacompare.render import Camera, render  # noqa: E402

# name -> (python interpreter of the model's env, runner script). Order = tile order.
MODELS = {
    "alpamayo": (ROOT / "third_party/alpamayo2-edge/.venv/bin/python",
                 ROOT / "runners/run_alpamayo2_edge.py"),
    "qwen_drive": (ROOT / "third_party/Qwen-Drive-1.0/.venv/bin/python",
                   ROOT / "runners/run_qwen_drive.py"),
    "impromptu": (ROOT / "envs/qwen25vl/bin/python", ROOT / "runners/run_impromptu.py"),
    "nord": (ROOT / "envs/qwen25vl/bin/python", ROOT / "runners/run_nord.py"),
}
NAV_MAP = {
    "alpamayo": {"straight": "Continue straight", "left": "Turn left at the next intersection",
                 "right": "Turn right at the next intersection"},
    "qwen_drive": {"straight": "straight", "left": "left", "right": "right"},
    "impromptu": {"straight": "straight", "left": "left", "right": "right"},   # ignored
    "nord": {"straight": "straight", "left": "left", "right": "right"},
}


def build_cmd(name: str, a, out: Path, device: str) -> list[str]:
    py, runner = MODELS[name]
    cmd = [str(py), str(runner), "--video", a.video, "--output", str(out),
           "--device", device, "--dtype", a.dtype, "--stride", str(a.stride),
           "--nav", NAV_MAP[name][a.nav], "--speed", str(a.speed),
           "--threads", str(a.threads)]
    if a.max_queries:
        cmd += ["--max-queries", str(a.max_queries)]
    return cmd


def run_cpu(jobs, a) -> None:
    for name, out in jobs:
        cmd = build_cmd(name, a, out, "cpu")
        if a.mem_cap:
            # Own cgroup with a hard RAM cap: if we overshoot, the kernel kills this
            # runner (it resumes on the next call) instead of anything else.
            cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.mem_cap}",
                   "-p", "MemorySwapMax=0"] + cmd
        print(f"== {name} [cpu]: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def run_gpus(jobs, a, gpus: list[str]) -> None:
    pending, running, failed = list(jobs), {}, []
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    while pending or running:
        free = [g for g in gpus if g not in running]
        while pending and free:
            name, out = pending.pop(0)
            gpu = free.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            log = open(logdir / f"{Path(a.video).stem}_{name}.log", "a")
            cmd = build_cmd(name, a, out, "cuda")
            print(f"== {name} [gpu {gpu}]: {' '.join(cmd)}", flush=True)
            running[gpu] = (name, subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log)
        time.sleep(2)
        for gpu, (name, proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                print(f"== {name} [gpu {gpu}] exited {proc.returncode}", flush=True)
                if proc.returncode:
                    failed.append(name)
                del running[gpu]
    if failed:
        raise SystemExit(f"failed: {failed} (see logs/{Path(a.video).stem}_<model>.log)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--gpus", help="comma-separated GPU ids, e.g. 0,1,2; omit for CPU-only")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--stride", type=float, default=1.0, help="seconds between model queries")
    p.add_argument("--max-queries", type=int)
    p.add_argument("--nav", default="straight", choices=["straight", "left", "right"])
    p.add_argument("--speed", type=float, default=8.0, help="assumed ego speed, m/s")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mem-cap", default="28G",
                   help="CPU mode: hard RAM cap per model run (systemd-run); '' disables")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--output")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--input-tile", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--hfov", type=float, default=65.0)
    p.add_argument("--horizon", type=float, default=0.5)
    a = p.parse_args()

    clip = Path(a.video).stem
    rdir = Path(a.results_dir) / clip
    rdir.mkdir(parents=True, exist_ok=True)
    jobs = [(name, rdir / f"{name}.jsonl") for name in a.models]
    if not a.render_only:
        if a.gpus:
            run_gpus(jobs, a, [g.strip() for g in a.gpus.split(",") if g.strip()])
        else:
            run_cpu(jobs, a)

    results = [str(out) for _, out in jobs if out.exists()]
    mode = "realtime" if a.realtime else "offline"
    output = a.output or str(ROOT / "outputs" / f"{clip}_{mode}.mp4")
    # Raw-input tile only when it fills an otherwise empty grid slot.
    include_input = {"auto": len(results) % 2 == 1, "on": True, "off": False}[a.input_tile]
    render(a.video, results, output, realtime=a.realtime, include_input=include_input,
           cam=Camera(hfov_deg=a.hfov, horizon=a.horizon))
    print(output)


if __name__ == "__main__":
    main()
