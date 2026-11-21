"""Run Qwen-Drive-1.0-4B on a single front-camera video.

Executed with third_party/Qwen-Drive-1.0/.venv/bin/python.

Model settings follow the upstream quick start: bf16, flash_attention_2 (SDPA on
CPU / when flash-attn is absent), planner-rl in REASONING_PLANNING mode, the model
config's default pixel budgets, reasoning length, flow steps and noise seed.

The model needs FRONT / FRONT LEFT / FRONT RIGHT views at 4 timestamps (1.5 s at
2 Hz), 1.5 s of 10 Hz ego history with velocity/acceleration, and a nav command.
From a dashcam we only have the front view, so:
  - FRONT                   <- the video frame
  - FRONT LEFT / RIGHT      <- black frames of the same size
  - ego history / status    <- straight line at a constant assumed speed, zero accel
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlacompare.common import (  # noqa: E402
    Prediction, ResultWriter, Timer, add_runtime_args, configure_device,
    constant_speed_history, probe_video, query_times, read_frames, sync, time_to_index,
)

MODEL_NAME = "Qwen-Drive-1.0-4B"
OFFSETS = [-1.5, -1.0, -0.5, 0.0]
HISTORY_S = 2.0   # keep query times aligned with Alpamayo (it needs 2 s)
NAV = {"straight": 0, "left": 1, "right": 2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/Qwen-Drive-1.0-4B"))
    p.add_argument("--planner", default="planner-rl", choices=["planner-rl", "planner-sft"])
    p.add_argument("--mode", default="reasoning_planning",
                   choices=["reasoning_planning", "direct_planning"])
    p.add_argument("--nav", default="straight", choices=list(NAV))
    p.add_argument("--attn", choices=["auto", "sdpa", "flash_attention_2"], default="auto")
    add_runtime_args(p)
    p.set_defaults(seed=None)   # None -> the model config's noise_seed (upstream default)
    args = p.parse_args()

    device = configure_device(args.device, args.threads)
    import numpy as np
    import torch
    from PIL import Image

    torch.set_num_threads(args.threads)
    from qwen_drive import CameraFrame, DrivingScene, InferenceMode, QwenDriveForPlanning

    attn = args.attn
    if attn == "auto":
        has_fa = importlib.util.find_spec("flash_attn") is not None
        attn = "flash_attention_2" if device == "cuda" and has_fa else "sdpa"

    info = probe_video(args.video)
    fps, n = info["fps"], info["num_frames"]
    times = query_times(info["duration"], HISTORY_S, args.stride, args.max_queries)
    writer = ResultWriter(args.output, header={
        "model": MODEL_NAME, "video": str(args.video), "video_info": info,
        "nav": args.nav, "assumed_speed_mps": args.speed, "stride_s": args.stride,
        "planner": args.planner, "mode": args.mode,
        "traj_frame": "ego at query time, x forward / y left, metres",
    })
    todo = [t for t in times if round(t, 3) not in writer.done]
    print(f"[qwen-drive] {len(times)} queries, {len(todo)} to run on {device}/{args.dtype}/{attn}",
          flush=True)
    if not todo:
        return

    with Timer() as load_t:
        model = QwenDriveForPlanning.from_pretrained(
            args.weights, planner=str(Path(args.weights) / args.planner),
            dtype=getattr(torch, args.dtype), attn_implementation=attn,
        ).to(device).eval()
    print(f"[qwen-drive] loaded in {load_t.elapsed:.0f}s", flush=True)

    needed = {time_to_index(t + off, fps, n) for t in todo for off in OFFSETS}
    frames = read_frames(args.video, needed)
    black = Image.new("RGB", (info["width"], info["height"]))

    hist_t = np.round(np.arange(-1.5, 0.0001, 0.1), 2)       # 16 steps
    hist = constant_speed_history(args.speed, hist_t)
    vel = np.tile([args.speed, 0.0], (len(hist_t), 1))
    acc = np.zeros((len(hist_t), 2))
    command = [0.0, 0.0, 0.0, 0.0]                            # one-hot left/straight/right/unk
    command[{0: 1, 1: 0, 2: 2}[NAV[args.nav]]] = 1.0
    mode = InferenceMode(args.mode)

    for t in todo:
        idx0 = time_to_index(t, fps, n)
        front = [CameraFrame(Image.fromarray(frames[time_to_index(t + o, fps, n)]))
                 for o in OFFSETS]
        scene = DrivingScene(
            views={
                "<FRONT VIEW>": front,
                "<FRONT LEFT VIEW>": [CameraFrame(black) for _ in OFFSETS],
                "<FRONT RIGHT VIEW>": [CameraFrame(black) for _ in OFFSETS],
            },
            history=hist, history_velocity=vel, history_acceleration=acc,
            ego_velocity=(args.speed, 0.0), ego_acceleration=(0.0, 0.0),
            driving_command=command, nav_command=NAV[args.nav], token=f"t{t:.2f}",
        )
        kwargs = {"num_samples": 1}
        if args.seed is not None:
            kwargs["seed"] = args.seed
        with Timer() as q:
            out = model.run(mode, scene=scene, **kwargs)
            sync(device)
        traj = np.asarray(out.trajectories[0])
        writer.write(Prediction(
            model=MODEL_NAME, t=t, frame_index=idx0, latency_s=q.elapsed,
            traj_xy=traj[:, :2].round(3).tolist(), traj_dt=0.1, text=out.reasoning,
            device=device,
            synthesized_inputs=[
                "front-left/right = black",
                f"ego history = straight @ {args.speed:g} m/s", f"nav = {args.nav}",
            ],
            extra={"dtype": args.dtype, "attn": attn, "threads": args.threads,
                   "planner": args.planner, "heading": traj[:, 2].round(4).tolist()},
        ))
        print(f"[qwen-drive] t={t:.2f}s  {q.elapsed:.1f}s  end=({traj[-1,0]:.1f},{traj[-1,1]:.1f})"
              f"  reasoning={out.reasoning!r:.120}", flush=True)


if __name__ == "__main__":
    main()
