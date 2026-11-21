"""Run Impromptu VLA 3B (Base+Impromptu+nuScenes) on a front-camera video (CPU only).

Executed with envs/qwen25vl/bin/python.

The checkpoint's planning prompt (nuscenes_test.json in the Impromptu repo) takes a
single front-view image plus current speed, acceleration and steering angle, and
answers with 6 waypoints at 0.5 s over 3 s. That matches a dashcam well; only the
ego state is synthesized (constant speed, zero accel, zero steering).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runners"))

from vlacompare.common import (  # noqa: E402
    Prediction, ResultWriter, Timer, add_runtime_args, configure_device, probe_video,
    query_times, read_frames, sync, time_to_index,
)

MODEL_NAME = "Impromptu VLA 3B"
HISTORY_S = 2.0   # align query times with the other models
PROMPT = (
    "You are an autonomous driving agent. You have access to a front view camera image of a "
    "vehicle <image>. Your task is to do your best to predict future waypoints for the vehicle "
    "over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are "
    "the previous ego vehicle status recorded over the last 0.0 seconds (at 0.5-second "
    "intervals). This includes the x and y coordinates of the ego vehicle. Positive x means "
    "forward direction while positive y means leftwards. The data is presented in the format "
    "[x, y]:.(t-0.0s) [0.0, 0.0], Acceleration: X {ax:.2f}, Y {ay:.2f} m/s^2, Velocity: "
    "{v:.2f} m/s, Steering angle: {steer:.2f} (positive: left turn, negative: right turn)\n"
)
PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_waypoints(text: str):
    body = text.split("formatted as [x, y]:", 1)[-1]
    return [(float(x), float(y)) for x, y in PAIR.findall(body)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/ImpromptuVLA/3B_AD_finetune"))
    p.add_argument("--nav", default="straight", help="unused: the prompt has no route input")
    p.add_argument("--attn", choices=["auto", "sdpa", "flash_attention_2"], default="auto")
    # Upstream train/inference_scripts/sglang_infer.py: image_max_pixels=262144,
    # image_min_pixels=1024, temperature 0.01 / top_p 0.001 / top_k 1 (= greedy),
    # max_new_tokens 3000, repetition_penalty 1.0, LLaMA-Factory qwen2_vl template.
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--min-pixels", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=3000)
    add_runtime_args(p)
    args = p.parse_args()

    device = configure_device(args.device, args.threads)
    import numpy as np
    import torch
    from PIL import Image

    import _qwen25vl

    torch.set_num_threads(args.threads)
    info = probe_video(args.video)
    fps, n = info["fps"], info["num_frames"]
    times = query_times(info["duration"], HISTORY_S, args.stride, args.max_queries)
    writer = ResultWriter(args.output, header={
        "model": MODEL_NAME, "video": str(args.video), "video_info": info,
        "assumed_speed_mps": args.speed, "stride_s": args.stride,
        "traj_frame": "ego at query time, x forward / y left, metres; resampled 0.5s->0.1s",
    })
    todo = [t for t in times if round(t, 3) not in writer.done]
    dtype = args.dtype
    print(f"[impromptu] {len(times)} queries, {len(todo)} to run on {device}/{dtype}", flush=True)
    if not todo:
        return
    with Timer() as lt:
        model, processor = _qwen25vl.load(args.weights, dtype, device, args.attn,
                                          args.min_pixels, args.max_pixels)
    print(f"[impromptu] loaded in {lt.elapsed:.0f}s", flush=True)

    frames = read_frames(args.video, {time_to_index(t, fps, n) for t in todo})
    before, after = PROMPT.split("<image>")
    for t in todo:
        idx = time_to_index(t, fps, n)
        prompt_after = after.format(ax=0.0, ay=0.0, v=args.speed, steer=0.0)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": before}, {"type": "image"},
            {"type": "text", "text": prompt_after},
        ]}]
        with Timer() as q:
            answer = _qwen25vl.generate(model, processor, messages,
                                        [Image.fromarray(frames[idx])],
                                        max_new_tokens=args.max_new_tokens)
            sync(device)
        pts = parse_waypoints(answer)
        if pts:
            raw = np.array(pts, float)
            t_raw = np.arange(len(raw) + 1) * 0.5
            path = np.vstack([[0.0, 0.0], raw])
            t_new = np.arange(1, int(t_raw[-1] * 10) + 1) * 0.1
            traj = np.c_[np.interp(t_new, t_raw, path[:, 0]), np.interp(t_new, t_raw, path[:, 1])]
        else:
            traj = np.zeros((0, 2))
        writer.write(Prediction(
            model=MODEL_NAME, t=t, frame_index=idx, latency_s=q.elapsed,
            traj_xy=traj.round(3).tolist(), traj_dt=0.1, text=None, device=device,
            synthesized_inputs=[f"speed = {args.speed:g} m/s", "accel = 0", "steer = 0",
                                "no route input"],
            extra={"dtype": dtype, "threads": args.threads, "raw_answer": answer,
                   "raw_waypoints_0p5s": pts},
        ))
        end = f"({traj[-1,0]:.1f},{traj[-1,1]:.1f})" if len(traj) else "PARSE-FAIL"
        print(f"[impromptu] t={t:.2f}s  {q.elapsed:.1f}s  end={end}  raw={answer!r:.160}",
              flush=True)


if __name__ == "__main__":
    main()
