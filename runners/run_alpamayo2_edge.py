"""Run NVIDIA Alpamayo 2 Edge on a single front-camera video.

Executed with third_party/alpamayo2-edge/.venv/bin/python.

Model settings are the upstream CLI defaults (alpamayo2_edge.inference): bf16, SDPA,
1 trajectory set x 1 sample, top_p 0.98, temperature 0.6, 256 generated tokens,
10 flow-matching steps, seed 43.

The model needs six cameras with 22 frames, 20 ego-history poses and a navigation
string. From a dashcam we only have the front view, so:
  - camera 1 (front wide)  <- the video frame
  - camera 6 (front tele)  <- central 2x crop of the video frame, upscaled
  - cameras 0/2/3/5        <- black frames
  - ego history            <- straight line at a constant assumed speed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlacompare.common import (  # noqa: E402
    Prediction, ResultWriter, Timer, add_runtime_args, center_crop_zoom, configure_device,
    constant_speed_history, probe_video, query_times, read_frames, sync, time_to_index,
)

MODEL_NAME = "Alpamayo 2 Edge"
FRONT_OFFSETS = [-2.0, -1.0, -0.5, -0.1, 0.0]   # cameras 1 and 6
SIDE_OFFSETS = [-1.0, -0.1, 0.0]                # cameras 0, 2, 3, 5
SIDE_CAMERAS = [0, 2, 3, 5]
HISTORY_S = 2.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/Alpamayo2-Edge"))
    p.add_argument("--nav", default="Continue straight")
    p.add_argument("--attn", choices=["sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--diffusion-steps", type=int, default=10)
    add_runtime_args(p)
    p.set_defaults(seed=43)
    args = p.parse_args()

    device = configure_device(args.device, args.threads)
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(args.threads)
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.inference import predict, validate_predictions

    info = probe_video(args.video)
    fps, n = info["fps"], info["num_frames"]
    times = query_times(info["duration"], HISTORY_S, args.stride, args.max_queries)
    writer = ResultWriter(args.output, header={
        "model": MODEL_NAME, "video": str(args.video), "video_info": info,
        "nav": args.nav, "assumed_speed_mps": args.speed, "stride_s": args.stride,
        "traj_frame": "ego at query time, rear axle, x forward / y left, metres",
    })
    todo = [t for t in times if round(t, 3) not in writer.done]
    print(f"[alpamayo] {len(times)} queries, {len(todo)} to run on {device}/{args.dtype}",
          flush=True)
    if not todo:
        return

    with Timer() as load_t:
        model, _ = load_model(args.weights, device=device, dtype=args.dtype,
                              attn_implementation=args.attn)
    print(f"[alpamayo] loaded in {load_t.elapsed:.0f}s", flush=True)

    needed = {time_to_index(t + off, fps, n)
              for t in todo for off in set(FRONT_OFFSETS) | set(SIDE_OFFSETS)}
    frames = read_frames(args.video, needed)
    h, w = info["height"], info["width"]
    black = np.zeros((h, w, 3), np.uint8)

    def tele(img: np.ndarray) -> np.ndarray:
        crop = torch.from_numpy(center_crop_zoom(img)).permute(2, 0, 1)[None].float()
        up = F.interpolate(crop, size=(h, w), mode="bicubic", align_corners=False)
        return up.clamp(0, 255).round().byte()[0].permute(1, 2, 0).numpy()

    settings = dict(
        seed=args.seed, top_p=0.98, top_k=None, temperature=0.6,
        num_traj_sets=1, num_traj_samples=1,
        max_generation_length=args.max_generation_length,
        diffusion_kwargs={"inference_step": args.diffusion_steps},
    )
    hist = constant_speed_history(args.speed, np.arange(-1.9, 0.0001, 0.1))

    for t in todo:
        idx0 = time_to_index(t, fps, n)
        images, cams, stamps = [], [], []
        for cam, offsets in [(1, FRONT_OFFSETS), (6, FRONT_OFFSETS)] + [
            (c, SIDE_OFFSETS) for c in SIDE_CAMERAS
        ]:
            for off in offsets:
                fi = time_to_index(t + off, fps, n)
                img = frames[fi] if cam == 1 else tele(frames[fi]) if cam == 6 else black
                images.append(torch.from_numpy(img).permute(2, 0, 1)[None])
                cams.append(cam)
                stamps.append(int(round(fi / fps * 1e6)))
        data = {
            "image_frames": torch.stack(images).contiguous(),        # [22, 1, 3, H, W]
            "camera_indices": torch.tensor(cams, dtype=torch.int64),
            "absolute_timestamps": torch.tensor(stamps, dtype=torch.int64),
            "ego_history_xyz": torch.tensor(
                np.c_[hist[:, :2], np.zeros(len(hist))], dtype=torch.float32)[None],
            "ego_history_rot": torch.eye(3).repeat(len(hist), 1, 1)[None],
            "nav_text": [args.nav],
        }
        with Timer() as q:
            prepared = to_device(prepare_model_inputs(data, model.config, model.tokenizer),
                                 device=device)
            xyz, rot, extra = predict(model, prepared, settings, device)
            sync(device)
        validate_predictions(xyz, rot, 1, 1)
        traj = xyz[0, 0, 0, :, :2].float().cpu().numpy()
        cot = extra.get("cot")
        text = str(np.asarray(cot).reshape(-1)[0]) if cot is not None else None
        writer.write(Prediction(
            model=MODEL_NAME, t=t, frame_index=idx0, latency_s=q.elapsed,
            traj_xy=traj.round(3).tolist(), traj_dt=0.1, text=text, device=device,
            synthesized_inputs=[
                "front-tele = 2x center crop", "side/rear cams = black",
                f"ego history = straight @ {args.speed:g} m/s", f"nav = '{args.nav}'",
            ],
            extra={"dtype": args.dtype, "threads": args.threads, "settings": settings},
        ))
        print(f"[alpamayo] t={t:.2f}s  {q.elapsed:.1f}s  end=({traj[-1,0]:.1f},{traj[-1,1]:.1f})"
              f"  cot={text!r:.120}", flush=True)


if __name__ == "__main__":
    main()
