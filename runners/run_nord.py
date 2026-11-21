"""Run NoRD (Applied Intuition, Qwen2.5-VL-3B + Dr.GRPO) on a front-camera video.

Executed with envs/qwen25vl/bin/python.

Upstream serves the model with vLLM (bf16) and queries it through nord.NordAgent.
Here the identical request is reproduced with transformers so the pipeline runs on
CPU or a pinned GPU without a server:
  - same system prompt / user text (nord.tokenizer.SYSTEM_PROMPT, format_past_tokens)
  - images JPEG-encoded exactly as NordAgent._image_to_b64 does
  - temperature 1.0, top_p 1.0, repetition_penalty 1.0 (NordAgent defaults)
  - guided decoding to ^((TRAJ_\\d{4} ){7}TRAJ_\\d{4})$ (vLLM guided_regex equivalent)
  - 8 K-Disc tokens decoded with nord's own tokenizer -> (40, 3) at 10 Hz
Sampling is seeded (--seed + query index) so reruns are reproducible.

Synthesized inputs from a dashcam:
  - front-left / front-right    <- black frames, same size as the front frame
  - past 1.5 s trajectory        <- straight line at the assumed speed, K-Disc encoded
  - velocity / acceleration      <- (speed, 0) / (0, 0)
"""

from __future__ import annotations

import argparse
import io
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

MODEL_NAME = "NoRD"
HISTORY_S = 2.0
TEMPLATE = " ".join(["TRAJ_####"] * 8)    # == ^((TRAJ_\d{4} ){7}TRAJ_\d{4})$


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights/nord"))
    p.add_argument("--nav", default="straight", choices=["straight", "left", "right"])
    p.add_argument("--attn", choices=["auto", "sdpa", "flash_attention_2"], default="auto")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-guided", action="store_true")
    add_runtime_args(p)
    args = p.parse_args()

    device = configure_device(args.device, args.threads)
    import numpy as np
    import torch
    from PIL import Image

    import _qwen25vl
    from nord.hub import BUNDLED_VOCAB
    from nord.tokenizer import SYSTEM_PROMPT, KDiscTokenizer, format_past_tokens

    torch.set_num_threads(args.threads)
    info = probe_video(args.video)
    fps, n = info["fps"], info["num_frames"]
    times = query_times(info["duration"], HISTORY_S, args.stride, args.max_queries)
    writer = ResultWriter(args.output, header={
        "model": MODEL_NAME, "video": str(args.video), "video_info": info,
        "nav": args.nav, "assumed_speed_mps": args.speed, "stride_s": args.stride,
        "traj_frame": "ego at query time, x forward / y left, metres",
    })
    todo = [t for t in times if round(t, 3) not in writer.done]
    print(f"[nord] {len(times)} queries, {len(todo)} to run on {device}/{args.dtype}",
          flush=True)
    if not todo:
        return
    with Timer() as lt:
        model, processor = _qwen25vl.load(args.weights, args.dtype, device, args.attn)
        guide = None
        if not args.no_guided:
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            guide = _qwen25vl.FixedTemplateGuide(processor.tokenizer, TEMPLATE, eos)
    print(f"[nord] loaded in {lt.elapsed:.0f}s"
          f"{'' if guide is None else f', {len(guide.candidates)} guide tokens'}", flush=True)

    def as_sent(img: Image.Image) -> Image.Image:
        """Round-trip through JPEG like NordAgent._image_to_b64 (PIL default quality)."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    tok = KDiscTokenizer.from_vocab_file(str(BUNDLED_VOCAB))
    t_hist = np.arange(16) * 0.1     # 1.5 s at 10 Hz, origin = oldest pose
    past_ids = tok.encode(np.c_[args.speed * t_hist, 0 * t_hist, 0 * t_hist].astype("float32"))
    user_text = (
        f"Past 1.5 seconds trajectory: {format_past_tokens(past_ids)}\n"
        f"Current [x, y] velocity: [{args.speed:.3f}, {0.0:.3f}] m/s\n"
        f"Current [x, y] acceleration: [{0.0:.3f}, {0.0:.3f}] m/s^2\n"
        f"Driving command: {args.nav}"
    )
    black = as_sent(Image.new("RGB", (info["width"], info["height"])))
    frames = read_frames(args.video, {time_to_index(t, fps, n) for t in todo})
    for t in todo:
        idx = time_to_index(t, fps, n)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "image"},
                                         {"type": "text", "text": user_text}]},
        ]
        with Timer() as q:
            answer = _qwen25vl.generate(
                model, processor, messages,
                [black, as_sent(Image.fromarray(frames[idx])), black], max_new_tokens=64,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                seed=args.seed + int(round(t * 1000)), prefix_allowed_tokens_fn=guide)
            sync(device)
        ids = [int(m) for m in re.findall(r"TRAJ_(\d{4})", answer)][:8]
        ok = len(ids) == 8
        ids += [0] * (8 - len(ids))
        traj = np.asarray(tok.decode(ids))
        writer.write(Prediction(
            model=MODEL_NAME, t=t, frame_index=idx, latency_s=q.elapsed,
            traj_xy=traj[:, :2].round(3).tolist(), traj_dt=0.1, text=None, device=device,
            synthesized_inputs=["front-left/right = black",
                                f"past traj = straight @ {args.speed:g} m/s",
                                f"cmd = {args.nav}"],
            extra={"dtype": args.dtype, "threads": args.threads, "token_ids": ids,
                   "parsed_ok": ok, "raw_answer": answer,
                   "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                "guided": guide is not None}},
        ))
        print(f"[nord] t={t:.2f}s  {q.elapsed:.1f}s  end=({traj[-1,0]:.1f},{traj[-1,1]:.1f})"
              f"  tokens={ids} ok={ok}", flush=True)


if __name__ == "__main__":
    main()
