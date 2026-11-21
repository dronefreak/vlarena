"""Shared loader/generator for Qwen2.5-VL-based driving checkpoints.

Used by the Impromptu VLA and NoRD runners (env: envs/qwen25vl).
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable


def pick_attn(attn: str, device: str) -> str:
    if attn != "auto":
        return attn
    has_fa = importlib.util.find_spec("flash_attn") is not None
    return "flash_attention_2" if device == "cuda" and has_fa else "sdpa"


def load(path: str, dtype_name: str, device: str, attn: str = "auto",
         min_pixels: int | None = None, max_pixels: int | None = None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, dtype=getattr(torch, dtype_name), attn_implementation=pick_attn(attn, device),
    ).to(device).eval()
    kwargs = {}
    if min_pixels:
        kwargs["min_pixels"] = min_pixels
    if max_pixels:
        kwargs["max_pixels"] = max_pixels
    try:
        processor = AutoProcessor.from_pretrained(path, **kwargs)
    except ValueError:
        # Checkpoints saved by transformers 4.49.dev name the image processor
        # "Qwen2_5_VLImageProcessor", which released versions call Qwen2VLImageProcessor.
        import json
        from pathlib import Path

        from transformers import (
            AutoTokenizer, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
        )

        tok = AutoTokenizer.from_pretrained(path)
        tmpl_file = Path(path) / "chat_template.json"
        tmpl = json.loads(tmpl_file.read_text())["chat_template"] if tmpl_file.exists() \
            else tok.chat_template
        processor = Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor.from_pretrained(path, **kwargs),
            tokenizer=tok, video_processor=Qwen2VLVideoProcessor(), chat_template=tmpl)
    return model, processor


def generate(model, processor, messages: list[dict[str, Any]], images: list,
             max_new_tokens: int, *, do_sample: bool = False, temperature: float = 1.0,
             top_p: float = 1.0, top_k: int | None = None, seed: int | None = None,
             prefix_allowed_tokens_fn: Callable | None = None) -> str:
    import torch

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    if seed is not None:
        torch.manual_seed(seed)
    # use_cache=True: some training checkpoints save use_cache=false in config.json;
    # serving engines (vLLM/sglang) always use a KV cache, so this matches upstream.
    kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                                  repetition_penalty=1.0, use_cache=True)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k or 0)
    else:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    if prefix_allowed_tokens_fn is not None:
        prompt_len = inputs["input_ids"].shape[1]
        kwargs["prefix_allowed_tokens_fn"] = (
            lambda b, ids: prefix_allowed_tokens_fn(ids[prompt_len:].tolist()))
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


class FixedTemplateGuide:
    """Constrain decoding to strings matching a fixed-width character template.

    Equivalent to vLLM ``guided_regex`` for patterns like
    ``^((TRAJ_\\d{4} ){7}TRAJ_\\d{4})$``: ``template`` uses ``#`` for any digit and
    literal characters elsewhere; EOS is only allowed once the template is complete.
    """

    def __init__(self, tokenizer, template: str, eos_ids: list[int]):
        self.tokenizer = tokenizer
        self.template = template
        self.eos_ids = eos_ids
        charset = set(c for c in template if c != "#") | set("0123456789")
        self.candidates: list[tuple[int, str]] = []
        for tid in range(len(tokenizer)):
            s = tokenizer.decode([tid])
            if s and set(s) <= charset:
                self.candidates.append((tid, s))

    def _ok(self, s: str) -> bool:
        if len(s) > len(self.template):
            return False
        return all(c.isdigit() if t == "#" else c == t for c, t in zip(s, self.template))

    def __call__(self, generated_ids: list[int]) -> list[int]:
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if len(text) >= len(self.template):
            return self.eos_ids
        allowed = [tid for tid, s in self.candidates if self._ok(text + s)]
        return allowed or self.eos_ids
