# vlarena

Run several driving VLAs on the **same front-camera video** and render a tiled
comparison video (one tile per model). No ground truth; this is a qualitative
side-by-side of trajectories, reasoning text and speed.

![Alpamayo 2 Edge, Qwen-Drive, Impromptu VLA and NoRD on one BDD-A clip](assets/demo_1003.gif)


## Get started

1. **Clone**
   ```bash
   git clone https://github.com/dronefreak/vlarena.git && cd vlarena
   ```
2. **Log in to Hugging Face** (weights are downloaded from the Hub)
   ```bash
   hf auth login
   ```
3. **Build** envs + fetch weights (~45 GB disk; needs `git`, `ffmpeg`)
   ```bash
   ./setup.sh cuda    # GPU servers  (or: ./setup.sh cpu)
   ```
4. **Quick test**: one query per model
   ```bash
   .venv/bin/python compare.py --video clip.mp4 --gpus 0,1,2 --max-queries 1
   ```
5. **Run** the whole clip → `outputs/<clip>_offline.mp4`
   ```bash
   .venv/bin/python compare.py --video clip.mp4 --gpus 0,1,2
   .venv/bin/python compare.py --video clip.mp4 --render-only --realtime   # latency-aware view
   ```

```
video.mp4 ─► runners/run_<model>.py (each in its own env) ─► results/<clip>/<model>.jsonl
                                                                    │
                                         vlacompare/render.py ◄─────┘ ─► outputs/<clip>_<mode>.mp4
```

## Setup

```bash
./setup.sh cuda    # GPU servers: CUDA torch + flash-attn / causal-conv1d / fla kernels
./setup.sh cpu     # CPU-only machines
```

## Quick start

```bash
# GPU server (e.g. 3x A5000): one model per GPU, concurrently; the 4th queues
.venv/bin/python compare.py --video clip.mp4 --gpus 0,1,2

# CPU only (default when --gpus is omitted): one model at a time, hard RAM cap each
.venv/bin/python compare.py --video clip.mp4

# re-render from cached results (seconds), e.g. latency-aware "realtime" mode
.venv/bin/python compare.py --video clip.mp4 --render-only --realtime
```

Useful flags: `--models alpamayo qwen_drive impromptu nord`, `--stride 1.0` (seconds
between queries), `--max-queries N`, `--nav straight|left|right`, `--speed 8` (assumed
ego m/s), `--dtype bfloat16`, `--hfov 65 --horizon 0.5` (drawing-only camera guess).
GPU runs log to `logs/<clip>_<model>.log`.

**Latency numbers are hardware-specific**: the FPS shown on each tile is what that
run measured (CPU runs are orders of magnitude slower than GPU).

## Model settings (upstream originals)

| model | settings |
| --- | --- |
| Alpamayo 2 Edge | bf16, SDPA, 1 set × 1 sample, temp 0.6, top-p 0.98, 256 tokens, 10 flow steps, seed 43 (upstream CLI defaults) |
| Qwen-Drive-1.0-4B | bf16, FlashAttention-2 on GPU (SDPA otherwise), `planner-rl` + reasoning planning, config pixel budgets / reasoning length / noise seed |
| Impromptu VLA 3B | `3B_AD_finetune`, bf16, image pixels 1024–262144, greedy (upstream temp 0.01 / top-k 1), up to 3000 new tokens, upstream nuScenes planning prompt |
| NoRD | bf16, temp 1.0, top-p 1.0, guided decoding to 8 `TRAJ_####` tokens, JPEG-encoded images as the vLLM client sends; served via transformers instead of vLLM |

## Render modes

- **offline** (default): each model's latest prediction is shown from its query time
  onward, so models are compared on identical frames.
- **realtime** (`--realtime`): a prediction appears only after its measured latency
  has passed, so slow models visibly lag the scene.

## Models

| key | model | env | inputs synthesized from a single dashcam |
| --- | --- | --- | --- |
| `alpamayo` | NVIDIA Alpamayo 2 Edge (2B) | `third_party/alpamayo2-edge/.venv` (uv, locked) | tele = 2× center crop; 4 side/rear cams black; straight ego history; nav text |
| `qwen_drive` | Qwen-Drive-1.0-4B + planner-rl | `third_party/Qwen-Drive-1.0/.venv` | front-left/right black; straight ego history, zero accel; nav command |
| `impromptu` | Impromptu VLA 3B (Qwen2.5-VL) | `envs/qwen25vl` | speed only (accel/steer = 0); no route input in its prompt |
| `nord` | NoRD (Qwen2.5-VL-3B + Dr.GRPO) | `envs/qwen25vl` | front-left/right black; straight past trajectory; command |

In CPU mode runners hide all GPUs (`CUDA_VISIBLE_DEVICES=""`), run at `nice 19`
with capped threads, inside a `systemd-run` scope with a hard RAM cap, so a GPU
job on the same machine is never touched.

These models are being fed out-of-distribution input (missing cameras, fake ego
history), so the video shows *how each copes with front-cam-only input*, not
benchmark-quality behaviour.

## Prompts

Each model was trained on a fixed prompt template, so the runners reproduce it exactly
and only fill in its slots (images, ego history / speed, navigation command). Changing
the wording would push the model off its training distribution. Below are the prompts as
sent for an 8 m/s straight ego history with `--nav straight` (`[IMG]` = one image,
system/role markers simplified).

What you can change from `compare.py`: **`--nav straight|left|right`** (route command;
Impromptu's prompt has none) and **`--speed`** (every history / velocity number).

<details>
<summary><b>Alpamayo 2 Edge</b>: 22 images over 6 cameras, ego-history tokens, navigation text</summary>

```text
<system> You are a driving assistant that generates safe and accurate actions.
<user>   Front left camera: <t=-1.0s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         Front camera: <t=-2.0s>[IMG]<t=-1.0s>[IMG]<t=-0.5s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         Front right camera: <t=-1.0s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         Rear left camera: <t=-1.0s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         Rear right camera: <t=-1.0s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         Front telephoto camera: <t=-2.0s>[IMG]<t=-1.0s>[IMG]<t=-0.5s>[IMG]<t=-0.1s>[IMG]<t=0.0s>[IMG]
         <|traj_history_start|>[57 ego-history tokens]<|traj_history_end|>
         Navigation input is: Continue straight
         Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory.
<assistant> <think></think>
```

Output: chain-of-causation reasoning (or `<NOMINAL>` for routine scenes), then
trajectory tokens decoded by the flow-matching action expert into 64 waypoints (6.4 s).
Built by `alpamayo2_edge.chat_template.build_conversation`.
</details>

<details>
<summary><b>Qwen-Drive-1.0-4B</b>: 3 views × 4 frames, ego history as text, navigation command</summary>

```text
<user> <FRONT VIEW>frame: 0[IMG]frame: 1[IMG]frame: 2[IMG]frame: 3[IMG]
       <FRONT LEFT VIEW>frame: 0[IMG]frame: 1[IMG]frame: 2[IMG]frame: 3[IMG]
       <FRONT RIGHT VIEW>frame: 0[IMG]frame: 1[IMG]frame: 2[IMG]frame: 3[IMG]
       The input images are organized by camera view. Each view contains 4 temporal frames captured at 0.5s intervals (frame 0 at t-1.5s, frame 3 is the current frame at t=0s).
       1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. Positive x points forward, positive y points left, and a positive heading indicates a left turn：
        -t-1.5s: (-12.0000, -0.0000, -0.0000);
        -t-1.0s: (-8.0000, -0.0000, -0.0000);
        -t-0.5s: (-4.0000, -0.0000, -0.0000);
        -t-0s: (0.0000, 0.0000, 0.0000);
       2. Active navigation command: [GO STRAIGHT]

       Give a one-sentence brief reasoning of the ego's future driving decision ONLY.
```

Output: a one-sentence rationale; the planning expert then reads the VLM's cache and
produces 50 waypoints (5 s). Built by `qwen_drive.QwenDriveProcessor` (the last line is
added in reasoning-planning mode).
</details>

<details>
<summary><b>Impromptu VLA 3B</b>: one front image, current speed / acceleration / steering</summary>

```text
<system> You are a helpful assistant.
<user>   You are an autonomous driving agent. You have access to a front view camera image of a vehicle [IMG]. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last 0.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:.(t-0.0s) [0.0, 0.0], Acceleration: X 0.00, Y 0.00 m/s^2, Velocity: 8.00 m/s, Steering angle: 0.00 (positive: left turn, negative: right turn)
```

Output: `<PLANNING>Predicted future movement details for the next 3 seconds … [x, y]: [..], …</PLANNING>`,
6 waypoints at 0.5 s. Verbatim from the upstream nuScenes test set (`nuscenes_test.json`),
including its typos; no navigation input.
</details>

<details>
<summary><b>NoRD</b>: 3 images, past-trajectory tokens, speed / acceleration, command</summary>

```text
<system> You are an expert driver. The current position [x, y, yaw] of the vehicle is [0, 0, 0].
         Input:
         - 3 frames of multi-view images collected from the ego-vehicle at the present timestep. The images are in the order: 1. front left, 2. front, 3. front right.
         - 3 tokens of past 1.5 seconds trajectory
         - Current speed [x, y] m/s
         - Current acceleration [x, y] m/s^2
         - High level driving command (left, right, straight)
         Given the inputs, predict the optimal 4-second future trajectory at 10Hz containing exactly 8 tokens. Output format is raw text.
<user>   [IMG][IMG][IMG]
         Past 1.5 seconds trajectory: TRAJ_0230 TRAJ_0230 TRAJ_0467
         Current [x, y] velocity: [8.000, 0.000] m/s
         Current [x, y] acceleration: [0.000, 0.000] m/s^2
         Driving command: straight
```

Output: exactly 8 tokens (`TRAJ_1500 TRAJ_0568 …`, guided decoding), decoded by
NoRD's K-Disc tokenizer into 40 waypoints (4 s). No reasoning text. From
`nord.tokenizer.SYSTEM_PROMPT` and `nord.NordAgent._build_messages`.
</details>

## Adding a model

1. Put the repo in `third_party/`, create its env.
2. Write `runners/run_<name>.py`: decode frames with `vlacompare.common.read_frames`,
   write one `Prediction` per query time with `ResultWriter` (ego frame, x forward,
   y left, metres).
3. Register it in `MODELS` / `NAV_MAP` in `compare.py`.

## Layout

- `vlacompare/common.py` — video decode (ffmpeg), synthetic inputs, `Prediction`, JSONL I/O
- `vlacompare/render.py` — overlays (trajectory ribbon, BEV inset, text) and tiling
- `runners/` — one script per model, executed with that model's interpreter
- `third_party/`, `weights/` — upstream repos and checkpoints (not ours)
- `results/`, `outputs/`, `logs/` — generated

## Contributing

Contributions are welcome, especially new model runners. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

vlarena's code is licensed under the [Apache License 2.0](LICENSE).
Copyright 2026 [dronefreak](https://github.com/dronefreak) (kumaar324@gmail.com).

**Model weights are not part of this repo**: they are downloaded from upstream and
keep their own licenses. Some are **non-commercial** (e.g. NoRD: CC BY-NC-SA 4.0).
See [NOTICE](NOTICE) for each model's license.

vlarena is a research/visualisation tool, not a driving system; never use model
outputs to control a vehicle.

## References

**Models**

- **Alpamayo 2 Edge**: NVIDIA, 2026. [Model card](https://huggingface.co/nvidia/Alpamayo2-Edge) ·
  [Code](https://github.com/NVlabs/alpamayo2-edge) ·
  [Blog: Running AV Reasoning Models Within Onboard Compute and Latency Limits](https://developer.nvidia.com/blog/running-av-reasoning-models-within-onboard-compute-and-latency-limits/)
- **Qwen-Drive-1.0**: Qwen Team. *Qwen-Drive-1.0: An Initial Step towards a
  Vision-Language Foundation Model for Autonomous Driving*, 2026.
  [arXiv:2609.00111](https://arxiv.org/abs/2609.00111) ·
  [Code](https://github.com/QwenLM/Qwen-Drive-1.0) ·
  [Model card](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B)
- **Impromptu VLA**: *Impromptu VLA: Open Weights and Open Data for Driving
  Vision-Language-Action Models*, NeurIPS 2025 (Datasets & Benchmarks).
  [arXiv:2505.23757](https://arxiv.org/abs/2505.23757) ·
  [Code](https://github.com/ahydchh/Impromptu-VLA) ·
  [Weights](https://huggingface.co/aaaaaap/ImpromptuVLAModel)
- **NoRD**: I. Rawal, S. Gupta, Y. Hu, W. Zhan. *NoRD: A Data-Efficient
  Vision-Language-Action Model that Drives without Reasoning*, CVPR 2026.
  [arXiv:2602.21172](https://arxiv.org/abs/2602.21172) ·
  [Code](https://github.com/Applied-Intuition-Open-Source/nord) ·
  [Model card](https://huggingface.co/AppliedIntuitionResearch/nord)

**Data used in the demo**

- **BDD-A**: Y. Xia, D. Zhang, J. Kim, K. Nakayama, K. Zipser, D. Whitney.
  *Predicting Driver Attention in Critical Situations*, ACCV 2018.
  [arXiv:1711.06406](https://arxiv.org/abs/1711.06406)
- **BDD100K**: F. Yu et al. *BDD100K: A Diverse Driving Dataset for Heterogeneous
  Multitask Learning*, CVPR 2020. [arXiv:1805.04687](https://arxiv.org/abs/1805.04687)
