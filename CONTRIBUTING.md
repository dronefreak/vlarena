# Contributing to vlarena

Thanks for helping! Bug reports, new model runners and rendering improvements are
all welcome.

## Ground rules

- **Run models with their upstream settings.** dtype, sampling, prompts and image
  sizes should match the model authors' reference inference. If a setting has to
  differ (e.g. no vLLM server), keep the behaviour equivalent and document it in the
  runner's docstring and the README's settings table.
- **Label synthesized inputs honestly.** Anything a model needs that a single
  dashcam can't provide (side cameras, ego history, navigation) goes into
  `Prediction.synthesized_inputs`, so it appears on the video tile.
- **Never commit weights, datasets, videos or results.** `.gitignore` covers
  `weights/`, `third_party/`, `results/`, `outputs/` and `logs/`.

## Development setup

```bash
git clone https://github.com/dronefreak/vlarena.git && cd vlarena
./setup.sh cpu     # or: ./setup.sh cuda
```

Check a change end-to-end with one query per model:

```bash
.venv/bin/python compare.py --video clip.mp4 --max-queries 1            # CPU
.venv/bin/python compare.py --video clip.mp4 --max-queries 1 --gpus 0   # GPU
```

Rendering-only changes don't need the models; re-render from cached results:

```bash
.venv/bin/python compare.py --video clip.mp4 --render-only
```

## Adding a model

1. Add the upstream repo and weights to `setup.sh` (pin a commit / revision).
2. Write `runners/run_<name>.py`:
   - call `configure_device()` **before** importing torch;
   - accept the shared flags via `add_runtime_args()`;
   - decode frames with `vlacompare.common.read_frames`;
   - write one `Prediction` per query with `ResultWriter` (ego frame: x forward,
     y left, metres; 10 Hz trajectories preferred).
3. Register it in `MODELS` and `NAV_MAP` in `compare.py`.
4. Add a row to the README tables (inputs synthesized, upstream settings) and to
   `NOTICE` (license).
5. In the PR, include a frame from the rendered video.

## Code style

- Python 3.12, type hints where they help, line length 100.
- Keep `vlacompare/common.py` dependency-light (stdlib + numpy): it is imported by
  every model's own environment.

## Pull requests

- One topic per PR, with a short description of what changed and how you tested it
  (CPU or GPU, which clip, `--max-queries`).
- By contributing, you agree that your contributions are licensed under the
  Apache License 2.0.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Questions: open an issue or
contact the maintainer, dronefreak ([kumaar324@gmail.com](mailto:kumaar324@gmail.com)).
