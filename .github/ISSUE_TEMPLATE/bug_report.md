---
name: Bug report
about: Something crashes or produces wrong output
labels: bug
---

**What happened**
<!-- Error message / traceback, or what looks wrong in the rendered video. -->

**Command**
```bash
.venv/bin/python compare.py --video 1003.mp4 --gpus 0,1,2 --max-queries 1   # replace with the exact command you ran
```

**Environment**
- Device: CPU / GPU (model, VRAM):
- `./setup.sh` mode: cpu / cuda
- Model(s) affected: alpamayo / qwen_drive / impromptu / nord
- vlarena commit:

**Logs**
<!-- logs/<clip>_<model>.log (GPU runs) or the console output. -->
