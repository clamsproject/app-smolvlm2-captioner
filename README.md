# SmolVLM2 Captioner

A CLAMS app that captions video TimeFrames with the
[SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
multimodal model.

This app ships only the 2.2B-Instruct variant: the largest and most
general-purpose model in the SmolVLM2 family. The smaller (256M and 500M)
SmolVLM2 releases are post-trained specifically for video-QA tasks and we
do not expect them to generalize well, given their size.

The canonical user-facing documentation (app description, runtime
parameters, I/O specs, GPU memory estimates) lives in
[`metadata.py`](metadata.py), rendered at the
[CLAMS App Directory](https://apps.clams.ai). For general CLAMS app
usage, see the [CLAMS App Manual](https://apps.clams.ai/clamsapp).

## System requirements

- CUDA-capable NVIDIA GPU (see `est_gpu_mem_min` / `est_gpu_mem_typ` in
  `metadata.py` for VRAM expectations).
- `ffmpeg` (used by `mmif-python[cv]`).
