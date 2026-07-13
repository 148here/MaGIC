# MaGIC Agent Notes

This repository is the local checkout of `148here/MaGIC` for smoke inference
and later adaptation to the target inpainting dataset.

## Primary Runtime Requirement

Use the two-GPU runtime instance, and use only GPU ID 1 for MaGIC work.

Connect from local Windows with:

```powershell
ssh -i C:\Users\29868\.ssh\codex_zwz_42312_10_193_2_99_ed25519 -p 30286 zwz_42312@10.193.2.99
```

Expected server project path:

```bash
/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/MaGIC/
```

Before any inference or training command:

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=1
```

Never use GPU 0 for this project unless the user explicitly changes the
requirement. Never stop another user's GPU process.

## Git Synchronization

Prefer local edits, then synchronize through Git:

1. Commit local changes.
2. Push to `https://github.com/148here/MaGIC` or the equivalent SSH remote.
3. Pull on the server checkout.

On the server, use:

```bash
export GIT_SSH_COMMAND="ssh -i /cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/sshkey -o IdentitiesOnly=yes"
```

Do not commit generated images, logs, checkpoints, downloaded weights, caches,
or temporary run artifacts.

## Environment And Weights

Create a new remote conda environment. Prefer the author recipe first:

```bash
conda create -n magic python=3.8.5
conda activate magic
pip install -r requirements.txt
```

If the server GPU needs the README's newer CUDA wheel path, use the PyTorch
1.12.1 + CUDA 11.3 requirement variant before making broader dependency
changes.

Known working environment adjustments on the A100 two-GPU instance:

- use `pip==24.0` before installing `pytorch_lightning==1.5.9`;
- replace `opencv-python` with `opencv-python-headless==4.11.0.86`.
- install `efficientnet_pytorch==0.7.1` for online MuGE sketch preparation.

The user uploaded the official sketch tau-net checkpoint here:

```bash
/home/zwz_42312/temp/tau_net_sketch.pth
```

The SD 2.1 inpainting backbone may be downloaded online into:

```bash
checkpoints/512-inpainting-ema.ckpt
```

Downloader entrypoint:

```bash
python scripts/download_sd_inpaint.py
```

Use `HF_ENDPOINT=https://hf-mirror.com` if Hugging Face access is blocked or
slow.

## Data And Adaptation Rules

Use the target data layout and paths consistently with SketchInpainter Stage3.
The data root is expected under:

```bash
/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data
```

Mask semantics are `white_is_hole`: white / 1 is regenerated, black / 0 is
preserved.

For the adapted MaGIC path, use online MuGE + random masks from the
SketchInpainter Stage3 workflow. Feed MaGIC sketch guidance directly with:

```yaml
sketch_inp_type: sketch
```

Do not rely on MaGIC's image-to-sketch PiDiNet path unless explicitly needed.

## Work Plan Defaults

First run a one-image smoke inference test. A smoke test passes if it produces
an output image without crashing; the user will judge output quality.

Adapted Stage3 sample preparation:

```bash
python scripts/prepare_stage3_smoke.py --config configs/adapt_stage3_sketch.yaml --limit 1
```

Adapted sketch-only inference:

```bash
python infer_adapt.py --config configs/adapt_stage3_sketch.yaml --limit 1
```

For later training, remember that official MaGIC does not release tau-net
training code. Implement fine-tuning in this fork only after making that
assumption explicit. Default fine-tuning assumptions:

- freeze Stable Diffusion inpainting backbone, VAE, and text encoder;
- continue training only the sketch tau-net from `tau_net_sketch.pth`;
- use SketchInpainter Stage3 online MuGE sketches and random masks;
- use a slightly smaller learning rate than the presumed original recipe;
- choose batch size based on GPU 1 memory, never above 32;
- save run outputs and checkpoints under `output/`.

Implemented training smoke entrypoint:

```bash
python train_tau_sketch.py --config configs/adapt_stage3_sketch.yaml --max-steps 1 --batch-size 1
```

If dependency, network, disk, permission, GPU availability, or long-runtime
problems occur, pause and ask the user instead of changing the core recipe.
