# MaGIC Adaptation Requirements

This document records the agreed requirements for adapting the forked
`148here/MaGIC` repository for smoke inference and later fine-tuning on the
target inpainting dataset.

## Repository And Sync

- Target repository: `https://github.com/148here/MaGIC`
- Local checkout: `D:\Coding\lab\TSA-inpainting\codes\MaGIC`
- The checkout must stay at the workspace root level, not inside another
  project directory.
- Prefer local code edits, then synchronize to the server through Git:
  local commit -> push to GitHub -> server pull.
- Existing unrelated local commits in sibling repositories can be ignored
  unless they are needed for this MaGIC task.
- Generated outputs, logs, checkpoints, downloaded model weights, caches, and
  temporary data should not be committed unless explicitly requested.

## Remote Runtime

- Use the two-GPU runtime instance recorded by `inpainting_demo`.
- SSH command from local Windows:

```powershell
ssh -i C:\Users\29868\.ssh\codex_zwz_42312_10_193_2_99_ed25519 -p 30286 zwz_42312@10.193.2.99
```

- Use a single GPU only: `GPU ID 1`.
- Verify `nvidia-smi` before running inference or training.
- Do not stop or interfere with another user's GPU process.
- Do not use GPU 0 for MaGIC unless the user explicitly changes this
  requirement.
- Expected remote project path:

```bash
/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/MaGIC/
```

- On the server, use the dedicated GitHub SSH key for Git operations:

```bash
export GIT_SSH_COMMAND="ssh -i /cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/sshkey -o IdentitiesOnly=yes"
```

## Environment

- Create a new conda environment on the remote server.
- Prefer the author's environment first:

```bash
conda create -n magic python=3.8.5
conda activate magic
pip install -r requirements.txt
```

- If the server GPU requires a newer CUDA-compatible wheel, follow the
  repository README guidance and use the PyTorch 1.12.1 + CUDA 11.3 option
  before making larger dependency changes.
- On the recorded two-GPU A100 instance, the working dependency adjustments
  were:
  - install `torch==1.12.1+cu113` and `torchvision==0.13.1+cu113`;
  - downgrade pip to `pip==24.0` before installing
    `pytorch_lightning==1.5.9`, because newer pip rejects its old dependency
    metadata;
  - replace GUI `opencv-python` with `opencv-python-headless==4.11.0.86`,
    because the server does not provide the Qt library needed by the GUI wheel.
  - install `efficientnet_pytorch==0.7.1` for the SketchInpainter online MuGE
    edge extractor.
- If direct Hugging Face access is blocked or slow, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

- If dependency, disk, permission, or network problems occur, stop and report
  the issue instead of silently changing the training recipe.

## Weights And Downloads

- Official sketch tau-net weight has already been uploaded on the server:

```bash
/home/zwz_42312/temp/tau_net_sketch.pth
```

- Copy or link it into the MaGIC project under:

```bash
checkpoints/tau_net_sketch.pth
```

- The original `stabilityai/stable-diffusion-2-inpainting` repository is no
  longer publicly accessible from the recorded environment. Use the community
  mirror instead:

```text
sd2-community/stable-diffusion-2-inpainting
```

- The Stable Diffusion inpainting backbone may be downloaded online:

```bash
checkpoints/512-inpainting-ema.safetensors
```

- MaGIC can load `.safetensors` through `ldm.util.read_state_dict`; the adapted
  config therefore uses:

```yaml
general:
  sd_ckpt: checkpoints/512-inpainting-ema.safetensors
```

- Expected SHA256 for `512-inpainting-ema.safetensors` from the mirror:

```text
2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3
```

- Do not commit raw model weights.
- Download helper:

```bash
python scripts/download_sd_inpaint.py
```

or, when a compatible endpoint is available:

```bash
python scripts/download_sd_inpaint.py --endpoint https://hf-mirror.com
```

## Smoke Inference

- First goal: run a minimal MaGIC inference smoke test.
- The smoke test is considered acceptable if it runs without crashing and
  produces one output image for user inspection.
- Output should be written inside this project, preferably under `output/`.
- Use the server single GPU:

```bash
CUDA_VISIBLE_DEVICES=1
```

- MaGIC's current `infer.py` hardcodes `CUDA_VISIBLE_DEVICES = '1'`; preserve
  the effective GPU 1 behavior, but prefer making this configurable in code if
  edits are needed.
- Prompt text can come from the target dataset prompt files. If prompt mapping
  is inconvenient for the first smoke test, an empty prompt is acceptable only
  for the crash/no-crash check.
- Adapted smoke samples are prepared with:

```bash
python scripts/prepare_stage3_smoke.py --config configs/adapt_stage3_sketch.yaml --limit 1
```

- Adapted sketch-only inference is run with:

```bash
python infer_adapt.py --config configs/adapt_stage3_sketch.yaml --limit 1
```

## Data And Mask Semantics

- Target data should follow the same path and layout conventions used by
  SketchInpainter Stage3.
- Actual data root should stay consistent with the existing server setup, e.g.
  the SketchInpainter data root under:

```bash
/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/data
```

- Use the same target dataset split/path as the existing SketchInpainter
  workflow unless a more specific split is requested.
- Mask semantics are confirmed from MaGIC code:
  white / value 1 means hole / region to regenerate, black / value 0 means
  keep original image.
- This matches SketchInpainter Stage3 `white_is_hole`.
- Use random masks and online MuGE-based sketch construction following
  SketchInpainter Stage3 conventions.

## Sketch Condition Strategy

- Use sketch guidance.
- Do not use MaGIC's default `sketch_inp_type: image` route for the adapted
  workflow unless explicitly needed.
- Instead, generate or provide sketch images from the SketchInpainter Stage3
  online MuGE + sketch construction pipeline and configure MaGIC as:

```yaml
sketch_inp_type: sketch
```

- This avoids requiring MaGIC's PiDiNet image-to-guidance checkpoint for the
  primary adapted path.
- SketchInpainter/MuGE sketches are black lines on white background, while
  MaGIC's Canny/PiDiNet-style direct condition expects white lines on black
  background. The adaptation therefore keeps the Stage3 construction process
  but saves the MaGIC condition sketch with inverted polarity by default.

## Training Plan And Known Ambiguity

The official MaGIC repository currently releases inference code and tau-net
checkpoints, but not tau-net training code. Therefore later fine-tuning will
require implementing training code in this fork.

Accepted training assumptions:

- Follow the official MaGIC design as closely as practical.
- Continue training from `/home/zwz_42312/temp/tau_net_sketch.pth`.
- Freeze the Stable Diffusion inpainting backbone, VAE, and text encoder by
  default.
- Train the sketch tau-net only unless the user explicitly requests broader
  fine-tuning.
- Use the target dataset with SketchInpainter Stage3 online MuGE sketch
  construction and random masks.
- Use a slightly smaller learning rate than the likely original/default
  recipe. Start conservatively, such as `5e-6` or `1e-5`, then adjust after a
  bounded smoke run.
- Choose batch size based on GPU 1 memory usage, but never exceed 32.
- Save outputs and checkpoints under this project, preferably:

```bash
output/
```

Training implementation choices that must remain explicit in code/docs:

- Whether the loss is diffusion noise-prediction loss or a reconstruction loss.
- How MaGIC's inference-time guide-signal injection is made differentiable for
  training.
- Which fields from SketchInpainter Stage3 are consumed by MaGIC. The default
  mapping is image, mask, prompt, and sketch; edge, noun, and T5 features are
  not direct MaGIC inputs unless the implementation is extended.
- Whether checkpoints save only tau-net weights or full optimizer/scheduler
  state. Prefer saving both for resume support.

The initial implemented training route uses diffusion noise-prediction loss
with frozen SD inpainting components and trainable sketch tau-net only:

```bash
python train_tau_sketch.py --config configs/adapt_stage3_sketch.yaml --max-steps 1 --batch-size 1
```

## Failure Policy

- For bounded smoke tests, use a timeout or a small sample count.
- If a single run is expected to exceed one hour, warn the user before starting
  it.
- If GPU availability, network access, environment creation, dependency
  installation, disk space, or permission problems occur, pause and ask how to
  proceed.
- Do not lower core training semantics, switch GPUs, delete non-cache files, or
  change global server settings without explicit approval.
