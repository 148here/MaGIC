"""Fine-tune MaGIC sketch tau-net with frozen SD inpainting noise loss."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adapt_stage3_sketch.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--resume-state", default="")
    return parser.parse_args()


def _cfg_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value)


def _image_tensor(images_rgb, device):
    import torch

    array = torch.from_numpy(images_rgb).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return array.to(device)


def _mask_tensor(masks, device):
    import torch

    array = torch.from_numpy(masks[:, None, :, :]).float() / 255.0
    array = torch.where(array >= 0.5, torch.ones_like(array), torch.zeros_like(array))
    return array.to(device)


def _sketch_tensor(sketches, device):
    import torch

    array = torch.from_numpy(sketches[:, None, :, :]).float() / 255.0
    array = torch.where(array >= 0.5, torch.ones_like(array), torch.zeros_like(array))
    return array.to(device)


def _target_for_model(model, x_start, noise, t):
    if model.parameterization == "x0":
        return x_start
    if model.parameterization == "eps":
        return noise
    if model.parameterization == "v":
        return model.get_v(x_start, noise, t)
    raise NotImplementedError("Unsupported parameterization: %s" % model.parameterization)


def _disable_checkpointing(module) -> int:
    disabled = 0
    for child in module.modules():
        if hasattr(child, "use_checkpoint"):
            child.use_checkpoint = False
            disabled += 1
    return disabled


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cuda_visible = args.cuda_visible_devices or str(cfg.general.get("cuda_visible_devices", "1"))
    if cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    os.environ.setdefault("DISABLE_XFORMERS", "true")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast

    from annotator.api import ExtraCondition
    from modules.stage3_adapter import (
        build_online_muge_sketches,
        find_stage3_records,
        load_prompt,
        load_random_mask,
        load_rgb,
    )
    from modules.utils import get_sd_models, get_tau_nets

    train_cfg = cfg.get("train", {})
    output_dir = Path(args.output_dir or str(train_cfg.get("output_dir", "output/train_tau_sketch_smoke"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 1))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 1))
    if batch_size > 32:
        raise ValueError("batch_size must not exceed 32")
    lr = float(args.lr if args.lr is not None else train_cfg.get("learning_rate", 5e-6))
    device_text = args.device or str(cfg.general.get("device", "cuda"))
    device = torch.device(device_text if torch.cuda.is_available() else "cpu")
    seed = int(train_cfg.get("seed", 20260425))
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    records = find_stage3_records(
        str(train_cfg["data_root"]),
        [str(item) for item in _cfg_list(train_cfg.get("splits", ["train"]))],
        max_samples=int(train_cfg.get("max_samples_to_scan", 0) or 0),
    )

    sd_model, _ = get_sd_models(cfg.general, False, device)
    sd_model.eval()
    for param in sd_model.parameters():
        param.requires_grad_(False)
    disabled_checkpoint_modules = _disable_checkpointing(sd_model)
    print("disabled_sd_checkpoint_modules=%d" % disabled_checkpoint_modules, flush=True)

    tau_net = get_tau_nets(cfg, device, ExtraCondition.sketch)
    tau_model = tau_net["model"]
    tau_model.train()
    optimizer = torch.optim.AdamW(tau_model.parameters(), lr=lr, weight_decay=0.0)
    start_step = 0
    if args.resume_state:
        state = torch.load(args.resume_state, map_location="cpu")
        tau_model.load_state_dict(state["tau_net"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))

    use_amp = args.precision != "fp32" and device.type == "cuda"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = GradScaler(enabled=(use_amp and args.precision == "fp16"))
    losses: List[Dict[str, Any]] = []
    resolution = int(train_cfg.get("resolution", 512))
    mask_dirs = [str(item) for item in _cfg_list(train_cfg.get("random_mask_dirs", []))]

    for step in range(start_step, max_steps):
        batch_records = [rng.choice(records) for _ in range(batch_size)]
        images = []
        masks = []
        prompts = []
        source_masks = []
        for record in batch_records:
            images.append(load_rgb(record.image_path, resolution=resolution))
            mask, mask_path = load_random_mask(mask_dirs, resolution=resolution, rng=rng, rotate_90=True)
            masks.append(mask)
            source_masks.append(str(mask_path))
            prompts.append(load_prompt(record, fallback=str(cfg.infer.get("prompt_text", ""))))

        _, magic_sketches = build_online_muge_sketches(
            images,
            masks,
            sketchinpainter_root=str(train_cfg["sketchinpainter_root"]),
            sketch_config_path=str(train_cfg["sketch_config_path"]),
            sketch_overrides=dict(train_cfg.get("sketch_overrides", {}) or {}),
            muge_source_root=str(train_cfg.get("muge_source_root", "")) or None,
            muge_checkpoint=str(train_cfg.get("muge_checkpoint", "")) or None,
            device=str(device),
            seed=seed + step,
            magic_sketch_polarity=str(train_cfg.get("magic_sketch_polarity", "white_on_black")),
        )

        image = _image_tensor(np.stack(images, axis=0), device)
        mask = _mask_tensor(np.stack(masks, axis=0), device)
        sketch = _sketch_tensor(np.stack(magic_sketches, axis=0), device)
        masked_image = image * (mask < 0.5)

        with torch.no_grad():
            z = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))
            masked_z = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(masked_image))
            c = sd_model.get_learned_conditioning(prompts)
            mask_latent = F.interpolate(mask, size=z.shape[-2:], mode="nearest")
            c_cat = torch.cat([mask_latent, masked_z], dim=1)
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            t = torch.randint(0, sd_model.num_timesteps, (z.shape[0],), device=device).long()
            noise = torch.randn_like(z)
            x_noisy = sd_model.q_sample(x_start=z, t=t, noise=noise)
            target = _target_for_model(sd_model, z, noise, t)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp, dtype=amp_dtype):
            guide_signals = [item * float(tau_net["cond_weight"]) for item in tau_model(sketch)]
            model_output = sd_model.apply_model(x_noisy, t, cond, guide_signals=guide_signals)
            loss = F.mse_loss(model_output.float(), target.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_item = float(loss.detach().cpu())
        row = {
            "step": step + 1,
            "loss": loss_item,
            "batch_size": batch_size,
            "records": [record.rel_key for record in batch_records],
            "masks": source_masks,
        }
        losses.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

        if (step + 1) % int(train_cfg.get("save_every_steps", 1)) == 0 or (step + 1) == max_steps:
            torch.save(tau_model.state_dict(), output_dir / "latest_tau_net_sketch.pth")
            torch.save(
                {
                    "step": step + 1,
                    "tau_net": tau_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                output_dir / "latest_training_state.pth",
            )
            (output_dir / "losses.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=True) + "\n" for item in losses),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
