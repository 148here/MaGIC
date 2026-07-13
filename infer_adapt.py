"""Sketch-only MaGIC inference entrypoint for the Stage3 adaptation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adapt_stage3_sketch.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def _load_prompt(prompt_dir: str, image_path: str, fallback: str) -> str:
    if prompt_dir:
        prompt_path = Path(prompt_dir) / ("%s.txt" % Path(image_path).stem)
        if prompt_path.is_file():
            return prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
    return str(fallback or "")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cuda_visible = args.cuda_visible_devices or str(cfg.general.get("cuda_visible_devices", "1"))
    if cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    os.environ.setdefault("DISABLE_XFORMERS", "true")

    import cv2
    import torch
    from basicsr.utils import tensor2img
    from pytorch_lightning import seed_everything
    from torch import autocast
    from tqdm import tqdm

    from annotator import api
    from annotator.api import ExtraCondition, get_weigted_guide_signal
    from ldm.data.utils import load_flist
    from modules.inference_base import diffusion_inference, diffusion_inference_pad
    from modules.utils import get_sd_models, get_tau_nets, setup_logger

    if args.output_dir:
        cfg.general.save_path = args.output_dir
    if args.steps is not None:
        cfg.infer.steps = int(args.steps)
    device_text = args.device or str(cfg.general.get("device", "cuda"))
    device = torch.device(device_text if torch.cuda.is_available() else "cpu")

    condition_names = [str(name) for name in cfg.infer.auxiliary.get("conditions", ["sketch"])]
    activated_conds: List[str] = []
    cond_flists: List[List[str]] = []
    cond_inp_types: List[str] = []
    process_cond_modules = []
    for cond_name in condition_names:
        if not hasattr(ExtraCondition, cond_name):
            raise ValueError("Unsupported condition: %s" % cond_name)
        cond_path = getattr(cfg.infer.auxiliary, "%s_path" % cond_name)
        ckpt_path = getattr(cfg.infer.auxiliary, "%s_ckpt" % cond_name)
        if not ckpt_path:
            raise ValueError("Missing checkpoint for condition: %s" % cond_name)
        activated_conds.append(cond_name)
        cond_flists.append(list(load_flist(cond_path)))
        cond_inp_types.append(str(getattr(cfg.infer.auxiliary, "%s_inp_type" % cond_name, "sketch")))
        process_cond_modules.append(getattr(api, "get_cond_%s" % cond_name))

    tau_nets = [get_tau_nets(cfg, device, getattr(ExtraCondition, name)) for name in activated_conds]
    cond_transformers = [None for _ in activated_conds]
    sd_model, sampler = get_sd_models(cfg.general, bool(cfg.infer.CMB.enable), device)
    infer_fn = diffusion_inference if bool(cfg.infer.fixed_resolution) else diffusion_inference_pad

    img_flist = list(load_flist(cfg.infer.data.img_path))
    mask_flist = list(load_flist(cfg.infer.data.mask_path))
    limit = int(args.limit or 0)
    if limit > 0:
        img_flist = img_flist[:limit]
        mask_flist = mask_flist[:limit]
        cond_flists = [items[:limit] for items in cond_flists]

    save_path = Path(str(cfg.general.save_path))
    save_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(save_path), __name__)
    OmegaConf.save(cfg, save_path / "config.yaml")

    seed_everything(int(cfg.general.rand_seed))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with sd_model.ema_scope(), autocast(autocast_device):
        for img_path, mask_path, *cond_paths in tqdm(zip(img_flist, mask_flist, *cond_flists)):
            basename = Path(img_path).name
            prompt = _load_prompt(
                str(cfg.infer.get("prompt_dir", "")),
                img_path,
                str(cfg.infer.get("prompt_text", "")),
            )
            logger.info("img_path: %s\nprompt: %s", img_path, prompt)

            guide_signals = []
            for cond_idx, cond_name in enumerate(activated_conds):
                cond = process_cond_modules[cond_idx](
                    cfg,
                    cond_paths[cond_idx],
                    device,
                    cond_inp_types[cond_idx],
                    cond_transformers[cond_idx],
                )
                guide_signals.append(
                    get_weigted_guide_signal(cond, tau_nets[cond_idx], int(cfg.infer.batch_size))
                )
            guides = guide_signals if guide_signals else None
            if guides is not None and not bool(cfg.infer.CMB.enable):
                guides = guides[0]

            for idx in range(int(cfg.infer.n_samples)):
                result = infer_fn(cfg, (img_path, mask_path, sd_model, sampler, prompt, guides))
                for batch_index, res in enumerate(result):
                    out_path = save_path / ("%s_%03d.png" % (Path(basename).stem, idx * int(cfg.infer.batch_size) + batch_index))
                    cv2.imwrite(str(out_path), tensor2img(res))
                    logger.info("save image to %s", out_path)


if __name__ == "__main__":
    main()
