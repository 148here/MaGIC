"""Prepare one or more SketchInpainter Stage3-style MaGIC smoke samples."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from modules.stage3_adapter import (
    build_online_muge_sketches,
    find_stage3_records,
    load_prompt,
    load_random_mask,
    load_rgb,
    save_prepared_sample,
)


def _cfg_list(value: Any) -> list:
    if value is None:
        return []
    return list(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adapt_stage3_sketch.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    train_cfg = cfg.get("train", {})
    output_dir = args.output_dir or str(cfg.infer.data.img_path).rsplit("/", 1)[0]
    device = args.device or str(cfg.general.get("device", "cuda"))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 20260425))
    rng = random.Random(seed)

    records = find_stage3_records(
        str(train_cfg["data_root"]),
        [str(item) for item in _cfg_list(train_cfg.get("splits", ["train"]))],
        max_samples=int(train_cfg.get("max_samples_to_scan", 0) or 0),
    )
    start = max(0, int(args.sample_index))
    selected = records[start : start + max(1, int(args.limit))]
    if not selected:
        raise ValueError("sample-index is out of range for %d records" % len(records))

    images = []
    masks = []
    prompts = []
    mask_paths = []
    resolution = int(train_cfg.get("resolution", 512))
    for record in selected:
        images.append(load_rgb(record.image_path, resolution=resolution))
        mask, mask_path = load_random_mask(
            [str(item) for item in _cfg_list(train_cfg.get("random_mask_dirs", []))],
            resolution=resolution,
            rng=rng,
            rotate_90=True,
        )
        masks.append(mask)
        mask_paths.append(mask_path)
        prompts.append(load_prompt(record, fallback=str(cfg.infer.get("prompt_text", ""))))

    _, magic_sketches = build_online_muge_sketches(
        images,
        masks,
        sketchinpainter_root=str(train_cfg["sketchinpainter_root"]),
        sketch_config_path=str(train_cfg["sketch_config_path"]),
        sketch_overrides=dict(train_cfg.get("sketch_overrides", {}) or {}),
        muge_source_root=str(train_cfg.get("muge_source_root", "")) or None,
        muge_checkpoint=str(train_cfg.get("muge_checkpoint", "")) or None,
        device=device,
        seed=seed,
        magic_sketch_polarity=str(train_cfg.get("magic_sketch_polarity", "white_on_black")),
    )

    for offset, record in enumerate(selected):
        stem = "sample_%03d_%s" % (start + offset, record.image_path.stem)
        metadata: Dict[str, Any] = {
            "source_image": str(record.image_path),
            "source_prompt": str(record.prompt_path) if record.prompt_path else "",
            "source_mask": str(mask_paths[offset]),
            "rel_key": record.rel_key,
            "seed": seed,
            "resolution": resolution,
            "mask_semantics": "white_is_hole",
            "sketch_source": "SketchInpainter online_muge + stage3 sketch overrides",
            "magic_sketch_polarity": str(train_cfg.get("magic_sketch_polarity", "white_on_black")),
        }
        paths = save_prepared_sample(
            output_dir,
            stem=stem,
            image_rgb=images[offset],
            mask=masks[offset],
            sketch=magic_sketches[offset],
            prompt=prompts[offset],
            metadata=metadata,
        )
        print(paths)


if __name__ == "__main__":
    main()
