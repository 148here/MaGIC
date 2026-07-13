"""Stage3-style data preparation helpers for the MaGIC adaptation."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Stage3Record:
    image_path: Path
    prompt_path: Optional[Path]
    rel_key: str


def _as_path(path: str) -> Path:
    return Path(str(path)).expanduser().resolve()


def _iter_image_dirs(root: Path, splits: Sequence[str]) -> Iterable[Path]:
    for split in splits:
        split_root = root / str(split)
        if not split_root.exists():
            continue
        for image_dir in split_root.rglob("images"):
            if image_dir.is_dir():
                yield image_dir


def find_stage3_records(
    data_root: str,
    splits: Sequence[str],
    *,
    max_samples: int = 0,
) -> List[Stage3Record]:
    root = _as_path(data_root)
    records: List[Stage3Record] = []
    for image_dir in _iter_image_dirs(root, splits):
        group_dir = image_dir.parent
        prompt_dir = group_dir / "prompts"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            prompt_path = prompt_dir / ("%s.txt" % image_path.stem)
            records.append(
                Stage3Record(
                    image_path=image_path,
                    prompt_path=prompt_path if prompt_path.is_file() else None,
                    rel_key=str(image_path.relative_to(root)).replace("\\", "/"),
                )
            )
            if max_samples > 0 and len(records) >= max_samples:
                return records
    if not records:
        raise FileNotFoundError(
            "No images found under %s for splits %s using **/images layout."
            % (root, list(splits))
        )
    return records


def load_prompt(record: Stage3Record, *, fallback: str = "") -> str:
    if record.prompt_path is None:
        return str(fallback)
    try:
        return record.prompt_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return record.prompt_path.read_text(encoding="utf-8", errors="ignore").strip()


def load_rgb(path: Path, *, resolution: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((int(resolution), int(resolution)), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _list_masks(mask_dirs: Sequence[str]) -> List[Path]:
    masks: List[Path] = []
    for raw_dir in mask_dirs:
        directory = _as_path(raw_dir)
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                masks.append(path)
    if not masks:
        raise FileNotFoundError("No random mask images found in: %s" % list(mask_dirs))
    return masks


def load_random_mask(
    mask_dirs: Sequence[str],
    *,
    resolution: int,
    rng: random.Random,
    rotate_90: bool = True,
) -> Tuple[np.ndarray, Path]:
    mask_path = rng.choice(_list_masks(mask_dirs))
    mask = Image.open(mask_path).convert("L").resize((int(resolution), int(resolution)), Image.NEAREST)
    mask_array = np.asarray(mask, dtype=np.uint8)
    if rotate_90:
        mask_array = np.rot90(mask_array, k=rng.randrange(4)).copy()
    mask_array = np.where(mask_array >= 128, 255, 0).astype(np.uint8)
    return mask_array, mask_path


def _ensure_sketchinpainter_import(sketchinpainter_root: str) -> None:
    root = _as_path(sketchinpainter_root)
    if not root.is_dir():
        raise FileNotFoundError("SketchInpainter root does not exist: %s" % root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _magic_sketch(sketch: np.ndarray, polarity: str) -> np.ndarray:
    gray = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY) if sketch.ndim == 3 else sketch
    gray = np.asarray(gray, dtype=np.uint8)
    mode = str(polarity or "white_on_black").strip().lower()
    if mode == "white_on_black":
        # SketchInpainter/MuGE sketches are black lines on white background.
        # MaGIC's direct sketch tensor follows Canny/PiDiNet convention:
        # white lines on black background.
        return 255 - gray
    if mode == "black_on_white":
        return gray
    raise ValueError("magic_sketch_polarity must be white_on_black or black_on_white")


def build_online_muge_sketches(
    images_rgb: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    sketchinpainter_root: str,
    sketch_config_path: str,
    sketch_overrides: Optional[Dict[str, Any]] = None,
    muge_source_root: Optional[str] = None,
    muge_checkpoint: Optional[str] = None,
    device: str = "cuda",
    seed: int = 20260425,
    magic_sketch_polarity: str = "white_on_black",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    _ensure_sketchinpainter_import(sketchinpainter_root)
    from dataset.makeedge.muge import extract_muge_batch
    from dataset.makesketch import make_sketch_from_edge

    batch = np.stack([np.asarray(image, dtype=np.uint8) for image in images_rgb], axis=0)
    extractor_kwargs: Dict[str, Any] = {"device": str(device)}
    if muge_source_root:
        extractor_kwargs["source_root"] = str(muge_source_root)
    if muge_checkpoint:
        extractor_kwargs["checkpoint_path"] = str(muge_checkpoint)
    edges = extract_muge_batch(
        batch,
        alpha=1.0,
        inference_seed=int(seed),
        line_polarity="black_on_white",
        **extractor_kwargs,
    )
    sketches: List[np.ndarray] = []
    magic_sketches: List[np.ndarray] = []
    overrides = dict(sketch_overrides or {})
    for index, edge in enumerate(edges):
        sketch = make_sketch_from_edge(
            edge,
            seed=int(seed) + int(index),
            config_path=str(sketch_config_path),
            mask=np.asarray(masks[index], dtype=np.uint8),
            mask_mode="mask_region",
            boundary_pin_px=12.0,
            **overrides,
        )
        sketches.append(np.asarray(sketch, dtype=np.uint8))
        magic_sketches.append(_magic_sketch(sketch, magic_sketch_polarity))
    return sketches, magic_sketches


def save_prepared_sample(
    output_dir: str,
    *,
    stem: str,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    sketch: np.ndarray,
    prompt: str,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    root = _as_path(output_dir)
    img_dir = root / "img"
    mask_dir = root / "mask"
    sketch_dir = root / "sketch"
    prompt_dir = root / "prompt"
    meta_dir = root / "metadata"
    for directory in (img_dir, mask_dir, sketch_dir, prompt_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = img_dir / ("%s.png" % stem)
    mask_path = mask_dir / ("%s.png" % stem)
    sketch_path = sketch_dir / ("%s.png" % stem)
    prompt_path = prompt_dir / ("%s.txt" % stem)
    metadata_path = meta_dir / ("%s.json" % stem)

    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(image_path)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(mask_path)
    Image.fromarray(np.asarray(sketch, dtype=np.uint8), mode="L").save(sketch_path)
    prompt_path.write_text(str(prompt), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "sketch_path": str(sketch_path),
        "prompt_path": str(prompt_path),
        "metadata_path": str(metadata_path),
    }
