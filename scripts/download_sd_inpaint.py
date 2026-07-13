"""Download the SD 2.1 inpainting checkpoint expected by MaGIC."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="sd2-community/stable-diffusion-2-inpainting")
    parser.add_argument("--filename", default="512-inpainting-ema.safetensors")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--endpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = str(args.endpoint)
    from huggingface_hub import hf_hub_download

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=str(args.repo_id),
        filename=str(args.filename),
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(path)


if __name__ == "__main__":
    main()
