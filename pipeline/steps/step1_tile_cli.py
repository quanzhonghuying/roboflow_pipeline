"""Step1: tile (slice) images.

Parameterised CLI version of your original `step1_tile.py`.

Default directory layout (your choice A):
  inputs/ : original images
  tiles/  : output tiles

By default we output to:
  tiles/<image_stem>/tile_x{X}_y{Y}.png
so multiple original images won't collide.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def tile_one_image(
    img_path: Path,
    out_root: Path,
    tile_size: int,
    overlap_ratio: float,
    pad_mode: str,
    pad_color: str,
    nested: bool,
) -> int:
    """Slice one image into fixed-size tiles.

    Args:
        img_path: original image
        out_root: output folder root
        tile_size: width/height of each tile
        overlap_ratio: 0.0~0.9, larger -> denser
        pad_mode: 'skip' (drop incomplete edge tiles) or 'pad' (pad to tile_size)
        pad_color: 'white' or 'black'
        nested: True -> out_root/<stem>/..., False -> out_root/...
    """

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] cannot read: {img_path}")
        return 0

    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap_ratio))
    if step <= 0:
        raise ValueError("overlap_ratio too large -> step <= 0")

    out_dir = (out_root / img_path.stem) if nested else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    value = (255, 255, 255) if pad_color == "white" else (0, 0, 0)

    count = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img[y : y + tile_size, x : x + tile_size]
            th, tw = tile.shape[:2]

            if th != tile_size or tw != tile_size:
                if pad_mode == "skip":
                    continue
                pad_bottom = max(0, tile_size - th)
                pad_right = max(0, tile_size - tw)
                tile = cv2.copyMakeBorder(
                    tile,
                    top=0,
                    bottom=pad_bottom,
                    left=0,
                    right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=value,
                )

            out_name = f"tile_x{x}_y{y}.png"
            cv2.imwrite(str(out_dir / out_name), tile)
            count += 1

    print(f"[OK] {img_path.name} -> {count} tiles -> {out_dir}")
    return count


def main() -> None:
    # NOTE:
    # argparse allows option abbreviation by default.
    # When UI/user passes "--pad", it could match both "--pad_mode" and "--pad_color",
    # causing: "ambiguous option: --pad could match --pad_mode, --pad_color".
    # We add an explicit "--pad" alias for convenience/compat, and keep the original
    # "--pad_mode" / "--pad_color" options.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", default="inputs", help="folder containing original images"
    )
    ap.add_argument("--out_dir", default="tiles", help="output root folder")
    ap.add_argument("--tile_size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--pad_mode", choices=["skip", "pad"], default="skip")
    ap.add_argument("--pad_color", choices=["white", "black"], default="white")
    ap.add_argument(
        "--pad",
        action="store_true",
        help="alias of --pad_mode pad (edge padding to avoid missing objects on tile borders)",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="output directly into out_dir (no per-image subfolder)",
    )
    args = ap.parse_args()

    # Backward-compatible alias: --pad => --pad_mode pad
    if getattr(args, "pad", False):
        args.pad_mode = "pad"

    in_dir = Path(args.input_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(in_dir))
    if not images:
        raise FileNotFoundError(f"No images in {in_dir} (supported: {sorted(EXTS)})")

    total = 0
    for img_path in images:
        total += tile_one_image(
            img_path,
            out_root,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
            pad_mode=args.pad_mode,
            pad_color=args.pad_color,
            nested=not args.flat,
        )

    print(f"Done. total_tiles={total} out_dir={out_root.resolve()}")


if __name__ == "__main__":
    main()
