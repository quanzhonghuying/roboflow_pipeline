"""
Step2: Run Roboflow detection on tiles (Step1 output) and save pred_tiles.json.

✅ 正确用途（完整 pipeline）：
    python step2_infer_tiles_cli.py --tiles_dir <Step1 tiles dir> ...

✅ Debug 用途（只测一张图能不能推理）：
    python step2_infer_tiles_cli.py --input_file <any image> ...

注意：
- 只用 --input_file 推理“源文件”时，输出仍会写 pred_tiles.json，
  但这不保证能无缝进入 Step3 合并（因为可能缺少 tile offset 信息）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import requests


# -----------------------------
# Roboflow Detect API
# -----------------------------
def infer_image(api_key: str, model: str, version: str, img_path: str) -> Dict:
    """
    Call Roboflow Detect API for ONE image.
    Return JSON like:
      {"predictions":[{"x","y","width","height","class","confidence"}, ...]}
    """
    url = f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    return resp.json()


# -----------------------------
# Image discovery (support *.png.rf.*)
# -----------------------------
def looks_like_image(path: str) -> bool:
    name = os.path.basename(path).lower()
    normal_ext = (".png", ".jpg", ".jpeg", ".webp")
    if name.endswith(normal_ext):
        return True
    rf_markers = (".png.rf.", ".jpg.rf.", ".jpeg.rf.", ".webp.rf.")
    if any(m in name for m in rf_markers):
        return True
    return False


def collect_images_from_dir(dir_path: str) -> List[str]:
    img_paths: List[str] = []
    for root, _, files in os.walk(dir_path):
        for fn in files:
            p = os.path.join(root, fn)
            if looks_like_image(p):
                img_paths.append(p)
    img_paths.sort()
    return img_paths


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--api_key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)

    # ✅ 不再 required：允许单文件模式
    ap.add_argument("--tiles_dir", default="", help="Step1 output tiles folder")
    ap.add_argument(
        "--input_file", default="", help="Infer only one image (debug / single)"
    )

    ap.add_argument("--out_dir", default="outputs/tiles_infer")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--max_tiles", type=int, default=0, help="0 means no limit")

    # 你原本 usage 里有 --draw_overlay（不影响的话保留）
    ap.add_argument(
        "--draw_overlay",
        action="store_true",
        help="(optional) draw overlay if implemented",
    )

    args = ap.parse_args()

    # ---------- validate ----------
    if not args.api_key:
        raise ValueError("api_key is empty. Provide --api_key or set ROBOFLOW_API_KEY.")

    if not args.tiles_dir and not args.input_file:
        ap.error("either --tiles_dir OR --input_file is required")

    # decide images
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"--input_file not found: {args.input_file}")
        img_paths = [args.input_file]
        mode = "single_file"
    else:
        if not os.path.isdir(args.tiles_dir):
            raise FileNotFoundError(f"--tiles_dir not found: {args.tiles_dir}")
        img_paths = collect_images_from_dir(args.tiles_dir)
        mode = "tiles_dir"

    if not img_paths:
        raise FileNotFoundError("No images found to infer.")

    if args.max_tiles and args.max_tiles > 0:
        img_paths = img_paths[: args.max_tiles]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Step2] mode={mode}")
    print(f"[Step2] images={len(img_paths)} out_dir={args.out_dir}")

    # ---------- infer ----------
    pred_tiles: List[Dict] = []

    for img_path in img_paths:
        tile_name = os.path.basename(img_path)

        try:
            pred = infer_image(args.api_key, args.model, args.version, img_path)
        except requests.HTTPError as e:
            print(f"[ERROR] API failed for: {img_path}")
            print(f"        HTTPError: {e}")
            continue

        for p in pred.get("predictions", []):
            conf = float(p.get("confidence", 0.0))
            if conf < args.conf:
                continue

            cls = p.get("class", "")

            # Roboflow bbox: center (x,y) + (w,h)
            cx, cy = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])

            # to xyxy (tile-local)
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2

            pred_tiles.append(
                {
                    "tile_name": tile_name,
                    "file_path": img_path,
                    "class": cls,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    # 重要：告诉后续这是“单文件 debug”还是“tile 推理”
                    "mode": mode,
                }
            )

    out_path = os.path.join(args.out_dir, "pred_tiles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred_tiles, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_path} boxes={len(pred_tiles)}")

    if mode == "single_file":
        print(
            "[NOTE] single_file mode is mainly for debug. "
            "If this file is NOT a real Step1 tile, Step3 merge may not behave as expected."
        )


if __name__ == "__main__":
    main()
