"""
Step2: infer tiles via Roboflow Detect API + (optional) draw overlay images.

- tiles_dir mode: infer all images under tiles_dir (recursive)
- single mode: infer one image via --input_file (or legacy --single)
- output: <out_dir>/pred_tiles.json
- overlay: <out_dir>/overlay_tiles/<basename>.png  (when --draw_overlay)

注意：
- overlay 绘制是“目视确认用”，不会影响 Step3/Step4。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import requests


def infer_image(api_key: str, model: str, version: str, img_path: str) -> Dict:
    url = f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    return resp.json()


def looks_like_image(path: str) -> bool:
    name = os.path.basename(path).lower()
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return True
    if any(m in name for m in (".png.rf.", ".jpg.rf.", ".jpeg.rf.", ".webp.rf.")):
        return True
    return False


def collect_images_from_dir(tiles_dir: str) -> List[str]:
    p = Path(tiles_dir)
    if not p.exists():
        return []
    files: List[str] = []
    for f in p.rglob("*"):
        if f.is_file() and looks_like_image(str(f)):
            files.append(str(f))
    files.sort()
    return files


def safe_overlay_name(img_path: str) -> str:
    """
    出力ファイル名を安全にする：
    - tile_x0_y0.png.rf.xxx みたいな名前でも、拡張子を .png に統一して保存
    """
    base = os.path.basename(img_path)
    # 末尾が .png/.jpg じゃないケース(.png.rf.xxx 等)にも対応
    # ここでは「.png を含むなら .png まで」を優先
    lower = base.lower()
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        idx = lower.find(ext)
        if idx != -1:
            return base[: idx + len(ext)]
    # どれも見つからなければ、そのまま + .png
    return base + ".png"


def draw_overlay(img_path: str, preds: List[Dict], out_path: str) -> None:
    """
    できるだけ依存を増やさないため、Pillow があれば Pillow、なければ OpenCV で描画する。
    """
    # ---------- try Pillow ----------
    try:
        from PIL import Image, ImageDraw  # type: ignore

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)

        for p in preds:
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            cls = p.get("class", "")
            conf = p.get("confidence", 0.0)

            # 枠
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            # ラベル
            txt = f"{cls} {conf:.2f}"
            draw.text((x1 + 2, max(0, y1 - 14)), txt, fill=(255, 0, 0))

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)
        return
    except Exception:
        pass

    # ---------- fallback OpenCV ----------
    try:
        import cv2  # type: ignore

        img = cv2.imread(img_path)
        if img is None:
            return

        for p in preds:
            x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
            cls = p.get("class", "")
            conf = float(p.get("confidence", 0.0))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{cls} {conf:.2f}",
                (x1 + 2, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, img)
        return
    except Exception:
        # Pillow も OpenCV も無い場合は描画できない（でも推論自体は成功させる）
        return


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--api_key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)

    ap.add_argument("--tiles_dir", default="", help="tiles folder (Step1 output)")
    ap.add_argument("--input_file", default="", help="infer only one image file")

    # legacy alias (older UI)
    ap.add_argument(
        "--single",
        dest="input_file",
        default="",
        help="(legacy) alias of --input_file",
    )

    ap.add_argument("--out_dir", default="outputs/tiles_infer")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--max_tiles", type=int, default=0)
    ap.add_argument("--draw_overlay", action="store_true")

    args = ap.parse_args()

    if not args.api_key:
        raise ValueError("api_key is empty. Provide --api_key or set ROBOFLOW_API_KEY.")

    if not args.tiles_dir and not args.input_file:
        ap.error("either --tiles_dir OR --input_file/--single is required")

    mode = "single_file" if args.input_file else "tiles_dir"

    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"input_file not found: {args.input_file}")
        img_paths = [args.input_file]
    else:
        img_paths = collect_images_from_dir(args.tiles_dir)

    if not img_paths:
        raise FileNotFoundError("No images found to infer.")

    if args.max_tiles and args.max_tiles > 0:
        img_paths = img_paths[: args.max_tiles]

    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = str(Path(args.out_dir) / "overlay_tiles")

    print(f"[Step2] mode={mode} images={len(img_paths)} out_dir={args.out_dir}")

    pred_tiles: List[Dict] = []

    for img_path in img_paths:
        tile_name = os.path.basename(img_path)

        try:
            pred = infer_image(args.api_key, args.model, args.version, img_path)
        except requests.HTTPError as e:
            print(f"[ERROR] API failed for: {img_path}")
            print(f"        HTTPError: {e}")
            continue

        # 这一张图的 bbox，用于 overlay
        one_img_preds: List[Dict] = []

        for p in pred.get("predictions", []):
            conf = float(p.get("confidence", 0.0))
            if conf < args.conf:
                continue

            cls = p.get("class", "")
            cx, cy = float(p["x"]), float(p["y"])
            bw, bh = float(p["width"]), float(p["height"])

            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2

            item = {
                "tile_name": tile_name,
                "file_path": img_path,
                "class": cls,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mode": mode,
            }
            pred_tiles.append(item)
            one_img_preds.append(item)

        # overlay 出力
        if args.draw_overlay:
            out_name = safe_overlay_name(img_path)
            out_path = str(Path(overlay_dir) / out_name)
            draw_overlay(img_path, one_img_preds, out_path)

    out_path = os.path.join(args.out_dir, "pred_tiles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred_tiles, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_path} boxes={len(pred_tiles)}")
    if args.draw_overlay:
        print(f"[OK] overlay dir: {overlay_dir}")


if __name__ == "__main__":
    main()
