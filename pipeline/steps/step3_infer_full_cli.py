import argparse
import json
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


def parse_xy(name: str) -> Optional[Tuple[int, int]]:
    """Parse tile origin (ox, oy) from a tile file name.

    We intentionally accept a *wide* range of naming schemes.

    Why:
      - Some pipelines produce standard names like: tile_x0_y512.png
      - Roboflow exports / caches can create names like: tile_x0_y512.png.rf.<hash>
      - Users may have tiles in nested folders.

    So we look for the pattern `tile_x<digits>_y<digits>` anywhere in the name.
    """
    m = re.search(r"tile_x(\d+)_y(\d+)", os.path.basename(name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def list_tile_paths(tiles_dir: str) -> List[str]:
    """Recursively list tile image file paths under tiles_dir.

    Robust to:
      - nested directories
      - weird extensions (e.g., `.png.rf.<hash>`)

    Returns a sorted list of file paths.
    """
    if not os.path.isdir(tiles_dir):
        return []

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    out: List[str] = []
    for root, _dirs, files in os.walk(tiles_dir):
        for fn in files:
            p = os.path.join(root, fn)
            low = fn.lower()
            _, ext = os.path.splitext(low)

            # Normal image extensions
            if ext in exts:
                out.append(p)
                continue

            # Roboflow-ish naming where extension is "broken"
            if (
                ".png.rf." in low
                or ".jpg.rf." in low
                or ".jpeg.rf." in low
                or ".webp.rf." in low
            ):
                out.append(p)
                continue

            # If it still looks like a tile, accept it
            if re.search(r"tile_x\d+_y\d+", low):
                out.append(p)

    out.sort()
    return out


def infer_tile(api_key: str, model: str, version: str, tile_path: str) -> Dict:
    url = f"https://detect.roboflow.com/{model}/{version}?api_key={api_key}"
    with open(tile_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    return resp.json()


def iou_xyxy(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = aa + bb - inter
    return 0.0 if u <= 0 else inter / u


def nms_classwise(boxes: List[Dict], iou_th: float) -> List[Dict]:
    out: List[Dict] = []
    for cls in sorted(set(b.get("class", "") for b in boxes)):
        group = [b for b in boxes if b.get("class", "") == cls]
        group.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        kept: List[Dict] = []
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [b for b in group if iou_xyxy(best, b) < iou_th]
        out.extend(kept)
    return out


def is_mostly_white_bgr(img: np.ndarray, thr: int = 245, ratio: float = 0.95) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white = (gray >= thr).mean()
    return white >= ratio


def tile_nonwhite_ratio(gray: np.ndarray, white_thr: int = 245) -> float:
    """Ratio of pixels that are NOT white-ish.

    For CAD / catalog line drawings, a tile can be "mostly white" but still contain
    thin lines or small symbols (e.g., a small window at the corner). Using the
    non-white ratio with a very small threshold (SAFE) avoids false skipping.
    """
    # gray in [0..255]; treat anything darker than white_thr as "ink".
    return float((gray < white_thr).mean())


def tile_edge_ratio(gray: np.ndarray) -> float:
    """Edge pixel ratio using Canny. More robust than plain whiteness for line art."""
    # Auto-ish thresholds based on median intensity.
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    return float((edges > 0).mean())


def is_blank_tile(
    bgr: np.ndarray,
    mode: str,
    white_thr: int,
    min_nonwhite_ratio: float,
    min_edge_ratio: float,
) -> bool:
    """Return True if tile is considered blank under the selected mode.

    OFF  : never blank
    SAFE : only blank if BOTH non-white ratio and edge ratio are extremely low
    FAST : more aggressive skipping (still uses BOTH signals)
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    nw = tile_nonwhite_ratio(gray, white_thr=white_thr)
    er = tile_edge_ratio(gray)
    return (nw < float(min_nonwhite_ratio)) and (er < float(min_edge_ratio))


def box_area_ratio(full_w: int, full_h: int, b: Dict) -> float:
    w = max(0, int(b["x2"]) - int(b["x1"]))
    h = max(0, int(b["y2"]) - int(b["y1"]))
    return (w * h) / max(1, full_w * full_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.getenv("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--input_image", required=True)
    ap.add_argument("--tiles_dir", default="tiles")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--conf", type=float, default=0.60)
    ap.add_argument("--nms_iou", type=float, default=0.50)
    ap.add_argument("--max_tiles", type=int, default=0, help="0 means no limit")

    # --- Tile skipping (recommended: use --skip_tiles_mode) ---
    # New mode-based strategy: OFF/SAFE/FAST with thresholds.
    ap.add_argument(
        "--skip_tiles_mode",
        choices=["off", "safe", "fast"],
        default="off",
        help="Tile skipping mode. off=no skip; safe=very conservative; fast=more aggressive.",
    )
    ap.add_argument(
        "--tile_white_thr",
        type=int,
        default=245,
        help="Gray threshold to consider as white-ish",
    )
    ap.add_argument(
        "--min_nonwhite_ratio",
        type=float,
        default=None,
        help="Override non-white ratio threshold (0..1). If omitted, uses mode defaults.",
    )
    ap.add_argument(
        "--min_edge_ratio",
        type=float,
        default=None,
        help="Override edge ratio threshold (0..1). If omitted, uses mode defaults.",
    )

    # Legacy flags kept for backward compatibility.
    ap.add_argument(
        "--skip_white_tiles",
        action="store_true",
        help="[legacy] skip tiles by white ratio",
    )
    ap.add_argument(
        "--tile_white_ratio",
        type=float,
        default=0.98,
        help="[legacy] white ratio threshold",
    )
    ap.add_argument("--filter_big_box", action="store_true")
    ap.add_argument("--big_box_area_ratio", type=float, default=0.15)
    ap.add_argument("--big_box_keep_conf", type=float, default=0.95)
    args = ap.parse_args()

    if not args.api_key:
        raise ValueError("api_key is empty. Provide --api_key or set ROBOFLOW_API_KEY.")

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(args.input_image)
    if img is None:
        raise FileNotFoundError(args.input_image)
    full_h, full_w = img.shape[:2]

    # NOTE: tiles can be nested or have Roboflow-style names like `*.png.rf.<hash>`.
    # So we use a robust recursive enumerator instead of a strict `endswith('.png')`.
    tile_paths = list_tile_paths(args.tiles_dir)
    if not tile_paths:
        raise FileNotFoundError(f"no tiles in {args.tiles_dir}")

    if args.max_tiles and args.max_tiles > 0:
        tile_paths = tile_paths[: args.max_tiles]

    all_boxes_raw: List[Dict] = []
    skipped_legacy_white = 0
    skipped_mode_blank = 0

    # Mode defaults tuned for CAD / catalog line drawings.
    # SAFE: almost never false-skip; FAST: skips more aggressively.
    if args.skip_tiles_mode == "safe":
        default_nonwhite = 0.0005  # 0.05%
        default_edge = 0.0002
    elif args.skip_tiles_mode == "fast":
        default_nonwhite = 0.003  # 0.3%
        default_edge = 0.001
    else:
        default_nonwhite = 0.0
        default_edge = 0.0

    min_nonwhite = (
        float(args.min_nonwhite_ratio)
        if args.min_nonwhite_ratio is not None
        else float(default_nonwhite)
    )
    min_edge = (
        float(args.min_edge_ratio)
        if args.min_edge_ratio is not None
        else float(default_edge)
    )

    for tile_path in tile_paths:
        fname = os.path.basename(tile_path)
        xy = parse_xy(fname)
        if xy is None:
            continue
        ox, oy = xy

        timg = None

        # Legacy strategy: whiteness ratio (can false-skip small corner objects).
        if args.skip_white_tiles:
            timg = cv2.imread(tile_path)
            if timg is None:
                continue
            if is_mostly_white_bgr(timg, args.tile_white_thr, args.tile_white_ratio):
                skipped_legacy_white += 1
                continue

        # New strategy: OFF/SAFE/FAST based on non-white ratio + edge ratio.
        if (not args.skip_white_tiles) and args.skip_tiles_mode != "off":
            if timg is None:
                timg = cv2.imread(tile_path)
            if timg is None:
                continue
            if is_blank_tile(
                timg,
                mode=args.skip_tiles_mode,
                white_thr=int(args.tile_white_thr),
                min_nonwhite_ratio=min_nonwhite,
                min_edge_ratio=min_edge,
            ):
                skipped_mode_blank += 1
                continue

        pred = infer_tile(args.api_key, args.model, args.version, tile_path)
        for p in pred.get("predictions", []):
            conf = float(p.get("confidence", 0.0))
            if conf < args.conf:
                continue
            cls = p.get("class", "")
            cx = float(p["x"]) + ox
            cy = float(p["y"]) + oy
            bw = float(p["width"])
            bh = float(p["height"])
            x1 = int(max(0, min(full_w - 1, cx - bw / 2)))
            y1 = int(max(0, min(full_h - 1, cy - bh / 2)))
            x2 = int(max(0, min(full_w - 1, cx + bw / 2)))
            y2 = int(max(0, min(full_h - 1, cy + bh / 2)))
            if x2 <= x1 or y2 <= y1:
                continue
            all_boxes_raw.append(
                {
                    "class": cls,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "tile_file": fname,
                }
            )

    boxes_nms = nms_classwise(all_boxes_raw, args.nms_iou)

    boxes_final: List[Dict] = []
    for b in boxes_nms:
        if (
            args.filter_big_box
            and box_area_ratio(full_w, full_h, b) >= args.big_box_area_ratio
        ):
            if float(b.get("confidence", 0.0)) < args.big_box_keep_conf:
                continue
        boxes_final.append(b)

    raw_path = os.path.join(args.out_dir, "result_boxes_raw.json")
    nms_path = os.path.join(args.out_dir, "result_boxes.json")
    final_path = os.path.join(args.out_dir, "result_boxes_final.json")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_boxes_raw, f, ensure_ascii=False, indent=2)
    with open(nms_path, "w", encoding="utf-8") as f:
        json.dump(boxes_nms, f, ensure_ascii=False, indent=2)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(boxes_final, f, ensure_ascii=False, indent=2)

    vis = img.copy()
    for b in boxes_final:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{b['class']}:{float(b['confidence']):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    out_img = os.path.join(args.out_dir, "result_drawn.png")
    cv2.imwrite(out_img, vis)

    print(
        f"[OK] tiles inferred: {len(tile_paths)}, skipped_legacy_white: {skipped_legacy_white}, skipped_mode_blank: {skipped_mode_blank}"
    )
    print(
        f"[OK] boxes raw: {len(all_boxes_raw)}, nms: {len(boxes_nms)}, final: {len(boxes_final)}"
    )
    print(f"[OUT] {out_img}")
    print(f"[OUT] {final_path}")


if __name__ == "__main__":
    main()
