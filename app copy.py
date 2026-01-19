#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""legacy/step4_tile_eval.py (FINAL)

目的
- COCO(教師データ) と pred JSON(推論結果) を突き合わせて TP / FP / FN を計算
- タイル画像へ overlay を出力
- TP/FP/FN の詳細を CSV にも出力（Review/分析で使う）

重要な前提
- “正しい評価” の絶対条件は、GT(COCO) と Pred が
  1) 同じ画像集合
  2) 同じ座標系
  を使っていることです。

本スクリプトは「ズレが起きがちな実務ケース」への自動救済を入れています:
- pred が global 座標のまま混じる
- pred の tile_name が誤って付く
- 512/640 の tile_size が混じる

ただし最も確実なのは、評価用 pred は “COCO に含まれる tile 画像” をそのまま推論して作ることです。
(＝ step2_infer_coco_tiles_cli.py を使う)

入力
- --coco: coco_valid/_annotations.coco.json
- --pred: outputs/.../pred_tiles.json (list or dict)

出力
- out_dir/
  - eval_report.json
  - tp.csv / fp.csv / fn.csv
  - overlay_tile_x{X}_y{Y}_{cls}.png (タイルごと)

"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

# -------------------------
# 基本ユーティリティ
# -------------------------


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + bb - inter
    return float(inter / denom) if denom > 0 else 0.0


def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def parse_tile_origin(text: str) -> Optional[Tuple[int, int]]:
    """tile_x{X}_y{Y} を含む文字列から origin を抽出"""
    m = re.search(r"tile_x(\d+)_y(\d+)", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def canonical_tile_key_from_origin(x0: int, y0: int) -> str:
    """内部キー（.png で統一）"""
    return f"tile_x{x0}_y{y0}.png"


def canonical_tile_key_from_any(name: str) -> Optional[str]:
    origin = parse_tile_origin(name)
    if not origin:
        return None
    return canonical_tile_key_from_origin(*origin)


def list_images(root_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    out = []
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


# -------------------------
# データ構造
# -------------------------


@dataclass
class Box:
    xyxy: Tuple[float, float, float, float]
    cls: str
    conf: float


# -------------------------
# COCO 読み込み
# -------------------------


def build_coco_index(coco: dict):
    # categories
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # images: tile_key -> (image_id, file_name)
    tile_to_img = {}
    imgid_to_file = {}
    imgid_to_wh = {}
    for im in coco.get("images", []):
        file_name = im.get("file_name", "")
        img_id = im.get("id")
        imgid_to_file[img_id] = file_name
        imgid_to_wh[img_id] = (int(im.get("width", 0)), int(im.get("height", 0)))
        tile_key = canonical_tile_key_from_any(file_name)
        if tile_key:
            tile_to_img[tile_key] = (img_id, file_name)

    # annotations: img_id -> list[Box]
    gt_by_img = {im.get("id"): [] for im in coco.get("images", [])}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        cat = cat_id_to_name.get(ann.get("category_id"), str(ann.get("category_id")))
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        gt_by_img.setdefault(img_id, []).append(Box((x1, y1, x2, y2), cat, 1.0))

    return {
        "cat_id_to_name": cat_id_to_name,
        "tile_to_img": tile_to_img,
        "imgid_to_file": imgid_to_file,
        "imgid_to_wh": imgid_to_wh,
        "gt_by_img": gt_by_img,
    }


# -------------------------
# Pred 読み込み（多形式対応）
# -------------------------


def normalize_pred_list(pred_obj) -> List[dict]:
    """pred の JSON 形式ゆらぎを list[dict] に正規化"""
    if isinstance(pred_obj, list):
        return pred_obj
    if isinstance(pred_obj, dict):
        # ありがちな形式
        for k in ("pred", "predictions", "items", "results"):
            if k in pred_obj and isinstance(pred_obj[k], list):
                return pred_obj[k]
        # tile_key -> list
        if all(isinstance(v, list) for v in pred_obj.values()):
            out = []
            for k, vs in pred_obj.items():
                for p in vs:
                    if isinstance(p, dict):
                        p = dict(p)
                        p.setdefault("tile_name", k)
                        out.append(p)
            return out
    raise ValueError("Unsupported pred JSON format")


def pred_to_xyxy(p: dict) -> Tuple[float, float, float, float]:
    # 1) already xyxy
    if all(k in p for k in ("x1", "y1", "x2", "y2")):
        return (
            safe_float(p["x1"]),
            safe_float(p["y1"]),
            safe_float(p["x2"]),
            safe_float(p["y2"]),
        )

    # 2) center format from Roboflow
    if all(k in p for k in ("x", "y", "width", "height")):
        cx = safe_float(p["x"])  # center
        cy = safe_float(p["y"])
        w = safe_float(p["width"])
        h = safe_float(p["height"])
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    # 3) bbox list [x,y,w,h]
    if "bbox" in p and isinstance(p["bbox"], (list, tuple)) and len(p["bbox"]) >= 4:
        x, y, w, h = p["bbox"][:4]
        x, y, w, h = safe_float(x), safe_float(y), safe_float(w), safe_float(h)
        return x, y, x + w, y + h

    raise ValueError(f"Unknown box format: keys={list(p.keys())}")


def pred_class(p: dict) -> str:
    for k in ("class", "label", "name", "category"):
        if k in p and p[k] is not None:
            return str(p[k])
    # numeric class id
    if "class_id" in p:
        return str(p["class_id"])
    return "unknown"


def pred_conf(p: dict) -> float:
    for k in ("confidence", "conf", "score", "prob"):
        if k in p:
            return safe_float(p[k], 0.0)
    return 1.0


def pred_tile_hint(p: dict) -> str:
    for k in ("tile_name", "image", "file_name", "path", "file_path"):
        if k in p and p[k]:
            return str(p[k])
    return ""


# -------------------------
# 画像パス探索
# -------------------------


def find_image_path_for_tile(
    tile_key: str, coco_dir: str, extra_search_dirs: List[str]
) -> Optional[str]:
    """tile_key(tile_x.._y..png) に対応する実ファイルを探す。

    Roboflow の COCO では file_name が
      tile_x0_y0.png.rf.<hash>.jpg
    のようになるので、tile_x.._y.. を含むファイルを優先的に探す。
    """
    origin = parse_tile_origin(tile_key)
    if not origin:
        return None
    x0, y0 = origin

    patterns = [
        os.path.join(coco_dir, f"**/*tile_x{x0}_y{y0}*"),
    ]
    for d in extra_search_dirs:
        patterns.append(os.path.join(d, f"**/*tile_x{x0}_y{y0}*"))

    exts = (".png", ".jpg", ".jpeg", ".webp")
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p) and p.lower().endswith(exts):
                return p
    return None


def read_image_wh(path: str) -> Optional[Tuple[int, int]]:
    if cv2 is None:
        return None
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return None


# -------------------------
# 座標系の自動判定・補正
# -------------------------


def guess_coord_mode(preds: List[dict], tile_size: int) -> str:
    """tile_size に対して座標が大きくはみ出るなら global と判定"""
    mx = 0.0
    my = 0.0
    for p in preds[: min(len(preds), 5000)]:
        try:
            x1, y1, x2, y2 = pred_to_xyxy(p)
            mx = max(mx, x1, x2)
            my = max(my, y1, y2)
        except Exception:
            continue
    if mx > tile_size * 1.15 or my > tile_size * 1.15:
        return "global"
    return "tile"


def assign_tile_by_center_global(x1, y1, x2, y2, tile_size: int) -> Tuple[int, int]:
    """global 座標の box 中心から属する tile origin を推定"""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ox = int(cx // tile_size) * tile_size
    oy = int(cy // tile_size) * tile_size
    return ox, oy


# -------------------------
# 描画
# -------------------------


def draw_boxes(
    img,
    boxes: List[Tuple[Tuple[float, float, float, float], Tuple[int, int, int], str]],
):
    """boxes: list[(xyxy, color(BGR), text)]"""
    if cv2 is None:
        return img
    for (x1, y1, x2, y2), color, text in boxes:
        x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, 2)
        if text:
            cv2.putText(
                img,
                text,
                (x1i, max(0, y1i - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return img


# -------------------------
# メイン評価
# -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO json path")
    ap.add_argument("--pred", required=True, help="pred json path")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--iou", type=float, default=0.5)
    # クラス指定
    # - 新 UI: --classes
    # - 旧 UI: --cls
    ap.add_argument(
        "--classes", nargs="*", default=None, help="evaluate only these classes (names)"
    )
    ap.add_argument("--cls", nargs="*", dest="classes", help=argparse.SUPPRESS)
    ap.add_argument("--all_classes", action="store_true", help="evaluate all classes")
    ap.add_argument(
        "--tile_size",
        type=int,
        default=None,
        help="(optional) override tile size for global assignment",
    )
    # 座標系
    # - 新 UI: --coord
    # - 旧 UI: --pred_coord
    ap.add_argument("--coord", choices=["auto", "tile", "global"], default="auto")
    # 旧 UI 互換: --pred_coord
    ap.add_argument(
        "--pred_coord",
        dest="coord",
        choices=["auto", "tile", "global"],
        help=argparse.SUPPRESS,
    )
    # Overlay is ON by default (because most users expect an image output).
    # Use --no_overlay to disable overlay generation.
    ap.add_argument(
        "--no_overlay", action="store_true", help="disable overlay image output"
    )
    # Backward compatible flag (older UI/scripts may still pass this).
    ap.add_argument("--make_overlay", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max_tiles", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    # Backward compatible: --make_overlay exists, but overlay is enabled by default
    args.make_overlay = (not getattr(args, "no_overlay", False)) or getattr(
        args, "make_overlay", False
    )
    # Effective overlay flag: default ON unless explicitly disabled.
    args.make_overlay_effective = (not args.no_overlay) or bool(
        getattr(args, "make_overlay", False)
    )

    ensure_dir(args.out)

    coco = load_json(args.coco)
    idx = build_coco_index(coco)

    coco_dir = os.path.dirname(os.path.abspath(args.coco))

    pred_obj = load_json(args.pred)
    pred_list = normalize_pred_list(pred_obj)

    # classes selection
    coco_class_names = sorted({c["name"] for c in coco.get("categories", [])})
    if args.all_classes or (args.classes is None and not args.all_classes):
        # UI から classes が空で来ることがあるので、"空" のときは全部ではなく coco の全クラス
        selected = (
            set(coco_class_names)
            if args.all_classes
            else set(args.classes or coco_class_names)
        )
    else:
        selected = set(args.classes or [])

    # determine tile_size for global assignment
    tile_size_guess = args.tile_size
    if tile_size_guess is None:
        # COCO images から most common width を採用
        widths = [
            int(im.get("width", 0))
            for im in coco.get("images", [])
            if int(im.get("width", 0)) > 0
        ]
        if widths:
            # mode-ish
            tile_size_guess = sorted(widths)[len(widths) // 2]
        else:
            tile_size_guess = 512

    coord_mode = args.coord
    if coord_mode == "auto":
        coord_mode = guess_coord_mode(pred_list, tile_size_guess)

    # pred を tile_key -> list[Box] に集約
    pred_by_tile: Dict[str, List[Box]] = {}

    # 追加探索: 典型の tiles/ や project root
    project_root = os.getcwd()
    extra_dirs = [os.path.join(project_root, "tiles"), project_root]

    # pred 座標が tile_size を超えるケース（= tile_size mismatch）検出用
    max_seen = {"x": 0.0, "y": 0.0}

    for p in pred_list:
        try:
            cls = pred_class(p)
            if cls not in selected:
                continue

            x1, y1, x2, y2 = pred_to_xyxy(p)
            conf = pred_conf(p)
            max_seen["x"] = max(max_seen["x"], x1, x2)
            max_seen["y"] = max(max_seen["y"], y1, y2)

            hint = pred_tile_hint(p)

            if coord_mode == "global":
                # tile_name が間違うことがあるので、中心から tile を決める
                ox, oy = assign_tile_by_center_global(x1, y1, x2, y2, tile_size_guess)
                tile_key = canonical_tile_key_from_origin(ox, oy)
                x1, y1, x2, y2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
            else:
                tile_key = canonical_tile_key_from_any(
                    hint
                ) or canonical_tile_key_from_any(str(p))
                if tile_key is None:
                    # tile_hint から取れない場合、global と同様に中心から決める（安全側）
                    ox, oy = assign_tile_by_center_global(
                        x1, y1, x2, y2, tile_size_guess
                    )
                    tile_key = canonical_tile_key_from_origin(ox, oy)
                    x1, y1, x2, y2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy

            pred_by_tile.setdefault(tile_key, []).append(
                Box((x1, y1, x2, y2), cls, conf)
            )

        except Exception:
            continue

    # tile_size mismatch のヒント
    mismatch_hint = None
    if max_seen["x"] > tile_size_guess * 1.05 or max_seen["y"] > tile_size_guess * 1.05:
        mismatch_hint = {
            "msg": "Pred coordinates exceed COCO tile_size guess. This often means you inferred on different tile_size (e.g., 640 tiles) than COCO (e.g., 512). Best fix: regenerate pred by inferring directly on COCO tile images.",
            "tile_size_guess": tile_size_guess,
            "pred_max": max_seen,
        }

    # 以降、COCO に含まれる tile を基準に評価
    tp_rows, fp_rows, fn_rows = [], [], []

    total_tp = total_fp = total_fn = 0

    # tiles list from COCO
    tiles = list(idx["tile_to_img"].items())
    if args.max_tiles is not None:
        tiles = tiles[: args.max_tiles]

    for tile_key, (img_id, file_name) in tiles:
        gt_list = [b for b in idx["gt_by_img"].get(img_id, []) if b.cls in selected]
        pr_list = [b for b in pred_by_tile.get(tile_key, []) if b.cls in selected]

        # locate image file
        img_path = find_image_path_for_tile(
            tile_key, coco_dir=coco_dir, extra_search_dirs=extra_dirs
        )

        # determine per-tile size for clamping
        tile_w, tile_h = idx["imgid_to_wh"].get(
            img_id, (tile_size_guess, tile_size_guess)
        )
        if img_path:
            wh = read_image_wh(img_path)
            if wh:
                tile_w, tile_h = wh

        # clamp all boxes
        gt_xy = []
        for g in gt_list:
            x1, y1, x2, y2 = clamp_xyxy(*g.xyxy, tile_w, tile_h)
            gt_xy.append(Box((x1, y1, x2, y2), g.cls, 1.0))

        pr_xy = []
        for pbox in pr_list:
            x1, y1, x2, y2 = pbox.xyxy
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, tile_w, tile_h)
            pr_xy.append(Box((x1, y1, x2, y2), pbox.cls, pbox.conf))

        matched_gt = set()
        matched_pr = set()

        # greedy match by confidence desc
        pr_order = sorted(range(len(pr_xy)), key=lambda i: pr_xy[i].conf, reverse=True)

        for pi in pr_order:
            p = pr_xy[pi]
            best_iou = 0.0
            best_gi = None
            for gi, g in enumerate(gt_xy):
                if gi in matched_gt:
                    continue
                if g.cls != p.cls:
                    continue
                iou = iou_xyxy(p.xyxy, g.xyxy)
                if iou >= args.iou and iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi is not None:
                matched_gt.add(best_gi)
                matched_pr.add(pi)
                total_tp += 1
                tp_rows.append(
                    {
                        "tile": tile_key,
                        "img_file": file_name,
                        "class": p.cls,
                        "conf": round(p.conf, 4),
                        "iou": round(best_iou, 4),
                        "x1": round(p.xyxy[0], 2),
                        "y1": round(p.xyxy[1], 2),
                        "x2": round(p.xyxy[2], 2),
                        "y2": round(p.xyxy[3], 2),
                    }
                )

        # FP
        for pi, p in enumerate(pr_xy):
            if pi in matched_pr:
                continue
            total_fp += 1
            fp_rows.append(
                {
                    "tile": tile_key,
                    "img_file": file_name,
                    "class": p.cls,
                    "conf": round(p.conf, 4),
                    "x1": round(p.xyxy[0], 2),
                    "y1": round(p.xyxy[1], 2),
                    "x2": round(p.xyxy[2], 2),
                    "y2": round(p.xyxy[3], 2),
                }
            )

        # FN
        for gi, g in enumerate(gt_xy):
            if gi in matched_gt:
                continue
            total_fn += 1
            fn_rows.append(
                {
                    "tile": tile_key,
                    "img_file": file_name,
                    "class": g.cls,
                    "x1": round(g.xyxy[0], 2),
                    "y1": round(g.xyxy[1], 2),
                    "x2": round(g.xyxy[2], 2),
                    "y2": round(g.xyxy[3], 2),
                }
            )

        # overlay
        if args.make_overlay_effective and img_path and cv2 is not None:
            img = cv2.imread(img_path)
            if img is not None:
                # TP: green, FP: red, FN: yellow
                draw_list = []
                # FN
                for gi, g in enumerate(gt_xy):
                    if gi not in matched_gt:
                        draw_list.append((g.xyxy, (0, 255, 255), f"FN[{g.cls}]"))
                # FP
                for pi, p in enumerate(pr_xy):
                    if pi not in matched_pr:
                        draw_list.append(
                            (p.xyxy, (0, 0, 255), f"FP[{p.cls}] {p.conf:.2f}")
                        )
                # TP
                for row in tp_rows[-len(matched_pr) :]:
                    pass
                # TP boxes as green for this tile
                for pi in matched_pr:
                    p = pr_xy[pi]
                    draw_list.append((p.xyxy, (0, 255, 0), f"TP[{p.cls}] {p.conf:.2f}"))

                img2 = draw_boxes(img, draw_list)
                out_name = (
                    f"overlay_{os.path.basename(tile_key).replace('.png','')}_"
                    + "_".join(sorted(selected))
                    + ".png"
                )
                cv2.imwrite(os.path.join(args.out, out_name), img2)

    # CSV 出力（空でもヘッダ作る）
    def write_csv(rows: List[dict], path: str, fieldnames: List[str]):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(
        tp_rows,
        os.path.join(args.out, "tp.csv"),
        ["tile", "img_file", "class", "conf", "iou", "x1", "y1", "x2", "y2"],
    )
    write_csv(
        fp_rows,
        os.path.join(args.out, "fp.csv"),
        ["tile", "img_file", "class", "conf", "x1", "y1", "x2", "y2"],
    )
    write_csv(
        fn_rows,
        os.path.join(args.out, "fn.csv"),
        ["tile", "img_file", "class", "x1", "y1", "x2", "y2"],
    )

    report = {
        "coco": os.path.abspath(args.coco),
        "pred": os.path.abspath(args.pred),
        "out": os.path.abspath(args.out),
        "iou": args.iou,
        "coord_mode": coord_mode,
        "tile_size_guess": tile_size_guess,
        "selected_classes": sorted(selected),
        "tiles_in_coco": len(idx["tile_to_img"]),
        "tiles_with_pred": len(pred_by_tile),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "mismatch_hint": mismatch_hint,
    }

    dump_json(report, os.path.join(args.out, "eval_report.json"))

    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"[OK] TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"[OUT] {os.path.join(args.out, 'eval_report.json')}")


if __name__ == "__main__":
    main()
