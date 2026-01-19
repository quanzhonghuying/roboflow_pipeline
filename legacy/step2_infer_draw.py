import os
import re
import json
import cv2
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
from itertools import combinations

# ============ 需要填的配置 ============
ROBOFLOW_API_KEY = "4s4HpbmcTQoex2Hm1o5P".strip()
MODEL = "window-door-demo"
VERSION = "4"
# =====================================

INPUT_IMAGE = "input.png"
TILES_DIR = "tiles"
OUT_DIR = "outputs"

MAX_TILES_TO_INFER = 50

CONF_TH = 0.60
NMS_IOU_TH = 0.5

SKIP_MOSTLY_WHITE_TILES = True
TILE_WHITE_THR = 245
TILE_WHITE_RATIO = 0.995

FILTER_MOSTLY_WHITE_BOX = True
BOX_WHITE_THR = 245
BOX_WHITE_RATIO = 0.98

FILTER_TOO_LARGE_BOX = True
BIG_BOX_AREA_RATIO = 0.15
BIG_BOX_CONF_KEEP = 0.85

# 统计信息开关（打开会输出重复框、类别数量、Top IoU 对等信息）
DEBUG_STATS = True
DUP_TOPK = 8


def parse_xy(filename: str) -> Optional[Tuple[int, int]]:
    """
    从文件名 tile_x{X}_y{Y}.png 解析出切块在原图中的左上角偏移量 (ox, oy)
    """
    m = re.search(r"tile_x(\d+)_y(\d+)\.png$", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def infer_tile(tile_path: str) -> Dict:
    """
    调用 Roboflow Hosted Detect API，对单个 tile 图片做推理，返回 JSON(dict)
    """
    url = f"https://detect.roboflow.com/{MODEL}/{VERSION}?api_key={ROBOFLOW_API_KEY}"
    with open(tile_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    return resp.json()


def iou_xyxy(a: Dict, b: Dict) -> float:
    """
    计算两个框的 IoU（交并比）
    a,b 均为 {"x1","y1","x2","y2"} 格式（像素坐标，原图坐标系）
    """
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms_classwise(boxes: List[Dict], iou_th: float) -> List[Dict]:
    """
    对 boxes 做 NMS（按 class 分组分别做）
    同类框按 confidence 从高到低排序，依次保留，删除与保留框 IoU>=阈值 的其他框
    """
    by_cls: Dict[str, List[Dict]] = {}
    for b in boxes:
        by_cls.setdefault(b["class"], []).append(b)

    kept_all: List[Dict] = []

    for _, cls_boxes in by_cls.items():
        cls_boxes = sorted(cls_boxes, key=lambda x: x["confidence"], reverse=True)

        kept: List[Dict] = []
        for b in cls_boxes:
            should_keep = True
            for kb in kept:
                if iou_xyxy(b, kb) >= iou_th:
                    should_keep = False
                    break
            if should_keep:
                kept.append(b)

        kept_all.extend(kept)

    kept_all.sort(key=lambda x: x["confidence"], reverse=True)
    return kept_all


def is_mostly_white_bgr(
    img_bgr: np.ndarray, white_thr: int, white_ratio: float
) -> bool:
    """
    判断整张图片是否几乎全白（用于 tile 跳过）
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ratio = float((gray >= white_thr).mean())
    return ratio >= white_ratio


def is_mostly_white_box(
    full_img_bgr: np.ndarray, box: Dict, white_thr: int, white_ratio: float
) -> bool:
    """
    判断框区域是否几乎全白（用于过滤误检大框）
    """
    h, w = full_img_bgr.shape[:2]
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return True

    roi = full_img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return True

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ratio = float((gray >= white_thr).mean())
    return ratio >= white_ratio


def is_too_large_box(full_w: int, full_h: int, box: Dict, area_ratio_th: float) -> bool:
    """
    判断框是否“异常超大”（面积占比超过阈值）
    """
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area = bw * bh
    full_area = full_w * full_h
    if full_area <= 0:
        return False
    return (area / full_area) >= area_ratio_th


def summarize_by_class(boxes: List[Dict]) -> Dict[str, int]:
    """
    统计每个类别的数量
    """
    cnt: Dict[str, int] = {}
    for b in boxes:
        cls = b.get("class", "")
        cnt[cls] = cnt.get(cls, 0) + 1
    return cnt


def top_duplicate_pairs_by_iou(
    boxes: List[Dict], iou_th: float, topk: int
) -> Tuple[List[Tuple[float, Dict, Dict]], int]:
    """
    统计同类框中 IoU >= iou_th 的候选重复对
    返回：(TopK 对列表, 总对数)
    """
    pairs: List[Tuple[float, Dict, Dict]] = []
    for a, b in combinations(boxes, 2):
        if a.get("class") != b.get("class"):
            continue
        v = iou_xyxy(a, b)
        if v >= iou_th:
            pairs.append((v, a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:topk], len(pairs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY 为空，请填入或用环境变量提供。")

    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        raise FileNotFoundError(f"找不到或无法读取图片：{INPUT_IMAGE}")

    full_h, full_w = img.shape[:2]
    print(f"原图尺寸：{full_w}x{full_h}")

    tile_files = [f for f in os.listdir(TILES_DIR) if f.lower().endswith(".png")]
    if not tile_files:
        raise FileNotFoundError(
            f"{TILES_DIR} 目录为空，请先运行 step1_tile.py 生成切块，或检查 TILES_DIR 配置"
        )

    tile_files.sort()

    if MAX_TILES_TO_INFER is None:
        target_tiles = tile_files
    else:
        target_tiles = tile_files[:MAX_TILES_TO_INFER]

    print(f"准备推理 tiles 数量：{len(target_tiles)}（TILES_DIR={TILES_DIR}）")

    all_boxes_raw: List[Dict] = []
    skipped_white_tiles = 0

    for fname in target_tiles:
        xy = parse_xy(fname)
        if xy is None:
            print(f"跳过不符合命名规则的文件：{fname}")
            continue

        ox, oy = xy
        tile_path = os.path.join(TILES_DIR, fname)

        if SKIP_MOSTLY_WHITE_TILES:
            tile_img = cv2.imread(tile_path)
            if tile_img is None:
                print(f"跳过（无法读取tile）：{fname}")
                continue
            if is_mostly_white_bgr(tile_img, TILE_WHITE_THR, TILE_WHITE_RATIO):
                skipped_white_tiles += 1
                print(f"跳过（几乎全白tile）：{fname}")
                continue

        print(f"推理：{fname} ...", end=" ")
        pred = infer_tile(tile_path)
        print("OK")

        for p in pred.get("predictions", []):
            cls = p.get("class", "")
            conf = float(p.get("confidence", 0.0))
            if conf < CONF_TH:
                continue

            cx_local = float(p["x"])
            cy_local = float(p["y"])
            bw = float(p["width"])
            bh = float(p["height"])

            cx = cx_local + ox
            cy = cy_local + oy

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            x1 = max(0, min(full_w - 1, x1))
            y1 = max(0, min(full_h - 1, y1))
            x2 = max(0, min(full_w - 1, x2))
            y2 = max(0, min(full_h - 1, y2))

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

    print(f"（NMS前）收集到预测框数量：{len(all_boxes_raw)}")
    if SKIP_MOSTLY_WHITE_TILES:
        print(f"（统计）跳过的全白tile数量：{skipped_white_tiles}")

    if DEBUG_STATS:
        print(f"（raw类别统计）{summarize_by_class(all_boxes_raw)}")
        top_pairs, total_pairs = top_duplicate_pairs_by_iou(
            all_boxes_raw, iou_th=NMS_IOU_TH, topk=DUP_TOPK
        )
        print(f"（raw重复对统计）IoU>={NMS_IOU_TH} 的同类框对数量：{total_pairs}")
        for v, a, b in top_pairs:
            print(
                f"  IoU={v:.3f} {a['class']} "
                f"{a['confidence']:.2f}({a['tile_file']}) <-> {b['confidence']:.2f}({b['tile_file']})"
            )

    all_boxes = nms_classwise(all_boxes_raw, NMS_IOU_TH)
    print(
        f"（NMS后）保留预测框数量：{len(all_boxes)}  |  CONF_TH={CONF_TH}  NMS_IOU_TH={NMS_IOU_TH}"
    )

    if DEBUG_STATS:
        print(f"（NMS后类别统计）{summarize_by_class(all_boxes)}")
        removed_by_nms = len(all_boxes_raw) - len(all_boxes)
        print(f"（NMS去重数量）{removed_by_nms}")

    filtered_boxes: List[Dict] = []
    removed_white = 0
    removed_big = 0

    for b in all_boxes:
        if FILTER_TOO_LARGE_BOX and is_too_large_box(
            full_w, full_h, b, BIG_BOX_AREA_RATIO
        ):
            if b["confidence"] < BIG_BOX_CONF_KEEP:
                removed_big += 1
                continue

        if FILTER_MOSTLY_WHITE_BOX and is_mostly_white_box(
            img, b, BOX_WHITE_THR, BOX_WHITE_RATIO
        ):
            removed_white += 1
            continue

        filtered_boxes.append(b)

    if FILTER_MOSTLY_WHITE_BOX or FILTER_TOO_LARGE_BOX:
        print(
            f"（过滤）移除全白框数量：{removed_white}  |  移除超大低分框数量：{removed_big}"
        )
        print(f"（最终）保留预测框数量：{len(filtered_boxes)}")

    if DEBUG_STATS:
        print(f"（最终类别统计）{summarize_by_class(filtered_boxes)}")

    out_json_raw = os.path.join(OUT_DIR, "result_boxes_raw.json")
    with open(out_json_raw, "w", encoding="utf-8") as f:
        json.dump(all_boxes_raw, f, ensure_ascii=False, indent=2)

    out_json_nms = os.path.join(OUT_DIR, "result_boxes.json")
    with open(out_json_nms, "w", encoding="utf-8") as f:
        json.dump(all_boxes, f, ensure_ascii=False, indent=2)

    out_json_final = os.path.join(OUT_DIR, "result_boxes_final.json")
    with open(out_json_final, "w", encoding="utf-8") as f:
        json.dump(filtered_boxes, f, ensure_ascii=False, indent=2)

    vis = img.copy()
    for b in filtered_boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cls = b["class"]
        conf = b["confidence"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{cls}:{conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    out_img = os.path.join(OUT_DIR, "result_drawn.png")
    cv2.imwrite(out_img, vis)

    print(f"✅ 输出完成：{out_img}")
    print(f"✅ 输出完成：{out_json_nms}（NMS后，未二次过滤）")
    print(f"✅ 输出完成：{out_json_final}（最终过滤后，推荐用这个）")
    print(f"✅ 输出完成：{out_json_raw}（NMS前，对比用）")


if __name__ == "__main__":
    main()
