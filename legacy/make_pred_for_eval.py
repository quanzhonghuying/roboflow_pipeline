import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

# =========================
# 输入：Step2 输出（NMS前或NMS后都可）
SRC = "outputs/result_boxes_raw.json"  # 你现在是从 raw 来转
DST = "pred_for_eval.json"
# =========================


def parse_xy_from_tile_name(tile_name: str) -> Optional[Tuple[int, int]]:
    """
    从 tile 文件名解析偏移：tile_x{ox}_y{oy}.png 或 .jpg
    """
    m = re.search(r"tile_x(\d+)_y(\d+)\.(png|jpg|jpeg)$", tile_name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f"not found: {SRC}")

    data = json.load(open(SRC, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("SRC must be a list of dicts")

    print("SRC len =", len(data))
    if data:
        print("keys sample =", list(data[0].keys()))

    # 你提到：tile_file 存在，tile_name 不存在
    has_tile_file = sum(1 for x in data if "tile_file" in x)
    has_tile_name = sum(1 for x in data if "tile_name" in x)
    print(f"has tile_file: {has_tile_file} / has tile_name: {has_tile_name}")

    # 统计一下 tile_file 分布
    c = Counter([x.get("tile_file") for x in data if x.get("tile_file")])
    print("top tile_file:", c.most_common(10))

    out: List[Dict] = []

    for b in data:
        tile = b.get("tile_name") or b.get("tile_file")
        if not tile:
            continue

        # Step4 使用 tile_name 字段
        tile_name = tile

        # Step2 的框是“全图坐标”，Step4 在 group_pred_by_tile 里会减去 offset
        # 所以这里保持全图坐标 x1..y2 不变，交给 step4 去转 tile-local
        x1 = float(b["x1"])
        y1 = float(b["y1"])
        x2 = float(b["x2"])
        y2 = float(b["y2"])

        # 基本合法性检查
        if x2 <= x1 or y2 <= y1:
            continue

        out.append(
            {
                "tile_name": tile_name,
                "class": b.get("class", ""),
                "confidence": float(b.get("confidence", 0.0)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    json.dump(out, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ wrote: {DST}  count={len(out)}")
    if out:
        print("sample:", out[0])


if __name__ == "__main__":
    main()
