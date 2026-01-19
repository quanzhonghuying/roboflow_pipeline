import os
import json
import cv2
import requests
from typing import List, Dict

# ============ 你需要填的配置 ============
ROBOFLOW_API_KEY = "4s4HpbmcTQoex2Hm1o5P".strip()
MODEL = "window-door-demo"
VERSION = "4"
# =====================================

# ====== 改这里：使用 Roboflow COCO 导出的 tiles ======
COCO_JSON = "coco_valid/_annotations.coco.json"
COCO_DIR = "coco_valid"
# =====================================================

OUT_DIR = "outputs"
MAX_TILES_TO_INFER = None  # 想限制就填数字，例如 50
CONF_TH = 0.60  # 你现在用的阈值


def infer_image(image_path: str) -> Dict:
    url = f"https://detect.roboflow.com/{MODEL}/{VERSION}?api_key={ROBOFLOW_API_KEY}"
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": f})
    resp.raise_for_status()
    return resp.json()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 读 COCO，拿到 tile 图片列表
    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if not images:
        raise FileNotFoundError("COCO JSON 里没有 images。")

    # 每张 tile：file_name 是实际 jpg，extra.name 是逻辑名 tile_x0_y0.png
    tiles = []
    for im in images:
        file_name = im.get("file_name", "")
        extra = im.get("extra", {}) or {}
        tile_name = extra.get("name", file_name)  # 没有 extra.name 就退化为 file_name
        if not file_name:
            continue
        tiles.append((tile_name, file_name))

    if MAX_TILES_TO_INFER is not None:
        tiles = tiles[:MAX_TILES_TO_INFER]

    print(f"COCO tiles 数量：{len(tiles)}")

    # 预测结果：tile 内坐标（与 COCO bbox 对齐）
    pred_tiles: List[Dict] = []

    for tile_name, file_name in tiles:
        img_path = os.path.join(COCO_DIR, file_name)
        if not os.path.exists(img_path):
            print(f"跳过（文件不存在）：{img_path}")
            continue

        print(f"推理：{tile_name} ({file_name}) ...", end=" ")
        pred = infer_image(img_path)
        print("OK")

        for p in pred.get("predictions", []):
            cls = p.get("class", "")
            conf = float(p.get("confidence", 0.0))
            if conf < CONF_TH:
                continue

            # Roboflow 返回的是 tile 内坐标：中心点 + 宽高
            cx = float(p["x"])
            cy = float(p["y"])
            bw = float(p["width"])
            bh = float(p["height"])

            x1 = float(cx - bw / 2)
            y1 = float(cy - bh / 2)
            x2 = float(cx + bw / 2)
            y2 = float(cy + bh / 2)

            pred_tiles.append(
                {
                    "tile_name": tile_name,  # 用于和 COCO 的 extra.name 对齐
                    "file_name": file_name,  # 实际图片文件名
                    "class": cls,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    out_pred = os.path.join(OUT_DIR, "pred_tiles.json")
    with open(out_pred, "w", encoding="utf-8") as f:
        json.dump(pred_tiles, f, ensure_ascii=False, indent=2)

    print(f"✅ 输出完成：{out_pred}")


if __name__ == "__main__":
    main()
