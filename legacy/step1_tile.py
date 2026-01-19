import os
from pathlib import Path
import cv2

# ====== 你只改这三个参数就行 ======
INPUT_DIR = "inputs"  # 放原图的文件夹（里面放多张图）
OUT_DIR = "tiles_out"  # 输出 tiles 的总文件夹
tile_size = 640  # 640 / 800 都常用
overlap_ratio = 0.6  # 重叠比例：0.2 少、0.6 多（越大切得越密）
# ==================================

# 支持的图片扩展名
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def tile_one_image(
    img_path: Path, out_root: Path, tile_size: int, overlap_ratio: float
):
    """把单张图片切成 tiles，输出到 out_root/原图名/ 下面"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] 无法读取: {img_path}")
        return 0

    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap_ratio))
    if step <= 0:
        raise ValueError(
            "overlap_ratio 太大导致 step<=0，请调小 overlap_ratio（比如 0.6 或 0.5）"
        )

    # 为这张图创建一个独立输出文件夹，避免不同原图 tile 文件名冲突
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img[y : y + tile_size, x : x + tile_size]

            # 边缘可能不够 tile_size，直接跳过（你也可以改成补黑边，但先跑通就跳过）
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                continue

            out_name = f"tile_x{x}_y{y}.png"
            cv2.imwrite(str(out_dir / out_name), tile)
            count += 1

    print(f"[OK] {img_path.name} -> {count} tiles, saved to {out_dir}")
    return count


def main():
    in_dir = Path(INPUT_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # 找到文件夹里的所有图片
    images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if not images:
        raise FileNotFoundError(f"在 {INPUT_DIR} 没找到图片（支持: {sorted(EXTS)}）")

    total = 0
    for img_path in images:
        total += tile_one_image(img_path, out_root, tile_size, overlap_ratio)

    print(f"\nDone. Total tiles: {total}. Output folder: {out_root.resolve()}")


if __name__ == "__main__":
    main()
