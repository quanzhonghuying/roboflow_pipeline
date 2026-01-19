import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pred", required=True, help="pred_tiles.json path")
    ap.add_argument("--out_pred", required=True, help="output path")
    args = ap.parse_args()

    with open(args.in_pred, "r", encoding="utf-8") as f:
        preds = json.load(f)

    out = []
    for p in preds:
        # 把 tile_name 映射成 step4 更常见的 tile_file
        tile_file = p.get("tile_file") or p.get("tile_name") or p.get("file_name")
        if not tile_file:
            continue

        out.append(
            {
                "tile_file": tile_file,
                "class": p.get("class", ""),
                "confidence": float(p.get("confidence", 0.0)),
                "x1": float(p.get("x1", 0.0)),
                "y1": float(p.get("y1", 0.0)),
                "x2": float(p.get("x2", 0.0)),
                "y2": float(p.get("y2", 0.0)),
            }
        )

    with open(args.out_pred, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"converted: {len(out)} boxes -> {args.out_pred}")


if __name__ == "__main__":
    main()
