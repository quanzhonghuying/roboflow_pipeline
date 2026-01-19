# Roboflow パイプラインUI（日本語版 / v4）

このUIは、**タイル分割 → タイル推論 → 原図へ復元 → 評価** の流れを、手順どおりに実行できるようにした Streamlit アプリです。

---

## 1. セットアップ

### 1.1 Python 仮想環境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 1.2 起動

```bash
streamlit run app.py
```

---

## 2. 使い方（推奨フロー）

### Step1: タイル分割
- `inputs/` に元画像を置く（例: `inputs/input.png`）
- Step1 を実行すると `tiles/` 配下にタイル画像が生成されます

### Step2: タイル推論（COCO不要）
- Step1 の `tiles/` を入力にして Roboflow 推論を実行します
- 出力: `outputs/tiles_infer/pred_tiles.json`
- 目的: **漏れを減らしたい**場合は `conf` を少し下げる（例: 0.6 → 0.4）

> ※ Roboflow の「Dataset/COCO」とは無関係に、ローカルの tiles 画像へ推論します。

### Step3: 原図へ合成して描画
- Step2 の推論結果（pred_tiles.json）を原図座標へ戻し、`result_drawn.png` を出力します

### Step4: 評価（COCOがある場合のみ）
- Roboflow から COCO を export して `coco_valid/_annotations.coco.json` を配置すると評価可能です

---

## 3. よくある原因（Step2 が空になる）

### 症状
- `pred_tiles.json` が `[]` になる

### 原因の典型
- Step2 の入力が **tiles画像ではなくフォルダ**になっている
- `tiles/` の下に画像ファイル（png/jpg）が存在しない

### 対応
- UI の Step2 で「tiles 内の画像枚数」を表示します。0 の場合は tiles の中身を確認してください。

---

## 4. OCR（型番/シリーズ抽出）を次段で追加する設計

- 詳細は `docs/ocr_design_ja.md` を参照してください。

