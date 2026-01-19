# -*- coding: utf-8 -*-
"""Roboflow パイプラインUI（日本語版 / v4）

目的
- Step1(タイル分割) → Step2(タイル推論) → Step3(原図へ合成/描画) → Step4(COCO評価)
  の一連を、迷わず実行できるUIにする。

方針
- 画面表記は日本語のみ（英語を混ぜない）。
- Step2 は「Step1 の tiles/ 出力」をそのまま入力にできる（COCO不要）。
- 各入力には help(ツールチップ) を付け、左サイドバーに“操作ヒント”を集約する。
- 設定は state/last_run.json に自動保存。

実行
  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st

from pipeline.runner import run_cli
from pipeline.config_store import load_last_config, save_last_config

# ----------------------------
# 基本設定
# ----------------------------
APP_TITLE = "Roboflow パイプラインUI"
STATE_DIR = "state"
DEFAULT_CFG: Dict[str, Any] = {
    # Roboflow
    "api_key": os.environ.get("ROBOFLOW_API_KEY", ""),
    "model": "window-door-demo",
    "version": 4,
    # フォルダ
    "inputs_dir": "inputs",  # 原図を置く
    "tiles_dir": "tiles",  # Step1 出力
    "outputs_dir": "outputs",  # 出力
    "coco_dir": "coco_valid",  # COCO (任意)
    # Step1
    "tile_size": 640,
    "overlap": 0.2,
    "pad": True,
    "flat": False,
    # Step2 (tiles 推論)
    "step2_out_dir": "outputs/tiles_infer",
    "conf": 0.6,
    "max_tiles": 0,
    "draw_overlay": True,
    # Step3 (原図推論)
    "step3_out_dir": "outputs/full_infer",
    # Step4 (評価)
    "eval_out_dir": "outputs/eval",
    "iou": 0.5,
    "eval_cls": "window",
}


# ----------------------------
# ユーティリティ
# ----------------------------


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["inputs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["tiles_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["outputs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)


def open_in_finder(path: str) -> None:
    """Mac の Finder でフォルダを開く（失敗しても無視）。"""
    try:
        subprocess.run(["open", path], check=False)
    except Exception:
        pass


def count_images_recursive(root: str) -> int:
    p = Path(root)
    if not p.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts)


def find_first_file(root: str, patterns: List[str]) -> Optional[str]:
    p = Path(root)
    if not p.exists():
        return None
    for pat in patterns:
        hits = list(p.rglob(pat))
        if hits:
            return str(hits[0])
    return None


def list_images(root: str) -> list[str]:
    """Recursively list image files under root. Returns sorted posix paths."""
    p = Path(root)
    if not p.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts]
    # Prefer deterministic order
    return sorted([x.as_posix() for x in files])


def safe_read_json(path: Path):
    """Read JSON file safely. Returns (obj, err)."""
    try:
        if not path.exists():
            return None, f"file not found: {path}"
        if path.stat().st_size == 0:
            return None, f"empty json file: {path.name}"
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None, f"empty json file: {path.name}"
        import json

        return json.loads(text), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def save_uploaded_file_to_tmp(uploaded, out_dir: str) -> str:
    """Streamlit のアップロード画像を out_dir/tmp_inputs/ に保存してパスを返す。"""
    tmp_dir = Path(out_dir) / "tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / uploaded.name
    out_path.write_bytes(uploaded.getbuffer())
    return str(out_path)


# ----------------------------
# UI
# ----------------------------


def sidebar(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("設定")

    st.sidebar.markdown("### 操作ヒント")
    with st.sidebar.expander("このUIの使い方（基本）", expanded=False):
        st.markdown(
            """
- **Step0（Setup）**：フォルダの準備状況を確認。
- **Step1**：原図を tiles/ に分割。
- **Step2**：tiles/ の画像を Roboflow に推論させ、pred_tiles.json を作成。
- **Step3**：原図を自動で切って推論→合成して result_drawn.png を作成。
- **Step4**：COCO（教師データ）がある場合に評価（TP/FP/FN, overlay）。

※ 迷ったら上から順に進めればOKです。
            """
        )

    cfg["api_key"] = st.sidebar.text_input(
        "Roboflow API Key",
        value=cfg.get("api_key", ""),
        type="password",
        help="Roboflow の API Key を入力します（環境変数 ROBOFLOW_API_KEY でも可）。",
    )
    cfg["model"] = st.sidebar.text_input(
        "モデル名（例: window-door-demo）",
        value=cfg.get("model", "window-door-demo"),
        help="Roboflow のモデルID（プロジェクト名）です。",
    )
    cfg["version"] = st.sidebar.number_input(
        "Version（例: 4）",
        min_value=1,
        step=1,
        value=int(cfg.get("version", 4)),
        help="Roboflow モデルのバージョン番号です（例: 4）。",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### フォルダ")
    cfg["inputs_dir"] = st.sidebar.text_input(
        "inputs_dir（原図）",
        value=cfg.get("inputs_dir", "inputs"),
        help="Step1/Step3 で使う原図フォルダです。",
    )
    cfg["tiles_dir"] = st.sidebar.text_input(
        "tiles_dir（タイル）",
        value=cfg.get("tiles_dir", "tiles"),
        help="Step1 出力、Step2 入力のタイルフォルダです。",
    )
    cfg["outputs_dir"] = st.sidebar.text_input(
        "outputs_dir（出力）",
        value=cfg.get("outputs_dir", "outputs"),
        help="各 Step の出力をまとめるフォルダです。",
    )
    cfg["coco_dir"] = st.sidebar.text_input(
        "coco_dir（COCO: 任意）",
        value=cfg.get("coco_dir", "coco_valid"),
        help="Step4 評価で使用する COCO フォルダ（任意）です。",
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ツールチップ（簡易）")
    st.sidebar.info("各入力欄の右上にマウスを置くと説明が表示されます。")

    return cfg


def tab_setup(cfg: Dict[str, Any]) -> None:
    st.subheader("Setup：接続/環境チェック")
    ensure_dirs(cfg)

    inputs_n = count_images_recursive(cfg["inputs_dir"])
    tiles_n = count_images_recursive(cfg["tiles_dir"])

    coco_json = find_first_file(
        cfg["coco_dir"], ["*_annotations.coco.json", "_annotations.coco.json"]
    )  # 柔軟に探す

    st.markdown("#### 状態")
    st.write(f"- project root: `{os.getcwd()}`")
    st.write(f"- inputs 画像数: **{inputs_n}**（{cfg['inputs_dir']}）")
    st.write(f"- tiles 画像数: **{tiles_n}**（{cfg['tiles_dir']}）")
    st.write(f"- COCO JSON: **{('あり: ' + coco_json) if coco_json else 'なし'}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("inputs フォルダを開く", key="btn_1"):
            open_in_finder(cfg["inputs_dir"])
    with c2:
        if st.button("tiles フォルダを開く", key="btn_2"):
            open_in_finder(cfg["tiles_dir"])
    with c3:
        if st.button("outputs フォルダを開く", key="btn_3"):
            open_in_finder(cfg["outputs_dir"])

    st.markdown("---")
    st.markdown("#### 次にやること")
    st.markdown(
        """
- **inputs に原図を入れた → Step1** へ
- **Step1 を実行して tiles ができた → Step2** へ
- **COCO（教師データ）がある → Step4** で評価も可能
        """
    )


def tab_step1(cfg: Dict[str, Any]) -> None:
    st.subheader("Step1：タイル分割")
    st.caption("inputs/ の原図を tiles/ に分割します。")

    cfg["tile_size"] = st.number_input(
        "tile_size（タイルの一辺ピクセル）",
        min_value=128,
        step=32,
        value=int(cfg.get("tile_size", 640)),
        help="一般的に 640 が使いやすいです。小さすぎると枚数が増え、大きすぎると切り落としが増えます。",
    )
    cfg["overlap"] = st.slider(
        "overlap（重なり率）",
        min_value=0.0,
        max_value=0.8,
        value=float(cfg.get("overlap", 0.2)),
        step=0.05,
        help="20% 程度が定番。窓がタイル境界で切れるのを減らします。",
    )
    cfg["pad"] = st.checkbox(
        "pad（端を埋める）",
        value=bool(cfg.get("pad", True)),
        help="画像端をタイルサイズに合わせて埋めます。端の窓が欠けにくくなります。",
    )
    cfg["flat"] = st.checkbox(
        "flat（タイルを1フォルダにまとめる）",
        value=bool(cfg.get("flat", False)),
        help="OFF 推奨。原図ごとにサブフォルダへ出力します。",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Step1 を実行", key="btn_4"):
            cmd = [
                "python",
                "pipeline/steps/step1_tile_cli.py",
                "--input_dir",
                cfg["inputs_dir"],
                "--out_dir",
                cfg["tiles_dir"],
                "--tile_size",
                str(cfg["tile_size"]),
                "--overlap",
                str(cfg["overlap"]),
            ]
            if cfg["pad"]:
                cmd.append("--pad")
            if cfg["flat"]:
                cmd.append("--flat")

            ok, out = run_cli(cmd)
            st.code(out)
            st.success("Step1 完了" if ok else "Step1 失敗")

    with col2:
        if st.button("tiles フォルダを開く", key="btn_5"):
            open_in_finder(cfg["tiles_dir"])

    st.markdown("---")
    st.markdown("#### 期待する結果")
    st.markdown(
        """
- tiles/ 配下に `tile_x0_y0.png` のような画像が増える
- フォルダの中に“フォルダしかない”場合は、画像がさらに深い階層にある可能性があります（Step2 は再帰探索で対応）。
        """
    )


def tab_step2(cfg: Dict[str, Any]) -> None:
    st.subheader("Step2：タイル推論（COCO不要）")
    st.caption(
        "tiles/ の画像を Roboflow に推論させ、pred_tiles.json（タイル単位の予測）を出力します。"
    )

    cfg["step2_out_dir"] = st.text_input(
        "out_dir（出力フォルダ）",
        value=cfg.get("step2_out_dir", "outputs/tiles_infer"),
        help="pred_tiles.json や overlay（任意）を保存します。",
    )

    cfg["conf"] = st.slider(
        "conf（信頼度しきい値）",
        min_value=0.05,
        max_value=0.95,
        value=float(cfg.get("conf", 0.6)),
        step=0.05,
        help="漏れを減らしたい（稳妥モード）なら 0.3〜0.5 に下げます。",
    )
    cfg["max_tiles"] = st.number_input(
        "max_tiles（0=全件）",
        min_value=0,
        step=10,
        value=int(cfg.get("max_tiles", 0)),
        help="まず少数で動作確認したいときに使います。",
    )
    cfg["draw_overlay"] = st.checkbox(
        "overlay 画像も出力する（推奨）",
        value=bool(cfg.get("draw_overlay", True)),
        help="推論結果をタイル画像に描画して保存します（目視確認用）。",
    )

    st.markdown("##### 入力（tiles）")
    tiles_n = count_images_recursive(cfg["tiles_dir"])
    st.write(f"- tiles 画像数（再帰探索）: **{tiles_n}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("tiles フォルダを開く", key="btn_6"):
            open_in_finder(cfg["tiles_dir"])
    with c2:
        if st.button("出力フォルダを開く", key="btn_7"):
            open_in_finder(cfg["step2_out_dir"])

    st.markdown("##### 推論対象")
    mode = st.radio(
        "選択",
        options=["tiles フォルダ全件", "ファイルを1枚だけ選んで推論"],
        index=0,
        help="普段は tiles 全件。動作確認だけなら1枚推奨。",
    )

    single_path: Optional[str] = None
    if mode == "ファイルを1枚だけ選んで推論":
        uploaded = st.file_uploader(
            "画像ファイルを選択（png/jpg）",
            type=["png", "jpg", "jpeg", "webp"],
            help="ローカルの画像を1枚選択して推論します（tiles に入っていなくてもOK）。",
        )
        if uploaded is not None:
            single_path = save_uploaded_file_to_tmp(uploaded, cfg["step2_out_dir"])
            st.write(f"選択ファイル: `{single_path}`")

    if st.button("Step2 を実行", key="btn_8"):
        if not cfg["api_key"]:
            st.error("Roboflow API Key が未入力です。左サイドバーで入力してください。")
            return

        cmd = [
            "python",
            "pipeline/steps/step2_infer_tiles_cli.py",
            "--api_key",
            cfg["api_key"],
            "--model",
            cfg["model"],
            "--version",
            str(cfg["version"]),
            "--out_dir",
            cfg["step2_out_dir"],
            "--conf",
            str(cfg["conf"]),
            "--max_tiles",
            str(cfg["max_tiles"]),
        ]

        if mode == "tiles フォルダ全件":
            cmd += ["--tiles_dir", cfg["tiles_dir"]]
        else:
            if single_path is None:
                st.error("ファイルが未選択です。")
                return
            cmd += ["--single", single_path]

        if cfg["draw_overlay"]:
            cmd.append("--draw_overlay")

        ok, out = run_cli(cmd)
        st.code(out)
        st.success("Step2 完了" if ok else "Step2 失敗")

        pred_path = Path(cfg["step2_out_dir"]) / "pred_tiles.json"
        if pred_path.exists():
            st.info(f"出力: `{pred_path}`")


def tab_step3(cfg: Dict[str, Any]) -> None:
    st.subheader("Step3 : 原図へ推論結果を合成して描画")
    st.caption(
        "Step1で作ったtilesを使って推論結果を原図に合成し、result_drawn.png を出力します。"
    )

    out_dir = st.text_input(
        "out_dir（出力フォルダ）", cfg.get("full_out_dir", "outputs/full_infer")
    )

    # Select input image
    inputs_dir = cfg.get("inputs_dir", "inputs")
    candidates = list_images(inputs_dir)
    if not candidates:
        st.warning(f"inputs_dir に画像が見つかりません: {inputs_dir}")
        return

    default_img = cfg.get("input_image")
    if default_img not in candidates:
        default_img = candidates[0]

    input_image = st.selectbox(
        "推論する原図 (input_image)", candidates, index=candidates.index(default_img)
    )

    # tiles_dir is usually tiles/<stem>
    stem = Path(input_image).stem
    tiles_root = Path(cfg.get("tiles_dir", "tiles"))
    tiles_dir_candidate = (tiles_root / stem).as_posix()
    tiles_dir = (
        tiles_dir_candidate
        if Path(tiles_dir_candidate).exists()
        else tiles_root.as_posix()
    )

    st.caption(f"tiles_dir: {tiles_dir}")

    if st.button("Step3 を実行"):
        cfg["full_out_dir"] = out_dir
        cfg["input_image"] = input_image
        save_cfg(cfg)

        cmd = [
            "python",
            "pipeline/steps/step3_infer_full_cli.py",
            "--api_key",
            cfg.get("api_key", ""),
            "--model",
            cfg.get("model", "window-door-demo"),
            "--version",
            str(cfg.get("version", 4)),
            "--input_image",
            input_image,
            "--tiles_dir",
            tiles_dir,
            "--out_dir",
            out_dir,
            "--conf",
            str(cfg.get("conf", 0.3)),
            "--nms_iou",
            str(cfg.get("nms_iou", 0.5)),
        ]
        # Optional knobs (only add when present)
        if cfg.get("max_tiles"):
            cmd += ["--max_tiles", str(cfg["max_tiles"])]
        if cfg.get("skip_tiles_mode"):
            cmd += ["--skip_tiles_mode", str(cfg["skip_tiles_mode"])]
        if cfg.get("tile_white_thr") is not None:
            cmd += ["--tile_white_thr", str(cfg["tile_white_thr"])]
        if cfg.get("min_nonwhite_ratio") is not None:
            cmd += ["--min_nonwhite_ratio", str(cfg["min_nonwhite_ratio"])]
        if cfg.get("min_edge_ratio") is not None:
            cmd += ["--min_edge_ratio", str(cfg["min_edge_ratio"])]
        if cfg.get("skip_white_tiles"):
            cmd += ["--skip_white_tiles"]
        if cfg.get("filter_big_box"):
            cmd += ["--filter_big_box"]
        if cfg.get("big_box_area_ratio") is not None:
            cmd += ["--big_box_area_ratio", str(cfg["big_box_area_ratio"])]
        if cfg.get("big_box_keep_conf") is not None:
            cmd += ["--big_box_keep_conf", str(cfg["big_box_keep_conf"])]

        run_cli(cmd)

        out_img = Path(out_dir) / "result_drawn.png"
        if out_img.exists():
            st.success("完了: result_drawn.png")
            st.image(str(out_img))
        else:
            st.info("完了しました。outputs を確認してください。")


def tab_step4(cfg: Dict[str, Any]) -> None:
    st.subheader("Step4：評価（COCOがある場合）")
    st.caption(
        "COCO（教師データ）がある場合に、TP/FP/FN を計算して overlay を出します。"
    )

    coco_json = find_first_file(
        cfg["coco_dir"], ["*_annotations.coco.json", "_annotations.coco.json"]
    )
    if coco_json is None:
        st.warning("coco_dir に COCO JSON が見つかりません。Step4 はスキップできます。")
        return

    cfg["eval_out_dir"] = st.text_input(
        "out_dir（評価出力フォルダ）",
        value=cfg.get("eval_out_dir", "outputs/eval"),
        help="eval_report.json / overlay / FP/FN などを保存します。",
    )
    cfg["iou"] = st.slider(
        "IoU（しきい値）",
        min_value=0.3,
        max_value=0.9,
        value=float(cfg.get("iou", 0.5)),
        step=0.05,
        help="一般的に 0.5 が基準。",
    )
    cfg["eval_cls"] = st.text_input(
        "評価対象クラス（例: window）",
        value=cfg.get("eval_cls", "window"),
        help="1クラス評価のときに指定します。",
    )

    # pred の選択（Step2 の pred_tiles.json or Step3 の result_boxes_raw.json）
    st.markdown("##### pred JSON を選ぶ")
    candidates: List[str] = []
    p1 = str(Path(cfg.get("step2_out_dir", "outputs/tiles_infer")) / "pred_tiles.json")
    if Path(p1).exists():
        candidates.append(p1)
    p2 = find_first_file(
        cfg.get("step3_out_dir", "outputs/full_infer"),
        ["result_boxes_raw.json", "*boxes*.json"],
    )
    if p2:
        candidates.append(p2)

    if not candidates:
        st.warning(
            "評価する pred JSON が見つかりません（Step2 または Step3 を先に実行してください）。"
        )
        return

    pred_path = st.selectbox(
        "pred",
        options=candidates,
        help="Step2 の pred_tiles.json（タイル座標）か、Step3 の raw json（原図座標）を選びます。",
    )

    # pred 座標系
    pred_coord = "tile" if pred_path.endswith("pred_tiles.json") else "global"

    if st.button("Step4 を実行", key="btn_10"):
        cmd = [
            "python",
            "legacy/step4_tile_eval.py",
            "--coco",
            coco_json,
            "--pred",
            pred_path,
            "--out",
            cfg["eval_out_dir"],
            "--iou",
            str(cfg["iou"]),
            "--pred_coord",
            pred_coord,
            "--cls",
            cfg["eval_cls"],
            "--tile_size",
            str(cfg["tile_size"]),
        ]

        ok, out = run_cli(cmd)
        st.code(out)
        st.success("Step4 完了" if ok else "Step4 失敗")

        rep = Path(cfg["eval_out_dir"]) / "eval_report.json"
        if rep.exists():
            st.info(f"レポート: `{rep}`")


def tab_review(cfg: Dict[str, Any]) -> None:
    st.subheader("Review：FP/FN 確認")
    st.caption("Step4 の出力（FP/FN/overlay）を確認します。")

    out_dir = cfg.get("eval_out_dir", "outputs/eval")
    rep = Path(out_dir) / "eval_report.json"
    fp = Path(out_dir) / "false_positives.json"
    fn = Path(out_dir) / "false_negatives.json"
    overlay_dir = Path(out_dir) / "tile_overlay"

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("評価出力フォルダを開く", key="btn_11"):
            open_in_finder(out_dir)
    with c2:
        if st.button("overlay フォルダを開く", key="btn_12"):
            open_in_finder(str(overlay_dir))
    with c3:
        if st.button("outputs を開く", key="btn_13"):
            open_in_finder(cfg["outputs_dir"])

    if rep.exists():
        st.markdown("#### eval_report")
        obj, err = safe_read_json(rep)
        if err:
            st.warning(f"eval_report.json is not readable: {err}")
        else:
            st.json(obj)

    if fp.exists():
        st.markdown("#### false_positives（FP）")
        st.write(f"件数: {len(json.loads(fp.read_text(encoding='utf-8')))}")

    if fn.exists():
        st.markdown("#### false_negatives（FN）")
        st.write(f"件数: {len(json.loads(fn.read_text(encoding='utf-8')))}")


def tab_flow3(cfg: Dict[str, Any]) -> None:
    st.subheader("Flow③：一括実行（推奨）")
    st.caption(
        "Step1 → Step2 →（任意で Step4）を、迷わず実行するためのショートカットです。"
    )

    st.markdown("- まず Step1 で tiles を作り、次に Step2 で tiles を推論します。")
    st.markdown("- COCO がある場合は Step4 で評価まで可能です。")

    do_step1 = st.checkbox("Step1 も実行する", value=False)
    do_step4 = st.checkbox("COCO がある場合、Step4 も実行する", value=False)

    if st.button("Flow③ を実行", key="btn_14"):
        if do_step1:
            # Step1
            cmd1 = [
                "python",
                "pipeline/steps/step1_tile_cli.py",
                "--input_dir",
                cfg["inputs_dir"],
                "--out_dir",
                cfg["tiles_dir"],
                "--tile_size",
                str(cfg["tile_size"]),
                "--overlap",
                str(cfg["overlap"]),
            ]
            if cfg["pad"]:
                cmd1.append("--pad")
            if cfg["flat"]:
                cmd1.append("--flat")
            ok1, out1 = run_cli(cmd1)
            st.code(out1)
            if not ok1:
                st.error("Step1 で停止")
                return

        # Step2
        if not cfg["api_key"]:
            st.error("Roboflow API Key が未入力です。")
            return

        cmd2 = [
            "python",
            "pipeline/steps/step2_infer_tiles_cli.py",
            "--api_key",
            cfg["api_key"],
            "--model",
            cfg["model"],
            "--version",
            str(cfg["version"]),
            "--tiles_dir",
            cfg["tiles_dir"],
            "--out_dir",
            cfg["step2_out_dir"],
            "--conf",
            str(cfg["conf"]),
            "--max_tiles",
            str(cfg["max_tiles"]),
        ]
        if cfg["draw_overlay"]:
            cmd2.append("--draw_overlay")
        ok2, out2 = run_cli(cmd2)
        st.code(out2)
        if not ok2:
            st.error("Step2 で停止")
            return

        if do_step4:
            # Step4
            coco_json = find_first_file(
                cfg["coco_dir"], ["*_annotations.coco.json", "_annotations.coco.json"]
            )
            if coco_json is None:
                st.warning("COCO JSON が見つからないため Step4 はスキップしました。")
            else:
                pred_path = str(Path(cfg["step2_out_dir"]) / "pred_tiles.json")
                cmd4 = [
                    "python",
                    "legacy/step4_tile_eval.py",
                    "--coco",
                    coco_json,
                    "--pred",
                    pred_path,
                    "--out",
                    cfg["eval_out_dir"],
                    "--iou",
                    str(cfg["iou"]),
                    "--pred_coord",
                    "tile",
                    "--cls",
                    cfg["eval_cls"],
                    "--tile_size",
                    str(cfg["tile_size"]),
                ]
                ok4, out4 = run_cli(cmd4)
                st.code(out4)
                st.success("Flow③ 完了" if ok4 else "Step4 失敗")
                return

        st.success("Flow③ 完了")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # 設定ロード → サイドバー反映
    cfg = DEFAULT_CFG.copy()
    cfg.update(load_last_config(STATE_DIR) or {})
    cfg = sidebar(cfg)
    ensure_dirs(cfg)

    # 変更があるたびに保存（軽量）
    save_last_config(STATE_DIR, cfg)

    tabs = st.tabs(
        [
            "Setup",
            "Step1：タイル分割",
            "Step2：タイル推論",
            "Step3：原図合成/描画",
            "Step4：評価（COCO）",
            "Review：FP/FN",
            "Flow③：一括実行",
        ]
    )

    with tabs[0]:
        tab_setup(cfg)
    with tabs[1]:
        tab_step1(cfg)
    with tabs[2]:
        tab_step2(cfg)
    with tabs[3]:
        tab_step3(cfg)
    with tabs[4]:
        tab_step4(cfg)
    with tabs[5]:
        tab_review(cfg)
    with tabs[6]:
        tab_flow3(cfg)


if __name__ == "__main__":
    main()
