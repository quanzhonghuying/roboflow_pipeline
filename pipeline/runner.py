# pipeline/runner.py
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple


def cmd_to_str(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_cli(
    cmd: List[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    CLI コマンド（配列）を実行するランナー。
    返り値: (成功/失敗, 標準出力+標準エラー)
    """
    try:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        out = []
        out.append(f"$ {cmd_to_str(cmd)}")
        if p.stdout:
            out.append(p.stdout)
        if p.stderr:
            out.append(p.stderr)
        out_text = "\n".join(out).strip()

        ok = (p.returncode == 0)
        if not ok:
            out_text += f"\n[終了コード] {p.returncode}"
        return ok, out_text

    except FileNotFoundError as e:
        return False, f"$ {cmd_to_str(cmd)}\n[FileNotFoundError] {e}"
    except subprocess.TimeoutExpired:
        return False, f"$ {cmd_to_str(cmd)}\n[Timeout] {timeout} 秒でタイムアウトしました"
    except Exception as e:
        return False, f"$ {cmd_to_str(cmd)}\n[例外] {type(e).__name__}: {e}"
