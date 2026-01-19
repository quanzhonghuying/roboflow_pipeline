import json
from pathlib import Path
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
STATE_PATH = os.path.join(ROOT, "state", "last_run.json")

# Streamlit UI state files (relative to a chosen state_dir)
CONFIG_FILENAME_NEW = "config.json"
LAST_RESULT_FILENAME = "last_result.json"



def load(default_cfg):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    if not os.path.exists(STATE_PATH):
        return {"config": default_cfg, "last_result": {}, "updated_at": None}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = dict(default_cfg)
        cfg.update(data.get("config", {}))
        return {
            "config": cfg,
            "last_result": data.get("last_result", {}),
            "updated_at": data.get("updated_at"),
        }
    except Exception:
        return {"config": default_cfg, "last_result": {}, "updated_at": None}


def save(config, last_result=None):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "last_result": last_result or {},
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_state_dir(state_dir: str | Path) -> Path:
    """Resolve state_dir to an absolute Path.

    - If state_dir is relative, it's resolved from project ROOT.
    """
    if isinstance(state_dir, dict):
        raise TypeError("state_dir must be a path, not a dict")
    p = Path(state_dir)
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_last_config(state_dir: str | Path, default_cfg: dict | None = None) -> dict:
    """Load last saved config for the Streamlit UI.

    Returns a dict. If no saved config exists, returns default_cfg or {}.
    """
    sd = _resolve_state_dir(state_dir)
    config_path = sd / CONFIG_FILENAME_NEW
    if not config_path.exists():
        return dict(default_cfg or {})
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return dict(default_cfg or {})


def save_last_config(state_dir: str | Path, config: dict, last_result: dict | None = None) -> None:
    """Save config (and optionally last_result) for the Streamlit UI."""
    sd = _resolve_state_dir(state_dir)
    # Keep compatibility with existing filenames.
    (sd / CONFIG_FILENAME_NEW).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if last_result is not None:
        (sd / LAST_RESULT_FILENAME).write_text(json.dumps(last_result, ensure_ascii=False, indent=2), encoding="utf-8")
