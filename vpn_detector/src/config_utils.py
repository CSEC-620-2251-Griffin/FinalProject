from pathlib import Path
from typing import Dict

import yaml


def _resolve_path(base_dir: Path, path_str: str) -> str:
    """
    Resolve a path string to an absolute path, using the config file's directory as the base
    for relative paths and expanding user (~). Returned as string for compatibility with
    downstream code that expects YAML-loaded types.
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_config(config_path: str) -> Dict:
    """
    Load YAML config and normalize any data paths so configs remain portable across machines.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    with cfg_path.open("r") as f:
        config = yaml.safe_load(f)

    base_dir = cfg_path.parent
    data_cfg = config.get("data", {})
    if "raw_dir" in data_cfg:
        data_cfg["raw_dir"] = _resolve_path(base_dir, data_cfg["raw_dir"])
    if "processed_path" in data_cfg:
        data_cfg["processed_path"] = _resolve_path(base_dir, data_cfg["processed_path"])
    config["data"] = data_cfg
    return config
