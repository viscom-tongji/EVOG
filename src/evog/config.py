import copy
import os
import re
from pathlib import Path

import yaml


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        return re.sub(
            r"\$\{([A-Z_][A-Z_0-9]*)(?::-([^}]*))?\}", lambda m: os.environ.get(m[1], m[2] or ""), value
        )
    return value


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(path, overrides=(), seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular configuration inheritance: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = cfg.pop("base", None)
    if parent:
        cfg = merge(load_config(path.parent / parent, seen=(*seen, path)), cfg)
    cfg = expand(cfg)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Expected dotted.key=value, got {override}")
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(key)
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(key)
        target[parts[-1]] = expand(yaml.safe_load(value))
    return cfg


def require_file(path, purpose):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{purpose} requires a nonempty file: {path}")
    return path
