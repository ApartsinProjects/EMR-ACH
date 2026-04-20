"""Central configuration loader. Reads config.yaml + .env overrides."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

_ROOT = Path(__file__).parent.parent


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class Config:
    def __init__(self, override_path: str | None = None):
        base = _load_yaml(_ROOT / "config.yaml")
        if override_path:
            override = _load_yaml(Path(override_path))
            base = _deep_merge(base, override)
        self._cfg = base

    def __getattr__(self, name):
        try:
            return self._cfg[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def get(self, *keys, default=None):
        val = self._cfg
        for k in keys:
            if not isinstance(val, dict) or k not in val:
                return default
            val = val[k]
        return val

    # Convenience properties
    @property
    def openai_api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        return key

    @property
    def smoke_model(self) -> str:
        return os.environ.get("EMRACH_SMOKE_MODEL", self._cfg["model"]["smoke"])

    @property
    def experiment_model(self) -> str:
        return os.environ.get("EMRACH_MODEL", self._cfg["model"]["experiment"])

    @property
    def results_dir(self) -> Path:
        return _ROOT / self._cfg["paths"]["results_dir"]

    @property
    def prompts_dir(self) -> Path:
        return _ROOT / self._cfg["paths"]["prompts_dir"]

    @property
    def data_dir(self) -> Path:
        return _ROOT / self._cfg["paths"]["data_dir"]

    @property
    def mirai_queries_path(self) -> Path:
        return _ROOT / self._cfg["paths"]["mirai_queries"]

    @property
    def mirai_articles_path(self) -> Path:
        return _ROOT / self._cfg["paths"]["mirai_articles"]

    @property
    def forecastbench_path(self) -> Path:
        return _ROOT / self._cfg["paths"]["forecastbench_queries"]


# Singleton
_cfg: Config | None = None


def get_config(override_path: str | None = None) -> Config:
    global _cfg
    if _cfg is None or override_path:
        _cfg = Config(override_path)
    return _cfg
