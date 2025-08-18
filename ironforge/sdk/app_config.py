from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    shards_glob: str = "data/shards/*.parquet"
    symbol: str = "ES"
    timeframe: str = "1m"


@dataclass
class OutputsCfg:
    run_dir: str = "runs/{date}"
    overwrite: bool = True


@dataclass
class WeightsCfg:
    cluster_z: float = 0.30
    htf_prox: float = 0.25
    structure: float = 0.20
    cycle: float = 0.15
    precursor: float = 0.10


@dataclass
class ScoringCfg:
    weights: WeightsCfg = field(default_factory=WeightsCfg)


@dataclass
class MinidashCfg:
    out_html: str = "minidash.html"
    out_png: str = "minidash.png"
    width: int = 1200
    height: int = 700


@dataclass
class ReportingCfg:
    minidash: MinidashCfg = field(default_factory=MinidashCfg)


@dataclass
class ValidationCfg:
    folds: int = 5
    purge_bars: int = 20


@dataclass
class Config:
    workspace: str | None = None
    data: DataCfg = field(default_factory=DataCfg)
    outputs: OutputsCfg = field(default_factory=OutputsCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    reporting: ReportingCfg = field(default_factory=ReportingCfg)
    validation: ValidationCfg = field(default_factory=ValidationCfg)


def _coerce(value: str) -> Any:
    lv = value.lower()
    if lv in {"true", "false"}:
        return lv == "true"
    try:
        return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    return value


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _env_overrides(prefix: str = "IFG_") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        cur: dict[str, Any] = out
        for p in path[:-1]:
            cur = cur.setdefault(p, {})  # type: ignore[assignment]
        cur[path[-1]] = _coerce(val)
    return out


def _to_dc(dc, d: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    # Get module for resolving string type annotations
    import sys
    dc_module = sys.modules[dc.__module__]
    
    for f in dc.__dataclass_fields__:  # type: ignore[attr-defined]
        val = d.get(f)
        field_type_str = getattr(dc, "__annotations__", {}).get(f)
        
        # Resolve string annotation to actual type
        field_type = field_type_str
        if isinstance(field_type_str, str):
            field_type = getattr(dc_module, field_type_str, None)
        
        if val is None:
            # Use default factory by instantiating nested dataclass when available
            if field_type and hasattr(field_type, "__dataclass_fields__"):
                kwargs[f] = field_type()  # type: ignore[call-arg]
            else:
                # Fall back to attribute default via a temporary instance
                kwargs[f] = getattr(dc(), f)  # type: ignore[misc]
            continue
        if field_type and hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[f] = _to_dc(field_type, val)  # type: ignore[arg-type]
        else:
            kwargs[f] = val
    return dc(**kwargs)  # type: ignore[call-arg]


def load_config(
    path: str | Path | None = None, cli_overrides: dict[str, Any] | None = None
) -> Config:
    base: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
    env = _env_overrides()
    merged = _merge(_merge(base, env), cli_overrides or {})
    cfg = _to_dc(Config, merged)
    return cfg


def materialize_run_dir(cfg: Config) -> Path:
    date = datetime.date.today().isoformat()
    run_dir = cfg.outputs.run_dir.replace("{date}", date)
    p = Path(run_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
