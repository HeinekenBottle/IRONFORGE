from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataCfg:
    shards_base: str = "data/shards"
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
class ArchaeologicalCfg:
    """Archaeological zone significance configuration for pattern graduation"""
    enabled: bool = False  # Feature flag for archaeological graduation enhancement
    zone_percentages: list[float] = field(default_factory=lambda: [0.236, 0.382, 0.40, 0.618, 0.786])
    zone_influence_weight: float = 0.10  # Weight for archaeological_zone_significance in graduation
    significance_threshold: float = 0.75  # Minimum zone significance to apply boost


@dataclass
class OracleCfg:
    enabled: bool = False  # Disabled by default
    early_pct: float = 0.20  # Must be in (0, 0.5]
    output_path: str = "oracle_predictions.parquet"


@dataclass
class Config:
    workspace: str | None = None
    data: DataCfg = field(default_factory=DataCfg)
    outputs: OutputsCfg = field(default_factory=OutputsCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    reporting: ReportingCfg = field(default_factory=ReportingCfg)
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    oracle: OracleCfg = field(default_factory=OracleCfg)
    archaeological: ArchaeologicalCfg = field(default_factory=ArchaeologicalCfg)


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
    type_hints = get_type_hints(dc)
    
    for f in dc.__dataclass_fields__:  # type: ignore[attr-defined]
        val = d.get(f)
        field_type = type_hints.get(f)
        if val is None:
            # Use default factory by instantiating nested dataclass when available
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[f] = field_type()  # type: ignore[call-arg]
            else:
                # Fall back to attribute default via a temporary instance
                kwargs[f] = getattr(dc(), f)  # type: ignore[misc]
            continue
        if hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
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


def validate_config(cfg: Config) -> None:
    """Light validation for 1.0 public config surface.

    Asserts basic types and ranges without being overly restrictive.
    Raises ValueError on invalid configuration.
    """
    # Data
    if not isinstance(cfg.data.symbol, str) or not cfg.data.symbol:
        raise ValueError("data.symbol must be a non-empty string")
    if not isinstance(cfg.data.timeframe, str) or not cfg.data.timeframe:
        raise ValueError("data.timeframe must be a non-empty string")

    # Outputs
    if not isinstance(cfg.outputs.run_dir, str) or not cfg.outputs.run_dir:
        raise ValueError("outputs.run_dir must be a non-empty string")

    # Reporting
    if cfg.reporting.minidash.width <= 0 or cfg.reporting.minidash.height <= 0:
        raise ValueError("reporting.minidash width/height must be positive integers")

    # Validation
    if cfg.validation.folds < 1:
        raise ValueError("validation.folds must be >= 1")
    if cfg.validation.purge_bars < 0:
        raise ValueError("validation.purge_bars must be >= 0")

    # Scoring weights sanity (0..1 bounds)
    w = cfg.scoring.weights
    for name in ("cluster_z", "htf_prox", "structure", "cycle", "precursor"):
        val = getattr(w, name)
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"scoring.weights.{name} must be in [0,1]")
