from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoaderCfg:
    fanouts: tuple[int, int] = (10, 10)
    batch_size: int = 2048
    time_attr: str = "t"


@dataclass
class Paths:
    shards_dir: str
    out_dir: str


@dataclass
class ConfluenceCfg:
    weights: dict[str, float] | None = None
    threshold: float = 65.0


@dataclass
class RunCfg:
    paths: Paths
    loader: LoaderCfg = field(default_factory=LoaderCfg)
    confluence: ConfluenceCfg = field(default_factory=ConfluenceCfg)


def _dict_to_loader(raw: dict[str, Any]) -> LoaderCfg:
    return LoaderCfg(
        fanouts=tuple(raw.get("fanouts", (10, 10))),
        batch_size=raw.get("batch_size", 2048),
        time_attr=raw.get("time_attr", "t"),
    )


def _dict_to_confluence(raw: dict[str, Any]) -> ConfluenceCfg:
    return ConfluenceCfg(
        weights=raw.get("weights"),
        threshold=raw.get("threshold", 65.0),
    )


def load_cfg(path: str) -> RunCfg:
    """Load a run configuration from YAML."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    paths_raw = raw.get("paths", {})
    loader_raw = raw.get("loader", {})
    confluence_raw = raw.get("confluence", {})

    paths = Paths(**paths_raw)
    loader = _dict_to_loader(loader_raw)
    confluence = _dict_to_confluence(confluence_raw)
    return RunCfg(paths=paths, loader=loader, confluence=confluence)
