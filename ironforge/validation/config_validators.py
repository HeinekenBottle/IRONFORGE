"""Configuration validators built on the unified framework."""

from __future__ import annotations

from typing import Any

from .framework import ValidationCheck, ValidationResult


def validate_sdk_config(cfg: Any) -> ValidationResult:
    checks: dict[str, ValidationCheck] = {}

    try:
        # Data
        ok = isinstance(cfg.data.symbol, str) and bool(cfg.data.symbol)
        checks["data.symbol"] = ValidationCheck(
            name="data.symbol",
            status="pass" if ok else "fail",
            message=None if ok else "data.symbol must be a non-empty string",
        )

        ok = isinstance(cfg.data.timeframe, str) and bool(cfg.data.timeframe)
        checks["data.timeframe"] = ValidationCheck(
            name="data.timeframe",
            status="pass" if ok else "fail",
            message=None if ok else "data.timeframe must be a non-empty string",
        )

        # Outputs
        ok = isinstance(cfg.outputs.run_dir, str) and bool(cfg.outputs.run_dir)
        checks["outputs.run_dir"] = ValidationCheck(
            name="outputs.run_dir",
            status="pass" if ok else "fail",
            message=None if ok else "outputs.run_dir must be a non-empty string",
        )

        # Reporting
        ok = cfg.reporting.minidash.width > 0 and cfg.reporting.minidash.height > 0
        checks["reporting.minidash"] = ValidationCheck(
            name="reporting.minidash",
            status="pass" if ok else "fail",
            message=None if ok else "minidash width/height must be positive",
        )

        # Validation
        ok = cfg.validation.folds >= 1 and cfg.validation.purge_bars >= 0
        checks["validation.params"] = ValidationCheck(
            name="validation.params",
            status="pass" if ok else "fail",
            message=None if ok else "folds>=1 and purge_bars>=0",
        )

        # Scoring weights range checks
        w = cfg.scoring.weights
        names = ("cluster_z", "htf_prox", "structure", "cycle", "precursor")
        ok = True
        for n in names:
            try:
                v = float(getattr(w, n))
                if not (0.0 <= v <= 1.0):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        checks["scoring.weights.bounds"] = ValidationCheck(
            name="scoring.weights.bounds",
            status="pass" if ok else "fail",
            message=None if ok else "weights must be within [0,1]",
        )

    except Exception as exc:
        # If something unexpected happens, surface it as a failure
        checks["exception"] = ValidationCheck(
            name="exception",
            status="fail",
            message=str(exc),
        )

    overall = "pass" if all(c.status == "pass" for c in checks.values()) else "fail"
    return ValidationResult(status=overall, checks=checks)

