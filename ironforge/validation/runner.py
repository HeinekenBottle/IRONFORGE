"""
IRONFORGE Validation Runner
==========================

Validation pipeline for TGAT discovery results and confluence scoring.
Implements quality gates and validation rails for archaeological discovery.
"""

import json
from pathlib import Path
from typing import Any


def validate_run(run_dir: str, config: dict | None = None) -> dict[str, Any]:
    """
    Validate a complete IRONFORGE run for quality and consistency.

    Args:
        run_dir: Path to run directory (runs/YYYY-MM-DD/)
        config: Optional validation configuration

    Returns:
        Dict containing validation results and quality metrics
    """

    run_path = Path(run_dir)

    if not run_path.exists():
        return {
            "status": "error",
            "message": f"Run directory not found: {run_dir}",
            "validations": {},
        }

    validations = {}

    # Validate embeddings
    validations["embeddings"] = _validate_embeddings(run_path / "embeddings")

    # Validate patterns
    validations["patterns"] = _validate_patterns(run_path / "patterns")

    # Validate confluence
    validations["confluence"] = _validate_confluence(run_path / "confluence")

    # Validate minidash
    validations["minidash"] = _validate_minidash(run_path)

    # Overall status
    all_passed = all(v.get("status") == "pass" for v in validations.values())

    return {
        "status": "pass" if all_passed else "fail",
        "run_dir": str(run_path),
        "validations": validations,
        "summary": _generate_summary(validations),
    }


def _validate_embeddings(embeddings_dir: Path) -> dict[str, Any]:
    """Validate TGAT embeddings output"""

    if not embeddings_dir.exists():
        return {"status": "fail", "message": "Embeddings directory missing"}

    checks = {}

    # Check for attention data
    attention_file = embeddings_dir / "attention_topk.parquet"
    if attention_file.exists():
        checks["attention_data"] = {"status": "pass", "file_size": attention_file.stat().st_size}
    else:
        checks["attention_data"] = {
            "status": "warning",
            "message": "No attention data (rank proxy mode)",
        }

    # Check for node embeddings (accept multiple per-session files)
    embeddings_candidates = list(embeddings_dir.glob("node_embeddings*.parquet"))
    if not embeddings_candidates:
        # Backward-compatible filename
        legacy_file = embeddings_dir / "node_embeddings.parquet"
        if legacy_file.exists():
            embeddings_candidates = [legacy_file]
    if embeddings_candidates:
        total_size = sum(f.stat().st_size for f in embeddings_candidates)
        checks["node_embeddings"] = {
            "status": "pass",
            "files": [f.name for f in embeddings_candidates],
            "total_size": total_size,
        }
    else:
        checks["node_embeddings"] = {"status": "fail", "message": "Missing node embeddings"}

    overall_status = (
        "pass" if all(c.get("status") == "pass" for c in checks.values()) else "warning"
    )

    return {"status": overall_status, "checks": checks}


def _validate_patterns(patterns_dir: Path) -> dict[str, Any]:
    """Validate discovered patterns"""

    if not patterns_dir.exists():
        return {"status": "fail", "message": "Patterns directory missing"}

    checks = {}

    # Check for pattern files
    pattern_files = list(patterns_dir.glob("*.parquet"))
    if pattern_files:
        checks["pattern_files"] = {
            "status": "pass",
            "count": len(pattern_files),
            "files": [f.name for f in pattern_files],
        }
    else:
        checks["pattern_files"] = {"status": "fail", "message": "No pattern files found"}

    return {
        "status": "pass" if checks.get("pattern_files", {}).get("status") == "pass" else "fail",
        "checks": checks,
    }


def _validate_confluence(confluence_dir: Path) -> dict[str, Any]:
    """Validate confluence scoring results"""

    if not confluence_dir.exists():
        return {"status": "fail", "message": "Confluence directory missing"}

    checks = {}

    # Check scores file
    scores_file = confluence_dir / "scores.parquet"
    if scores_file.exists():
        checks["scores"] = {"status": "pass", "file_size": scores_file.stat().st_size}
    else:
        checks["scores"] = {"status": "fail", "message": "Missing confluence scores"}

    # Check stats file
    stats_file = confluence_dir / "stats.json"
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                stats = json.load(f)

            # Validate stats structure
            required_fields = ["scale_type", "health_status"]
            missing_fields = [f for f in required_fields if f not in stats]

            if missing_fields:
                checks["stats"] = {
                    "status": "warning",
                    "message": f"Missing fields: {missing_fields}",
                    "data": stats,
                }
            else:
                checks["stats"] = {
                    "status": "pass",
                    "scale_type": stats.get("scale_type"),
                    "health_status": stats.get("health_status"),
                }

        except Exception as e:
            checks["stats"] = {"status": "fail", "message": f"Invalid stats.json: {e}"}
    else:
        checks["stats"] = {"status": "fail", "message": "Missing stats.json"}

    overall_status = "pass" if all(c.get("status") == "pass" for c in checks.values()) else "fail"

    return {"status": overall_status, "checks": checks}


def _validate_minidash(run_dir: Path) -> dict[str, Any]:
    """Validate minidash output"""

    checks = {}

    # Check HTML dashboard
    html_file = run_dir / "minidash.html"
    if html_file.exists():
        checks["html"] = {"status": "pass", "file_size": html_file.stat().st_size}
    else:
        checks["html"] = {"status": "fail", "message": "Missing minidash.html"}

    # Check PNG export (optional)
    png_file = run_dir / "minidash.png"
    if png_file.exists():
        checks["png"] = {"status": "pass", "file_size": png_file.stat().st_size}
    else:
        checks["png"] = {"status": "warning", "message": "No PNG export"}

    overall_status = "pass" if checks.get("html", {}).get("status") == "pass" else "fail"

    return {"status": overall_status, "checks": checks}


def _generate_summary(validations: dict[str, Any]) -> dict[str, Any]:
    """Generate validation summary"""

    total_checks = sum(len(v.get("checks", {})) for v in validations.values())
    passed_checks = sum(
        len([c for c in v.get("checks", {}).values() if c.get("status") == "pass"])
        for v in validations.values()
    )

    return {
        "total_validations": len(validations),
        "passed_validations": len([v for v in validations.values() if v.get("status") == "pass"]),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
    }


# CLI interface for validation
def main():
    """CLI entry point for validation runner"""
    import argparse

    parser = argparse.ArgumentParser(description="IRONFORGE Validation Runner")
    parser.add_argument("run_dir", help="Run directory to validate")
    parser.add_argument("--config", help="Validation configuration file")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Run validation
    results = validate_run(args.run_dir, config)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

    # Exit with appropriate code
    return 0 if results["status"] == "pass" else 1


if __name__ == "__main__":
    exit(main())
