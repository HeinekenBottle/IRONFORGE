from __future__ import annotations

import json
from pathlib import Path


def test_report_manifest_helper_env(tmp_path, monkeypatch):
    # Prepare empty run dir (no parquet files required for this smoke test)
    run = tmp_path / "NQ_5m"
    run.mkdir(parents=True)

    # Enable env flag (the CLI path uses it; here we call helper directly)
    monkeypatch.setenv("IRONFORGE_WRITE_MANIFEST", "1")

    # Write a minimal manifest via internal helper
    from ironforge.sdk.manifest import write_for_run

    out = write_for_run(str(run), window_bars=512, version="0.test")
    assert out.exists()

    data = json.loads(out.read_text())
    # Invariants always included
    inv = data["invariants"]
    assert inv["taxonomy_events"] == 6
    assert inv["edge_intents"] == 4

    # Params reflect naming convention
    params = data["params"]
    assert params["symbol"] == "NQ"
    assert params["tf"] == 5
    assert params["window_bars"] == 512

