import json

import pandas as pd
from typer.testing import CliRunner

from ironforge.sdk.cli import app


def test_report_minimal_smoke(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(run_dir / "patterns.parquet")
    pd.DataFrame({"score": [1.0]}).to_parquet(run_dir / "confluence_scores.parquet")
    (run_dir / "motifs.json").write_text(json.dumps([]))

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(f"paths:\n  shards_dir: {tmp_path}\n  out_dir: {run_dir}\n")

    runner = CliRunner()
    result = runner.invoke(app, ["report-minimal", "--cfg", str(cfg_file)])
    assert result.exit_code == 0
    assert (run_dir / "minidash.html").exists()
    assert (run_dir / "minidash.png").exists()
