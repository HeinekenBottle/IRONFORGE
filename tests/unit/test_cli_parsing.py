import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ironforge.sdk.cli import app


def _make_shard(base: Path) -> None:
    shard = base / "shard_0"
    shard.mkdir(parents=True)
    pd.DataFrame({"node_id": [1], "t": [1]}).to_parquet(shard / "nodes.parquet")
    pd.DataFrame({"src": [1], "dst": [1]}).to_parquet(shard / "edges.parquet")


def test_cli_commands(tmp_path):
    shards_dir = tmp_path / "shards"
    out_dir = tmp_path / "run"
    shards_dir.mkdir()
    _make_shard(shards_dir)

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"""
paths:
  shards_dir: {shards_dir}
  out_dir: {out_dir}
        """
    )

    runner = CliRunner()

    res = runner.invoke(app, ["discover-temporal", "--cfg", str(cfg_file)])
    assert res.exit_code == 0
    patt_file = out_dir / "patterns" / "patterns.parquet"
    assert patt_file.exists()

    res = runner.invoke(app, ["score-session", "--cfg", str(cfg_file)])
    assert res.exit_code == 0
    conf_file = out_dir / "confluence_scores.parquet"
    assert conf_file.exists()

    res = runner.invoke(app, ["validate-run", "--cfg", str(cfg_file)])
    assert res.exit_code == 0
    report_file = out_dir / "reports" / "validation.json"
    assert report_file.exists()

    res = runner.invoke(app, ["report-minimal", "--cfg", str(cfg_file)])
    assert res.exit_code == 0
    assert (out_dir / "minidash.html").exists()
    assert (out_dir / "minidash.png").exists()

    res = runner.invoke(app, ["status", "--cfg", str(cfg_file)])
    assert res.exit_code == 0
    assert json.loads(res.stdout)
