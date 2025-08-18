import json

import pandas as pd

from ironforge.reporting.minidash import build_minidash


def test_minidash_builds(tmp_path):
    run_dir = tmp_path
    pd.DataFrame({"a": [1]}).to_parquet(run_dir / "patterns.parquet")
    pd.DataFrame({"score": [1.0]}).to_parquet(run_dir / "confluence_scores.parquet")
    (run_dir / "motifs.json").write_text(json.dumps([{"pattern": "p1"}]))

    html = run_dir / "dash.html"
    png = run_dir / "dash.png"
    build_minidash(str(run_dir), str(html), str(png))

    assert html.exists()
    assert png.exists()
    assert html.stat().st_size > 0
    assert png.stat().st_size > 0
