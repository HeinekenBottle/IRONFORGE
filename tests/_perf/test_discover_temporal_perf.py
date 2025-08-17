import time

import pandas as pd

from ironforge.learning.discovery_pipeline import run_discovery
from ironforge.sdk.config import LoaderCfg


def test_discover_temporal_perf(tmp_path):
    shard = tmp_path / "shard_0"
    shard.mkdir()
    pd.DataFrame({"node_id": [1], "t": [1]}).to_parquet(shard / "nodes.parquet")
    pd.DataFrame({"src": [1], "dst": [1]}).to_parquet(shard / "edges.parquet")

    start = time.time()
    run_discovery([str(shard)], str(tmp_path / "out"), LoaderCfg())
    duration = time.time() - start
    assert duration < 5
