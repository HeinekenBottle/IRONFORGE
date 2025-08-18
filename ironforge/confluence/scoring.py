from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def score_confluence(
    pattern_paths: Sequence[str],
    out_dir: str,
    _weights: Mapping[str, float] | None,
    threshold: float,
) -> str:
    """Stub confluence scorer.

    Creates a simple dataframe with scores and writes it to a parquet file.
    Returns the path to the written parquet.
    """
    scores = pd.DataFrame({"pattern": pattern_paths, "score": [threshold] * len(pattern_paths)})
    out_path = Path(out_dir) / "confluence_scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    return str(out_path)
