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
    """Development stub for confluence scorer.

    TODO: Implement actual confluence scoring algorithm.
    
    Currently creates a simple dataframe with uniform scores for development/testing.
    
    Args:
        pattern_paths: List of pattern file paths to score
        out_dir: Output directory for results
        _weights: Weight mapping (currently unused)
        threshold: Score threshold value to use for all patterns
        
    Returns:
        Path to the written parquet file containing scores
    """
    scores = pd.DataFrame({"pattern": pattern_paths, "score": [threshold] * len(pattern_paths)})
    out_path = Path(out_dir) / "confluence_scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    return str(out_path)
