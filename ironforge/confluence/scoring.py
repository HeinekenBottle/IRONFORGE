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


# Alias for CLI compatibility
def score_session(cfg) -> None:
    """CLI-compatible wrapper for score_confluence"""
    # Extract patterns from shards directory
    from pathlib import Path
    import glob
    
    # Default pattern paths from config or fallback
    shards_glob = getattr(cfg.data, 'shards_glob', 'data/shards/*/shard_*')
    pattern_paths = glob.glob(shards_glob)
    
    # Get run directory for output
    try:
        from ironforge.sdk.app_config import materialize_run_dir
        run_dir = materialize_run_dir(cfg)
        out_dir = str(run_dir / "confluence")
    except:
        out_dir = "runs/confluence"
    
    # Extract weights and threshold from config
    weights = getattr(cfg.scoring, 'weights', None)
    if weights:
        weights = dict(weights.__dict__) if hasattr(weights, '__dict__') else dict(weights)
    
    threshold = 0.7  # Default threshold
    
    # Run confluence scoring
    result_path = score_confluence(
        pattern_paths=pattern_paths,
        out_dir=out_dir, 
        _weights=weights,
        threshold=threshold
    )
    
    print(f"[confluence] scored {len(pattern_paths)} patterns -> {result_path}")
