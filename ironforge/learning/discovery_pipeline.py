from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ironforge.data_engine.parquet_reader import read_nodes_edges
from ironforge.graph_builder.igraph_builder import from_parquet
from ironforge.graph_builder.pyg_converters import igraph_to_pyg

from .tgat_discovery import infer_shard_embeddings


def run_discovery(shard_paths: Iterable[str], out_dir: str, loader_cfg) -> list[str]:  # type: ignore[no-untyped-def]
    """Run discovery over a list of shard directories.

    Parameters
    ----------
    shard_paths: iterable of str
        Paths to shard directories containing ``nodes.parquet`` and ``edges.parquet``.
    out_dir: str
        Output directory where results will be stored.
    loader_cfg: LoaderCfg
        Configuration for loader. Only passed through to ``infer_shard_embeddings``.

    Returns
    -------
    list[str]
        List of pattern parquet paths produced for each shard.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for shard in shard_paths:
        nodes, edges = read_nodes_edges(shard)
        g = from_parquet(nodes, edges)
        data = igraph_to_pyg(g)

        # Use run layout subdirectories for each stage per docs
        # out_dir is expected to be the run directory (runs/YYYY-MM-DD)
        _, patt_path = infer_shard_embeddings(data, out_dir, loader_cfg)
        outputs.append(patt_path)
    return outputs
