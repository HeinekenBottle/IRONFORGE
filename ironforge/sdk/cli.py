from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import typer

from ironforge.confluence.scoring import score_confluence
from ironforge.learning.discovery_pipeline import run_discovery
from ironforge.motifs.scanner import scan_motifs
from ironforge.reporting.minidash import build_minidash
from ironforge.sdk.config import load_cfg
from ironforge.validation.oos import run_oos

app = typer.Typer(help="IRONFORGE SDK")


@app.command()
def discover_temporal(cfg: str = typer.Option(..., "--cfg")) -> None:
    c = load_cfg(cfg)
    shards = sorted(glob.glob(os.path.join(c.paths.shards_dir, "shard_*")))
    out_dir = c.paths.out_dir
    patt_paths = run_discovery(shards, out_dir, c.loader)
    typer.echo(json.dumps({"patterns": patt_paths, "out_dir": out_dir}))


@app.command()
def score_session(cfg: str = typer.Option(..., "--cfg")) -> None:
    c = load_cfg(cfg)
    run_dir = c.paths.out_dir
    pattern_paths = sorted(glob.glob(os.path.join(run_dir, "patterns", "*.parquet")))
    confluence_path = score_confluence(
        pattern_paths, run_dir, c.confluence.weights or {}, c.confluence.threshold
    )
    motifs_path = scan_motifs(confluence_path, run_dir)
    typer.echo(json.dumps({"confluence": confluence_path, "motifs": motifs_path}))


@app.command()
def validate_run(cfg: str = typer.Option(..., "--cfg")) -> None:
    c = load_cfg(cfg)
    report = run_oos(c.paths.out_dir)
    typer.echo(json.dumps(report))


@app.command()
def report_minimal(cfg: str = typer.Option(..., "--cfg")) -> None:
    c = load_cfg(cfg)
    html = os.path.join(c.paths.out_dir, "minidash.html")
    png = os.path.join(c.paths.out_dir, "minidash.png")
    build_minidash(c.paths.out_dir, html, png)
    typer.echo(f"wrote {html} and {png}")


@app.command()
def status(cfg: str = typer.Option(..., "--cfg")) -> None:
    c = load_cfg(cfg)
    run_dir = Path(c.paths.out_dir)
    summary = {}
    for p in run_dir.glob("**/*"):
        if p.is_file():
            summary[str(p.relative_to(run_dir))] = p.stat().st_size
    typer.echo(json.dumps(summary))


if __name__ == "__main__":  # pragma: no cover
    app()
