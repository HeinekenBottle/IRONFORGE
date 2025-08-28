"""
Discovery Pipeline Adapter Layer
Bridges CLI configuration objects with discovery pipeline parameter contracts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

logger = logging.getLogger(__name__)


def find_shard_directories(shards_base: str, symbol: str = None, timeframe: str = None) -> List[str]:
    """
    Find shard directories containing nodes.parquet and edges.parquet
    
    Args:
        shards_base: Base directory for shards (e.g., "data/shards")
        symbol: Optional symbol filter (e.g., "ES", "NQ")
        timeframe: Optional timeframe filter (e.g., "1m", "M5")
        
    Returns:
        List of shard directory paths containing valid node/edge parquet files
    """
    base_path = Path(shards_base)
    if not base_path.exists():
        logger.warning(f"Shards base directory does not exist: {shards_base}")
        return []
    
    shard_dirs = []
    
    # Pattern: data/shards/{symbol}_{timeframe}/session_*/
    if symbol and timeframe:
        # Look for specific symbol/timeframe combination
        symbol_tf_pattern = f"{symbol}_{timeframe}"
        symbol_tf_dir = base_path / symbol_tf_pattern
        
        if symbol_tf_dir.exists():
            # Find session directories within symbol/timeframe directory
            for session_dir in symbol_tf_dir.iterdir():
                if session_dir.is_dir() and _is_valid_shard_directory(session_dir):
                    shard_dirs.append(str(session_dir))
        else:
            logger.warning(f"Symbol/timeframe directory not found: {symbol_tf_dir}")
    else:
        # Search all available shard directories
        for item in base_path.rglob("*"):
            if item.is_dir() and _is_valid_shard_directory(item):
                shard_dirs.append(str(item))
    
    logger.info(f"Found {len(shard_dirs)} valid shard directories in {shards_base}")
    
    if len(shard_dirs) == 0:
        logger.warning("No valid shard directories found. Expected structure:")
        logger.warning("  data/shards/{symbol}_{timeframe}/session_*/nodes.parquet")
        logger.warning("  data/shards/{symbol}_{timeframe}/session_*/edges.parquet")
    
    return sorted(shard_dirs)


def _is_valid_shard_directory(directory: Path) -> bool:
    """Check if directory contains required nodes.parquet and edges.parquet files"""
    nodes_file = directory / "nodes.parquet"
    edges_file = directory / "edges.parquet"
    
    return nodes_file.exists() and edges_file.exists()


def discovery_config_adapter(config: Any) -> Tuple[List[str], str, Dict[str, Any]]:
    """
    Convert configuration object to discovery pipeline parameters
    
    Args:
        config: Configuration object with data, outputs, and other settings
        
    Returns:
        Tuple of (shard_paths, out_dir, loader_cfg)
        - shard_paths: List of shard directory paths
        - out_dir: Output directory for discovery results  
        - loader_cfg: Configuration dictionary for loader
        
    Raises:
        ValueError: If configuration is missing required fields
    """
    # Extract required configuration fields
    try:
        shards_base = getattr(config.data, 'shards_base', None)
        symbol = getattr(config.data, 'symbol', None)
        timeframe = getattr(config.data, 'timeframe', None)
        output_dir = getattr(config.outputs, 'run_dir', 'runs/default')
        
    except AttributeError as e:
        raise ValueError(f"Configuration missing required fields: {e}")
    
    if not shards_base:
        raise ValueError("Configuration missing 'data.shards_base' field")
    
    # Find shard directories
    shard_paths = find_shard_directories(shards_base, symbol, timeframe)
    
    if not shard_paths:
        logger.error(f"No valid shard directories found in {shards_base}")
        logger.error("Run prep-shards to generate shard directories before discovery")
    
    # Create output directory (run directory). The discovery CLI will override this with
    # materialize_run_dir(cfg) to ensure documented run layout.
    out_dir = str(Path(output_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert config object to dictionary for loader
    if hasattr(config, 'to_dict'):
        loader_cfg = config.to_dict()
    elif hasattr(config, '__dict__'):
        loader_cfg = config.__dict__.copy()
    else:
        # Fallback: extract basic fields
        loader_cfg = {
            'data': {
                'symbol': symbol,
                'timeframe': timeframe,
                'shards_base': shards_base
            }
        }
    
    logger.info(f"Discovery adapter: {len(shard_paths)} shards → {out_dir}")
    
    return shard_paths, out_dir, loader_cfg


def validate_discovery_inputs(shard_paths: List[str], out_dir: str) -> bool:
    """
    Validate that discovery inputs meet contract requirements
    
    Args:
        shard_paths: List of shard directory paths
        out_dir: Output directory path
        
    Returns:
        True if inputs are valid, False otherwise
    """
    # Check that we have shard paths
    if not shard_paths:
        logger.error("No shard paths provided to discovery pipeline")
        return False
    
    # Check that all shard paths exist and contain required files
    invalid_shards = []
    for shard_path in shard_paths:
        shard_dir = Path(shard_path)
        if not shard_dir.exists():
            invalid_shards.append(f"{shard_path} (does not exist)")
        elif not _is_valid_shard_directory(shard_dir):
            invalid_shards.append(f"{shard_path} (missing nodes.parquet or edges.parquet)")
    
    if invalid_shards:
        logger.error("Invalid shard directories found:")
        for invalid in invalid_shards:
            logger.error(f"  {invalid}")
        return False
    
    # Check output directory
    out_path = Path(out_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory {out_dir}: {e}")
        return False
    
    logger.info(f"Discovery inputs validated: {len(shard_paths)} shards → {out_dir}")
    return True