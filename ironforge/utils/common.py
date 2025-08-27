"""Common utility functions extracted from large modules."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any


def maybe_import(module_name: str, attribute_name: str) -> Any | None:
    """
    Safely import an attribute from a module, returning None if import fails.
    
    This pattern was duplicated in cli.py and other modules.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)
    except Exception:
        return None


def normalize_timeframe(timeframe: str) -> tuple[str, str]:
    """
    Normalize timeframe input to both numeric and string formats.
    
    Args:
        timeframe: Input like "5", "M5", etc.
    
    Returns:
        Tuple of (numeric_string, formatted_string)
        
    Raises:
        ValueError: If timeframe format is invalid
    """
    if timeframe.upper().startswith('M'):
        tf_numeric = timeframe[1:]
        tf_string = timeframe.upper()
    elif timeframe.isdigit():
        tf_numeric = timeframe
        tf_string = f"M{timeframe}"
    else:
        raise ValueError(f"Invalid timeframe format: {timeframe}. Use '5' or 'M5'")
    
    return tf_numeric, tf_string


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def validate_path_exists(path: str | Path, path_type: str = "path") -> Path:
    """
    Validate that a path exists and return Path object.
    
    Args:
        path: Path to validate
        path_type: Description of path type for error messages
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{path_type.capitalize()} not found: {path_obj}")
    return path_obj


def get_legacy_entrypoint(module_paths: list[str], function_name: str, current_module: str) -> Any | None:
    """
    Get legacy entrypoint with deprecation warning.
    
    This pattern was used in multiple CLI commands.
    """
    for module_path in module_paths:
        fn = maybe_import(module_path, function_name)
        if fn is not None:
            warnings.warn(
                f"Legacy {function_name} entrypoint from {module_path} is deprecated "
                f"and will be removed in 2.0; use {current_module}:{function_name}",
                DeprecationWarning,
                stacklevel=3,
            )
            return fn
    return None


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_string(text: str, max_length: int = 50) -> str:
    """
    Truncate string with ellipsis if too long.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return f"{text[:max_length-3]}..."


def create_progress_indicator(current: int, total: int, width: int = 20) -> str:
    """
    Create a simple text progress indicator.
    
    Args:
        current: Current progress value
        total: Total progress value
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "?" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + ">" if filled < width else "=" * width
    bar = bar.ljust(width)
    
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}%"