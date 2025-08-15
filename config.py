#!/usr/bin/env python3
"""
IRONFORGE Configuration Management
=================================

Centralized configuration system to eliminate hardcoded paths and make
IRONFORGE deployable across different environments.

Supports:
- Environment variables
- Configuration files
- Default fallbacks
- Path validation
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class IRONFORGEConfig:
    """
    Configuration manager for IRONFORGE system.
    
    Eliminates hardcoded paths and provides environment-specific configuration.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger('ironforge.config')
        
        # Default configuration
        self._defaults = {
            'data_path': 'data',
            'preservation_path': 'IRONFORGE/preservation',
            'graphs_path': 'IRONFORGE/preservation/full_graph_store',
            'embeddings_path': 'IRONFORGE/preservation/embeddings',
            'discoveries_path': 'IRONFORGE/discoveries',
            'reports_path': 'IRONFORGE/reports',
            'htf_data_path': 'data/sessions/htf_relativity',
            'session_data_path': 'data/sessions/level_1',
            'integration_path': 'integration'
        }
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Validate and create directories
        self._validate_and_create_paths()
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from environment variables and config file."""
        config = self._defaults.copy()
        
        # Get workspace root from environment or current directory
        workspace_root = os.environ.get('IRONFORGE_WORKSPACE_ROOT', os.getcwd())
        config['workspace_root'] = workspace_root
        
        # Override with environment variables
        env_mappings = {
            'IRONFORGE_DATA_PATH': 'data_path',
            'IRONFORGE_PRESERVATION_PATH': 'preservation_path',
            'IRONFORGE_GRAPHS_PATH': 'graphs_path',
            'IRONFORGE_EMBEDDINGS_PATH': 'embeddings_path',
            'IRONFORGE_DISCOVERIES_PATH': 'discoveries_path',
            'IRONFORGE_REPORTS_PATH': 'reports_path',
            'IRONFORGE_HTF_DATA_PATH': 'htf_data_path',
            'IRONFORGE_SESSION_DATA_PATH': 'session_data_path',
            'IRONFORGE_INTEGRATION_PATH': 'integration_path'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config[config_key] = os.environ[env_var]
                self.logger.info(f"Using environment variable {env_var}: {config[config_key]}")
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")
        
        # Convert relative paths to absolute paths
        workspace_root = Path(config['workspace_root'])
        for key, value in config.items():
            if key != 'workspace_root' and isinstance(value, str):
                if not os.path.isabs(value):
                    config[key] = str(workspace_root / value)
        
        return config
    
    def _validate_and_create_paths(self):
        """Validate configuration and create necessary directories."""
        required_dirs = [
            'preservation_path',
            'graphs_path', 
            'embeddings_path',
            'discoveries_path',
            'reports_path'
        ]
        
        for dir_key in required_dirs:
            dir_path = Path(self.config[dir_key])
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {dir_path}: {e}")
                raise
    
    def get_path(self, key: str) -> str:
        """Get configured path by key."""
        if key not in self.config:
            raise ValueError(f"Unknown configuration key: {key}")
        return self.config[key]
    
    def get_data_path(self) -> str:
        """Get main data directory path."""
        return self.get_path('data_path')
    
    def get_preservation_path(self) -> str:
        """Get preservation directory path."""
        return self.get_path('preservation_path')
    
    def get_graphs_path(self) -> str:
        """Get graphs storage path."""
        return self.get_path('graphs_path')
    
    def get_embeddings_path(self) -> str:
        """Get embeddings storage path."""
        return self.get_path('embeddings_path')
    
    def get_discoveries_path(self) -> str:
        """Get discoveries storage path."""
        return self.get_path('discoveries_path')
    
    def get_reports_path(self) -> str:
        """Get reports storage path."""
        return self.get_path('reports_path')
    
    def get_htf_data_path(self) -> str:
        """Get HTF data directory path."""
        return self.get_path('htf_data_path')
    
    def get_session_data_path(self) -> str:
        """Get session data directory path."""
        return self.get_path('session_data_path')
    
    def get_integration_path(self) -> str:
        """Get integration directory path."""
        return self.get_path('integration_path')
    
    def get_workspace_root(self) -> str:
        """Get workspace root directory."""
        return self.get_path('workspace_root')
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise


# Global configuration instance
_config: Optional[IRONFORGEConfig] = None

def get_config(config_file: Optional[str] = None) -> IRONFORGEConfig:
    """Get or create global IRONFORGE configuration."""
    global _config
    
    if _config is None:
        _config = IRONFORGEConfig(config_file)
    
    return _config

def initialize_config(config_file: Optional[str] = None) -> IRONFORGEConfig:
    """Initialize IRONFORGE configuration system."""
    global _config
    _config = IRONFORGEConfig(config_file)
    return _config


if __name__ == "__main__":
    # Test configuration system
    print("🔧 Testing IRONFORGE Configuration System")
    print("=" * 50)
    
    config = get_config()
    
    print(f"Workspace Root: {config.get_workspace_root()}")
    print(f"Data Path: {config.get_data_path()}")
    print(f"Preservation Path: {config.get_preservation_path()}")
    print(f"Graphs Path: {config.get_graphs_path()}")
    print(f"HTF Data Path: {config.get_htf_data_path()}")
    
    print("\n✅ Configuration system working correctly")
