#!/usr/bin/env python3
"""
IRONFORGE Temporal Analysis Module

This module provides temporal pattern analysis capabilities with archaeological zone calculations,
Theory B temporal non-locality detection, and comprehensive session management.

The module is organized into focused components:
- SessionDataManager: Session loading, caching, and data preprocessing
- PriceRelativityEngine: Archaeological zone calculations and Theory B analysis
- TemporalQueryCore: Core temporal querying and pattern matching
- VisualizationManager: Display, plotting, and reporting functionality
- EnhancedTemporalQueryEngine: Main interface maintaining backward compatibility

Usage:
    from ironforge.temporal import EnhancedTemporalQueryEngine
    
    engine = EnhancedTemporalQueryEngine()
    results = engine.ask("What happens after liquidity sweeps in NY_AM sessions?")
    engine.display_results(results)
"""

# Import main classes for public API
from .enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from .price_relativity import PriceRelativityEngine
from .query_core import TemporalQueryCore
from .session_manager import SessionDataManager
from .visualization import VisualizationManager

# Maintain backward compatibility - the main class should be importable as before
# This ensures that existing code like:
# from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
# continues to work through the new module structure

__all__ = [
    'EnhancedTemporalQueryEngine',
    'SessionDataManager', 
    'PriceRelativityEngine',
    'TemporalQueryCore',
    'VisualizationManager'
]

# Version information
__version__ = '1.1.0'
__author__ = 'IRONFORGE Team'
__description__ = 'Enhanced Temporal Query Engine with Archaeological Zone Analysis'

# Module metadata
MODULE_INFO = {
    "name": "ironforge.temporal",
    "version": __version__,
    "description": __description__,
    "components": {
        "EnhancedTemporalQueryEngine": "Main interface for temporal pattern analysis",
        "SessionDataManager": "Session data loading and management",
        "PriceRelativityEngine": "Archaeological zone and Theory B analysis", 
        "TemporalQueryCore": "Core temporal querying logic",
        "VisualizationManager": "Display and plotting functionality"
    },
    "features": [
        "Archaeological zone percentage calculations",
        "Theory B temporal non-locality detection",
        "RD@40% sequence path classification",
        "Session opening pattern analysis",
        "Temporal sequence analysis with price relativity",
        "Interactive visualization and reporting",
        "Backward compatibility with original interface"
    ],
    "dependencies": [
        "pandas >= 2.2.0",
        "numpy >= 1.24.0", 
        "matplotlib >= 3.5.0",
        "seaborn >= 0.11.0"
    ]
}

def get_module_info() -> dict:
    """Get information about the temporal analysis module"""
    return MODULE_INFO

def list_available_queries() -> list:
    """List available query types and patterns"""
    return [
        "Temporal Sequence Analysis: 'What happens after [pattern] in [session_type] sessions?'",
        "Archaeological Zone Analysis: 'Show zone distribution for [criteria]'",
        "Theory B Pattern Detection: 'Find Theory B events with [precision] criteria'",
        "RD@40% Sequence Analysis: 'Analyze RD@40% paths in [timeframe]'",
        "Opening Pattern Analysis: 'When sessions start with [pattern]'",
        "Relative Positioning: 'Analyze positioning patterns for [criteria]'",
        "Pattern Search: 'Search for [pattern] with [conditions]'",
        "Liquidity Analysis: 'Analyze liquidity sweeps in [context]'",
        "HTF Analysis: 'Show HTF taps at [levels]'",
        "FVG Analysis: 'Analyze FVG follow-through patterns'",
        "Event Chains: 'Show event chain sequences for [criteria]'",
        "Minute Hotspots: 'Find minute hotspots in [session_type]'",
        "News Impact: 'Analyze RD@40% day news matrix'",
        "Feature Interactions: 'Show F8 interactions with [features]'",
        "ML Predictions: 'Analyze ML predictions for [scenario]'"
    ]

def get_example_usage() -> str:
    """Get example usage code for the temporal analysis module"""
    return '''
# Example Usage of IRONFORGE Temporal Analysis Module

from ironforge.temporal import EnhancedTemporalQueryEngine

# Initialize the engine
engine = EnhancedTemporalQueryEngine(
    shard_dir="data/shards/NQ_M5",
    adapted_dir="data/adapted"
)

# Example queries
queries = [
    "What happens after liquidity sweeps in NY_AM sessions?",
    "Show archaeological zone distribution for 40% events",
    "Find Theory B events with high precision in LONDON sessions",
    "Analyze RD@40% continuation paths",
    "When NY_PM sessions start with gap patterns"
]

# Run analysis
for query in queries:
    print(f"\\n🔍 Query: {query}")
    results = engine.ask(query)
    engine.display_results(results)
    
    # Optional: Create visualizations
    engine.plot_results(results, save_path=f"analysis_{query[:20]}.png")

# Get session information
session_info = engine.get_session_statistics()
print(f"\\n📊 Loaded {session_info['total_sessions']} sessions")
print(f"📈 Total events: {session_info['total_events']}")

# List available sessions
sessions = engine.list_sessions()
print(f"\\n📋 Available sessions: {len(sessions)}")
for session in sessions[:5]:  # Show first 5
    print(f"  • {session}")
'''

# Convenience function for quick access
def create_engine(shard_dir: str = "data/shards/NQ_M5", 
                 adapted_dir: str = "data/adapted") -> EnhancedTemporalQueryEngine:
    """
    Create and initialize an EnhancedTemporalQueryEngine instance
    
    Args:
        shard_dir: Directory containing parquet shard data
        adapted_dir: Directory containing adapted JSON session data
        
    Returns:
        Initialized EnhancedTemporalQueryEngine instance
    """
    return EnhancedTemporalQueryEngine(shard_dir=shard_dir, adapted_dir=adapted_dir)

# Module initialization message
def _print_module_info():
    """Print module initialization information"""
    print("🔍 IRONFORGE Temporal Analysis Module Loaded")
    print(f"   Version: {__version__}")
    print(f"   Components: {len(MODULE_INFO['components'])} specialized modules")
    print(f"   Features: {len(MODULE_INFO['features'])} analysis capabilities")
    print("   Ready for temporal pattern analysis with archaeological zones")

# Print info when module is imported (optional - can be disabled)
# _print_module_info()
