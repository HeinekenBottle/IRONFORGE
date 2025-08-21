# New Internal Imports Between Split Modules

## Overview
This document lists the new internal imports created between the refactored modules in the IRONFORGE temporal analysis system.

## Module Import Dependencies

### 1. `query_core.py` → Other Modules
```python
from .session_manager import SessionDataManager
from .price_relativity import PriceRelativityEngine
```
**Reason**: TemporalQueryCore orchestrates session management and price analysis

### 2. `enhanced_temporal_query_engine.py` → All Modules
```python
from .session_manager import SessionDataManager
from .price_relativity import PriceRelativityEngine
from .query_core import TemporalQueryCore
from .visualization import VisualizationManager
```
**Reason**: Main interface coordinates all specialized modules

### 3. `__init__.py` → All Modules
```python
from .enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from .session_manager import SessionDataManager
from .price_relativity import PriceRelativityEngine
from .query_core import TemporalQueryCore
from .visualization import VisualizationManager
```
**Reason**: Public API exports for backward compatibility

## External Dependencies Preserved

All modules maintain their original external dependencies:

### Common External Imports
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
```

### Specialized External Imports
```python
# session_manager.py
import glob
import json
from pathlib import Path
from session_time_manager import SessionTimeManager

# price_relativity.py  
from archaeological_zone_calculator import ArchaeologicalZoneCalculator
from experiment_e_analyzer import ExperimentEAnalyzer

# query_core.py
import re
from ml_path_predictor import MLPathPredictor
from liquidity_htf_analyzer import LiquidityHTFAnalyzer

# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
```

## Import Flow Architecture

```
enhanced_temporal_query_engine.py (Main Interface)
├── session_manager.py (Data Management)
├── price_relativity.py (Analysis Engine)
├── query_core.py (Query Processing)
│   ├── session_manager.py (Data Access)
│   └── price_relativity.py (Analysis Delegation)
└── visualization.py (Display & Plotting)
```

## Dependency Injection Pattern

The refactored architecture uses dependency injection to maintain loose coupling:

### In `query_core.py`:
```python
class TemporalQueryCore:
    def __init__(self, session_manager: SessionDataManager, price_engine: PriceRelativityEngine):
        self.session_manager = session_manager
        self.price_engine = price_engine
```

### In `enhanced_temporal_query_engine.py`:
```python
class EnhancedTemporalQueryEngine:
    def __init__(self, shard_dir: str = "data/shards/NQ_M5", adapted_dir: str = "data/adapted"):
        # Initialize core modules
        self.session_manager = SessionDataManager(shard_dir, adapted_dir)
        self.price_engine = PriceRelativityEngine()
        self.query_core = TemporalQueryCore(self.session_manager, self.price_engine)
        self.visualization = VisualizationManager()
```

## Benefits of New Import Structure

### 1. **Clear Separation of Concerns**
- Each module imports only what it needs
- No circular dependencies
- Clean dependency hierarchy

### 2. **Maintainable Dependencies**
- Easy to track module relationships
- Simple to modify individual modules
- Clear impact analysis for changes

### 3. **Testable Architecture**
- Modules can be mocked easily
- Dependencies can be injected for testing
- Isolated unit testing possible

### 4. **Extensible Design**
- New modules can be added without affecting existing imports
- Plugin architecture potential
- Modular replacement of components

## Import Verification

All internal imports have been verified to:
- ✅ Use relative imports (`.module_name`) for internal modules
- ✅ Maintain absolute imports for external dependencies
- ✅ Follow Python import best practices
- ✅ Avoid circular dependencies
- ✅ Support proper module initialization order

## Backward Compatibility

The new import structure maintains full backward compatibility:

### Original Import (Still Works)
```python
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
```

### New Recommended Import
```python
from ironforge.temporal import EnhancedTemporalQueryEngine
```

### Advanced Usage (New Capability)
```python
from ironforge.temporal import SessionDataManager, PriceRelativityEngine
# Use individual modules directly
```

## Summary

The refactored import structure creates a clean, maintainable architecture while preserving all existing functionality. The new internal imports enable better separation of concerns and improved testability without breaking any existing code.
