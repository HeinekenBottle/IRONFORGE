# IRONFORGE Comprehensive Refactoring Plan

## Current State Analysis

### Critical Issues Identified
1. **Root Directory Chaos**: 100+ loose files including tests, scripts, data files
2. **Import Inconsistencies**: Mix of relative/absolute imports, circular dependencies  
3. **Duplicate Functionality**: Multiple test files, redundant analysis scripts
4. **Data Disorganization**: Scattered data across multiple directories
5. **Dependency Conflicts**: PyTorch installation issues, iron-core version conflicts

### Architecture Assessment
- **IRON-CORE**: ✅ Well-structured, production-ready infrastructure
- **IRONFORGE**: ❌ Needs complete reorganization
- **Integration Layer**: ⚠️ Functional but needs cleanup
- **Data Pipeline**: ❌ Fragmented across multiple locations

## Phase 1: Directory Structure Consolidation

### Target Structure
```
/IRONFORGE/
├── iron_core/                    # Keep existing - well structured
├── ironforge/                    # Main application package
│   ├── __init__.py
│   ├── core/                     # Core business logic
│   ├── learning/                 # ML/TGAT components  
│   ├── analysis/                 # Pattern analysis
│   ├── synthesis/                # Pattern validation
│   └── integration/              # System integration
├── data/                         # Consolidated data storage
│   ├── raw/                      # Level 1 raw market data
│   ├── enhanced/                 # Enhanced/processed sessions
│   ├── adapted/                  # Adapted sessions with relativity
│   └── discoveries/              # Pattern discoveries
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test data
├── scripts/                      # Utility and runner scripts
│   ├── analysis/                 # Analysis runners
│   ├── data_processing/          # Data pipeline scripts
│   └── utilities/                # General utilities
├── docs/                         # Documentation
├── reports/                      # Generated reports
└── config/                       # Configuration files
```

### Files to Consolidate/Move

#### Root Level Cleanup (Move to appropriate directories)
- `test_*.py` → `tests/integration/`
- `run_*.py` → `scripts/analysis/`
- `analyze_*.py` → `scripts/analysis/`
- `phase*.py` → `scripts/analysis/`
- `*.json` data files → `data/discoveries/`
- `*.png` visualizations → `reports/visualizations/`
- Documentation `.md` files → `docs/`

#### Data Directory Consolidation
- `enhanced_sessions/` → `data/enhanced/`
- `enhanced_sessions_with_relativity/` → `data/enhanced/`
- `adapted_enhanced_sessions/` → `data/adapted/`
- `discoveries/` → `data/discoveries/`
- `data/sessions/` → `data/raw/`

## Phase 2: Import System Standardization

### Import Strategy
1. **Absolute Imports Only**: All imports from package root
2. **Clean Dependencies**: Remove circular imports
3. **Lazy Loading**: Maintain iron-core lazy loading benefits
4. **Type Hints**: Add comprehensive type annotations

### Import Patterns to Fix
```python
# BEFORE (problematic)
from ..analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from learning.enhanced_graph_builder import EnhancedGraphBuilder
import sys; sys.path.append(...)

# AFTER (standardized)
from ironforge.analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
```

## Phase 3: Dependency Management

### Requirements Consolidation
- Single `requirements.txt` at root
- Remove version conflicts
- Add missing dependencies
- Clean up unused packages

### Iron-Core Integration
- Ensure proper iron-core package installation
- Maintain lazy loading architecture
- Fix container initialization issues

## Phase 4: Code Quality Standardization

### Naming Conventions
- **Classes**: PascalCase
- **Functions/Methods**: snake_case  
- **Constants**: UPPER_SNAKE_CASE
- **Modules**: lowercase_with_underscores

### Documentation Standards
```python
def discover_patterns(session_data: Dict[str, Any]) -> List[Pattern]:
    """
    Discover archaeological patterns in market session data.
    
    Args:
        session_data: Level 1 market session data with OHLC and events
        
    Returns:
        List of discovered patterns with confidence scores
        
    Raises:
        ValidationError: If session data format is invalid
    """
```

### Error Handling Patterns
- Consistent exception types
- Proper logging throughout
- No silent failures
- Graceful degradation where appropriate

## Phase 5: Testing Infrastructure

### Test Organization
```
tests/
├── unit/
│   ├── test_learning/
│   ├── test_analysis/
│   └── test_synthesis/
├── integration/
│   ├── test_full_pipeline/
│   ├── test_data_flow/
│   └── test_discovery_workflow/
└── fixtures/
    ├── sample_sessions/
    └── expected_outputs/
```

### Test Coverage Goals
- Unit tests: >90% coverage
- Integration tests: All major workflows
- Performance tests: Regression prevention
- Data validation tests: All input/output formats

## Implementation Priority

### High Priority (Phase 1)
1. Move all loose root files to appropriate directories
2. Create proper package structure with `__init__.py` files
3. Consolidate data directories
4. Fix critical import issues

### Medium Priority (Phase 2)  
1. Standardize all import statements
2. Remove duplicate functionality
3. Update requirements.txt
4. Add comprehensive logging

### Low Priority (Phase 3)
1. Add type hints throughout
2. Improve documentation
3. Optimize performance bottlenecks
4. Enhance test coverage

## Success Metrics

### Technical Metrics
- Zero circular imports
- All tests passing
- <3 second import time
- Clean `pytest` run with no warnings

### Organizational Metrics  
- <10 files in repository root
- All data properly categorized
- Consistent naming throughout
- Complete documentation coverage

## Risk Mitigation

### Backup Strategy
- Create `backup/` directory with current state
- Incremental commits for each phase
- Rollback procedures documented

### Testing Strategy
- Run full test suite after each phase
- Validate all data processing pipelines
- Ensure no functionality regression

### Performance Monitoring
- Benchmark import times before/after
- Monitor memory usage patterns
- Track discovery accuracy metrics
