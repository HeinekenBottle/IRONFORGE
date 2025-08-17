# IRONFORGE Semantic Codebase Indexer

A deep semantic analysis tool for the IRONFORGE multi-engine archaeological discovery system. Generates AI-friendly reports and human-readable summaries of the codebase architecture, dependencies, and complexity patterns.

## Quick Start

From the IRONFORGE project root:

```bash
# Full analysis with all reports
./index-project

# Quick markdown summary 
./index-project summary

# Focus on engine architecture
./index-project engines

# Analyze dependencies only
./index-project dependencies

# Fast complexity check
./index-project quick
```

## Features

- **AST-based Analysis**: Deep Python code parsing with classes, functions, imports, and complexity metrics
- **Multi-Engine Classification**: Automatically categorizes components into IRONFORGE's engine architecture
- **Dependency Mapping**: Tracks cross-engine flows, circular dependencies, and coupling patterns
- **Pattern Detection**: Identifies design patterns (Factory, Builder, Container, Strategy, etc.)
- **AI-Optimized Output**: Generates semantic reports specifically designed for AI assistant consumption
- **Terminal Native**: Pure Python stdlib, no external dependencies, optimized for CLI workflows

## Analysis Modes

### Full Analysis (`./index-project` or `./index-project full`)

Performs comprehensive analysis and generates all report types:

```bash
./index-project
# Generates:
# - ironforge_index.json (AI-optimized semantic analysis)
# - ironforge_summary.md (human-readable overview)
# - ironforge_engines.json (engine-specific details)
# - ironforge_dependencies.json (dependency analysis)
```

**Output Example:**
```
📊 PROJECT OVERVIEW
   Files Analyzed: 131
   Lines of Code: 45,731
   Functions: 206
   Classes: 180
   Avg Complexity: 6.4

🏗️  ENGINE ARCHITECTURE
   Total Engines: 8
   Analysis: 55 files
   Learning: 3 files
   Integration: 25 files
```

### Summary Mode (`./index-project summary`)

Quick analysis focused on generating the markdown summary:

```bash
./index-project summary
# Generates: ironforge_summary.md
```

Perfect for getting a quick architectural overview or sharing with team members.

### Engine Analysis (`./index-project engines`)

Focuses on the multi-engine architecture classification:

```bash
./index-project engines
# Output:
🔧 ANALYSIS ENGINE
   Files: 15
   Lines: 7,650
   Avg Complexity: 62.8
   Description: Pattern analysis and session adaptation components

🔧 LEARNING ENGINE
   Files: 3
   Lines: 1,346
   Avg Complexity: 72.0
   Description: Machine learning, TGAT discovery, and graph building
```

### Dependency Analysis (`./index-project dependencies`)

Analyzes import relationships and cross-engine dependencies:

```bash
./index-project dependencies
# Shows:
# - Total dependencies
# - Circular dependency detection
# - Hub modules (high centrality)
# - Cross-engine flow patterns
```

### Quick Mode (`./index-project quick`)

Fast complexity analysis with minimal processing:

```bash
./index-project quick
# Fast analysis focused on complexity hotspots
```

## Command-Line Options

### Output Control

```bash
# Custom output directory
./index-project --output ./reports

# JSON output to stdout (for piping)
./index-project --json-stdout | jq .project_overview

# Include test files in analysis
./index-project --include-tests

# Verbose logging
./index-project --verbose

# Quiet mode (minimal output)
./index-project --quiet
```

### Format Selection

```bash
# Generate only JSON reports
./index-project --format json

# Generate only Markdown reports  
./index-project --format markdown

# Generate both (default)
./index-project --format both
```

## Generated Reports

### `ironforge_index.json` - AI Assistant Optimized

Complete semantic analysis optimized for AI consumption:

```json
{
  "project_overview": {
    "name": "IRONFORGE",
    "architecture_type": "Multi-Engine Archaeological Discovery System",
    "total_files": 131,
    "key_technologies": ["TGAT", "PyTorch", "NetworkX"]
  },
  "engine_architecture": {
    "analysis": {
      "description": "Pattern analysis and session adaptation",
      "metrics": { "file_count": 15, "avg_complexity": 62.8 },
      "key_components": [...],
      "public_interfaces": [...]
    }
  },
  "dependency_map": {
    "cross_engine_flows": {...},
    "circular_dependencies": [...],
    "hub_modules": [...]
  }
}
```

### `ironforge_summary.md` - Human Readable

Markdown report with architecture overview:

```markdown
# IRONFORGE Codebase Analysis Report

## Engine Architecture
### Analysis Engine
**Description**: Pattern analysis and session adaptation components
**Key Classes**:
- **TimeframeLatticeMapper**: Maps temporal relationships...
- **EnhancedSessionAdapter**: Adapts sessions with features...

## Dependency Analysis
### Cross-Engine Flows
- **analysis → learning**: 15 dependencies
- **learning → synthesis**: 8 dependencies
```

### `ironforge_engines.json` - Engine Details

Detailed breakdown of each engine with components and metrics.

### `ironforge_dependencies.json` - Dependency Analysis

Complete dependency mapping with import graphs, circular dependencies, and coupling metrics.

## Integration Examples

### Terminal Workflows

```bash
# Daily architecture check
./index-project quick && echo "✅ Architecture healthy"

# Before major refactoring
./index-project dependencies > deps_before.json

# Generate reports for documentation
./index-project summary --output ./docs

# Pipeline integration
./index-project --json-stdout | jq '.complexity_analysis.hotspots'
```

### AI Assistant Integration

The generated reports are optimized for AI assistant consumption:

```bash
# Feed analysis to AI for architectural advice
./index-project --json-stdout | ai-assistant analyze-architecture

# Get refactoring suggestions
cat ironforge_index.json | ai-assistant suggest-refactoring
```

## Architecture Insights

The indexer automatically detects and reports:

### Engine Classification

- **Analysis Engine**: Pattern analysis, session adaptation (TimeframeLatticeMapper, etc.)
- **Learning Engine**: TGAT discovery, graph building (EnhancedGraphBuilder, etc.)
- **Synthesis Engine**: Pattern validation, graduation (PatternGraduation, etc.)
- **Integration Engine**: Container system, configuration (IRONContainer, etc.)
- **Validation Engine**: Testing, quality assurance
- **Utilities Engine**: Support tools, migration scripts

### Dependency Patterns

- **Cross-Engine Flows**: Dependencies between engines
- **Circular Dependencies**: Detected automatically with severity levels
- **Hub Modules**: High-centrality components that many others depend on
- **Coupling Metrics**: Afferent/efferent coupling, instability scores

### Complexity Analysis

- **Hotspots**: Functions with high cyclomatic complexity
- **Architecture Health**: Balance between engines, dependency health
- **Pattern Detection**: Factory, Builder, Container, Strategy patterns

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the IRONFORGE project root
cd /path/to/IRONFORGE
./index-project
```

**Permission Issues:**
```bash
# Make script executable
chmod +x index-project
```

**Python Path Issues:**
```bash
# The script automatically manages Python paths
# Ensure IRONFORGE directory structure is intact
```

### Performance

- **Full analysis**: ~7 seconds for 131 files
- **Quick analysis**: ~1-2 seconds  
- **Memory usage**: <100MB typical
- **Large codebases**: Automatically limits scope in quick mode

### Verbose Output

For debugging, use verbose mode:

```bash
./index-project --verbose
# Shows detailed progress:
# - File discovery process
# - AST parsing status
# - Engine classification logic
# - Dependency mapping progress
```

## Advanced Usage

### Custom Analysis Scripts

The indexer can be imported and used programmatically:

```python
from tools.indexer import IRONFORGEIndexer

indexer = IRONFORGEIndexer('/path/to/project')
results = indexer.analyze_codebase()

# Access specific data
engine_arch = results['engine_architecture']
complexity = results['complexity_analysis']
```

### Integration with CI/CD

```bash
# In CI pipeline
./index-project quick --quiet
if [ $? -eq 0 ]; then
    echo "Architecture analysis passed"
else
    echo "Architecture issues detected"
    exit 1
fi
```

### Monitoring Architecture Evolution

```bash
# Track changes over time
./index-project --json-stdout > analysis_$(date +%Y%m%d).json

# Compare with previous analysis
jq '.project_overview' analysis_*.json
```

## Technical Details

### Analysis Methodology

1. **File Discovery**: Recursively finds Python files in key directories
2. **AST Parsing**: Uses Python's `ast` module for syntax tree analysis
3. **Engine Classification**: Pattern-based classification using file paths and content
4. **Dependency Mapping**: Import statement analysis with relationship tracking
5. **Complexity Calculation**: Cyclomatic complexity using control flow analysis
6. **Pattern Detection**: Heuristic-based design pattern identification

### Supported Python Features

- Classes, functions, methods (including async)
- Type hints and annotations
- Decorators and properties
- Import statements (absolute and relative)
- Docstrings and comments
- Design patterns (Factory, Builder, Singleton, Strategy, Container/DI)

### Performance Optimizations

- **Lazy Loading**: Components loaded only when needed
- **Parallel Processing**: File analysis can be parallelized
- **Selective Analysis**: Quick mode processes subset of files
- **Efficient Storage**: Results cached to avoid reprocessing

---

## Support

This indexer is specifically designed for the IRONFORGE multi-engine archaeological discovery system. It understands the unique architecture patterns and provides targeted analysis for this codebase.

For issues or enhancements, the indexer code is located in `tools/indexer/` and can be modified to suit evolving analysis needs.

**Version**: 1.0.0  
**Compatibility**: Python 3.8+, pure stdlib  
**Performance**: <10s full analysis, <2s quick analysis  
**Output**: JSON (AI-optimized), Markdown (human-readable)