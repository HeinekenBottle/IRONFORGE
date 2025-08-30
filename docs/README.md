# IRONFORGE Documentation Index
**Version**: 1.1.0  
**Last Updated**: 2025-01-15

## 🎯 Quick Navigation

### 🚀 Getting Started
- **[01-QUICKSTART.md](01-QUICKSTART.md)** - Installation, first discovery, and basic usage
- **[02-USER-GUIDE.md](02-USER-GUIDE.md)** - Complete guide to daily usage and workflows
- **[03-API-REFERENCE.md](03-API-REFERENCE.md)** - Complete API documentation with examples

### 🏗️ Architecture & Technical
- **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** - Complete system design and components
- **[05-DEPLOYMENT.md](05-DEPLOYMENT.md)** - Production deployment and monitoring
- **[06-TROUBLESHOOTING.md](06-TROUBLESHOOTING.md)** - Common issues and solutions
 - **[AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md)** - Multi‑agent system overview and links

### 📚 Reference & Support
- **[07-CHANGELOG.md](07-CHANGELOG.md)** - Version history and breaking changes
- **[08-GLOSSARY.md](08-GLOSSARY.md)** - Archaeological discovery terminology

### 🔬 Specialized Documentation
- **[specialized/TGAT-ARCHITECTURE.md](specialized/TGAT-ARCHITECTURE.md)** - Neural network implementation details
- **[specialized/SEMANTIC-FEATURES.md](specialized/SEMANTIC-FEATURES.md)** - Rich contextual event preservation system
- **[specialized/PATTERN-DISCOVERY.md](specialized/PATTERN-DISCOVERY.md)** - Advanced pattern analysis techniques
- **[specialized/MCP-INTEGRATION.md](specialized/MCP-INTEGRATION.md)** - Context7 MCP integration guide

### 📊 Release Documentation
- **[releases/](releases/)** - Version-specific release notes and documentation

### 🗄️ Archive
- **[archive/](archive/)** - Historical documentation and completed project reports

---

## 🎯 What is IRONFORGE?

IRONFORGE is a sophisticated **archaeological discovery engine** for market pattern analysis that combines:

- **Rule-based preprocessing**: Enhanced session adapter with event detection
- **TGAT (single ML core)**: Temporal graph attention networks for pattern learning  
- **Rule-based scoring**: Confluence analysis and validation rails
- **Within-session learning**: No cross-session edges, preserves session boundaries

### Key Capabilities
- **Systematic Processing**: All enhanced sessions with cross-session analysis
- **Pattern Intelligence**: Advanced classification, trending, and relationship mapping
- **Daily Workflows**: Morning prep, session hunting, performance tracking
- **Real-time Analysis**: Sub-5s initialization, efficient processing
- **Archaeological Focus**: Discovery of existing patterns (no predictions)

## 🚀 Quick Start

### Installation
```bash
pip install -e .[dev]
```

### First Discovery Run
```bash
# Discover → Score → Validate → Report
python -m ironforge.sdk.cli discover-temporal --config configs/dev.yml
python -m ironforge.sdk.cli score-session     --config configs/dev.yml
python -m ironforge.sdk.cli validate-run      --config configs/dev.yml
python -m ironforge.sdk.cli report-minimal    --config configs/dev.yml

# Open dashboard
open runs/$(date +%F)/minidash.html
```

### Programmatic Usage
```python
from ironforge.api import (
    run_discovery, score_confluence, validate_run, build_minidash,
    LoaderCfg, Paths, RunCfg
)

# 1) Discovery: provide shard directories, output dir, and loader configuration
patterns = run_discovery(
    shard_paths=["data/shards/NQ_5m/shard_2024-12-15"],
    out_dir="runs/2025-01-15/NQ_5m",
    loader_cfg=LoaderCfg(fanouts=(10, 10), batch_size=2048),
)

# 2) Confluence: score produced patterns with optional weights and threshold
scores_path = score_confluence(
    pattern_paths=patterns,
    out_dir="runs/2025-01-15/NQ_5m/confluence",
    _weights=None,
    threshold=65.0,
)

# 3) Validation: high-level run validation using config
run_cfg = RunCfg(paths=Paths(shards_dir="data/shards/NQ_5m", out_dir="runs/2025-01-15/NQ_5m"))
validation = validate_run(run_cfg)

# 4) Reporting: build a minimal dashboard from dataframes (illustrative)
import pandas as pd
activity = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=10, freq="min"), "count": range(10)})
confluence_df = pd.DataFrame({"ts": activity["ts"], "score": range(0, 100, 10)})
build_minidash(activity, confluence_df, motifs=[], out_html="runs/2025-01-15/minidash.html", out_png="runs/2025-01-15/minidash.png")
```

### Golden Invariants

The following data contracts are invariant and must be preserved across the system:
- Events: 6
- Edge intents: 4
- Node features: 45D (default) / 51D (HTF ON)
- Edge features: 20D
- HTF sampling: last‑closed only
- Session isolation: enforced

## 📋 Documentation Standards

This documentation follows the [Documentation Standards](DOCUMENTATION_STANDARDS.md) for:
- **Naming Convention**: Hierarchical, purpose-driven file naming
- **Content Structure**: Consistent formatting and cross-references
- **Maintenance**: Regular updates and archival procedures

## 🔗 External Resources

- **GitHub Repository**: Source code and issue tracking
- **Context7 MCP**: Model Context Protocol integration
- **Iron-Core**: Shared mathematical infrastructure

---

**Last Updated**: 2025-01-15  
**Version**: 1.1.0  
**Maintainer**: IRONFORGE Team