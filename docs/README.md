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
from ironforge.api import run_discovery, score_confluence, validate_run, build_minidash
from ironforge.api import Config, load_config

# Load configuration
config = load_config('configs/dev.yml')

# Run complete pipeline
discovery_results = run_discovery(config)
confluence_results = score_confluence(config)
validation_results = validate_run(config)
dashboard = build_minidash(config)
```

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