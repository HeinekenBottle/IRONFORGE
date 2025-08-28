# 🚀 Enhanced Context Workflow Guide

Your practical toolkit for bug hunting and architecture decisions.

## Quick Start

```bash
# Navigate to your IRONFORGE directory
cd /Users/jack/IRONFORGE

# Quick overview of potential issues
./context_helpers/workflow_helper.sh quick

# Detailed bug hunting
./context_helpers/workflow_helper.sh bug

# Architecture analysis
./context_helpers/workflow_helper.sh arch [component_name]
```

## 🐛 Bug Hunting Workflow

### 1. Gather Context
```bash
# General bug hunting
./context_helpers/workflow_helper.sh bug

# Target specific issue
./context_helpers/workflow_helper.sh bug "authentication"
./context_helpers/workflow_helper.sh bug "performance"
./context_helpers/workflow_helper.sh bug "memory"
```

### 2. Use Claude Code Sub-agents
With the bug report generated, use Claude Code's Task tool:

```
"Use the general-purpose agent to analyze these flagged files for potential bugs and suggest fixes"
```

Or for data-driven analysis:
```
"Use the data-scientist agent to analyze these error patterns and identify the most critical issues to fix first"
```

### 3. Target Investigation
The bug hunter identifies:
- ❌ Recent changes that might introduce bugs
- 🚨 Files with high complexity
- 🔄 TODO/FIXME items requiring attention
- 📊 Test coverage gaps
- ⚡ Performance indicators

## 🏗️ Architecture Decision Workflow

### 1. Analyze Current State
```bash
# Full architecture analysis
./context_helpers/workflow_helper.sh arch

# Focus on specific component
./context_helpers/workflow_helper.sh arch "tgat_discovery"
./context_helpers/workflow_helper.sh arch "confluence"
```

### 2. Use Knowledge-Architect Agent
```
"Use the knowledge-architect agent to analyze this architecture report and document key decisions for the IRONFORGE system"
```

Or for recommendations:
```
"Based on this architecture analysis, suggest improvements for modularity and maintainability"
```

### 3. Architecture Insights
The analyzer provides:
- 📁 Component structure overview
- 🔗 Dependency patterns
- 🎨 Design pattern detection
- ⚙️ Configuration management analysis
- 🛡️ Error handling architecture
- ⚡ Performance architecture patterns

## 💡 Pro Tips

### Immediate Actions Based on Your IRONFORGE Analysis
Your quick scan showed:
- **1,188 TODO items** - High cleanup priority
- **8,940 debug prints** - Remove before production
- **861 large files** - Consider refactoring opportunities

### Workflow Integration
1. **Daily**: Run `./workflow_helper.sh quick` for overview
2. **Before releases**: Full bug hunt with `./workflow_helper.sh bug`
3. **Architecture reviews**: Use `./workflow_helper.sh arch [component]`
4. **Always**: Feed results to Claude Code sub-agents for analysis

### Claude Code Integration Examples

**Bug Investigation:**
```
"I found 15 recent changes and 50 TODO items in the TGAT discovery module. Use the general-purpose agent to search for potential issues in ironforge/learning/tgat_discovery.py and related files."
```

**Architecture Decision:**
```
"Based on this architecture analysis showing high coupling in the confluence scoring system, use the knowledge-architect agent to suggest a decoupling strategy and document the decision."
```

**Performance Investigation:**
```
"The bug hunter found 12 large files with high complexity. Use the data-scientist agent to analyze performance bottlenecks and suggest optimization priorities."
```

## 🎯 Your Success Metrics

- **Bug Prevention**: Catch issues before they reach production
- **Faster Debugging**: Context-driven investigation instead of random searching
- **Better Architecture**: Data-driven decisions instead of guesswork
- **Efficient Reviews**: Targeted analysis of high-risk areas

## 🔧 Customization

Edit the scripts in `/Users/jack/IRONFORGE/context_helpers/` to:
- Add IRONFORGE-specific patterns
- Include additional file types
- Customize thresholds and alerts
- Add integration with your specific tools

The system is designed to grow with your needs - start simple and add complexity as you identify patterns in your workflow.