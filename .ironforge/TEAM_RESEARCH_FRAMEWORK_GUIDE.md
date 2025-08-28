# IRONFORGE Research Framework - Team Guide

## 🎯 Overview

IRONFORGE is now configured as a **research-agnostic platform** that can investigate **any** market structure hypothesis with professional rigor. This guide ensures everyone uses the system correctly and consistently.

## ❗ CRITICAL: What Changed

### ❌ OLD WAY (Don't Do This Anymore)
```python
# WRONG - Hardcoded assumptions
zone_percentage = 0.40  # Assumes 40% zones always work
event_families = ["FVG", "liquidity", "expansion"]  # Assumes these families exist
if archaeological_zone:  # Assumes archaeological zones are always relevant
    temporal_non_locality = True  # Assumes this relationship always exists
```

### ✅ NEW WAY (Always Do This)
```python
# CORRECT - Configuration-driven research
research_config = {
    "research_question": "Do clustering events occur at specific percentage levels?",
    "hypothesis_parameters": {
        "percentage_levels": [20, 30, 40, 50, 60, 70, 80],  # Test multiple levels
        "time_windows": [30, 60, 120, 300],
        "clustering_metrics": ["intensity", "count", "duration"]
    },
    "discovery_method": "tgat_unsupervised_attention",  # Let TGAT find patterns
    "agents": ["data-scientist", "adjacent-possible-linker"]
}
```

## 🚀 Quick Start for Team Members

### 1. Always Start with Configuration
```bash
# Create research configuration first
cat > my_research.yml << 'EOF'
research_question: "What percentage levels show significant clustering?"
hypothesis_parameters:
  percentage_levels: [25, 35, 45, 55, 65, 75]
  time_windows: [60, 120, 300]
  clustering_metrics: ["intensity", "count"]
discovery_method: "tgat_unsupervised_attention"
validation_method: "statistical_significance"
agents: ["data-scientist"]
authenticity_threshold: 0.87
EOF
```

### 2. Use the Research Template
```python
from .ironforge.research_templates.configurable_research_template import (
    create_research_configuration, ConfigurableResearchFramework
)

# Create configuration
config = create_research_configuration(
    research_question="Your specific research question here",
    hypothesis_parameters={"param1": [values], "param2": [values]},
    agents=["data-scientist", "adjacent-possible-linker"]
)

# Execute research
framework = ConfigurableResearchFramework(config)
results = framework.execute_research()
```

### 3. Validate Your Research
```bash
# Always validate before execution
python .ironforge/research_framework_validator.py --directory .
```

## 🤝 Agent Coordination Standards

### When to Use Each Agent

**data-scientist** (Always recommended):
```python
# Use for: Statistical analysis, hypothesis testing, validation
research_config = {
    "agents": ["data-scientist"],
    "research_question": "Statistical relationship between X and Y?",
    "hypothesis_parameters": {"significance_level": 0.01}
}
```

**knowledge-architect** (For cross-session research):
```python
# Use for: Knowledge preservation, context continuity, documentation
research_config = {
    "agents": ["data-scientist", "knowledge-architect"],
    "research_question": "How do patterns evolve across sessions?",
    "coordination_method": "cross_session_analysis"
}
```

**adjacent-possible-linker** (For creative research):
```python
# Use for: Creative connections, novel insights, emergent patterns
research_config = {
    "agents": ["data-scientist", "adjacent-possible-linker"],
    "research_question": "What unexpected relationships exist between...?",
    "discovery_method": "creative_pattern_discovery"
}
```

**scrum-master** (For complex projects):
```python
# Use for: Multi-phase research, sprint management, team coordination
research_config = {
    "agents": ["data-scientist", "knowledge-architect", "adjacent-possible-linker", "scrum-master"],
    "research_question": "Comprehensive analysis of market structure patterns",
    "coordination_method": "agile_research_management"
}
```

## 🔒 Quality Gates (ALWAYS ENFORCE)

### Required Quality Standards
```python
quality_gates = {
    "pattern_authenticity": 0.87,        # 87% minimum authenticity
    "statistical_significance": 0.01,    # p < 0.01 for significance
    "confidence_minimum": 0.95,          # 95% confidence intervals
    "cross_validation_score": 0.80       # 80% cross-validation score
}
```

### Quality Gate Checklist
- [ ] Research question explicitly defined (not assumed)
- [ ] Hypothesis parameters configurable (not hardcoded)
- [ ] Statistical validation method specified
- [ ] TGAT discovery without pattern assumptions
- [ ] 87% authenticity threshold enforced
- [ ] Agent coordination for complex research
- [ ] Results pass quality gates before production

## 📊 Research Examples (Team Reference)

### Example 1: Percentage Level Research
```python
# Research any percentage levels (not just 40%)
percentage_research = create_research_configuration(
    research_question="Do events cluster at specific percentage retracement levels?",
    hypothesis_parameters={
        "percentage_levels": [23.6, 38.2, 50.0, 61.8, 78.6],  # Fibonacci levels
        "event_types": ["reversal", "continuation", "breakout"],
        "time_precision": [30, 60, 120],  # seconds
        "clustering_thresholds": [2.0, 3.0, 5.0]  # minimum clustering strength
    },
    agents=["data-scientist", "adjacent-possible-linker"]
)
```

### Example 2: Time-Based Pattern Research
```python
# Research temporal relationships without assumptions
temporal_research = create_research_configuration(
    research_question="What time-based patterns exist in market structure events?",
    hypothesis_parameters={
        "time_windows": [60, 300, 900, 1800, 3600],  # 1min to 1hour
        "event_sequences": ["AB", "ABC", "ABCD"],  # pattern lengths
        "temporal_relationships": ["leading", "lagging", "simultaneous"],
        "correlation_strengths": [0.3, 0.5, 0.7, 0.9]
    },
    agents=["data-scientist", "knowledge-architect"]
)
```

### Example 3: Volume-Price Relationship Research
```python
# Research volume-price relationships without assumptions
volume_price_research = create_research_configuration(
    research_question="How do volume patterns relate to price movement patterns?",
    hypothesis_parameters={
        "volume_patterns": ["surge", "decline", "stability", "spike"],
        "price_patterns": ["trend", "reversal", "consolidation", "breakout"],
        "relationship_delays": [0, 30, 60, 120, 300],  # seconds
        "correlation_methods": ["pearson", "spearman", "kendall"]
    },
    agents=["data-scientist", "adjacent-possible-linker", "knowledge-architect"]
)
```

## 🚨 Common Violations (What NOT to Do)

### ❌ Hardcoded Assumptions
```python
# DON'T DO THIS - Hardcoded patterns
if zone_percentage == 0.40:  # Assumes 40% zones are special
    apply_archaeological_analysis()

# DON'T DO THIS - Assumed event families
event_families = ["FVG", "liquidity", "expansion"]  # Assumes these exist

# DON'T DO THIS - Hardcoded temporal relationships
temporal_non_locality = True  # Assumes this always works
```

### ❌ Bypassing Agent Coordination
```python
# DON'T DO THIS - Manual analysis for complex research
def manual_pattern_analysis():  # Should use agent coordination
    # Complex statistical analysis...
    # Hypothesis testing...
    # Cross-session learning...
    pass  # This should use data-scientist agent
```

### ❌ Skipping Statistical Validation
```python
# DON'T DO THIS - No statistical validation
patterns = discover_patterns()  # No significance testing
return patterns  # No quality gates, no confidence intervals
```

### ❌ Ignoring Quality Thresholds
```python
# DON'T DO THIS - Ignoring authenticity
def accept_any_pattern(pattern):
    return True  # Should enforce 87% authenticity threshold
```

## ✅ Framework Compliance Checklist

Before any research execution:

### Configuration Phase
- [ ] Research question explicitly defined (not "investigate 40% zones")
- [ ] Hypothesis parameters are lists/ranges (not single values)
- [ ] Discovery method specified (usually "tgat_unsupervised_attention")
- [ ] Validation method specified (usually "statistical_significance")
- [ ] No hardcoded pattern assumptions in configuration

### Agent Coordination Phase
- [ ] Appropriate agents selected based on research complexity
- [ ] data-scientist agent included for statistical rigor
- [ ] knowledge-architect agent included for cross-session research
- [ ] adjacent-possible-linker agent included for creative research
- [ ] Coordination method specified

### Execution Phase
- [ ] TGAT discovery runs without pattern assumptions
- [ ] Statistical validation applied to all discovered patterns
- [ ] Quality gates enforced (87% authenticity minimum)
- [ ] Results documented with methodology
- [ ] Framework validator passes without violations

### Results Phase
- [ ] Statistical significance reported (p-values, confidence intervals)
- [ ] Pattern authenticity scores included
- [ ] Quality gate results documented
- [ ] Recommendations provided for next steps
- [ ] Knowledge preserved for future research

## 🔧 Tools & Validation

### Framework Validator
```bash
# Run validator before any research
python .ironforge/research_framework_validator.py

# Generate compliance report
python .ironforge/research_framework_validator.py --directory . --strict

# Check specific files
python .ironforge/research_framework_validator.py --directory scripts/analysis/
```

### Research Templates
```bash
# Use templates for consistent methodology
cp .ironforge/research_templates/configurable_research_template.py my_research.py

# Modify template for your specific research question
# Keep the framework structure, change the parameters
```

### Quality Monitoring
```python
# Always check quality before production use
if not results['quality_assessment']['production_ready']:
    print("⚠️ Results failed quality gates - review before use")
    print(f"Gates failed: {results['quality_assessment']['gates_failed']}")
else:
    print("✅ Results passed quality gates - ready for production")
```

## 📈 Success Metrics

### Team Framework Adoption
- **Compliance Rate**: >80% of research files pass framework validation
- **Agent Usage**: Complex research uses appropriate agent coordination
- **Quality Gates**: All production research passes 87% authenticity threshold
- **Configuration-Driven**: No hardcoded pattern assumptions in research code

### Research Quality Indicators
- **Statistical Rigor**: All pattern discoveries include significance testing
- **Reproducibility**: Research configurations enable result reproduction
- **Methodology**: Systematic approach using professional research standards
- **Knowledge Preservation**: Cross-session learning via knowledge-architect agent

## 🆘 Getting Help

### Framework Issues
1. **Run the validator**: `python .ironforge/research_framework_validator.py`
2. **Check the compliance report**: Review generated markdown report
3. **Use templates**: Start with configurable_research_template.py
4. **Follow examples**: Use the provided research examples as guides

### Agent Coordination Issues
1. **Review agent descriptions**: Check .claude/agents/ for each agent's capabilities
2. **Match agents to complexity**: Use appropriate agents for research complexity
3. **Check coordination patterns**: Ensure agents work together systematically
4. **Validate quality gates**: Ensure agents enforce quality standards

### Research Methodology Questions
1. **Configuration-first approach**: Always start with research_question and hypothesis_parameters
2. **Statistical validation**: Include significance testing and confidence intervals
3. **TGAT discovery**: Let TGAT find patterns without assumptions
4. **Quality enforcement**: Apply 87% authenticity thresholds consistently

## 🎯 Team Success Formula

**Configuration + Agents + Statistics + Quality = Professional Research**

1. **Configure** your research question and hypothesis parameters explicitly
2. **Coordinate** appropriate agents for systematic analysis
3. **Validate** statistically with significance testing and confidence intervals
4. **Enforce** quality gates and authenticity thresholds

This ensures IRONFORGE operates as a flexible, professional research platform that can investigate any market structure hypothesis with rigorous methodology.

---

**Remember: IRONFORGE discovered 40% zones, temporal non-locality, and FVG families - but it's capable of discovering completely different patterns. Don't limit the system to its past discoveries.**