# IRONFORGE Pipeline Contract Mismatch Problem Statement

## Root Cause Summary

The IRONFORGE system experienced a critical pipeline contract mismatch between CLI command interfaces and the discovery pipeline implementation. The CLI layer expected structured configuration objects with explicit attribute access patterns (e.g., `config.participating_agents`), while the discovery pipeline components were designed to receive raw dictionary parameters. This fundamental interface mismatch caused the system to fall back to weak JSON-only processing instead of leveraging the powerful TGAT-grounded analysis capabilities.

**Core Issue**: Missing adapter layers between CLI configuration parsing and discovery pipeline execution. When CLI commands passed configuration objects directly to discovery methods that expected dictionaries, attribute access failures occurred (`'dict' object has no attribute 'participating_agents'`, `'CoordinationResults' object is not subscriptable`). This forced the system to bypass TGAT discovery entirely, resulting in 0-2 metamorphosis pattern detections instead of the expected 10+ patterns.

## Impact on Metamorphosis Detection Performance

The contract mismatch caused a catastrophic degradation in pattern discovery capabilities:

- **Expected Performance**: 10+ metamorphosis patterns with TGAT-grounded analysis achieving 92.3/100 authenticity scores
- **Actual Performance**: 0-2 weak patterns using fallback JSON processing with minimal analytical depth
- **System Behavior**: Discovery pipeline silently failed over to simulation mode instead of proper TGAT memory workflows
- **Quality Impact**: Pattern graduation thresholds (87% authenticity) could not be met, preventing production-ready discoveries

## Prevention: Architectural Contract Enforcement

**Key Principle**: All pipeline stage interfaces must enforce explicit data contracts through adapter pattern implementation. CLI commands should never pass raw configuration objects directly to discovery pipelines without translation layers.

**Prevention Measures**:
1. Implement strict interface adapters between CLI and core discovery components
2. Enforce data contract validation at pipeline boundaries
3. Fail fast with explicit error messages when contract mismatches occur (instead of silent fallback to weak processing)
4. Use type hints and runtime validation to catch interface mismatches during development

This issue demonstrates the critical importance of maintaining clean separation between command interfaces and core processing pipelines in complex ML systems.