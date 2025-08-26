# STRICT vs COMPAT Execution Modes

This guide explains IRONFORGE's dual execution modes for handling hardware acceleration requirements and fallback behavior.

## Overview

IRONFORGE supports two runtime execution modes:

- **STRICT Mode**: Fail fast when required acceleration is unavailable
- **COMPAT Mode**: Graceful fallback with degraded performance warnings

## Quick Start

### Environment Variables

```bash
# Set runtime mode (default: strict)
export IRONFORGE_RUNTIME_MODE=strict  # or 'compat'

# Control individual acceleration features
export IRONFORGE_ENABLE_SDPA=true
export IRONFORGE_ENABLE_FLASH=true
export IRONFORGE_ENABLE_AMP=true
export IRONFORGE_ENABLE_CDC=true

# Audit and performance settings
export IRONFORGE_ENABLE_AUDIT=true
export IRONFORGE_AUDIT_FILE=audit_run.json
export IRONFORGE_TIMEOUT=300
```

### YAML Configuration

```yaml
# config.yml
runtime:
  mode: strict  # or 'compat'
  enable_sdpa: true
  enable_flash_attention: true
  enable_amp: true
  enable_cdc: true
  operation_timeout_seconds: 300
  memory_limit_gb: 8.0
  enable_audit_logging: true
  audit_file_path: audit_run.json
```

## Acceleration Components

### SDPA (Scaled Dot Product Attention)
- **Requirement**: PyTorch >= 2.0
- **Benefit**: GPU-optimized attention computation
- **Fallback**: Manual attention implementation

### Flash Attention
- **Requirement**: CUDA + PyTorch with Flash Attention support
- **Benefit**: Memory-efficient attention for large sequences
- **Fallback**: Standard SDPA or manual attention

### AMP (Automatic Mixed Precision)
- **Requirement**: CUDA + `torch.cuda.amp`
- **Benefit**: FP16 computation for faster training
- **Fallback**: FP32 computation

### CDC (Compile Dynamic Cache)
- **Requirement**: PyTorch >= 2.0 with `torch.compile`
- **Benefit**: JIT compilation optimizations
- **Fallback**: Eager execution

## STRICT Mode

### Behavior
- **Fail Fast**: Raises `RuntimeError` if required acceleration is unavailable
- **No Degradation**: Ensures consistent performance across environments
- **Explicit Requirements**: Clear error messages with setup instructions

### Use Cases
- Production deployments with guaranteed hardware
- Performance-critical applications
- CI/CD validation of optimal configurations
- Benchmarking and performance regression testing

### Example

```python
from ironforge.sdk.runtime_config import initialize_runtime_system

# Will fail if acceleration requirements not met
config, accel_state, enforcer = initialize_runtime_system()

# Use with TGAT
from ironforge.learning.runtime_optimized_tgat import create_runtime_optimized_tgat
from ironforge.learning.dual_graph_config import TGATConfig

base_config = TGATConfig(input_dim=45, hidden_dim=44, num_heads=4)
tgat_engine = create_runtime_optimized_tgat(base_config, config)
```

### Error Messages

```
STRICT mode acceleration requirements not met:
  - SDPA (Scaled Dot Product Attention) is required but unavailable. 
    Please upgrade to PyTorch >= 2.0 or set IRONFORGE_RUNTIME_MODE=compat
  - Flash Attention is required but unavailable. 
    Please ensure CUDA is available and PyTorch has Flash Attention support, 
    or set IRONFORGE_RUNTIME_MODE=compat
```

## COMPAT Mode

### Behavior
- **Graceful Fallback**: Continues with degraded performance
- **Degradation Logging**: Warns about missing acceleration
- **Broad Compatibility**: Works across diverse environments

### Use Cases
- Development and testing environments
- Cloud deployments with variable hardware
- Legacy system compatibility
- Exploratory data analysis

### Example

```python
import os
os.environ['IRONFORGE_RUNTIME_MODE'] = 'compat'

from ironforge.sdk.runtime_config import initialize_runtime_system

# Will not fail, but may run with degraded performance
config, accel_state, enforcer = initialize_runtime_system()

if accel_state.is_degraded():
    reasons = accel_state.get_degradation_reasons()
    print("Running with degraded performance:")
    for reason in reasons:
        print(f"  - {reason}")
```

### Warning Messages

```
COMPAT mode: Running with degraded performance
  - SDPA (Scaled Dot Product Attention) unavailable
  - Flash Attention backend unavailable
  - Automatic Mixed Precision disabled
```

## Audit System

### Audit Run Ledger

The audit system captures detailed execution metadata:

```json
{
  "format_version": "1.0",
  "runs": [
    {
      "mode": "strict",
      "start_time": "2024-01-15 14:30:00",
      "wall_time": 12.5,
      "peak_mem": 3.2,
      "sdpa": "used",
      "flash": "used",
      "amp": "used",
      "cdc": "used",
      "degraded": false,
      "reasons": [],
      "pytorch_version": "2.2.0",
      "cuda_available": true,
      "device_name": "NVIDIA A100-SXM4-40GB"
    }
  ]
}
```

### Audit Fields

| Field | Description |
|-------|-------------|
| `mode` | Runtime mode: "strict" or "compat" |
| `wall_time` | Execution time in seconds |
| `peak_mem` | Peak memory usage in GB |
| `sdpa` | SDPA status: "used", "missing", "off" |
| `flash` | Flash Attention status |
| `amp` | AMP status |
| `cdc` | Compile Dynamic Cache status |
| `degraded` | Boolean: any acceleration missing |
| `reasons` | List of degradation reasons |

## CI/CD Integration

### GitHub Actions Workflows

Two separate workflows validate each mode:

#### STRICT Mode Validation
- **File**: `.github/workflows/strict-mode-validation.yml`
- **Purpose**: Ensure optimal acceleration on golden shards
- **Criteria**: `degraded=false` or pipeline fails
- **Timeout**: 30 minutes maximum

#### COMPAT Mode Validation  
- **File**: `.github/workflows/compat-mode-validation.yml`
- **Purpose**: Test graceful degradation scenarios
- **Criteria**: Runs complete with reasonable performance bounds
- **Timeout**: 45 minutes maximum

### Pytest Integration

```bash
# Run STRICT mode tests
pytest -m strict_mode --timeout=300

# Run COMPAT mode tests  
pytest -m compat_mode --timeout=300

# Run slow tests with extended timeout
pytest -m slow --timeout=300
```

### Test Markers

```python
import pytest

@pytest.mark.strict_mode
def test_strict_mode_requirements():
    """Test requiring optimal acceleration"""
    pass

@pytest.mark.compat_mode
def test_compat_mode_fallback():
    """Test graceful degradation handling"""
    pass

@pytest.mark.timeout(300)
@pytest.mark.slow
def test_performance_bounds():
    """Test with extended timeout"""
    pass
```

## Performance Expectations

### STRICT Mode Performance Bounds
- **Wall Time**: ≤ 5 minutes for 20+ session shards
- **Memory Usage**: ≤ 8GB peak
- **Parity Difference**: ≤ 1e-4 between runs

### COMPAT Mode Performance Bounds  
- **Wall Time**: ≤ 15 minutes for 10+ session shards (degraded)
- **Memory Usage**: ≤ 12GB peak (higher tolerance)
- **Parity Difference**: Variable (degradation acceptable)

## Troubleshooting

### Common Issues

#### 1. SDPA Not Available
```
RuntimeError: SDPA (Scaled Dot Product Attention) is required but unavailable
```

**Solutions**:
- Upgrade to PyTorch >= 2.0
- Set `IRONFORGE_RUNTIME_MODE=compat`
- Disable SDPA: `IRONFORGE_ENABLE_SDPA=false`

#### 2. CUDA Not Available
```
COMPAT mode: Running with degraded performance
  - Flash Attention backend unavailable
  - Automatic Mixed Precision disabled
```

**Solutions**:
- Install CUDA toolkit and drivers
- Use CPU-only mode: `IRONFORGE_ENABLE_FLASH=false IRONFORGE_ENABLE_AMP=false`
- Accept degraded performance in COMPAT mode

#### 3. Memory Limit Exceeded
```
Peak memory 10.5GB exceeds STRICT limit 8.0GB
```

**Solutions**:
- Increase memory limit: `IRONFORGE_MEMORY_LIMIT_GB=16.0`
- Reduce batch size in loader configuration
- Use gradient checkpointing: enable in TGAT config

#### 4. Timeout Issues
```
Test session timed out after 300 seconds
```

**Solutions**:
- Increase timeout: `IRONFORGE_TIMEOUT=600`
- Use COMPAT mode for slower fallbacks
- Profile performance bottlenecks in audit ledger

### Debug Mode

```bash
# Enable verbose logging
export IRONFORGE_LOG_LEVEL=DEBUG

# Enable performance profiling
export IRONFORGE_PROFILE_PERFORMANCE=true

# Save intermediate results
export IRONFORGE_SAVE_INTERMEDIATE=true
```

## API Reference

### Core Classes

```python
from ironforge.sdk.runtime_config import (
    RuntimeMode, AccelStatus, AccelerationState,
    RuntimeConfig, AccelerationDetector, RuntimeModeEnforcer
)

# Runtime modes
RuntimeMode.STRICT  # "strict"
RuntimeMode.COMPAT  # "compat"

# Acceleration status
AccelStatus.USED     # "used" 
AccelStatus.MISSING  # "missing"
AccelStatus.OFF      # "off"
```

### Configuration Loading

```python
# From environment variables
config = RuntimeConfig.from_env()

# From YAML file
config = RuntimeConfig.from_yaml("config.yml")

# Manual configuration
config = RuntimeConfig(
    mode=RuntimeMode.STRICT,
    enable_sdpa=True,
    enable_flash_attention=True,
    enable_amp=True,
    enable_cdc=True
)
```

### Acceleration Detection

```python
from ironforge.sdk.runtime_config import AccelerationDetector

# Detect all acceleration components
accel_state = AccelerationDetector.detect_all()

# Check individual components
sdpa_status = AccelerationDetector.detect_sdpa()
flash_status = AccelerationDetector.detect_flash_attention()
amp_status = AccelerationDetector.detect_amp()
cdc_status = AccelerationDetector.detect_cdc()

# Check degradation
if accel_state.is_degraded():
    reasons = accel_state.get_degradation_reasons()
    for reason in reasons:
        print(f"Degradation: {reason}")
```

### Runtime Enforcement

```python
from ironforge.sdk.runtime_config import RuntimeModeEnforcer

enforcer = RuntimeModeEnforcer(config)

# Enforce requirements (may raise RuntimeError in STRICT mode)
enforcer.enforce_acceleration_requirements(accel_state)

# Create audit entry
audit_entry = enforcer.create_audit_entry(accel_state)

# Save to ledger
enforcer.save_audit_entry(audit_entry)
```

## Migration Guide

### From Legacy TGAT

```python
# Before (legacy)
from ironforge.learning.tgat_discovery import create_attention_layer

layer = create_attention_layer(input_dim=45, hidden_dim=44, num_heads=4)

# After (runtime-optimized)
from ironforge.learning.runtime_optimized_tgat import create_runtime_optimized_tgat
from ironforge.learning.dual_graph_config import TGATConfig

base_config = TGATConfig(input_dim=45, hidden_dim=44, num_heads=4)
tgat_engine = create_runtime_optimized_tgat(base_config)
```

### Environment Migration

```bash
# Legacy environment (no runtime control)
export CUDA_VISIBLE_DEVICES=0

# Runtime-optimized environment  
export IRONFORGE_RUNTIME_MODE=strict
export IRONFORGE_ENABLE_SDPA=true
export IRONFORGE_ENABLE_FLASH=true
export IRONFORGE_ENABLE_AMP=true
export IRONFORGE_ENABLE_AUDIT=true
```

## Best Practices

### Production Deployments
1. Use **STRICT mode** for consistent performance
2. Validate acceleration requirements in CI/CD
3. Monitor audit ledger for degradation detection
4. Set appropriate timeout and memory limits

### Development Environments
1. Use **COMPAT mode** for flexibility
2. Enable audit logging for performance awareness
3. Test both modes during development
4. Profile performance bottlenecks with debug mode

### Testing Strategy
1. Test STRICT mode with optimal hardware configurations
2. Test COMPAT mode with degraded scenarios
3. Use pytest markers for mode-specific tests
4. Validate performance bounds in CI/CD

### Performance Optimization
1. Enable all acceleration when hardware supports it
2. Use appropriate batch sizes and memory limits
3. Profile wall time and memory usage in audit ledger
4. Consider gradient checkpointing for large models

## Summary

The STRICT vs COMPAT system provides:

- ✅ **Explicit Control**: Choose fail-fast or graceful degradation
- ✅ **Comprehensive Auditing**: Track acceleration and performance
- ✅ **CI/CD Integration**: Automated validation across scenarios  
- ✅ **Hardware Awareness**: Detect and utilize available acceleration
- ✅ **Backward Compatibility**: Graceful fallback for diverse environments

Choose **STRICT mode** for production deployments requiring consistent performance, and **COMPAT mode** for development and environments with variable hardware capabilities.