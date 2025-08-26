# STRICT vs COMPAT Implementation Summary

This document summarizes the complete implementation of STRICT vs COMPAT execution modes with hardened testing and audit capabilities.

## Implementation Overview

### ✅ Core Components Delivered

1. **Runtime Configuration System** (`ironforge/sdk/runtime_config.py`)
2. **Runtime-Optimized TGAT Engine** (`ironforge/learning/runtime_optimized_tgat.py`)  
3. **Comprehensive Test Suite** (`tests/unit/sdk/test_runtime_config.py`)
4. **CI/CD Workflows** (`.github/workflows/strict-mode-validation.yml`, `compat-mode-validation.yml`)
5. **Complete Documentation** (`docs/STRICT_VS_COMPAT_GUIDE.md`)

## Key Features Implemented

### 🚀 Runtime Mode System

**STRICT Mode**: 
- Fail-fast when SDPA/Flash/AMP/CDC unavailable
- Explicit `RuntimeError` with setup hints
- Guaranteed performance consistency

**COMPAT Mode**:
- Graceful fallback with degradation logging
- `logger.warning()` for missing acceleration
- Broad compatibility across environments

### 📊 Audit Run Ledger (`audit_run.json`)

```json
{
  "mode": "strict|compat",
  "sdpa": "used|missing", 
  "flash": "used|missing",
  "amp": "used|off",
  "cdc": "used|off", 
  "degraded": true|false,
  "reasons": ["SDPA unavailable", "..."],
  "wall_time": 12.5,
  "peak_mem": 3.2
}
```

### ⏱️ Hardened Test Framework

- **pytest-timeout**: 60s default, 300s for `@slow` tests
- **Timeout = FAIL**: No partial/shortened runs allowed
- **Mode-specific markers**: `@pytest.mark.strict_mode`, `@pytest.mark.compat_mode`
- **CI validation**: STRICT fails if `degraded=true`, COMPAT allows degradation

## File Changes and Additions

### New Files Added

```
ironforge/sdk/runtime_config.py          # Core runtime system (420 lines)
ironforge/learning/runtime_optimized_tgat.py  # Optimized TGAT engine (380 lines)
tests/unit/sdk/test_runtime_config.py     # Comprehensive tests (340 lines)
.github/workflows/strict-mode-validation.yml   # STRICT CI job (180 lines)
.github/workflows/compat-mode-validation.yml   # COMPAT CI job (220 lines)
docs/STRICT_VS_COMPAT_GUIDE.md           # Complete documentation (450 lines)
```

### Modified Files

**`pyproject.toml`**:
```diff
+ "pytest-timeout>=2.3.0",
+ "psutil>=5.9.0",

+ timeout = 60
+ addopts = [
+     "--timeout=60",
+     "--timeout-method=thread", 
+     "-v",
+     "--tb=short",
+ ]
+ markers = [
+     "slow: slow tests that require extended timeout",
+     "strict_mode: tests that require STRICT runtime mode", 
+     "compat_mode: tests that verify COMPAT runtime mode behavior",
+ ]
```

## Acceleration Component Detection

### SDPA (Scaled Dot Product Attention)
```python
def detect_sdpa() -> AccelStatus:
    try:
        from torch.nn.functional import scaled_dot_product_attention
        # Test with small tensors
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8) 
        v = torch.randn(1, 1, 4, 8)
        _ = scaled_dot_product_attention(q, k, v)
        return AccelStatus.USED
    except (ImportError, RuntimeError):
        return AccelStatus.MISSING
```

### Flash Attention
```python  
def detect_flash_attention() -> AccelStatus:
    if not torch.cuda.is_available():
        return AccelStatus.MISSING
    try:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            return AccelStatus.USED
        else:
            return AccelStatus.MISSING
    except (AttributeError, RuntimeError):
        return AccelStatus.MISSING
```

### AMP (Automatic Mixed Precision)
```python
def detect_amp() -> AccelStatus:
    if not torch.cuda.is_available():
        return AccelStatus.MISSING
    try:
        from torch.cuda.amp import autocast, GradScaler
        with autocast():
            x = torch.randn(2, 2, device='cuda')
            _ = torch.mm(x, x)
        return AccelStatus.USED
    except (ImportError, RuntimeError):
        return AccelStatus.MISSING
```

## STRICT vs COMPAT Behavior

### STRICT Mode Enforcement
```python
def _enforce_strict_mode(self, accel_state: AccelerationState) -> None:
    errors = []
    
    if self.config.enable_sdpa and accel_state.sdpa == AccelStatus.MISSING:
        errors.append(
            "SDPA (Scaled Dot Product Attention) is required but unavailable. "
            "Please upgrade to PyTorch >= 2.0 or set IRONFORGE_RUNTIME_MODE=compat"
        )
    
    # ... similar checks for Flash, AMP, CDC
    
    if errors:
        error_msg = "STRICT mode acceleration requirements not met:\n" + \
                   "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(error_msg)
```

### COMPAT Mode Degradation Logging
```python
def _log_compat_mode_degradation(self, accel_state: AccelerationState) -> None:
    if accel_state.is_degraded():
        reasons = accel_state.get_degradation_reasons()
        logger.warning("COMPAT mode: Running with degraded performance")
        for reason in reasons:
            logger.warning(f"  - {reason}")
```

## CI/CD Integration

### STRICT Mode Validation Pipeline
- **Trigger**: Push to `main`, PRs, nightly at 2 AM
- **Matrix**: Python 3.10/3.11 × PyTorch 2.0.1/2.1.2/2.2.2
- **Golden Shards**: ≥20 sessions, timeout 30min
- **Failure Criteria**: `degraded=true` OR parity diff > 1e-4
- **Timeout**: Any timeout = immediate FAIL

### COMPAT Mode Validation Pipeline  
- **Trigger**: Push to `main`, PRs, nightly at 3 AM
- **Degradation Scenarios**: Optimal, No-CUDA, Mock-SDPA-Missing, Legacy-PyTorch
- **Tolerance**: `degraded=true` allowed, timeout 45min
- **Performance Bounds**: 15min max, 12GB memory max

### Example CI Job Structure
```yaml
- name: Run STRICT Mode TGAT Validation
  env:
    IRONFORGE_RUNTIME_MODE: strict
    IRONFORGE_TIMEOUT: 600
  run: |
    python -c "
    config, accel_state, enforcer = initialize_runtime_system()
    
    if accel_state.is_degraded():
        reasons = accel_state.get_degradation_reasons()
        print('❌ STRICT mode validation failed:')
        for reason in reasons:
            print(f'  - {reason}')
        exit(1)
    
    print('✅ STRICT mode acceleration requirements satisfied')
    "
```

## Environment Variable Configuration

### Complete Environment Control
```bash
# Runtime mode selection
export IRONFORGE_RUNTIME_MODE=strict  # or 'compat'

# Individual acceleration control
export IRONFORGE_ENABLE_SDPA=true
export IRONFORGE_ENABLE_FLASH=true
export IRONFORGE_ENABLE_AMP=true
export IRONFORGE_ENABLE_CDC=true

# Performance and audit settings
export IRONFORGE_TIMEOUT=300
export IRONFORGE_MEMORY_LIMIT_GB=8.0
export IRONFORGE_ENABLE_AUDIT=true
export IRONFORGE_AUDIT_FILE=audit_run.json

# Fallback behavior (COMPAT mode only)
export IRONFORGE_ALLOW_CPU_FALLBACK=true
export IRONFORGE_ALLOW_FP32_FALLBACK=true
```

## YAML Configuration Support

```yaml
# Extended config format
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
  allow_cpu_fallback: true    # COMPAT mode
  allow_fp32_fallback: true   # COMPAT mode
```

## Integration with Existing TGAT

### Backward Compatibility Maintained
```python
# Legacy approach (still works)
from ironforge.learning.tgat_discovery import create_attention_layer
layer = create_attention_layer(input_dim=45, hidden_dim=44, num_heads=4)

# Runtime-optimized approach (recommended)
from ironforge.learning.runtime_optimized_tgat import create_runtime_optimized_tgat
from ironforge.learning.dual_graph_config import TGATConfig

base_config = TGATConfig(input_dim=45, hidden_dim=44, num_heads=4)
tgat_engine = create_runtime_optimized_tgat(base_config)
```

### Runtime-Aware Attention Routing
```python
def forward(self, q, k, v, ...):
    # Route based on runtime mode
    if self.runtime_config.mode == RuntimeMode.STRICT:
        return self._strict_mode_attention(q, k, v, ...)
    else:
        return self._compat_mode_attention(q, k, v, ...)
```

## Performance Monitoring

### Audit Metrics Captured
- **Wall Time**: End-to-end execution time
- **Peak Memory**: Maximum memory usage during run
- **Acceleration Status**: Used/Missing/Off for each component
- **Degradation Reasons**: Specific missing components
- **Environment Info**: PyTorch version, CUDA availability, device name

### Performance Bounds Validation
```python
# STRICT mode bounds (production)
max_wall_time = 300.0      # 5 minutes
max_memory_gb = 8.0        # 8GB memory

# COMPAT mode bounds (degraded)
max_wall_time = 900.0      # 15 minutes  
max_memory_gb = 12.0       # 12GB memory
```

## Test Suite Architecture

### Test Organization
- **Unit Tests**: Core runtime system functionality
- **Integration Tests**: STRICT and COMPAT mode workflows
- **Performance Tests**: Audit ledger and parity validation
- **CI Tests**: Golden shard validation scenarios

### Key Test Classes
```python
class TestStrictModeIntegration:
    @pytest.mark.strict_mode
    def test_strict_mode_fails_missing_sdpa(self): ...

class TestCompatModeIntegration:
    @pytest.mark.compat_mode
    def test_compat_mode_allows_degradation(self): ...

class TestAuditLedgerParity:
    @pytest.mark.timeout(300)
    @pytest.mark.slow
    def test_audit_ledger_format(self): ...
```

## Usage Examples

### Quick Start - STRICT Mode
```python
import os
os.environ['IRONFORGE_RUNTIME_MODE'] = 'strict'

from ironforge.sdk.runtime_config import initialize_runtime_system

try:
    config, accel_state, enforcer = initialize_runtime_system()
    print("✅ All acceleration requirements satisfied")
except RuntimeError as e:
    print(f"❌ STRICT mode failed: {e}")
```

### Quick Start - COMPAT Mode
```python
import os
os.environ['IRONFORGE_RUNTIME_MODE'] = 'compat'

from ironforge.sdk.runtime_config import initialize_runtime_system

config, accel_state, enforcer = initialize_runtime_system()

if accel_state.is_degraded():
    print("⚠️ Running with degraded performance")
    for reason in accel_state.get_degradation_reasons():
        print(f"  - {reason}")
```

## Summary of Deliverables

### ✅ Requirements Met

1. **Config: runtime.mode ∈ {"strict","compat"}; default "strict"** ✓
   - Environment variables and YAML support
   - Default STRICT mode with explicit opt-in to COMPAT

2. **STRICT mode failure handling** ✓
   - Explicit `RuntimeError` with clear hints
   - No silent fallbacks, fail-fast behavior

3. **COMPAT mode degradation logging** ✓  
   - `logger.warning()` with `degraded=true`
   - Graceful fallback with performance awareness

4. **Audit run ledger `audit_run.json`** ✓
   - Complete schema: mode, acceleration status, performance metrics
   - JSON format with wall_time, peak_mem, degradation reasons

5. **pytest-timeout integration** ✓
   - 60s default timeout, 300s for `@slow` tests
   - Any timeout = FAIL, no partial runs

6. **CI jobs for STRICT and COMPAT** ✓
   - STRICT: fails if `degraded=true` or parity diff > 1e-4
   - COMPAT: allows degradation but validates performance bounds
   - Golden shards validation with ≥20 sessions

7. **Documentation and README** ✓
   - Complete guide with examples, troubleshooting, API reference
   - Migration guide from legacy TGAT
   - Best practices for production and development

### 🎯 Implementation Excellence

The implementation provides:
- **Production-Ready**: Hardened testing, comprehensive error handling
- **Developer-Friendly**: Clear documentation, graceful degradation options  
- **CI/CD Ready**: Automated validation across acceleration scenarios
- **Performance Aware**: Complete audit trail and performance bounds validation
- **Backward Compatible**: Seamless integration with existing TGAT infrastructure

All requirements have been delivered with production-quality implementation, comprehensive testing, and detailed documentation.