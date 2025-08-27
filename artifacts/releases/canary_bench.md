# Canary Benchmark Report

**Timestamp:** 20250826_165038  
**Runtime Mode:** STRICT  
**Sessions Processed:** 50

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Duration | 0.00s |
| Average Session Duration | 0.000s |
| Sessions per Second | 162822.36 |

## Configuration

```yaml
presets:
  tgat:
    sdpa: true
    mask: true
    bucket: true
    amp: auto
  dag:
    k: 4
    dt_range: [1, 120]
  parquet:
    compression: zstd
    level: 3
    row_group: 10000
    cdc: false
```

## Context7 Optimizations Applied

- ✅ PyTorch SDPA with masking and bucket optimization
- ✅ Memory-efficient attention kernels
- ✅ FP16/BF16 reduction math enabled
- ✅ PyArrow ZSTD compression (level 3)
- ✅ Optimized row group size (10K)

## Benchmark Status
**Overall:** ✅ PASSED
