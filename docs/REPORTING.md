# IRONFORGE Wave 5 — Minimal Reporting

Timeline heatmaps and confluence strips for session visualization.

## Overview

Wave 5 provides minimal reporting artifacts for IRONFORGE discovery and validation results:

- **Timeline Heatmaps**: Event density visualization per minute across session timelines
- **Confluence Strips**: 0–100 gradient strips with optional event markers  
- **HTML Reports**: Single-page reports combining all visualizations
- **CLI Integration**: Simple `ironforge report` command for production workflows

## Quick Start

### Prerequisites

Wave 5 reporting requires:
- `numpy` — Array processing and numerical computation
- `Pillow` — PNG image generation and manipulation

Install these dependencies:
```bash
pip install numpy Pillow
```

### Basic Usage

```bash
# Generate PNG reports from session data
ironforge report --input-json session_data.json --out-dir reports/

# Include HTML summary page
ironforge report --input-json session_data.json --out-dir reports/ --html

# Custom dimensions and specific sessions
ironforge report \
  --input-json data.json \
  --out-dir custom_reports/ \
  --sessions "2025-08-12_NY-AM" "2025-08-12_LA-PM" \
  --width 512 \
  --height 80 \
  --html
```

## Input Data Format

The `--input-json` file should contain session data in this format:

```json
{
  "2025-08-12_NY-AM": {
    "minute_bins": [0, 1, 2, 3, 5, 10, 15, 20, 25, 30],
    "densities": [0.0, 2.0, 0.5, 1.8, 3.2, 1.1, 0.8, 2.5, 0.3, 1.0],
    "confluence": [55.0, 57.0, 62.0, 58.0, 65.0, 70.0, 75.0, 68.0, 60.0, 50.0],
    "markers": [15, 25]
  },
  "2025-08-12_LA-PM": {
    "minute_bins": [0, 2, 4, 6, 8, 12, 18, 24, 28, 35],
    "densities": [1.5, 0.8, 2.2, 1.7, 0.9, 2.8, 1.3, 3.1, 0.6, 2.0],
    "confluence": [40.0, 45.0, 50.0, 60.0, 55.0, 80.0, 85.0, 90.0, 75.0, 65.0],
    "markers": [12, 28]
  }
}
```

### Field Descriptions

- **`minute_bins`** *(required)*: Array of minute offsets from session start
- **`densities`** *(required)*: Event density values for each minute bin
- **`confluence`** *(optional)*: Confluence scores (0-100) for each minute bin
- **`markers`** *(optional)*: Array of minute offsets where significant events occurred

Missing optional fields will be handled gracefully:
- Missing `confluence` → filled with zeros
- Missing `markers` → no markers rendered

## CLI Command Reference

### `ironforge report`

Generate minimal reporting artifacts from session data.

```bash
ironforge report --input-json INPUT.json [OPTIONS]
```

#### Required Arguments

- `--input-json PATH` — JSON file containing session data

#### Optional Arguments

- `--out-dir PATH` — Output directory (default: `reports/minimal`)
- `--sessions ID [ID ...]` — Specific session IDs to render (default: all)
- `--width PIXELS` — Image width in pixels (default: 1024)
- `--height PIXELS` — Timeline heatmap height (default: 160)  
- `--strip-height PIXELS` — Confluence strip height (default: 54)
- `--html` — Generate single HTML page with all images embedded

#### Examples

**Basic report generation:**
```bash
ironforge report --input-json validation_results.json
```

**Custom output directory:**
```bash
ironforge report \
  --input-json discovery_output.json \
  --out-dir /var/reports/session_analysis/
```

**Specific sessions with HTML:**
```bash
ironforge report \
  --input-json large_dataset.json \
  --sessions "morning_session" "evening_session" \
  --html
```

**Custom dimensions for presentations:**
```bash
ironforge report \
  --input-json presentation_data.json \
  --width 1920 \
  --height 240 \
  --strip-height 80 \
  --html
```

## Output Files

For each session, Wave 5 generates:

### PNG Files
- `{session_id}_timeline.png` — Timeline heatmap showing event density
- `{session_id}_confluence.png` — Confluence strip with gradient and markers

### HTML Report (Optional)
- `index.html` — Single-page report with all images embedded as data URIs

### Example Output Structure
```
reports/minimal/
├── 2025-08-12_NY-AM_timeline.png
├── 2025-08-12_NY-AM_confluence.png
├── 2025-08-12_LA-PM_timeline.png  
├── 2025-08-12_LA-PM_confluence.png
└── index.html                     # (if --html specified)
```

## Programmatic API

Wave 5 components can also be used programmatically:

### Timeline Heatmaps

```python
import numpy as np
from ironforge.reporting.heatmap import build_session_heatmap, TimelineHeatmapSpec

# Create session data
minute_bins = np.array([0, 5, 10, 15, 20, 30])
densities = np.array([0.5, 2.1, 1.8, 3.2, 0.9, 1.5])

# Generate heatmap
spec = TimelineHeatmapSpec(width=1024, height=160, pad=8)
heatmap = build_session_heatmap(minute_bins, densities, spec)

# Save to file
from ironforge.reporting.writer import write_png
write_png(Path("session_timeline.png"), heatmap)
```

### Confluence Strips

```python
from ironforge.reporting.confluence import build_confluence_strip, ConfluenceStripSpec

# Create confluence data
minute_bins = np.array([0, 5, 10, 15, 20, 30])
scores_0_100 = np.array([45.0, 60.0, 75.0, 80.0, 65.0, 50.0])
marker_minutes = np.array([10, 20])  # Event markers

# Generate confluence strip
spec = ConfluenceStripSpec(width=1024, height=54, marker_radius=3)
strip = build_confluence_strip(minute_bins, scores_0_100, marker_minutes, spec)

# Save to file
write_png(Path("confluence_strip.png"), strip)
```

### HTML Reports

```python
from ironforge.reporting.html import build_report_html
from ironforge.reporting.writer import write_html

# Combine images into HTML report
images = [
    ("Session Timeline", heatmap),
    ("Confluence Analysis", strip)
]

html = build_report_html("IRONFORGE Session Analysis", images)
write_html(Path("report.html"), html)
```

## Configuration Specifications

### TimelineHeatmapSpec

```python
@dataclass(frozen=True)
class TimelineHeatmapSpec:
    width: int = 1024        # Image width in pixels
    height: int = 160        # Image height in pixels  
    pad: int = 8            # Padding around content
    colormap: str = "viridis"  # Colormap (placeholder for future)
```

### ConfluenceStripSpec

```python
@dataclass(frozen=True)
class ConfluenceStripSpec:
    width: int = 1024        # Image width in pixels
    height: int = 54         # Image height in pixels
    pad: int = 6            # Padding around content
    marker_radius: int = 3   # Radius of event markers
```

## Performance Characteristics

Wave 5 is designed for production use with performance budgets:

### Timing Budgets
- **Single heatmap**: <1 second generation
- **Single confluence strip**: <1 second generation  
- **5 sessions × 240 minutes**: <2 seconds total
- **File I/O**: PNG <1s, HTML <0.5s per operation

### Memory Budgets
- **Single image generation**: <50MB peak memory
- **Multi-session batch (5×240min)**: <150MB peak memory
- **Memory cleanup**: Proper garbage collection, no leaks

### Scalability Testing

Performance tests validate:
```python
# Large timeline (4 hours, every minute)
minute_bins = np.arange(0, 240, 1)  # 240 data points
densities = np.random.random(240) * 5.0

# Multi-session batch processing  
sessions = 5
timeline_length = 240  # 4 hours per session
total_points = sessions * timeline_length  # 1,200 data points
```

## Integration with Other Waves

### Wave 3 Integration (Discovery Pipeline)

Discovery results can be converted to Wave 5 format:

```python
# Convert discovery results to reporting format
def discovery_to_reporting_format(discovery_results):
    reporting_data = {}
    
    for session_id, results in discovery_results.items():
        # Extract timeline data from discovery results
        timeline = results.get('temporal_timeline', {})
        
        reporting_data[session_id] = {
            'minute_bins': timeline.get('minute_bins', []),
            'densities': timeline.get('event_density', []),
            'confluence': timeline.get('pattern_confidence', []),
            'markers': timeline.get('significant_events', [])
        }
    
    return reporting_data
```

### Wave 4 Integration (Validation Rails)

Validation results can include reporting data:

```python
# Add reporting data to validation output
validation_results['reporting'] = {
    'session_timelines': reporting_data,
    'summary_statistics': {
        'total_sessions': len(reporting_data),
        'avg_session_length': np.mean([len(s['minute_bins']) for s in reporting_data.values()]),
        'peak_density': max([max(s['densities']) for s in reporting_data.values()])
    }
}
```

## Troubleshooting

### Common Issues

**Import Error: "Wave 5 reporting requires numpy and Pillow"**
```bash
# Install missing dependencies
pip install numpy Pillow
```

**File Not Found: "Input JSON not found"**
```bash
# Verify file path and permissions
ls -la your_data_file.json
ironforge report --input-json $(pwd)/your_data_file.json
```

**Empty Output: No images generated**
- Check JSON format matches expected structure
- Verify `minute_bins` and `densities` arrays have same length
- Ensure arrays contain numeric data

**Performance Issues: Generation too slow**
- Reduce image dimensions: `--width 512 --height 80`
- Process fewer sessions: `--sessions session1 session2`
- Check available memory: Monitor peak usage <150MB

### Debug Mode

Enable detailed logging:
```bash
# Add verbose output (feature for future enhancement)
ironforge report --input-json data.json --verbose
```

### Validation

Validate input JSON before processing:
```python
import json
import numpy as np

def validate_reporting_data(data):
    """Validate reporting data format."""
    for session_id, session_data in data.items():
        # Required fields
        assert 'minute_bins' in session_data
        assert 'densities' in session_data
        
        # Array consistency  
        bins = np.array(session_data['minute_bins'])
        densities = np.array(session_data['densities'])
        assert bins.shape == densities.shape
        
        # Optional field validation
        if 'confluence' in session_data:
            conf = np.array(session_data['confluence'])
            assert conf.shape == bins.shape
            
        print(f"✓ {session_id}: {len(bins)} data points")

# Usage
with open('data.json') as f:
    data = json.load(f)
    validate_reporting_data(data)
```

## Future Enhancements

Wave 5 provides a minimal but complete reporting foundation. Future enhancements may include:

- **Advanced Colormaps**: Viridis, plasma, custom gradients
- **Interactive Elements**: Hover tooltips, zoom controls
- **Export Formats**: SVG, PDF, high-DPI variants
- **Dashboard Integration**: REST API, real-time updates
- **Annotation Support**: Text overlays, region highlighting
- **Performance Scaling**: GPU acceleration, parallel processing

## API Reference

See individual module documentation:
- `ironforge.reporting.heatmap` — Timeline heatmap generation
- `ironforge.reporting.confluence` — Confluence strip rendering  
- `ironforge.reporting.html` — HTML report assembly
- `ironforge.reporting.writer` — File I/O utilities

---

**Wave 5 Status**: ✅ **COMPLETE**  
**Performance**: <2s execution, <150MB memory  
**Dependencies**: numpy, Pillow  
**CLI Command**: `ironforge report`
