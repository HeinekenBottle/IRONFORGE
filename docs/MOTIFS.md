# IRONFORGE Motifs v0 - Wave 6 Documentation

**Canonical event sequence detection with confluence filtering for temporal pattern discovery**

## Overview

Wave 6 introduces **Motifs v0** and **Confluence v0** to IRONFORGE, providing systematic pattern matching and quality scoring for temporal market events. This system identifies specific event sequences (motifs) and scores their quality using a weighted confluence formula.

### Key Components

1. **Confluence v0**: Weighted scoring system (0-100) for pattern quality assessment
2. **Motifs v0**: Canonical event sequence cards with temporal and structural constraints
3. **CLI Integration**: Complete discovery → validation → motifs workflow
4. **Input Adapter**: Flexible JSON conversion for existing discovery outputs

## Confluence v0 - Quality Scoring System

### Score Formula

```
Confluence Score = Σ(weight_i × component_i) × 100
```

**Default Weights:**
- `cluster`: 0.35 (35%) - Event clustering intensity
- `htf_prox`: 0.25 (25%) - Higher timeframe proximity
- `structure`: 0.20 (20%) - Market structure strength  
- `cycle`: 0.10 (10%) - Cycle alignment
- `precursor`: 0.10 (10%) - Precursor presence

**Input Range**: All components expected in [0, 1]  
**Output Range**: [0, 100] (0 = no confluence, 100 = maximum confluence)

### Usage Examples

#### Basic Scoring
```python
from ironforge.metrics.confluence import compute_confluence_score

# Single score calculation
score = compute_confluence_score(
    cluster=0.8,     # Strong clustering
    htf_prox=0.6,    # Good HTF proximity
    structure=0.4,   # Moderate structure
    cycle=0.7,       # Good cycle alignment
    precursor=0.3    # Weak precursor
)
print(f"Confluence Score: {score:.1f}")  # Output: 61.0
```

#### Vectorized Scoring
```python
import numpy as np

# Batch scoring for multiple time windows
scores = compute_confluence_score(
    cluster=[0.5, 0.8, 0.9],
    htf_prox=[0.3, 0.9, 0.7],
    structure=[0.4, 0.6, 0.8],
    cycle=[0.2, 0.7, 0.6],
    precursor=[0.1, 0.5, 0.4]
)
print(f"Scores: {scores}")  # Array of 3 scores
```

#### Component Breakdown
```python
from ironforge.metrics.confluence import compute_confluence_components

inputs = {
    "cluster": 0.8,
    "htf_prox": 0.6,
    "structure": 0.4,
    "cycle": 0.7,
    "precursor": 0.3
}

components = compute_confluence_components(inputs)
print(f"Cluster contribution: {components['cluster'][0]:.1f}")  # 28.0
print(f"Total score: {components['total'][0]:.1f}")           # 61.0
```

#### Custom Weights
```python
from ironforge.metrics.confluence import ConfluenceWeights

# Emphasize clustering and structure
custom_weights = ConfluenceWeights(
    cluster=0.5, htf_prox=0.2, structure=0.3, cycle=0.0, precursor=0.0
)

score = compute_confluence_score(
    cluster=0.8, htf_prox=0.6, structure=0.4, cycle=0.0, precursor=0.0,
    weights=custom_weights
)
```

## Motifs v0 - Event Sequence Detection

### Motif Cards

Motifs are defined as **cards** containing:
- **Event sequence**: Ordered list of event types with timing constraints
- **Time windows**: Overall duration limits for the complete sequence
- **Structural guardrails**: Optional market structure requirements (e.g., HTF under midpoint)
- **Confluence thresholds**: Minimum/maximum quality scores for matches

### Default Motif Cards

#### Card C1: Sweep → FVG Redelivery
```python
MotifCard(
    id="c1",
    name="Sweep → FVG redelivery under HTF midpoint",
    steps=[
        MotifStep("sweep", (0, 0)),                              # Immediate start
        MotifStep("fvg_redelivery", (12, 30), htf_under_mid=True) # 12-30min later, under HTF mid
    ],
    window_minutes=(12, 30),     # Total sequence: 12-30 minutes
    min_confluence=65.0,         # Minimum quality threshold
    description="FPFVG redelivery 12–30m after sweep, under HTF midline."
)
```

#### Card C2: NY-AM Expansion Sequence
```python
MotifCard(
    id="c2", 
    name="Expansion → Consolidation → Redelivery (NY-AM)",
    steps=[
        MotifStep("expansion", (0, 0)),        # Session start
        MotifStep("consolidation", (5, 40)),   # 5-40min after expansion
        MotifStep("redelivery", (10, 40)),     # 10-40min after consolidation
    ],
    window_minutes=(15, 80),     # Total sequence: 15-80 minutes
    min_confluence=70.0,         # Higher threshold for complex sequence
    description="AM expansion sequence with mid-session redelivery."
)
```

#### Card C3: First Presentation FVG
```python
MotifCard(
    id="c3",
    name="First-presentation FVG after Open (Wk4 bias)", 
    steps=[
        MotifStep("fpfvg", (0, 0)),  # Single event
    ],
    window_minutes=(10, 25),     # Must occur 10-25min after reference
    min_confluence=65.0,
    description="First presentation FVG soon after open; weekly cycle bias optional."
)
```

### Event Input Format

Events must include:
- `type`: Event type string (e.g., "sweep", "fvg_redelivery", "expansion")
- `minute`: Integer minute offset from session start
- `htf_under_mid`: Boolean indicating if HTF is under midpoint (for guardrails)

```python
events = [
    {"type": "sweep", "minute": 5, "htf_under_mid": False},
    {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": True},
    {"type": "expansion", "minute": 45, "htf_under_mid": False}
]
```

### Scanning for Matches

```python
from ironforge.motifs.scanner import scan_session_for_cards
import numpy as np

# Confluence data (0-100 scores per minute)
confluence = np.array([70.0, 75.0, 80.0, 85.0, 90.0] + [75.0] * 50)

matches = scan_session_for_cards(
    session_id="NY_AM_2025_08_05",
    events=events,
    confluence=confluence,
    min_confluence=65.0
)

for match in matches:
    print(f"Card {match.card_id}: Score {match.score:.1f}, "
          f"Window {match.window}, Steps {match.steps_at}")
```

### Custom Motif Cards

```python
from ironforge.motifs.cards import MotifCard, MotifStep

custom_card = MotifCard(
    id="custom1",
    name="London Sweep Pattern",
    steps=[
        MotifStep("sweep", (0, 0)),
        MotifStep("consolidation", (15, 45)), 
        MotifStep("expansion", (20, 60))
    ],
    window_minutes=(35, 105),
    min_confluence=75.0,
    max_confluence=95.0,  # Also set maximum threshold
    description="London session sweep with delayed expansion."
)

# Use custom cards
matches = scan_session_for_cards(
    session_id="test",
    events=events,
    confluence=confluence,
    cards=[custom_card]  # Override default cards
)
```

## CLI Commands

### Discovery with Confluence

Generate discoveries with confluence scores attached:

```bash
# Standard discovery
ironforge discover-temporal \
  --data-path data/parquet_shards/ \
  --output-dir discoveries/ \
  --with-confluence

# Output includes confluence scores per session/minute
```

### Prepare Motifs Input

Convert discovery/validation outputs to motifs scanner format:

```bash
# Basic conversion (discovery only)
ironforge prepare-motifs-input \
  --discovery-json discoveries/temporal_discoveries_20250817.json \
  --out motifs_input.json

# With validation confluence data
ironforge prepare-motifs-input \
  --discovery-json discoveries/temporal_discoveries_20250817.json \
  --validation-json validation/validation_results_20250817.json \
  --out motifs_input.json
```

#### Input Format Flexibility

The adapter handles multiple discovery formats:

**Format A: Events List**
```json
{
  "session_001": {
    "events": [
      {"type": "sweep", "minute": 10, "htf_under_mid": true},
      {"type": "fvg_redelivery", "minute": 25, "htf_under_mid": false}
    ],
    "confluence": [70.0, 75.0, 80.0, ...]
  }
}
```

**Format B: Zipped Arrays**
```json
{
  "session_001": {
    "event_types": ["sweep", "fvg_redelivery"],
    "event_minutes": [10, 25],
    "event_htf_under_mid": [true, false]
  }
}
```

### Scan for Motifs

Find top motif matches in prepared data:

```bash
# Basic scanning
ironforge motifs \
  --input-json motifs_input.json \
  --min-confluence 65.0 \
  --top-k 3

# Higher quality threshold
ironforge motifs \
  --input-json motifs_input.json \
  --min-confluence 80.0 \
  --top-k 10 \
  --preset default
```

#### Example Output
```json
[
  {
    "session_id": "NY_AM_2025_08_05",
    "card_id": "c1", 
    "score": 78.5,
    "window": [5, 20],
    "steps_at": [5, 20]
  },
  {
    "session_id": "NY_PM_2025_08_05",
    "card_id": "c3",
    "score": 71.2,
    "window": [15, 15], 
    "steps_at": [15]
  }
]
```

## Complete Workflow Example

### 1. Discovery with Confluence
```bash
# Run temporal discovery with confluence scoring
ironforge discover-temporal \
  --data-path data/enhanced_sessions/ \
  --output-dir discoveries/ \
  --with-confluence \
  --fanouts 10 5 3 \
  --batch-size 128
```

### 2. Validation (Optional)
```bash
# Run validation with additional confluence metrics
ironforge validate \
  --data-path discoveries/ \
  --mode oos \
  --report-dir validation/
```

### 3. Prepare Motifs Input
```bash
# Convert discovery results to motifs format
ironforge prepare-motifs-input \
  --discovery-json discoveries/temporal_discoveries_20250817_143022.json \
  --validation-json validation/validation_results_20250817.json \
  --out analysis/motifs_input.json
```

### 4. Scan for Motifs
```bash
# Find top 5 motif matches with 70% minimum confluence
ironforge motifs \
  --input-json analysis/motifs_input.json \
  --min-confluence 70.0 \
  --top-k 5 > analysis/top_motifs.json
```

### 5. Generate Reports (Wave 5)
```bash
# Create visual reports
ironforge report \
  --discovery-file discoveries/temporal_discoveries_20250817_143022.json \
  --output-dir reports/ \
  --format both \
  --width 1200
```

## Programming API

### Direct Scanner Usage

```python
from ironforge.motifs.scanner import scan_session_for_cards
from ironforge.motifs.cards import default_cards
import numpy as np

# Prepare session data
session_events = [
    {"type": "sweep", "minute": 8, "htf_under_mid": False},
    {"type": "fvg_redelivery", "minute": 22, "htf_under_mid": True}
]

confluence_scores = np.array([65.0] * 10 + [85.0] * 20 + [70.0] * 30)

# Scan for matches
matches = scan_session_for_cards(
    session_id="NY_PM_2025_08_05",
    events=session_events,
    confluence=confluence_scores,
    cards=default_cards(),
    min_confluence=70.0
)

# Process results
for match in matches:
    print(f"Found {match.card_id} with score {match.score:.1f}")
    print(f"  Window: {match.window[1] - match.window[0]} minutes")
    print(f"  Steps: {len(match.steps_at)} events")
```

### Batch Processing

```python
from pathlib import Path
import json

def process_session_directory(session_dir: Path, output_file: Path):
    """Process all sessions in directory for motif matches."""
    all_matches = []
    
    for session_file in session_dir.glob("*.json"):
        with open(session_file) as f:
            session_data = json.load(f)
        
        events = session_data.get("events", [])
        confluence = np.array(session_data.get("confluence", []))
        
        matches = scan_session_for_cards(
            session_id=session_file.stem,
            events=events,
            confluence=confluence,
            min_confluence=65.0
        )
        
        all_matches.extend(matches)
    
    # Sort by score and save top matches
    all_matches.sort(key=lambda m: -m.score)
    top_matches = all_matches[:20]
    
    with open(output_file, 'w') as f:
        json.dump([
            {
                "session_id": m.session_id,
                "card_id": m.card_id, 
                "score": round(m.score, 2),
                "window": m.window,
                "steps_at": m.steps_at
            }
            for m in top_matches
        ], f, indent=2)

# Usage
process_session_directory(
    Path("discoveries/sessions/"),
    Path("analysis/batch_motifs.json")
)
```

## Performance Specifications

### Confluence v0
- **Throughput**: ≤2s for 1000 session-minutes
- **Memory**: ≤100MB peak usage
- **Accuracy**: Floating-point precision maintained through vectorization
- **Scaling**: Linear with input size

### Motifs v0
- **Throughput**: ≤1s per 100 events per session
- **Memory**: ≤50MB for 1000 events with full confluence data
- **Accuracy**: Exact temporal window matching with guardrail validation
- **Scaling**: O(n×m×k) where n=events, m=cards, k=candidate starts

### CLI Performance
- **prepare-motifs-input**: ≤5s for 100 sessions with full confluence
- **motifs scanner**: ≤3s for 50 sessions with 1000+ events total
- **End-to-end**: ≤30s from discovery JSON to top motifs output

## Error Handling

### Common Issues

**Missing Events**: Empty event lists return empty matches (no error)
```python
matches = scan_session_for_cards("test", [], None)  # Returns []
```

**Confluence Array Mismatch**: Handles gracefully with bounds checking
```python
# Short confluence array with late events
events = [{"type": "sweep", "minute": 100, "htf_under_mid": False}]
confluence = np.array([80.0, 85.0])  # Only 2 minutes
matches = scan_session_for_cards("test", events, confluence)  # Handles safely
```

**Invalid Input Types**: Automatic type coercion with fallbacks
```python
# String booleans are coerced
events = [{"type": "sweep", "minute": "10", "htf_under_mid": "true"}]
# Becomes: {"type": "sweep", "minute": 10, "htf_under_mid": True}
```

**Missing Required Fields**: Uses sensible defaults
```python
events = [{"type": "sweep"}]  # Missing minute, htf_under_mid
# Becomes: {"type": "sweep", "minute": 0, "htf_under_mid": False}
```

### Debugging

**Enable Detailed Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Scanner will log match attempts and filtering decisions
matches = scan_session_for_cards(session_id, events, confluence)
```

**Validate Input Data**:
```python
def validate_motifs_input(data):
    """Validate motifs input format."""
    for session_id, session_data in data.items():
        assert "events" in session_data, f"Missing events in {session_id}"
        
        for event in session_data["events"]:
            assert "type" in event, f"Missing type in {session_id} event"
            assert "minute" in event, f"Missing minute in {session_id} event"
            assert isinstance(event["minute"], int), f"Non-integer minute in {session_id}"
            
        if "confluence" in session_data:
            conf = session_data["confluence"]
            assert all(0 <= score <= 100 for score in conf), f"Invalid confluence range in {session_id}"
```

## Integration with Existing Waves

### Wave 3 Integration (Discovery Pipeline)
- `--with-confluence` flag in `discover-temporal` command
- Confluence scores automatically computed and attached to discovery outputs
- No breaking changes to existing discovery workflow

### Wave 4 Integration (Validation Rails)
- Confluence scores included in validation reports
- Negative controls can test confluence computation reliability
- Ablation studies can isolate confluence component contributions

### Wave 5 Integration (Reporting)
- Confluence scores feed into heatmap intensity and confluence strip visualization
- Motif matches can be overlaid on temporal heatmaps as markers
- Report generation includes motif statistics in summary

## Future Enhancements

### Planned for Wave 7
- **Custom Motif Builder**: GUI/CLI tool for creating custom motif cards
- **Statistical Validation**: Motif significance testing against random baselines
- **Adaptive Thresholds**: Dynamic confluence thresholds based on market regime
- **Pattern Evolution**: Track how motif frequencies change over time

### Research Directions
- **Multi-Session Motifs**: Cross-session pattern detection (e.g., Weekly Friday → Monday patterns)
- **Fuzzy Matching**: Approximate sequence matching with edit distance
- **Machine Learning**: Train neural networks to generate motif cards automatically
- **Real-Time Motifs**: Streaming motif detection for live trading systems

---

## Quick Reference

### Import Statements
```python
# Confluence
from ironforge.metrics.confluence import (
    compute_confluence_score, 
    compute_confluence_components,
    ConfluenceWeights
)

# Motifs
from ironforge.motifs.cards import MotifCard, MotifStep, default_cards
from ironforge.motifs.scanner import scan_session_for_cards, MotifMatch

# Input Preparation
from ironforge.scripts.prepare_motifs_input import build_motifs_input
```

### CLI Commands Summary
```bash
# Discovery with confluence
ironforge discover-temporal --data-path <path> --with-confluence

# Prepare motifs input
ironforge prepare-motifs-input --discovery-json <file> --out <output>

# Scan for motifs  
ironforge motifs --input-json <file> --min-confluence <score> --top-k <n>
```

### Key File Paths
- **Confluence**: `ironforge/metrics/confluence.py`
- **Motif Cards**: `ironforge/motifs/cards.py`
- **Scanner**: `ironforge/motifs/scanner.py`
- **Input Adapter**: `ironforge/scripts/prepare_motifs_input.py`
- **CLI Integration**: `ironforge/sdk/cli.py`

**Status**: Wave 6 Production-Ready  
**Last Updated**: August 17, 2025  
**Architecture**: Confluence v0 + Motifs v0 + CLI Integration