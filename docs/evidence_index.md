# Event Taxonomy Evidence Index

## Core Definitions Found

### Event Types (Primary Taxonomy)
- **EXPANSION**: Found in Pattern Discovery Guide and Motifs documentation
- **CONSOLIDATION**: Found in Semantic Features (lines 30, 95, 163-164) and Grok 4 System description
- **RETRACEMENT**: Limited references, needs canonical definition  
- **REVERSAL**: Limited references, needs canonical definition
- **LIQUIDITY_TAKEN**: Referenced as "liquidity sweeps" in archaeological analysis
- **REDELIVERY**: Extensively documented as "FVG redelivery" patterns

### Compound Terms
- **FPFVG**: "Fair Value Gap redelivery" - found in:
  - `docs/MOTIFS.md:121` - "FPFVG redelivery 12–30m after sweep"
  - `docs/SEMANTIC_FEATURES.md:28,68-81,214-218,308` - Core FVG redelivery detection
  - `CLAUDE.md:19` - "FVG redelivery, expansion phases, session anchoring"

- **ECRR**: Not explicitly found as acronym, inferred as Expansion/Consolidation/Retracement/Reversal

### Archaeological Terms
- **Zone/Anchor**: Extensively documented in:
  - `CLAUDE.local.md:1-31` - Theory B 40%/60% dimensional zones
  - Pattern analysis showing "archaeological zones" as predictive markers
  - Zone-based dimensional relationships to final session structure

### Current Implementation Evidence

**Semantic Features (8D)**:
- `fvg_redelivery_event` - Lines found in:
  - `docs/SEMANTIC_FEATURES.md:28,93,162`
  - `docs/ARCHITECTURE.md:162`
  - `docs/TGAT_ARCHITECTURE.md:93`
- `expansion_phase_event` - Market expansion identification
- `consolidation_event` - Consolidation pattern recognition  
- `pd_array_event` - Premium/Discount array detection

**Event Processing**:
- `docs/MOTIFS.md:158-167` - Event format: `type`, `minute`, `htf_under_mid`
- `docs/SEMANTIC_FEATURES.md:78,150` - Event type filtering in price movements
- Event types referenced: "sweep", "fvg_redelivery", "expansion_phase"

**Archaeological Discovery**:
- `CLAUDE.local.md:8-31` - 40% zone dimensional anchoring to final session range
- `docs/PATTERN_ANALYSIS_GUIDE.md:104-110` - FVG redelivery and expansion phase sessions
- Theory B predictive archaeology confirmed empirically

## Gap Analysis
- **RETRACEMENT/REVERSAL**: Need canonical definitions
- **LIQUIDITY_TAKEN**: Currently referenced as "sweeps", needs formal taxonomy
- **Standardization**: Event types scattered across documentation, need unified taxonomy
- **Node Mapping**: How events become Node.kind codes (0-5) needs documentation