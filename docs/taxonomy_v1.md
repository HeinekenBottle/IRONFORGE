# Event Taxonomy v1.0

## Canonical Event Types

**Complete canonical list**: Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery

### Core Market Events (ECRR)
1. **EXPANSION** - Directional market movement with increasing momentum
2. **CONSOLIDATION** - Sideways price action within defined range boundaries  
3. **RETRACEMENT** - Counter-trend pullback within established directional bias
4. **REVERSAL** - Change in primary market direction, invalidating previous bias

### Liquidity Events
5. **LIQUIDITY_TAKEN** - Price sweep through significant liquidity zones (stops, orders)
6. **REDELIVERY** - Return to previously identified fair value gaps or imbalances

## Event Structure

### Required Fields
- `session_id`: Unique session identifier
- `t`: Unix timestamp (milliseconds)
- `price`: Price level at event occurrence
- `symbol`: Trading symbol (e.g., "NQ", "ES")
- `tf`: Timeframe identifier (e.g., "M5", "M15")
- `event_type`: One of the six canonical types above
- `source`: Event detection source ("tgat", "confluence", "manual")

### Optional Fields
- `strength`: Event magnitude [0.0-1.0]
- `direction`: Market direction ("bullish", "bearish", "neutral")
- `location`: Price location context ("high", "mid", "low")

## Graph Integration

### Node Mapping (Node.kind uint8)
- `0`: EXPANSION
- `1`: CONSOLIDATION  
- `2`: RETRACEMENT
- `3`: REVERSAL
- `4`: LIQUIDITY_TAKEN
- `5`: REDELIVERY

### Edge Types (Edge.etype uint8) 
- `0`: TEMPORAL_NEXT - Sequential time relationship
- `1`: MOVEMENT_TRANSITION - Price movement causality
- `2`: LIQ_LINK - Liquidity-based connection
- `3`: CONTEXT - Contextual archaeological relationship

## HTF Context Integration
Events leverage HTF features (f45-f50) for enhanced context:
- Bar position within HTF timeframes
- Regime classification (consolidation/transition/expansion)
- Volume anomaly detection via synthetic volume z-scores
- Distance to daily midpoint for dimensional anchoring

## Version Control
- **taxonomy_version**: "v1.0"
- Compatible with Node Features v1.1 (51D)
- Maintains backward compatibility with 45D baseline