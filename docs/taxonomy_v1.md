# IRONFORGE Event Taxonomy v1.0

**Authoritative Reference — Golden Invariants**

This document defines the canonical event taxonomy and edge intents for IRONFORGE. These are **golden invariants** that must never change without major version increment.

## Event Types (Exactly 6)

### 1. Expansion
**Definition**: Market range extension beyond previous boundaries
**Characteristics**:
- Price breaks out of established range
- Volume typically increases
- Creates new high or low for the session
- Often accompanied by momentum acceleration

**Example**: Price breaks above resistance level with strong volume

### 2. Consolidation  
**Definition**: Range compression and sideways price action
**Characteristics**:
- Price oscillates within defined boundaries
- Volume typically decreases
- Volatility compression
- Preparation phase for next directional move

**Example**: Price trades in tight range after initial move

### 3. Retracement
**Definition**: Partial reversal within an established trend
**Characteristics**:
- Counter-trend movement
- Does not violate major trend structure
- Typically 23.6%, 38.2%, or 50% of prior move
- Temporary in nature

**Example**: Pullback to 38.2% Fibonacci level in uptrend

### 4. Reversal
**Definition**: Full directional change in market trend
**Characteristics**:
- Breaks significant trend structure
- Often accompanied by volume spike
- Creates new trend direction
- Invalidates previous trend assumptions

**Example**: Break of major support leading to downtrend

### 5. Liquidity Taken
**Definition**: Order flow absorption at key levels
**Characteristics**:
- Large volume at specific price levels
- Often at round numbers or technical levels
- May create temporary price stalls
- Indicates institutional activity

**Example**: Large volume spike at 18,000 level with price rejection

### 6. Redelivery
**Definition**: Return to previously significant price levels
**Characteristics**:
- Price revisits prior important levels
- May test previous support/resistance
- Often occurs with reduced volume initially
- Can lead to breakout or rejection

**Example**: Return to previous day's high for retest

## Edge Intents (Exactly 4)

### 1. TEMPORAL_NEXT
**Definition**: Sequential time progression between events
**Characteristics**:
- Direct chronological relationship
- Represents natural time flow
- Most common edge type
- Preserves temporal ordering

**Usage**: Connects consecutive events in time sequence

### 2. MOVEMENT_TRANSITION
**Definition**: Price movement relationships between events
**Characteristics**:
- Represents directional changes
- Captures momentum shifts
- Links cause-and-effect price movements
- Critical for trend analysis

**Usage**: Connects events that represent price transitions

### 3. LIQ_LINK
**Definition**: Liquidity flow connections between events
**Characteristics**:
- Represents order flow relationships
- Connects liquidity events
- Captures institutional activity patterns
- Important for volume analysis

**Usage**: Links events related to liquidity provision/consumption

### 4. CONTEXT
**Definition**: Contextual relationships between events
**Characteristics**:
- Non-temporal relationships
- Structural or thematic connections
- Provides additional context
- Enriches pattern understanding

**Usage**: Connects events with shared context or meaning

## Taxonomy Rules

### Event Classification Rules
1. **Mutual Exclusivity**: Each event belongs to exactly one type
2. **Temporal Ordering**: Events must maintain chronological sequence
3. **Session Boundaries**: Events cannot span across sessions
4. **Magnitude Thresholds**: Minimum significance requirements apply

### Edge Classification Rules
1. **Intent Clarity**: Each edge has exactly one intent
2. **Directional Consistency**: Edge direction must be meaningful
3. **Temporal Constraints**: TEMPORAL_NEXT edges must respect time order
4. **Session Isolation**: No cross-session edges allowed

## Feature Mapping

### Node Features (51D)
- **f0..f44**: Base event features (price, volume, time, structure)
- **f45..f50**: HTF v1.1 features (last-closed only)

### Edge Features (20D)  
- **e0..e19**: Edge relationship features based on intent type

## Validation Criteria

### Event Validation
```python
def validate_event_taxonomy(events):
    """Validate events against taxonomy v1.0"""
    valid_types = {
        'Expansion', 'Consolidation', 'Retracement', 
        'Reversal', 'Liquidity Taken', 'Redelivery'
    }
    
    for event in events:
        assert event.type in valid_types
        assert event.session_id is not None
        assert event.timestamp is not None
    
    return True
```

### Edge Validation
```python
def validate_edge_intents(edges):
    """Validate edges against intent taxonomy v1.0"""
    valid_intents = {
        'TEMPORAL_NEXT', 'MOVEMENT_TRANSITION', 
        'LIQ_LINK', 'CONTEXT'
    }
    
    for edge in edges:
        assert edge.intent in valid_intents
        assert edge.source_event is not None
        assert edge.target_event is not None
    
    return True
```

## Historical Context

### Version History
- **v1.0**: Initial taxonomy (6 events, 4 edge intents)
- **Future**: No changes planned (golden invariants)

### Design Principles
1. **Completeness**: Cover all relevant market behaviors
2. **Simplicity**: Minimal but sufficient taxonomy
3. **Clarity**: Unambiguous definitions
4. **Stability**: Long-term consistency

## Usage Guidelines

### For Developers
- Always validate against this taxonomy
- Never add new event types without major version change
- Preserve edge intent semantics
- Maintain session boundary constraints

### For Analysts
- Use taxonomy for pattern classification
- Reference definitions for event identification
- Apply validation criteria consistently
- Report taxonomy violations immediately

## Compliance

### Required Checks
1. Event type validation on all inputs
2. Edge intent validation on all graphs
3. Feature dimension verification (51D/20D)
4. Session boundary enforcement

### Error Handling
- **Invalid Event Type**: Reject with clear error message
- **Invalid Edge Intent**: Reject with clear error message  
- **Cross-Session Edge**: Reject with session boundary violation
- **Dimension Mismatch**: Reject with feature dimension error

---

**IRONFORGE Event Taxonomy v1.0** — Canonical and immutable reference
