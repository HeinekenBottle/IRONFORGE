"""
Event Taxonomy v1.0 Contracts
=============================

Canonical dataclasses for IRONFORGE event taxonomy standardization.
Supports the original taxonomy: Expansion, Consolidation, Retracement, 
Reversal, Liquidity Taken, Redelivery within TGAT pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class EventType(IntEnum):
    """Canonical event types for archaeological discovery"""
    EXPANSION = 0
    CONSOLIDATION = 1
    RETRACEMENT = 2
    REVERSAL = 3
    LIQUIDITY_TAKEN = 4
    REDELIVERY = 5


class EdgeType(IntEnum):
    """Graph edge relationship types"""
    TEMPORAL_NEXT = 0         # Sequential time relationship
    MOVEMENT_TRANSITION = 1   # Price movement causality  
    LIQ_LINK = 2             # Liquidity-based connection
    CONTEXT = 3              # Contextual archaeological relationship


class Direction(IntEnum):
    """Market direction classification"""
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


class Location(IntEnum):
    """Price location within session context"""
    LOW = 0
    MID = 1
    HIGH = 2


@dataclass
class MarketEvent:
    """
    Canonical market event structure for TGAT discovery pipeline
    
    Required fields for all events within the taxonomy.
    Maps directly to Node.kind for graph representation.
    """
    # Core identification
    session_id: str
    t: int                    # Unix timestamp milliseconds
    price: float
    symbol: str
    tf: str                   # Timeframe (M5, M15, H1, etc.)
    event_type: EventType
    source: str               # Detection source
    
    # Optional context
    strength: Optional[float] = None      # [0.0-1.0] event magnitude
    direction: Optional[Direction] = None # Market bias
    location: Optional[Location] = None   # Session location context
    
    # Archaeological context
    htf_regime: Optional[int] = None      # HTF regime (0=consol, 1=trans, 2=expan)
    zone_anchor: Optional[float] = None   # 40%/60% dimensional relationship
    
    def to_node_kind(self) -> int:
        """Convert event type to Node.kind uint8 code"""
        return int(self.event_type)
    
    def __post_init__(self):
        """Validate event data on construction"""
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)
        
        if self.strength is not None:
            self.strength = max(0.0, min(1.0, self.strength))


@dataclass 
class EventRelationship:
    """
    Relationship between two market events for graph edges
    
    Maps directly to Edge.etype for graph representation.
    """
    source_event_id: str
    target_event_id: str
    relationship_type: EdgeType
    strength: float = 1.0
    temporal_distance: Optional[int] = None  # Milliseconds between events
    
    def to_edge_type(self) -> int:
        """Convert relationship to Edge.etype uint8 code"""
        return int(self.relationship_type)


@dataclass
class TaxonomyMetadata:
    """Version and compatibility information"""
    taxonomy_version: str = "v1.0"
    node_features_version: str = "1.1"  # 51D with HTF context
    total_event_types: int = 6
    total_edge_types: int = 4
    
    def is_compatible_with(self, other_version: str) -> bool:
        """Check compatibility with other taxonomy versions"""
        return other_version in ["v1.0"]


# Global taxonomy metadata instance
TAXONOMY_V1 = TaxonomyMetadata()