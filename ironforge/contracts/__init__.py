"""
IRONFORGE Contracts
===================

Canonical data contracts and taxonomy definitions for archaeological discovery.
"""

from .taxonomy_v1 import (
    EventType,
    EdgeType, 
    Direction,
    Location,
    MarketEvent,
    EventRelationship,
    TaxonomyMetadata,
    TAXONOMY_V1
)

__all__ = [
    "EventType",
    "EdgeType", 
    "Direction",
    "Location",
    "MarketEvent",
    "EventRelationship", 
    "TaxonomyMetadata",
    "TAXONOMY_V1"
]