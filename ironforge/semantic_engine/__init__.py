"""
IRONFORGE Semantic Engine  
=========================

Semantic confluence scoring for archaeological significance assessment.
Thin re-export layer for canonical import paths.
"""

from ..confluence.scoring import score_confluence

__all__ = ["score_confluence"]