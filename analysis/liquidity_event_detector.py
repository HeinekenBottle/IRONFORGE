#!/usr/bin/env python3
"""
IRONFORGE Liquidity Event Detector
==================================

Core detection algorithms for market liquidity events based on archaeological pattern discovery.
Identifies and classifies the specific liquidity signatures that define each range cluster:

- FVG Events: redelivery, first_presented, continuation
- Sweep Events: buy_side, sell_side, double_sweep  
- PD Array: premium_rejection, discount_acceptance, equilibrium_test
- Consolidation: range_consolidation, accumulation, distribution
- Expansion: breakout_expansion, momentum_expansion, volatility_expansion

Based on discovered liquidity event DNA:
- 40% Range: 63.2% sweep events (acceleration zone)
- 60% Range: 61.1% FVG events (equilibrium zone) 
- 80% Range: 71.9% sweep events (completion zone)
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
import logging

class LiquidityEventType(Enum):
    """Enumeration of liquidity event types"""
    FVG_REDELIVERY = "fvg_redelivery"
    FVG_FIRST_PRESENTED = "fvg_first_presented"
    FVG_CONTINUATION = "fvg_continuation"
    
    SWEEP_BUY_SIDE = "buy_side_sweep"
    SWEEP_SELL_SIDE = "sell_side_sweep"
    SWEEP_DOUBLE = "double_sweep"
    
    PD_PREMIUM_REJECTION = "premium_rejection"
    PD_DISCOUNT_ACCEPTANCE = "discount_acceptance" 
    PD_EQUILIBRIUM_TEST = "equilibrium_test"
    
    CONSOLIDATION_RANGE = "range_consolidation"
    CONSOLIDATION_ACCUMULATION = "accumulation"
    CONSOLIDATION_DISTRIBUTION = "distribution"
    
    EXPANSION_BREAKOUT = "breakout_expansion"
    EXPANSION_MOMENTUM = "momentum_expansion"
    EXPANSION_VOLATILITY = "volatility_expansion"

@dataclass
class LiquidityEvent:
    """Individual liquidity event with classification and metadata"""
    event_type: LiquidityEventType
    range_level: float
    session_phase: str
    session_position: float
    pattern_strength: float
    htf_confluence: bool
    temporal_echo: bool
    evolution_strength: float
    archaeological_significance: float

@dataclass
class LiquiditySignature:
    """Complete liquidity signature for a pattern or range cluster"""
    range_level: str
    primary_events: List[LiquidityEventType]
    event_frequencies: Dict[LiquidityEventType, float]
    pd_array_interaction: bool
    htf_confluence_rate: float
    avg_evolution_strength: float
    velocity_consistency: float
    continuation_probability: float

class LiquidityEventDetector:
    """
    Core liquidity event detection and classification system
    """
    
    def __init__(self, patterns_file: str = None):
        self.logger = logging.getLogger('liquidity_event_detector')
        
        # Load discovered patterns
        if patterns_file is None:
            patterns_file = '/Users/jack/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json'
        
        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)
        
        # Initialize detection parameters based on archaeological discovery
        self.range_signatures = self._initialize_range_signatures()
        self.event_thresholds = self._initialize_event_thresholds()
        
        print(f"🔍 Liquidity Event Detector initialized")
        print(f"  Patterns loaded: {len(self.patterns)}")
        print(f"  Range signatures: {list(self.range_signatures.keys())}")
    
    def _initialize_range_signatures(self) -> Dict[str, LiquiditySignature]:
        """Initialize archaeological range signatures from discovered patterns"""
        return {
            "20%": LiquiditySignature(
                range_level="20%",
                primary_events=[LiquidityEventType.SWEEP_BUY_SIDE, LiquidityEventType.FVG_REDELIVERY],
                event_frequencies={
                    LiquidityEventType.FVG_REDELIVERY: 0.560,
                    LiquidityEventType.SWEEP_BUY_SIDE: 0.607,
                    LiquidityEventType.PD_PREMIUM_REJECTION: 1.000
                },
                pd_array_interaction=True,
                htf_confluence_rate=1.000,
                avg_evolution_strength=0.92,
                velocity_consistency=0.68,
                continuation_probability=1.00
            ),
            "40%": LiquiditySignature(
                range_level="40%", 
                primary_events=[LiquidityEventType.SWEEP_DOUBLE, LiquidityEventType.FVG_CONTINUATION],
                event_frequencies={
                    LiquidityEventType.FVG_CONTINUATION: 0.574,
                    LiquidityEventType.SWEEP_DOUBLE: 0.632,
                    LiquidityEventType.PD_DISCOUNT_ACCEPTANCE: 1.000
                },
                pd_array_interaction=True,
                htf_confluence_rate=1.000,
                avg_evolution_strength=0.89,
                velocity_consistency=1.00,  # Perfect consistency - acceleration zone
                continuation_probability=1.00
            ),
            "60%": LiquiditySignature(
                range_level="60%",
                primary_events=[LiquidityEventType.FVG_FIRST_PRESENTED, LiquidityEventType.PD_EQUILIBRIUM_TEST],
                event_frequencies={
                    LiquidityEventType.FVG_FIRST_PRESENTED: 0.611,
                    LiquidityEventType.SWEEP_SELL_SIDE: 0.574,
                    LiquidityEventType.PD_EQUILIBRIUM_TEST: 1.000
                },
                pd_array_interaction=True,
                htf_confluence_rate=1.000,
                avg_evolution_strength=0.93,  # Highest evolution strength
                velocity_consistency=0.89,
                continuation_probability=1.00
            ),
            "80%": LiquiditySignature(
                range_level="80%",
                primary_events=[LiquidityEventType.SWEEP_DOUBLE, LiquidityEventType.CONSOLIDATION_RANGE],
                event_frequencies={
                    LiquidityEventType.FVG_CONTINUATION: 0.438,  # Lowest - exhaustion
                    LiquidityEventType.SWEEP_DOUBLE: 0.719,      # Highest - completion
                    LiquidityEventType.CONSOLIDATION_RANGE: 0.391,
                    LiquidityEventType.PD_PREMIUM_REJECTION: 1.000
                },
                pd_array_interaction=True,
                htf_confluence_rate=1.000,
                avg_evolution_strength=0.92,
                velocity_consistency=1.00,  # Perfect terminal velocity
                continuation_probability=1.00
            )
        }
    
    def _initialize_event_thresholds(self) -> Dict[str, float]:
        """Initialize detection thresholds for liquidity events"""
        return {
            'fvg_confidence_threshold': 0.75,
            'sweep_momentum_threshold': 0.60,
            'pd_array_significance_threshold': 0.80,
            'consolidation_range_threshold': 0.25,
            'expansion_velocity_threshold': 0.70,
            'htf_confluence_strength_threshold': 0.50,
            'temporal_echo_threshold': 0.80,
            'evolution_strength_threshold': 0.85
        }
    
    def detect_pattern_liquidity_events(self, pattern: Dict) -> List[LiquidityEvent]:
        """Detect and classify liquidity events within a single pattern"""
        events = []
        
        # Extract pattern metadata
        range_level = self._extract_range_level(pattern)
        if not range_level:
            return events
        
        session_phase = pattern.get('phase_information', {}).get('primary_phase', 'unknown')
        session_position = pattern.get('phase_information', {}).get('session_position', 0.0)
        pattern_strength = pattern.get('semantic_context', {}).get('structural_context', {}).get('pattern_strength', 0.0)
        
        # HTF and temporal features
        htf_confluence = self._detect_htf_confluence(pattern)
        temporal_echo = self._detect_temporal_echo(pattern)
        evolution_strength = self._extract_evolution_strength(pattern)
        archaeological_sig = pattern.get('archaeological_significance', {}).get('overall_significance', 0.0)
        
        # Constant features indicate liquidity events that were filtered out (present)
        const_features = pattern.get('semantic_context', {}).get('constant_features_context', {}).get('constant_names', [])
        
        # FVG Event Detection
        if 'fvg_redelivery_flag' in const_features:
            fvg_type = self._classify_fvg_event(pattern, range_level)
            events.append(LiquidityEvent(
                event_type=fvg_type,
                range_level=range_level,
                session_phase=session_phase,
                session_position=session_position,
                pattern_strength=pattern_strength,
                htf_confluence=htf_confluence,
                temporal_echo=temporal_echo,
                evolution_strength=evolution_strength,
                archaeological_significance=archaeological_sig
            ))
        
        # Sweep Event Detection
        if 'liq_sweep_flag' in const_features:
            sweep_type = self._classify_sweep_event(pattern, range_level)
            events.append(LiquidityEvent(
                event_type=sweep_type,
                range_level=range_level,
                session_phase=session_phase,
                session_position=session_position,
                pattern_strength=pattern_strength,
                htf_confluence=htf_confluence,
                temporal_echo=temporal_echo,
                evolution_strength=evolution_strength,
                archaeological_significance=archaeological_sig
            ))
        
        # PD Array Event Detection
        if 'pd_array_interaction_flag' in const_features:
            pd_type = self._classify_pd_array_event(pattern, range_level)
            events.append(LiquidityEvent(
                event_type=pd_type,
                range_level=range_level,
                session_phase=session_phase,
                session_position=session_position,
                pattern_strength=pattern_strength,
                htf_confluence=htf_confluence,
                temporal_echo=temporal_echo,
                evolution_strength=evolution_strength,
                archaeological_significance=archaeological_sig
            ))
        
        # Consolidation Event Detection
        if 'consolidation_flag' in const_features:
            consolidation_type = self._classify_consolidation_event(pattern, range_level)
            events.append(LiquidityEvent(
                event_type=consolidation_type,
                range_level=range_level,
                session_phase=session_phase,
                session_position=session_position,
                pattern_strength=pattern_strength,
                htf_confluence=htf_confluence,
                temporal_echo=temporal_echo,
                evolution_strength=evolution_strength,
                archaeological_significance=archaeological_sig
            ))
        
        # Expansion Event Detection
        if 'expansion_phase_flag' in const_features:
            expansion_type = self._classify_expansion_event(pattern, range_level)
            events.append(LiquidityEvent(
                event_type=expansion_type,
                range_level=range_level,
                session_phase=session_phase,
                session_position=session_position,
                pattern_strength=pattern_strength,
                htf_confluence=htf_confluence,
                temporal_echo=temporal_echo,
                evolution_strength=evolution_strength,
                archaeological_significance=archaeological_sig
            ))
        
        return events
    
    def _extract_range_level(self, pattern: Dict) -> Optional[float]:
        """Extract range level percentage from pattern"""
        desc = pattern.get('description', '')
        range_match = re.search(r'(\d+\.?\d*)% of range', desc)
        return float(range_match.group(1)) if range_match else None
    
    def _detect_htf_confluence(self, pattern: Dict) -> bool:
        """Detect HTF confluence presence in pattern"""
        desc = pattern.get('description', '')
        const_features = pattern.get('semantic_context', {}).get('constant_features_context', {}).get('constant_names', [])
        
        return ('HTF confluence' in desc or 
                'cross_tf_confluence' in const_features or
                'temporal_echo_strength' in const_features)
    
    def _detect_temporal_echo(self, pattern: Dict) -> bool:
        """Detect temporal echo strength in pattern"""
        const_features = pattern.get('semantic_context', {}).get('constant_features_context', {}).get('constant_names', [])
        return 'temporal_echo_strength' in const_features
    
    def _extract_evolution_strength(self, pattern: Dict) -> float:
        """Extract evolution strength from pattern context"""
        # This would be enhanced with cross-session evolution analysis
        structural_context = pattern.get('semantic_context', {}).get('structural_context', {})
        return structural_context.get('evolution_strength', 0.5)  # Default archaeological average
    
    def _classify_fvg_event(self, pattern: Dict, range_level: float) -> LiquidityEventType:
        """Classify Fair Value Gap event type based on range characteristics"""
        if range_level <= 25:
            return LiquidityEventType.FVG_REDELIVERY
        elif 55 <= range_level <= 65:
            return LiquidityEventType.FVG_FIRST_PRESENTED  # Equilibrium zone signature
        else:
            return LiquidityEventType.FVG_CONTINUATION
    
    def _classify_sweep_event(self, pattern: Dict, range_level: float) -> LiquidityEventType:
        """Classify liquidity sweep event type based on range characteristics"""
        if 35 <= range_level <= 45:  # 40% range - acceleration zone
            return LiquidityEventType.SWEEP_DOUBLE
        elif 75 <= range_level <= 85:  # 80% range - completion zone
            return LiquidityEventType.SWEEP_DOUBLE  # Highest frequency
        elif range_level >= 60:
            return LiquidityEventType.SWEEP_SELL_SIDE
        else:
            return LiquidityEventType.SWEEP_BUY_SIDE
    
    def _classify_pd_array_event(self, pattern: Dict, range_level: float) -> LiquidityEventType:
        """Classify Premium/Discount Array interaction type"""
        if range_level <= 25:
            return LiquidityEventType.PD_PREMIUM_REJECTION
        elif 35 <= range_level <= 45:
            return LiquidityEventType.PD_DISCOUNT_ACCEPTANCE
        elif 55 <= range_level <= 65:
            return LiquidityEventType.PD_EQUILIBRIUM_TEST  # Perfect balance signature
        else:
            return LiquidityEventType.PD_PREMIUM_REJECTION
    
    def _classify_consolidation_event(self, pattern: Dict, range_level: float) -> LiquidityEventType:
        """Classify consolidation event type"""
        if range_level >= 75:  # 80% range has highest consolidation (39.1%)
            return LiquidityEventType.CONSOLIDATION_RANGE
        elif range_level <= 25:
            return LiquidityEventType.CONSOLIDATION_ACCUMULATION
        else:
            return LiquidityEventType.CONSOLIDATION_DISTRIBUTION
    
    def _classify_expansion_event(self, pattern: Dict, range_level: float) -> LiquidityEventType:
        """Classify expansion event type"""
        if range_level >= 60:  # Higher ranges show more expansion
            return LiquidityEventType.EXPANSION_BREAKOUT
        else:
            return LiquidityEventType.EXPANSION_MOMENTUM
    
    def analyze_pattern_liquidity_signature(self, pattern: Dict) -> Optional[LiquiditySignature]:
        """Analyze complete liquidity signature for a pattern"""
        events = self.detect_pattern_liquidity_events(pattern)
        if not events:
            return None
        
        range_level = self._extract_range_level(pattern)
        if not range_level:
            return None
        
        # Classify range bucket
        range_bucket = self._classify_range_bucket(range_level)
        
        # Calculate event frequencies
        event_types = [event.event_type for event in events]
        event_counts = Counter(event_types)
        total_events = len(events)
        event_frequencies = {event_type: count / total_events for event_type, count in event_counts.items()}
        
        # Extract signature characteristics
        pd_array_interaction = any(event.event_type.value.startswith('pd_') for event in events)
        htf_confluence_rate = sum(1 for event in events if event.htf_confluence) / len(events)
        avg_evolution_strength = np.mean([event.evolution_strength for event in events])
        
        return LiquiditySignature(
            range_level=range_bucket,
            primary_events=list(event_counts.keys())[:3],  # Top 3 events
            event_frequencies=event_frequencies,
            pd_array_interaction=pd_array_interaction,
            htf_confluence_rate=htf_confluence_rate,
            avg_evolution_strength=avg_evolution_strength,
            velocity_consistency=0.0,  # Would be calculated from velocity analysis
            continuation_probability=0.0  # Would be calculated from cross-session analysis
        )
    
    def _classify_range_bucket(self, range_level: float) -> str:
        """Classify range level into archaeological buckets"""
        if 15 <= range_level < 25:
            return "20%"
        elif 35 <= range_level < 45:
            return "40%"
        elif 55 <= range_level < 65:
            return "60%"
        elif 75 <= range_level < 85:
            return "80%"
        else:
            return f"{range_level:.0f}%"
    
    def match_signature_to_range(self, signature: LiquiditySignature) -> Dict[str, float]:
        """Match detected signature against known range signatures"""
        matches = {}
        
        for range_level, known_signature in self.range_signatures.items():
            similarity_score = self._calculate_signature_similarity(signature, known_signature)
            matches[range_level] = similarity_score
        
        return matches
    
    def _calculate_signature_similarity(self, sig1: LiquiditySignature, sig2: LiquiditySignature) -> float:
        """Calculate similarity score between two liquidity signatures"""
        score = 0.0
        
        # PD Array interaction match (binary)
        if sig1.pd_array_interaction == sig2.pd_array_interaction:
            score += 0.25
        
        # HTF confluence rate similarity
        confluence_diff = abs(sig1.htf_confluence_rate - sig2.htf_confluence_rate)
        score += 0.25 * (1.0 - confluence_diff)
        
        # Evolution strength similarity
        evolution_diff = abs(sig1.avg_evolution_strength - sig2.avg_evolution_strength)
        score += 0.25 * (1.0 - evolution_diff)
        
        # Primary event overlap
        common_events = set(sig1.primary_events) & set(sig2.primary_events)
        event_similarity = len(common_events) / max(len(sig1.primary_events), len(sig2.primary_events), 1)
        score += 0.25 * event_similarity
        
        return min(score, 1.0)
    
    def detect_all_pattern_events(self) -> Dict[str, List[LiquidityEvent]]:
        """Detect liquidity events across all patterns"""
        print("🔍 Detecting liquidity events across all patterns...")
        
        all_events = {}
        
        for i, pattern in enumerate(self.patterns):
            events = self.detect_pattern_liquidity_events(pattern)
            if events:
                all_events[f"pattern_{i}"] = events
        
        print(f"  ✅ Detected events in {len(all_events)} patterns")
        return all_events
    
    def generate_range_event_analysis(self) -> Dict[str, Dict]:
        """Generate comprehensive range-level event analysis"""
        print("📊 Generating range-level event analysis...")
        
        # Detect all events
        all_events = self.detect_all_pattern_events()
        
        # Group by range levels
        range_events = defaultdict(list)
        
        for pattern_id, events in all_events.items():
            for event in events:
                range_bucket = self._classify_range_bucket(event.range_level)
                range_events[range_bucket].append(event)
        
        # Analyze each range level
        analysis = {}
        for range_level, events in range_events.items():
            analysis[range_level] = self._analyze_range_events(range_level, events)
        
        return analysis
    
    def _analyze_range_events(self, range_level: str, events: List[LiquidityEvent]) -> Dict:
        """Analyze events for a specific range level"""
        if not events:
            return {}
        
        event_types = [event.event_type for event in events]
        event_counts = Counter(event_types)
        
        return {
            'total_events': len(events),
            'event_distribution': {event_type.value: count for event_type, count in event_counts.items()},
            'event_frequencies': {event_type.value: count / len(events) for event_type, count in event_counts.items()},
            'avg_pattern_strength': np.mean([event.pattern_strength for event in events]),
            'htf_confluence_rate': sum(1 for event in events if event.htf_confluence) / len(events),
            'temporal_echo_rate': sum(1 for event in events if event.temporal_echo) / len(events),
            'avg_evolution_strength': np.mean([event.evolution_strength for event in events]),
            'avg_archaeological_significance': np.mean([event.archaeological_significance for event in events]),
            'session_phase_distribution': dict(Counter([event.session_phase for event in events])),
            'dominant_event_types': [event_type.value for event_type, _ in event_counts.most_common(3)]
        }
    
    def save_event_analysis(self, output_path: str = None) -> str:
        """Save comprehensive liquidity event analysis"""
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/liquidity_events_analysis.json'
        
        # Generate analysis
        analysis = {
            'analysis_metadata': {
                'detector_version': '1.0',
                'patterns_analyzed': len(self.patterns),
                'range_signatures': {level: {
                    'primary_events': [event.value for event in sig.primary_events],
                    'pd_array_interaction': sig.pd_array_interaction,
                    'htf_confluence_rate': sig.htf_confluence_rate,
                    'avg_evolution_strength': sig.avg_evolution_strength,
                    'velocity_consistency': sig.velocity_consistency,
                    'continuation_probability': sig.continuation_probability
                } for level, sig in self.range_signatures.items()}
            },
            'range_event_analysis': self.generate_range_event_analysis(),
            'detection_thresholds': self.event_thresholds
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"💾 Liquidity event analysis saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    print("🔍 IRONFORGE Liquidity Event Detector")
    print("=" * 60)
    
    detector = LiquidityEventDetector()
    output_file = detector.save_event_analysis()
    
    print(f"\n✅ Liquidity event detection complete!")
    print(f"📊 Results saved to: {output_file}")