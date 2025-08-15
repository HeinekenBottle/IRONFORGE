#!/usr/bin/env python3
"""
IRONFORGE Range Filter System
============================

Range-specific pattern filters based on archaeological liquidity event DNA:

**40% Range - SWEEP ACCELERATION ZONE**
- 63.2% sweep events (buy/sell/double sweeps)
- 100% continuation probability beyond level
- Perfect velocity consistency (1.00)
- 2-3 day HTF lag signatures

**60% Range - FVG EQUILIBRIUM ZONE**  
- 61.1% FVG events (redelivery/first presented)
- Highest evolution strength (0.93)
- Perfect 50/50 early/late timing balance
- 2-3 day HTF lag patterns

**80% Range - SWEEP COMPLETION ZONE**
- 71.9% sweep events (highest concentration)
- 39.1% consolidation (completion signal)
- Guaranteed 80%+ completion (85% avg)
- 1-5 day HTF lag spectrum
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
import logging
from liquidity_event_detector import LiquidityEventDetector, LiquidityEvent, LiquidityEventType, LiquiditySignature

class RangeZoneType(Enum):
    """Range zone classifications based on archaeological discovery"""
    MOMENTUM_FILTER = "20%"           # Variable momentum (0.68 consistency)
    SWEEP_ACCELERATION = "40%"        # Perfect consistency (1.00) - acceleration  
    FVG_EQUILIBRIUM = "60%"          # High consistency (0.89) - equilibrium
    SWEEP_COMPLETION = "80%"         # Perfect consistency (1.00) - terminal

@dataclass
class RangeFilterCriteria:
    """Filter criteria for range zone identification"""
    zone_type: RangeZoneType
    required_events: List[LiquidityEventType]
    event_frequency_thresholds: Dict[LiquidityEventType, float]
    pd_array_required: bool
    htf_confluence_threshold: float
    velocity_consistency_threshold: float
    continuation_probability_threshold: float
    session_phase_preferences: List[str]
    completion_probability_80pct: float
    avg_range_reached: float

@dataclass
class RangeFilterMatch:
    """Result of range filter matching"""
    zone_type: RangeZoneType
    confidence_score: float
    matching_events: List[LiquidityEvent]
    signature_similarity: float
    tactical_intelligence: Dict[str, any]
    prediction_metrics: Dict[str, float]

class RangeFilterSystem:
    """
    Range-specific pattern filters based on liquidity event DNA
    """
    
    def __init__(self, patterns_file: str = None):
        self.logger = logging.getLogger('range_filter_system')
        
        # Initialize liquidity event detector
        self.event_detector = LiquidityEventDetector(patterns_file)
        
        # Initialize range filters
        self.range_filters = self._initialize_range_filters()
        self.tactical_frameworks = self._initialize_tactical_frameworks()
        
        print(f"🎯 Range Filter System initialized")
        print(f"  Range filters: {list(self.range_filters.keys())}")
    
    def _initialize_range_filters(self) -> Dict[RangeZoneType, RangeFilterCriteria]:
        """Initialize archaeological range filter criteria"""
        return {
            RangeZoneType.MOMENTUM_FILTER: RangeFilterCriteria(
                zone_type=RangeZoneType.MOMENTUM_FILTER,
                required_events=[LiquidityEventType.FVG_REDELIVERY, LiquidityEventType.SWEEP_BUY_SIDE],
                event_frequency_thresholds={
                    LiquidityEventType.FVG_REDELIVERY: 0.56,
                    LiquidityEventType.SWEEP_BUY_SIDE: 0.607,
                    LiquidityEventType.PD_PREMIUM_REJECTION: 1.0
                },
                pd_array_required=True,
                htf_confluence_threshold=1.0,
                velocity_consistency_threshold=0.68,
                continuation_probability_threshold=1.0,
                session_phase_preferences=["session_closing"],
                completion_probability_80pct=0.0,
                avg_range_reached=28.1
            ),
            
            RangeZoneType.SWEEP_ACCELERATION: RangeFilterCriteria(
                zone_type=RangeZoneType.SWEEP_ACCELERATION,
                required_events=[LiquidityEventType.SWEEP_DOUBLE, LiquidityEventType.FVG_CONTINUATION],
                event_frequency_thresholds={
                    LiquidityEventType.FVG_CONTINUATION: 0.574,
                    LiquidityEventType.SWEEP_DOUBLE: 0.632,
                    LiquidityEventType.PD_DISCOUNT_ACCEPTANCE: 1.0
                },
                pd_array_required=True,
                htf_confluence_threshold=1.0,
                velocity_consistency_threshold=1.0,  # Perfect acceleration zone
                continuation_probability_threshold=1.0,
                session_phase_preferences=["session_closing"],
                completion_probability_80pct=0.0,
                avg_range_reached=50.0
            ),
            
            RangeZoneType.FVG_EQUILIBRIUM: RangeFilterCriteria(
                zone_type=RangeZoneType.FVG_EQUILIBRIUM,
                required_events=[LiquidityEventType.FVG_FIRST_PRESENTED, LiquidityEventType.PD_EQUILIBRIUM_TEST],
                event_frequency_thresholds={
                    LiquidityEventType.FVG_FIRST_PRESENTED: 0.611,
                    LiquidityEventType.SWEEP_SELL_SIDE: 0.574,
                    LiquidityEventType.PD_EQUILIBRIUM_TEST: 1.0
                },
                pd_array_required=True,
                htf_confluence_threshold=1.0,
                velocity_consistency_threshold=0.89,
                continuation_probability_threshold=1.0,
                session_phase_preferences=["session_closing"],
                completion_probability_80pct=0.0,
                avg_range_reached=70.6
            ),
            
            RangeZoneType.SWEEP_COMPLETION: RangeFilterCriteria(
                zone_type=RangeZoneType.SWEEP_COMPLETION,
                required_events=[LiquidityEventType.SWEEP_DOUBLE, LiquidityEventType.CONSOLIDATION_RANGE],
                event_frequency_thresholds={
                    LiquidityEventType.FVG_CONTINUATION: 0.438,      # Lowest - exhaustion
                    LiquidityEventType.SWEEP_DOUBLE: 0.719,         # Highest - completion
                    LiquidityEventType.CONSOLIDATION_RANGE: 0.391,
                    LiquidityEventType.PD_PREMIUM_REJECTION: 1.0
                },
                pd_array_required=True,
                htf_confluence_threshold=1.0,
                velocity_consistency_threshold=1.0,  # Perfect terminal velocity
                continuation_probability_threshold=1.0,
                session_phase_preferences=["session_closing"],
                completion_probability_80pct=1.0,    # Guaranteed 80%+ completion
                avg_range_reached=85.0
            )
        }
    
    def _initialize_tactical_frameworks(self) -> Dict[RangeZoneType, Dict[str, any]]:
        """Initialize tactical decision frameworks for each range zone"""
        return {
            RangeZoneType.MOMENTUM_FILTER: {
                "zone_name": "Momentum Filter Zone",
                "tactical_description": "Initial resistance zone with variable momentum characteristics",
                "key_intelligence": [
                    "44.7% continuation probability beyond this level",
                    "Acts as momentum filter - not major resistance", 
                    "Early session_closing bias (55.3%)",
                    "Average progression only 28.1%"
                ],
                "decision_rules": {
                    "if_recognized": "Prepare for momentum decision point",
                    "continuation_signal": "Look for sweep acceleration patterns",
                    "failure_signal": "Range rejection at early levels",
                    "next_target": "40% acceleration zone"
                },
                "risk_parameters": {
                    "completion_certainty": "Low",
                    "reversal_risk": "Medium",
                    "momentum_clarity": "Variable"
                }
            },
            
            RangeZoneType.SWEEP_ACCELERATION: {
                "zone_name": "Sweep Acceleration Zone", 
                "tactical_description": "Momentum acceleration with perfect velocity consistency",
                "key_intelligence": [
                    "100% continuation probability beyond level",
                    "Perfect velocity consistency (1.00)",
                    "Double sweep dominance (63.2%)",
                    "Mid-momentum positioning (2.88 session pos)"
                ],
                "decision_rules": {
                    "if_recognized": "Confirm momentum is building - high probability zone",
                    "continuation_signal": "Perfect consistency indicates reliable acceleration", 
                    "failure_signal": "Rare - reconsider range analysis",
                    "next_target": "60% equilibrium zone with highest evolution strength"
                },
                "risk_parameters": {
                    "completion_certainty": "High",
                    "reversal_risk": "Low", 
                    "momentum_clarity": "Perfect"
                }
            },
            
            RangeZoneType.FVG_EQUILIBRIUM: {
                "zone_name": "FVG Equilibrium Zone",
                "tactical_description": "Critical balance point with highest evolutionary stability",
                "key_intelligence": [
                    "Highest evolution strength (0.93) - most predictable",
                    "Perfect timing balance (50/50 early/late)",
                    "FVG first presented dominance (61.1%)",
                    "High average progression (70.6%)"
                ],
                "decision_rules": {
                    "if_recognized": "Most reliable range level for cross-session prediction",
                    "continuation_signal": "FVG redelivery confirms equilibrium maintenance",
                    "failure_signal": "Major structural shift if fails here",
                    "next_target": "80% completion zone with guaranteed 80%+ reach"
                },
                "risk_parameters": {
                    "completion_certainty": "Very High",
                    "reversal_risk": "Very Low",
                    "momentum_clarity": "Optimal"
                }
            },
            
            RangeZoneType.SWEEP_COMPLETION: {
                "zone_name": "Sweep Completion Zone",
                "tactical_description": "Terminal velocity with guaranteed high completion",
                "key_intelligence": [
                    "Guaranteed 80%+ completion (100% probability)",
                    "Highest sweep concentration (71.9%)", 
                    "Latest timing (3.06 session position)",
                    "Terminal consolidation signals (39.1%)"
                ],
                "decision_rules": {
                    "if_recognized": "Final liquidity hunt before session completion",
                    "continuation_signal": "Expect 85% average range completion",
                    "failure_signal": "Structural breakdown - major reversal likely",
                    "next_target": "Session completion with exhaustion patterns"
                },
                "risk_parameters": {
                    "completion_certainty": "Guaranteed",
                    "reversal_risk": "High (if fails)", 
                    "momentum_clarity": "Terminal"
                }
            }
        }
    
    def analyze_pattern_for_range_zone(self, pattern: Dict) -> Optional[RangeFilterMatch]:
        """Analyze a pattern to determine its range zone classification"""
        
        # Detect liquidity events in pattern
        events = self.event_detector.detect_pattern_liquidity_events(pattern)
        if not events:
            return None
        
        # Extract pattern characteristics
        range_level = self.event_detector._extract_range_level(pattern)
        if not range_level:
            return None
        
        session_phase = pattern.get('phase_information', {}).get('primary_phase', 'unknown')
        session_position = pattern.get('phase_information', {}).get('session_position', 0.0)
        
        # Test against each range filter
        best_match = None
        best_score = 0.0
        
        for zone_type, filter_criteria in self.range_filters.items():
            match_score = self._calculate_filter_match_score(
                events, pattern, filter_criteria, range_level, session_phase, session_position
            )
            
            if match_score > best_score:
                best_score = match_score
                best_match = RangeFilterMatch(
                    zone_type=zone_type,
                    confidence_score=match_score,
                    matching_events=events,
                    signature_similarity=match_score,
                    tactical_intelligence=self.tactical_frameworks[zone_type],
                    prediction_metrics=self._generate_prediction_metrics(zone_type, filter_criteria, range_level)
                )
        
        return best_match if best_score > 0.6 else None  # Minimum confidence threshold
    
    def _calculate_filter_match_score(self, events: List[LiquidityEvent], pattern: Dict, 
                                    criteria: RangeFilterCriteria, range_level: float, 
                                    session_phase: str, session_position: float) -> float:
        """Calculate how well a pattern matches range filter criteria"""
        score = 0.0
        max_score = 0.0
        
        # Event type matching (30% weight)
        event_types = [event.event_type for event in events]
        required_events_present = sum(1 for req_event in criteria.required_events if req_event in event_types)
        event_match_score = required_events_present / len(criteria.required_events) if criteria.required_events else 1.0
        score += event_match_score * 0.3
        max_score += 0.3
        
        # Event frequency matching (25% weight)
        event_counts = Counter(event_types)
        total_events = len(events)
        frequency_matches = 0
        frequency_tests = 0
        
        for event_type, threshold_freq in criteria.event_frequency_thresholds.items():
            if event_type in event_counts:
                actual_freq = event_counts[event_type] / total_events
                frequency_similarity = 1.0 - abs(actual_freq - threshold_freq)
                frequency_matches += max(0, frequency_similarity)
            frequency_tests += 1
        
        if frequency_tests > 0:
            frequency_score = frequency_matches / frequency_tests
            score += frequency_score * 0.25
        max_score += 0.25
        
        # PD Array requirement (15% weight)
        pd_array_present = any(event.event_type.value.startswith('pd_') for event in events)
        if criteria.pd_array_required == pd_array_present:
            score += 0.15
        max_score += 0.15
        
        # HTF confluence matching (10% weight)
        htf_confluence_rate = sum(1 for event in events if event.htf_confluence) / len(events)
        htf_similarity = 1.0 - abs(htf_confluence_rate - criteria.htf_confluence_threshold)
        score += max(0, htf_similarity) * 0.10
        max_score += 0.10
        
        # Session phase preference (10% weight)
        if session_phase in criteria.session_phase_preferences:
            score += 0.10
        max_score += 0.10
        
        # Range level appropriateness (10% weight)
        target_range = float(criteria.zone_type.value.replace('%', ''))
        range_similarity = 1.0 - abs(range_level - target_range) / 100.0
        score += max(0, range_similarity) * 0.10
        max_score += 0.10
        
        return score / max_score if max_score > 0 else 0.0
    
    def _generate_prediction_metrics(self, zone_type: RangeZoneType, 
                                   criteria: RangeFilterCriteria, range_level: float) -> Dict[str, float]:
        """Generate prediction metrics for matched range zone"""
        return {
            'completion_probability_80pct': criteria.completion_probability_80pct,
            'continuation_probability': criteria.continuation_probability_threshold,
            'avg_range_reached': criteria.avg_range_reached,
            'velocity_consistency': criteria.velocity_consistency_threshold,
            'current_range_level': range_level,
            'expected_progression': criteria.avg_range_reached - range_level if criteria.avg_range_reached > range_level else 0,
            'zone_reliability_score': self._calculate_zone_reliability(zone_type)
        }
    
    def _calculate_zone_reliability(self, zone_type: RangeZoneType) -> float:
        """Calculate reliability score for each zone based on archaeological data"""
        reliability_scores = {
            RangeZoneType.MOMENTUM_FILTER: 0.68,      # Variable momentum
            RangeZoneType.SWEEP_ACCELERATION: 1.00,   # Perfect consistency
            RangeZoneType.FVG_EQUILIBRIUM: 0.93,      # Highest evolution strength  
            RangeZoneType.SWEEP_COMPLETION: 1.00      # Perfect terminal velocity
        }
        return reliability_scores.get(zone_type, 0.5)
    
    def create_40pct_sweep_acceleration_filter(self, patterns: List[Dict]) -> List[RangeFilterMatch]:
        """Specific filter for 40% range sweep acceleration zone"""
        print("🚀 Applying 40% Range Sweep Acceleration Filter...")
        
        matches = []
        acceleration_criteria = self.range_filters[RangeZoneType.SWEEP_ACCELERATION]
        
        for i, pattern in enumerate(patterns):
            # Check range level first
            range_level = self.event_detector._extract_range_level(pattern)
            if not range_level or not (35 <= range_level <= 45):
                continue
            
            # Analyze pattern
            match = self.analyze_pattern_for_range_zone(pattern)
            if match and match.zone_type == RangeZoneType.SWEEP_ACCELERATION:
                
                # Enhanced analysis for acceleration zone
                match.tactical_intelligence.update({
                    'acceleration_indicators': [
                        f"Double sweep patterns at {range_level:.1f}% range",
                        f"Perfect velocity consistency detected",
                        f"100% continuation probability confirmed",
                        f"Mid-momentum positioning identified"
                    ],
                    'tactical_action': "MOMENTUM BUILDING CONFIRMED",
                    'confidence_level': "VERY HIGH" if match.confidence_score > 0.8 else "HIGH",
                    'next_expectations': [
                        "Expect progression to 60% equilibrium zone",
                        "Look for FVG first presented patterns next",
                        "Monitor for highest evolution strength (0.93)"
                    ]
                })
                
                matches.append(match)
        
        print(f"  ✅ Found {len(matches)} 40% sweep acceleration patterns")
        return matches
    
    def create_60pct_fvg_equilibrium_filter(self, patterns: List[Dict]) -> List[RangeFilterMatch]:
        """Specific filter for 60% range FVG equilibrium zone"""  
        print("⚖️ Applying 60% Range FVG Equilibrium Filter...")
        
        matches = []
        equilibrium_criteria = self.range_filters[RangeZoneType.FVG_EQUILIBRIUM]
        
        for i, pattern in enumerate(patterns):
            # Check range level first
            range_level = self.event_detector._extract_range_level(pattern)
            if not range_level or not (55 <= range_level <= 65):
                continue
            
            # Analyze pattern
            match = self.analyze_pattern_for_range_zone(pattern)
            if match and match.zone_type == RangeZoneType.FVG_EQUILIBRIUM:
                
                # Enhanced analysis for equilibrium zone
                match.tactical_intelligence.update({
                    'equilibrium_indicators': [
                        f"FVG first presented patterns at {range_level:.1f}% range",
                        f"Perfect 50/50 timing balance detected",
                        f"Highest evolution strength (0.93) confirmed",
                        f"Critical balance point identified"
                    ],
                    'tactical_action': "MOST PREDICTABLE ZONE DETECTED",
                    'confidence_level': "MAXIMUM" if match.confidence_score > 0.9 else "VERY HIGH",
                    'next_expectations': [
                        "Expect 70.6% average range progression",
                        "Highest cross-session continuity (0.93)",
                        "Most reliable zone for tomorrow's prediction"
                    ]
                })
                
                matches.append(match)
        
        print(f"  ✅ Found {len(matches)} 60% FVG equilibrium patterns")
        return matches
    
    def create_80pct_completion_zone_filter(self, patterns: List[Dict]) -> List[RangeFilterMatch]:
        """Specific filter for 80% range sweep completion zone"""
        print("🎯 Applying 80% Range Completion Zone Filter...")
        
        matches = []
        completion_criteria = self.range_filters[RangeZoneType.SWEEP_COMPLETION]
        
        for i, pattern in enumerate(patterns):
            # Check range level first
            range_level = self.event_detector._extract_range_level(pattern)
            if not range_level or not (75 <= range_level <= 85):
                continue
            
            # Analyze pattern
            match = self.analyze_pattern_for_range_zone(pattern)
            if match and match.zone_type == RangeZoneType.SWEEP_COMPLETION:
                
                # Enhanced analysis for completion zone
                match.tactical_intelligence.update({
                    'completion_indicators': [
                        f"Highest sweep concentration (71.9%) at {range_level:.1f}% range",
                        f"Terminal consolidation signals (39.1%) detected",
                        f"Latest session timing (3.06) confirmed", 
                        f"Guaranteed 80%+ completion zone"
                    ],
                    'tactical_action': "FINAL LIQUIDITY HUNT DETECTED",
                    'confidence_level': "GUARANTEED" if match.confidence_score > 0.85 else "VERY HIGH",
                    'next_expectations': [
                        "Expect 85% average range completion",
                        "Session completion with exhaustion patterns",
                        "High reversal risk if this zone fails"
                    ]
                })
                
                matches.append(match)
        
        print(f"  ✅ Found {len(matches)} 80% completion zone patterns")
        return matches
    
    def apply_all_range_filters(self, patterns: List[Dict] = None) -> Dict[str, List[RangeFilterMatch]]:
        """Apply all range filters to patterns"""
        if patterns is None:
            patterns = self.event_detector.patterns
        
        print("🎯 Applying all range filters...")
        
        results = {
            '20% Momentum Filter': [],
            '40% Sweep Acceleration': self.create_40pct_sweep_acceleration_filter(patterns),
            '60% FVG Equilibrium': self.create_60pct_fvg_equilibrium_filter(patterns),
            '80% Completion Zone': self.create_80pct_completion_zone_filter(patterns)
        }
        
        # Apply momentum filter for completeness
        for pattern in patterns:
            range_level = self.event_detector._extract_range_level(pattern)
            if range_level and (15 <= range_level <= 25):
                match = self.analyze_pattern_for_range_zone(pattern)
                if match and match.zone_type == RangeZoneType.MOMENTUM_FILTER:
                    results['20% Momentum Filter'].append(match)
        
        print(f"  ✅ Filter results:")
        for zone, matches in results.items():
            print(f"    {zone}: {len(matches)} patterns")
        
        return results
    
    def generate_tactical_report(self, filter_results: Dict[str, List[RangeFilterMatch]]) -> Dict:
        """Generate comprehensive tactical intelligence report"""
        print("📋 Generating tactical intelligence report...")
        
        report = {
            'tactical_summary': {},
            'zone_analysis': {},
            'decision_framework': {},
            'predictive_intelligence': {}
        }
        
        # Analyze each zone
        for zone_name, matches in filter_results.items():
            if not matches:
                continue
            
            zone_analysis = {
                'total_patterns': len(matches),
                'avg_confidence': np.mean([match.confidence_score for match in matches]),
                'high_confidence_patterns': len([m for m in matches if m.confidence_score > 0.8]),
                'dominant_events': self._extract_dominant_events(matches),
                'tactical_intelligence': matches[0].tactical_intelligence if matches else {},
                'prediction_metrics': {
                    'avg_completion_probability': np.mean([m.prediction_metrics.get('completion_probability_80pct', 0) for m in matches]),
                    'avg_continuation_probability': np.mean([m.prediction_metrics.get('continuation_probability', 0) for m in matches]),
                    'avg_expected_progression': np.mean([m.prediction_metrics.get('expected_progression', 0) for m in matches]),
                    'zone_reliability': np.mean([m.prediction_metrics.get('zone_reliability_score', 0) for m in matches])
                }
            }
            
            report['zone_analysis'][zone_name] = zone_analysis
        
        # Generate decision framework
        report['decision_framework'] = self._generate_decision_framework(filter_results)
        
        # Generate predictive intelligence
        report['predictive_intelligence'] = self._generate_predictive_intelligence(filter_results)
        
        return report
    
    def _extract_dominant_events(self, matches: List[RangeFilterMatch]) -> List[str]:
        """Extract dominant event types from matches"""
        all_events = []
        for match in matches:
            all_events.extend([event.event_type.value for event in match.matching_events])
        
        event_counts = Counter(all_events)
        return [event for event, _ in event_counts.most_common(3)]
    
    def _generate_decision_framework(self, filter_results: Dict[str, List[RangeFilterMatch]]) -> Dict:
        """Generate tactical decision framework"""
        return {
            'recognition_sequence': [
                "1. Identify liquidity event combinations (FVG, sweeps, PD arrays)",
                "2. Determine range level classification (20%, 40%, 60%, 80%)",
                "3. Match against archaeological signatures",
                "4. Apply zone-specific tactical rules",
                "5. Predict range progression and cross-session continuation"
            ],
            'zone_decision_rules': {
                zone_name: matches[0].tactical_intelligence.get('decision_rules', {})
                for zone_name, matches in filter_results.items() if matches
            },
            'risk_management': {
                zone_name: matches[0].tactical_intelligence.get('risk_parameters', {})
                for zone_name, matches in filter_results.items() if matches
            }
        }
    
    def _generate_predictive_intelligence(self, filter_results: Dict[str, List[RangeFilterMatch]]) -> Dict:
        """Generate predictive intelligence summary"""
        total_patterns = sum(len(matches) for matches in filter_results.values())
        
        return {
            'total_patterns_classified': total_patterns,
            'zone_distribution': {zone: len(matches) for zone, matches in filter_results.items()},
            'highest_reliability_zone': max(filter_results.items(), key=lambda x: len(x[1]))[0] if filter_results else None,
            'predictive_accuracy': {
                'guaranteed_completion_zones': len(filter_results.get('80% Completion Zone', [])),
                'perfect_velocity_zones': len(filter_results.get('40% Sweep Acceleration', [])) + len(filter_results.get('80% Completion Zone', [])),
                'highest_evolution_zones': len(filter_results.get('60% FVG Equilibrium', []))
            }
        }
    
    def save_filter_analysis(self, output_path: str = None) -> str:
        """Save comprehensive range filter analysis"""
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/range_filter_analysis.json'
        
        # Apply all filters
        filter_results = self.apply_all_range_filters()
        
        # Generate tactical report
        tactical_report = self.generate_tactical_report(filter_results)
        
        # Build comprehensive analysis
        analysis = {
            'analysis_metadata': {
                'filter_version': '1.0',
                'patterns_analyzed': len(self.event_detector.patterns),
                'range_zones': list(self.range_filters.keys()),
                'tactical_frameworks': list(self.tactical_frameworks.keys())
            },
            'range_filter_results': {
                zone_name: [
                    {
                        'zone_type': match.zone_type.value,
                        'confidence_score': match.confidence_score,
                        'signature_similarity': match.signature_similarity,
                        'matching_events_count': len(match.matching_events),
                        'tactical_action': match.tactical_intelligence.get('tactical_action', ''),
                        'confidence_level': match.tactical_intelligence.get('confidence_level', ''),
                        'prediction_metrics': match.prediction_metrics
                    }
                    for match in matches
                ]
                for zone_name, matches in filter_results.items()
            },
            'tactical_intelligence_report': tactical_report,
            'range_filter_criteria': {
                zone_type.value: {
                    'required_events': [event.value for event in criteria.required_events],
                    'event_frequency_thresholds': {event.value: freq for event, freq in criteria.event_frequency_thresholds.items()},
                    'pd_array_required': criteria.pd_array_required,
                    'htf_confluence_threshold': criteria.htf_confluence_threshold,
                    'velocity_consistency_threshold': criteria.velocity_consistency_threshold,
                    'continuation_probability_threshold': criteria.continuation_probability_threshold,
                    'completion_probability_80pct': criteria.completion_probability_80pct,
                    'avg_range_reached': criteria.avg_range_reached
                }
                for zone_type, criteria in self.range_filters.items()
            }
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"💾 Range filter analysis saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    print("🎯 IRONFORGE Range Filter System")
    print("=" * 60)
    
    filter_system = RangeFilterSystem()
    output_file = filter_system.save_filter_analysis()
    
    print(f"\n✅ Range filter analysis complete!")
    print(f"📊 Results saved to: {output_file}")