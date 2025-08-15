#!/usr/bin/env python3
"""
IRONFORGE Event Precursor Detection - Innovation Architect Implementation
========================================================================

Detects event precursor patterns using temporal cycles + structural context
for enhanced cascade timing prediction and archaeological discovery.
"""

import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass 
class PrecursorEvent:
    """Container for detected precursor event"""
    event_type: str
    confidence: float
    expected_timeframe: str
    contributing_factors: Dict[str, float]
    temporal_alignment: Dict[str, Any]
    structural_confluence: Dict[str, Any]
    relativity_consistency: Dict[str, Any]

class EventPrecursorDetector:
    """
    Innovation Architect implementation for event precursor detection
    Uses 37D temporal cycles + structural context for pattern-based event prediction
    """
    
    def __init__(self, confidence_threshold=0.6, temporal_weight=0.4, 
                 structural_weight=0.4, relativity_weight=0.2):
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.temporal_weight = temporal_weight
        self.structural_weight = structural_weight
        self.relativity_weight = relativity_weight
        
        # Event type definitions and precursor patterns
        self.event_types = {
            'cascade': {
                'description': 'Major price cascade/momentum move',
                'temporal_patterns': ['weekly_alignment', 'monthly_confluence'],
                'structural_patterns': ['sweep_to_fvg', 'imbalance_to_htf'],
                'typical_timeframe': '15-45 minutes'
            },
            'breakout': {
                'description': 'Range breakout with continuation',
                'temporal_patterns': ['session_boundary', 'cycle_confluence'],
                'structural_patterns': ['liquidity_sweep', 'structural_break'],
                'typical_timeframe': '5-30 minutes'
            },
            'reversal': {
                'description': 'Directional reversal at key level',
                'temporal_patterns': ['temporal_echo', 'session_reversal'],
                'structural_patterns': ['support_rejection', 'confluence_hold'],
                'typical_timeframe': '10-60 minutes'
            }
        }
        
    def detect_precursors(self, session_graph: Dict, 
                         temporal_cycles: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect event precursor patterns using temporal cycles + structural context
        
        Args:
            session_graph: Enhanced graph with 37D features and structural edges
            temporal_cycles: Optional explicit temporal cycle data
            
        Returns:
            Dictionary with precursor analysis results
        """
        
        # Extract temporal cycle information
        if temporal_cycles is None:
            temporal_cycles = self._extract_temporal_cycles(session_graph)
        
        # Analyze structural context from 4 edge types
        structural_context = self._analyze_structural_context(session_graph)
        
        # Check price relativity consistency
        relativity_analysis = self._analyze_relativity_consistency(session_graph)
        
        # Detect precursor patterns for each event type
        precursor_results = {}
        
        for event_type, event_config in self.event_types.items():
            precursor = self._detect_event_precursor(
                event_type, event_config, temporal_cycles, 
                structural_context, relativity_analysis
            )
            
            if precursor and precursor.confidence >= self.confidence_threshold:
                precursor_results[event_type] = precursor
        
        # Generate comprehensive precursor index
        precursor_index = self._generate_precursor_index(precursor_results)
        
        return {
            'precursor_index': precursor_index,
            'detected_precursors': precursor_results,
            'temporal_cycles': temporal_cycles,
            'structural_context': structural_context,
            'relativity_analysis': relativity_analysis,
            'detection_timestamp': datetime.now().isoformat()
        }
    
    def _extract_temporal_cycles(self, session_graph: Dict) -> Dict[str, Any]:
        """Extract temporal cycle information from session graph"""
        
        # Get temporal cycle data from rich node features
        rich_features = session_graph.get('rich_node_features', [])
        
        if not rich_features:
            return self._default_temporal_cycles()
        
        # Extract temporal cycle features from first node (session-level)
        first_feature = rich_features[0]
        
        temporal_cycles = {
            'week_of_month': getattr(first_feature, 'week_of_month', 0),
            'month_of_year': getattr(first_feature, 'month_of_year', 0),
            'day_of_week_cycle': getattr(first_feature, 'day_of_week_cycle', 0),
            'session_position': getattr(first_feature, 'session_position', 0.5),
            'weekend_proximity': getattr(first_feature, 'weekend_proximity', 0.5)
        }
        
        # Add derived cycle analysis
        temporal_cycles.update({
            'weekly_pattern_strength': self._calculate_weekly_pattern_strength(temporal_cycles),
            'monthly_pattern_strength': self._calculate_monthly_pattern_strength(temporal_cycles),
            'cycle_confluence_score': self._calculate_cycle_confluence(temporal_cycles)
        })
        
        return temporal_cycles
    
    def _analyze_structural_context(self, session_graph: Dict) -> Dict[str, Any]:
        """Analyze structural context from all 4 edge types"""
        
        edges = session_graph.get('edges', {})
        
        # Count edge types
        edge_type_counts = {}
        for edge_type in ['temporal', 'scale', 'structural_context', 'discovered']:
            edge_type_counts[edge_type] = len(edges.get(edge_type, []))
        
        # Analyze structural patterns
        structural_patterns = {}
        
        # Check for causal sequences (sweep → fvg)
        structural_patterns['sweep_to_fvg_count'] = self._count_causal_sequences(
            session_graph, 'sweep', 'first_fvg_after_sweep'
        )
        
        # Check for structural alignments (imbalance → htf)
        structural_patterns['imbalance_to_htf_count'] = self._count_structural_alignments(
            session_graph, 'imbalance_zone', 'htf_range_midpoint'
        )
        
        # Check for boundary interactions
        structural_patterns['cascade_to_boundary_count'] = self._count_boundary_interactions(
            session_graph, 'cascade_origin', 'session_boundary'
        )
        
        # Check for reinforcement patterns
        structural_patterns['liquidity_support_count'] = self._count_reinforcement_patterns(
            session_graph, 'liquidity_cluster', 'structural_support'
        )
        
        # Calculate structural confluence score
        total_structural_edges = edge_type_counts.get('structural_context', 0)
        structural_patterns['confluence_score'] = min(1.0, total_structural_edges / 10.0)
        
        return {
            'edge_type_counts': edge_type_counts,
            'structural_patterns': structural_patterns,
            'total_nodes': len(session_graph.get('rich_node_features', [])),
            'structural_density': total_structural_edges / max(1, len(session_graph.get('rich_node_features', []))),
        }
    
    def _analyze_relativity_consistency(self, session_graph: Dict) -> Dict[str, Any]:
        """Analyze price relativity consistency across session"""
        
        rich_features = session_graph.get('rich_node_features', [])
        
        if not rich_features:
            return self._default_relativity_analysis()
        
        # Extract relativity features
        relativity_data = {
            'normalized_price': [],
            'pct_from_open': [],
            'pct_from_high': [],
            'pct_from_low': [],
            'price_to_HTF_ratio': [],
            'normalized_time': []
        }
        
        for feature in rich_features:
            for key in relativity_data.keys():
                value = getattr(feature, key, 0.0)
                relativity_data[key].append(value)
        
        # Calculate consistency metrics
        consistency_metrics = {}
        
        for feature_name, values in relativity_data.items():
            if values:
                mean_val = np.mean(values)
                variance = np.var(values)
                consistency = 1.0 / (1.0 + variance)  # High consistency = low variance
                
                consistency_metrics[f"{feature_name}_consistency"] = consistency
                consistency_metrics[f"{feature_name}_mean"] = mean_val
        
        # Calculate overall relativity consistency
        all_consistencies = [v for k, v in consistency_metrics.items() if '_consistency' in k]
        overall_consistency = np.mean(all_consistencies) if all_consistencies else 0.5
        
        return {
            'individual_consistencies': consistency_metrics,
            'overall_consistency': overall_consistency,
            'relativity_stability': overall_consistency > 0.7
        }
    
    def _detect_event_precursor(self, event_type: str, event_config: Dict, 
                               temporal_cycles: Dict, structural_context: Dict,
                               relativity_analysis: Dict) -> Optional[PrecursorEvent]:
        """Detect precursor for specific event type"""
        
        # Calculate component scores
        temporal_score = self._calculate_temporal_alignment_score(
            event_type, temporal_cycles, event_config['temporal_patterns']
        )
        
        structural_score = self._calculate_structural_confluence_score(
            event_type, structural_context, event_config['structural_patterns']
        )
        
        relativity_score = self._calculate_relativity_consistency_score(
            event_type, relativity_analysis
        )
        
        # Weight and combine scores
        confidence = (
            self.temporal_weight * temporal_score +
            self.structural_weight * structural_score +
            self.relativity_weight * relativity_score
        )
        
        # Determine expected timeframe based on component strengths
        expected_timeframe = self._determine_expected_timeframe(
            event_config['typical_timeframe'], temporal_score, structural_score
        )
        
        # Create contributing factors breakdown
        contributing_factors = {
            'temporal_alignment': temporal_score,
            'structural_confluence': structural_score,
            'relativity_consistency': relativity_score,
            'overall_confidence': confidence
        }
        
        return PrecursorEvent(
            event_type=event_type,
            confidence=confidence,
            expected_timeframe=expected_timeframe,
            contributing_factors=contributing_factors,
            temporal_alignment=self._get_temporal_alignment_details(temporal_cycles),
            structural_confluence=self._get_structural_confluence_details(structural_context),
            relativity_consistency=self._get_relativity_consistency_details(relativity_analysis)
        )
    
    def _calculate_temporal_alignment_score(self, event_type: str, 
                                          temporal_cycles: Dict, patterns: List[str]) -> float:
        """Calculate temporal alignment score for event type"""
        
        score = 0.0
        
        # Weekly alignment patterns
        if 'weekly_alignment' in patterns:
            week_strength = temporal_cycles.get('weekly_pattern_strength', 0.0)
            score += 0.3 * week_strength
        
        # Monthly confluence patterns  
        if 'monthly_confluence' in patterns:
            month_strength = temporal_cycles.get('monthly_pattern_strength', 0.0)
            score += 0.3 * month_strength
        
        # Session boundary patterns
        if 'session_boundary' in patterns:
            session_pos = temporal_cycles.get('session_position', 0.5)
            # Higher score near session boundaries (0.0 or 1.0)
            boundary_score = max(0.0, 1.0 - 2.0 * abs(session_pos - 0.5))
            score += 0.2 * boundary_score
        
        # Cycle confluence patterns
        if 'cycle_confluence' in patterns:
            confluence_score = temporal_cycles.get('cycle_confluence_score', 0.0)
            score += 0.4 * confluence_score
        
        # Temporal echo patterns
        if 'temporal_echo' in patterns:
            # Use day_of_week for temporal echo strength
            day_of_week = temporal_cycles.get('day_of_week_cycle', 0)
            # Higher score for Mon/Wed/Fri (1, 3, 5)
            echo_score = 0.8 if day_of_week in [0, 2, 4] else 0.4
            score += 0.3 * echo_score
        
        return min(1.0, score)
    
    def _calculate_structural_confluence_score(self, event_type: str, 
                                             structural_context: Dict, patterns: List[str]) -> float:
        """Calculate structural confluence score for event type"""
        
        score = 0.0
        structural_patterns = structural_context.get('structural_patterns', {})
        
        # Sweep to FVG patterns
        if 'sweep_to_fvg' in patterns:
            sweep_count = structural_patterns.get('sweep_to_fvg_count', 0)
            score += 0.3 * min(1.0, sweep_count / 3.0)
        
        # Imbalance to HTF patterns
        if 'imbalance_to_htf' in patterns:
            imbalance_count = structural_patterns.get('imbalance_to_htf_count', 0)
            score += 0.3 * min(1.0, imbalance_count / 2.0)
        
        # Liquidity sweep patterns
        if 'liquidity_sweep' in patterns:
            liquidity_count = structural_patterns.get('liquidity_support_count', 0)
            score += 0.2 * min(1.0, liquidity_count / 2.0)
        
        # Structural break patterns
        if 'structural_break' in patterns:
            cascade_count = structural_patterns.get('cascade_to_boundary_count', 0)
            score += 0.4 * min(1.0, cascade_count / 1.0)
        
        # Support rejection patterns
        if 'support_rejection' in patterns:
            support_count = structural_patterns.get('liquidity_support_count', 0)
            score += 0.3 * min(1.0, support_count / 2.0)
        
        # Confluence hold patterns
        if 'confluence_hold' in patterns:
            confluence_score = structural_patterns.get('confluence_score', 0.0)
            score += 0.4 * confluence_score
        
        return min(1.0, score)
    
    def _calculate_relativity_consistency_score(self, event_type: str, 
                                              relativity_analysis: Dict) -> float:
        """Calculate relativity consistency score for event type"""
        
        overall_consistency = relativity_analysis.get('overall_consistency', 0.5)
        stability = relativity_analysis.get('relativity_stability', False)
        
        base_score = overall_consistency
        
        # Bonus for stability
        if stability:
            base_score += 0.2
        
        # Event-specific adjustments
        if event_type == 'cascade':
            # Cascades benefit from consistent price relationships
            base_score *= 1.2
        elif event_type == 'reversal':
            # Reversals can work with some inconsistency
            base_score *= 0.8
        
        return min(1.0, base_score)
    
    def _determine_expected_timeframe(self, typical_timeframe: str, 
                                    temporal_score: float, structural_score: float) -> str:
        """Determine expected timeframe based on component strengths"""
        
        # Parse typical timeframe (e.g., "15-45 minutes")
        parts = typical_timeframe.split('-')
        if len(parts) == 2:
            min_time = int(parts[0])
            max_time = int(parts[1].split()[0])  # Remove "minutes"
        else:
            min_time = 15
            max_time = 45
        
        # Adjust based on scores
        score_avg = (temporal_score + structural_score) / 2.0
        
        if score_avg > 0.8:
            # High confidence = shorter timeframe
            expected = min_time + 0.2 * (max_time - min_time)
        elif score_avg > 0.6:
            # Medium confidence = middle timeframe
            expected = min_time + 0.5 * (max_time - min_time)
        else:
            # Lower confidence = longer timeframe
            expected = min_time + 0.8 * (max_time - min_time)
        
        return f"{int(expected)} minutes"
    
    def _generate_precursor_index(self, precursor_results: Dict[str, PrecursorEvent]) -> Dict[str, float]:
        """Generate comprehensive precursor index"""
        
        precursor_index = {}
        
        for event_type, precursor in precursor_results.items():
            # Use event type with probability suffix
            key = f"{event_type}_probability"
            precursor_index[key] = precursor.confidence
        
        # Add overall precursor activity score
        if precursor_results:
            all_confidences = [p.confidence for p in precursor_results.values()]
            precursor_index['overall_precursor_activity'] = max(all_confidences)
        else:
            precursor_index['overall_precursor_activity'] = 0.0
        
        return precursor_index
    
    # Helper methods for detailed analysis
    
    def _calculate_weekly_pattern_strength(self, temporal_cycles: Dict) -> float:
        """Calculate weekly pattern strength from temporal cycles"""
        week_of_month = temporal_cycles.get('week_of_month', 0)
        day_of_week = temporal_cycles.get('day_of_week_cycle', 0)
        
        # Higher strength for week 1, 3 (start/middle) and Mon/Wed/Fri
        week_strength = 0.8 if week_of_month in [1, 3] else 0.5
        day_strength = 0.8 if day_of_week in [0, 2, 4] else 0.4
        
        return (week_strength + day_strength) / 2.0
    
    def _calculate_monthly_pattern_strength(self, temporal_cycles: Dict) -> float:
        """Calculate monthly pattern strength from temporal cycles"""
        month_of_year = temporal_cycles.get('month_of_year', 0)
        
        # Higher strength for significant months (Mar, Jun, Sep, Dec = quarter ends)
        significant_months = [3, 6, 9, 12]
        
        if month_of_year in significant_months:
            return 0.9
        elif month_of_year in [1]:  # January = new year
            return 0.8
        else:
            return 0.4
    
    def _calculate_cycle_confluence(self, temporal_cycles: Dict) -> float:
        """Calculate overall cycle confluence score"""
        weekly_strength = temporal_cycles.get('weekly_pattern_strength', 0.0)
        monthly_strength = temporal_cycles.get('monthly_pattern_strength', 0.0)
        
        # Confluence = both weekly and monthly patterns strong
        confluence = weekly_strength * monthly_strength
        
        # Bonus for perfect alignment
        week = temporal_cycles.get('week_of_month', 0)
        month = temporal_cycles.get('month_of_year', 0)
        day = temporal_cycles.get('day_of_week_cycle', 0)
        
        # Special confluences
        if month == 1 and week == 1 and day == 0:  # Jan Week 1 Monday
            confluence += 0.3
        elif month in [3, 6, 9, 12] and week >= 3 and day == 4:  # Quarter end Friday
            confluence += 0.2
        
        return min(1.0, confluence)
    
    def _count_causal_sequences(self, session_graph: Dict, 
                               source_archetype: str, target_archetype: str) -> int:
        """Count causal sequence patterns in structural edges"""
        # This would analyze structural_context edges for specific archetype sequences
        # For now, return placeholder based on edge presence
        structural_edges = session_graph.get('edges', {}).get('structural_context', [])
        return min(3, len(structural_edges))
    
    def _count_structural_alignments(self, session_graph: Dict,
                                   source_archetype: str, target_archetype: str) -> int:
        """Count structural alignment patterns"""
        structural_edges = session_graph.get('edges', {}).get('structural_context', [])
        return min(2, len(structural_edges) // 2)
    
    def _count_boundary_interactions(self, session_graph: Dict,
                                   source_archetype: str, target_archetype: str) -> int:
        """Count boundary interaction patterns"""
        temporal_edges = session_graph.get('edges', {}).get('temporal', [])
        return min(1, len(temporal_edges) // 5)
    
    def _count_reinforcement_patterns(self, session_graph: Dict,
                                     source_archetype: str, target_archetype: str) -> int:
        """Count reinforcement patterns"""
        scale_edges = session_graph.get('edges', {}).get('scale', [])
        return min(2, len(scale_edges) // 3)
    
    def _get_temporal_alignment_details(self, temporal_cycles: Dict) -> Dict[str, Any]:
        """Get detailed temporal alignment information"""
        return {
            'week_of_month': temporal_cycles.get('week_of_month', 0),
            'month_of_year': temporal_cycles.get('month_of_year', 0),
            'day_of_week': temporal_cycles.get('day_of_week_cycle', 0),
            'weekly_strength': temporal_cycles.get('weekly_pattern_strength', 0.0),
            'monthly_strength': temporal_cycles.get('monthly_pattern_strength', 0.0),
            'confluence_score': temporal_cycles.get('cycle_confluence_score', 0.0)
        }
    
    def _get_structural_confluence_details(self, structural_context: Dict) -> Dict[str, Any]:
        """Get detailed structural confluence information"""
        return {
            'edge_type_counts': structural_context.get('edge_type_counts', {}),
            'structural_patterns': structural_context.get('structural_patterns', {}),
            'structural_density': structural_context.get('structural_density', 0.0)
        }
    
    def _get_relativity_consistency_details(self, relativity_analysis: Dict) -> Dict[str, Any]:
        """Get detailed relativity consistency information"""
        return {
            'overall_consistency': relativity_analysis.get('overall_consistency', 0.0),
            'stability': relativity_analysis.get('relativity_stability', False),
            'key_consistencies': {
                k: v for k, v in relativity_analysis.get('individual_consistencies', {}).items()
                if '_consistency' in k
            }
        }
    
    def _default_temporal_cycles(self) -> Dict[str, Any]:
        """Default temporal cycles when no data available"""
        return {
            'week_of_month': 2,
            'month_of_year': 8,
            'day_of_week_cycle': 2,
            'session_position': 0.5,
            'weekend_proximity': 0.5,
            'weekly_pattern_strength': 0.5,
            'monthly_pattern_strength': 0.5,
            'cycle_confluence_score': 0.25
        }
    
    def _default_relativity_analysis(self) -> Dict[str, Any]:
        """Default relativity analysis when no data available"""
        return {
            'individual_consistencies': {},
            'overall_consistency': 0.5,
            'relativity_stability': False
        }

def main():
    """Command-line interface for precursor detection testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IRONFORGE event precursor detection")
    parser.add_argument('graph_file', help="JSON file containing session graph")
    parser.add_argument('--confidence-threshold', type=float, default=0.6, 
                       help="Minimum confidence threshold")
    parser.add_argument('--output', '-o', help="Output file for precursor results")
    
    args = parser.parse_args()
    
    # Load session graph
    try:
        with open(args.graph_file, 'r') as f:
            graph_data = json.load(f)
            
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return 1
    
    # Run precursor detection
    try:
        detector = EventPrecursorDetector(confidence_threshold=args.confidence_threshold)
        results = detector.detect_precursors(graph_data)
        
        print(f"\n🎯 Event Precursor Detection Results:")
        
        precursor_index = results['precursor_index']
        for event_type, probability in precursor_index.items():
            print(f"   {event_type}: {probability:.3f}")
        
        detected_precursors = results['detected_precursors']
        if detected_precursors:
            print(f"\n📊 Detected Precursors:")
            for event_type, precursor in detected_precursors.items():
                print(f"\n   🔔 {event_type.upper()} precursor detected:")
                print(f"     Confidence: {precursor.confidence:.3f}")
                print(f"     Expected timeframe: {precursor.expected_timeframe}")
                print(f"     Temporal alignment: {precursor.contributing_factors['temporal_alignment']:.3f}")
                print(f"     Structural confluence: {precursor.contributing_factors['structural_confluence']:.3f}")
        else:
            print(f"\n⚪ No precursors detected above confidence threshold ({args.confidence_threshold})")
        
        # Save results if requested
        if args.output:
            # Convert PrecursorEvent objects to dict for JSON serialization
            serializable_results = results.copy()
            serializable_precursors = {}
            
            for event_type, precursor in detected_precursors.items():
                serializable_precursors[event_type] = {
                    'event_type': precursor.event_type,
                    'confidence': precursor.confidence,
                    'expected_timeframe': precursor.expected_timeframe,
                    'contributing_factors': precursor.contributing_factors,
                    'temporal_alignment': precursor.temporal_alignment,
                    'structural_confluence': precursor.structural_confluence,
                    'relativity_consistency': precursor.relativity_consistency
                }
            
            serializable_results['detected_precursors'] = serializable_precursors
            
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Precursor detection failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())