#!/usr/bin/env python3
"""
IRONFORGE Cascade Event Orchestrator Agent
==========================================

Coordinates cross-agent analysis and synthesizes cascade event discoveries
into actionable market structure intelligence.

Multi-Agent Role: Event Orchestrator 
Focus: Cross-agent coordination, results integration, cascade recipe generation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CascadeRecipe:
    """Probabilistic cascade event recipe"""
    recipe_name: str
    trigger_condition: str
    event_sequence: List[Dict]
    overall_probability: float
    timing_distribution: Dict
    confidence_interval: Tuple[float, float]
    success_criteria: Dict
    risk_factors: List[str]

@dataclass
class CrossAgentValidation:
    """Validation results combining multiple agent findings"""
    sequence_agent_confidence: float
    classification_agent_confidence: float
    cross_validation_score: float
    agreement_level: str
    conflicting_signals: List[str]
    integrated_confidence: float

class CascadeEventOrchestratorAgent:
    """Agent that orchestrates multi-agent cascade event analysis"""
    
    def __init__(self, results_output_path: str):
        self.results_output_path = Path(results_output_path)
        self.results_output_path.mkdir(parents=True, exist_ok=True)
        
        # Integration parameters
        self.minimum_agreement_threshold = 0.7
        self.high_confidence_threshold = 0.85
        self.recipe_probability_threshold = 0.6
        
        # Cascade recipe templates
        self.recipe_templates = {
            'liquidity_hunt_cascade': {
                'description': 'Archaeological touch → liquidity hunt → reversal sequence',
                'expected_duration': '5-15 minutes',
                'key_events': ['touch', 'liquidity_sweep', 'reversal']
            },
            'fvg_expansion_cascade': {
                'description': 'Archaeological touch → displacement → FVG formation sequence',
                'expected_duration': '10-30 minutes', 
                'key_events': ['touch', 'displacement', 'fvg_formation']
            },
            'multi_timeframe_cascade': {
                'description': 'Archaeological touch → session extreme → daily extreme sequence',
                'expected_duration': '15-60 minutes',
                'key_events': ['touch', 'session_extreme', 'daily_extreme']
            }
        }
    
    def integrate_agent_findings(self, sequence_results: Dict, classification_results: Dict) -> Dict:
        """Integrate findings from Sequence Discovery and Pattern Classification agents"""
        
        print("🤝 ORCHESTRATING AGENT INTEGRATION...")
        print("=" * 45)
        
        # Extract key metrics from each agent
        sequence_summary = sequence_results.get('analysis_summary', {})
        classification_summary = classification_results.get('classification_summary', {})
        
        # Cross-validate findings
        cross_validations = self._cross_validate_findings(sequence_results, classification_results)
        
        # Generate integrated cascade sequences
        integrated_sequences = self._integrate_sequence_classifications(
            sequence_results.get('cascade_sequences', []),
            classification_results.get('classified_sequences', [])
        )
        
        # Synthesize cascade recipes
        cascade_recipes = self._synthesize_cascade_recipes(integrated_sequences, cross_validations)
        
        return {
            'agent_role': 'Event Orchestrator Agent',
            'integration_summary': {
                'sequence_agent_sequences': sequence_summary.get('cascade_sequences_discovered', 0),
                'classification_agent_events': classification_summary.get('total_events_classified', 0),
                'integration_timestamp': datetime.now().isoformat(),
                'cross_validation_score': np.mean([cv.cross_validation_score for cv in cross_validations.values()])
            },
            'cross_agent_validation': {k: asdict(v) for k, v in cross_validations.items()},
            'integrated_sequences': integrated_sequences,
            'cascade_recipes': [asdict(recipe) for recipe in cascade_recipes],
            'orchestration_methodology': self._get_methodology_summary()
        }
    
    def _cross_validate_findings(self, sequence_results: Dict, classification_results: Dict) -> Dict[str, CrossAgentValidation]:
        """Cross-validate findings between sequence discovery and pattern classification"""
        
        validations = {}
        
        # Get statistical results from both agents
        seq_stats = sequence_results.get('statistical_validation', {})
        class_stats = classification_results.get('pattern_validations', {})
        
        if 'error' in seq_stats or not class_stats:
            # TODO(human): Implement cross-validation logic when one agent has limited data
            # This should handle cases where:
            # 1. Sequence agent found patterns but classification agent had data issues
            # 2. Different confidence levels between agents need reconciliation  
            # 3. Conflicting event counts or timing distributions need resolution
            #
            # Consider creating weighted confidence scores based on:
            # - Data quality from each agent
            # - Statistical significance of findings
            # - Consistency of timing distributions
            
            validations['limited_data_validation'] = CrossAgentValidation(
                sequence_agent_confidence=0.5,
                classification_agent_confidence=0.3,
                cross_validation_score=0.4,
                agreement_level='limited',
                conflicting_signals=['insufficient_data'],
                integrated_confidence=0.4
            )
            return validations
        
        # Compare event frequencies
        seq_events = seq_stats.get('most_common_events', [])
        
        for pattern_name, pattern_validation in class_stats.items():
            seq_confidence = 0.5  # Default if not found in sequence results
            
            # Find corresponding event in sequence results
            for event_type, probability in seq_events:
                if self._events_match(event_type, pattern_name):
                    seq_confidence = probability
                    break
            
            class_confidence = pattern_validation['structural_quality_score']
            
            # Calculate cross-validation metrics
            confidence_difference = abs(seq_confidence - class_confidence)
            agreement_score = 1.0 - confidence_difference
            
            # Determine agreement level
            if agreement_score >= 0.8:
                agreement_level = 'high'
            elif agreement_score >= 0.6:
                agreement_level = 'moderate'
            else:
                agreement_level = 'low'
            
            # Check for conflicting signals
            conflicting_signals = []
            if confidence_difference > 0.3:
                conflicting_signals.append('confidence_discrepancy')
            if pattern_validation['false_positive_rate'] > 0.2:
                conflicting_signals.append('high_false_positive_rate')
            
            # Calculate integrated confidence
            integrated_confidence = (seq_confidence + class_confidence) / 2.0
            if conflicting_signals:
                integrated_confidence *= 0.8  # Reduce confidence if conflicts exist
            
            validations[pattern_name] = CrossAgentValidation(
                sequence_agent_confidence=seq_confidence,
                classification_agent_confidence=class_confidence,
                cross_validation_score=agreement_score,
                agreement_level=agreement_level,
                conflicting_signals=conflicting_signals,
                integrated_confidence=integrated_confidence
            )
        
        return validations
    
    def _events_match(self, sequence_event_type: str, classification_pattern: str) -> bool:
        """Determine if sequence event type matches classification pattern"""
        # Simple matching logic - can be enhanced
        return sequence_event_type.upper() in classification_pattern.upper() or \
               classification_pattern.upper() in sequence_event_type.upper()
    
    def _integrate_sequence_classifications(self, sequence_data: List, classification_data: List) -> List[Dict]:
        """Integrate sequence discoveries with pattern classifications"""
        
        integrated_sequences = []
        
        for i, sequence in enumerate(sequence_data):
            # Find corresponding classified sequence
            classified_sequence = classification_data[i] if i < len(classification_data) else None
            
            if classified_sequence:
                # Merge sequence discovery with pattern classification
                integrated_sequence = {
                    'touch_info': {
                        'timestamp': sequence.get('touch_timestamp'),
                        'price': sequence.get('touch_price'),
                        'archaeological_level': sequence.get('archaeological_40_level')
                    },
                    'sequence_metrics': {
                        'total_duration_minutes': sequence.get('total_duration_minutes'),
                        'liquidity_events_count': sequence.get('liquidity_events_count'),
                        'fvg_events_count': sequence.get('fvg_events_count'),
                        'displacement_events_count': sequence.get('displacement_events_count'),
                        'sequence_success_score': sequence.get('sequence_success_score')
                    },
                    'classified_events': classified_sequence.get('classified_events', []),
                    'classification_summary': classified_sequence.get('classification_summary', {}),
                    'integration_quality': self._calculate_integration_quality(sequence, classified_sequence)
                }
                integrated_sequences.append(integrated_sequence)
        
        return integrated_sequences
    
    def _calculate_integration_quality(self, sequence_data: Dict, classification_data: Dict) -> Dict:
        """Calculate quality metrics for sequence-classification integration"""
        
        # Count events that were both discovered and classified
        discovered_events = len(sequence_data.get('sequence_events', []))
        classified_events = len(classification_data.get('classified_events', []))
        
        # Calculate coverage ratio
        coverage_ratio = min(classified_events / discovered_events, 1.0) if discovered_events > 0 else 0.0
        
        # Calculate confidence alignment
        seq_confidence = sequence_data.get('sequence_success_score', 0.5)
        class_confidence = classification_data.get('classification_summary', {}).get('average_confidence', 0.5)
        confidence_alignment = 1.0 - abs(seq_confidence - class_confidence)
        
        return {
            'coverage_ratio': coverage_ratio,
            'confidence_alignment': confidence_alignment,
            'integration_score': (coverage_ratio + confidence_alignment) / 2.0,
            'data_completeness': 'high' if coverage_ratio > 0.8 else 'moderate' if coverage_ratio > 0.5 else 'low'
        }
    
    def _synthesize_cascade_recipes(self, integrated_sequences: List[Dict], 
                                  cross_validations: Dict[str, CrossAgentValidation]) -> List[CascadeRecipe]:
        """Synthesize probabilistic cascade recipes from integrated findings"""
        
        recipes = []
        
        # Group sequences by similarity to identify common patterns
        pattern_groups = self._group_similar_sequences(integrated_sequences)
        
        for pattern_name, sequences in pattern_groups.items():
            if len(sequences) < 1:  # Need at least one sequence
                continue
                
            # For single sequences, treat as exploratory pattern
            is_single_sequence = len(sequences) == 1
            
            # Calculate recipe probability
            recipe_probability = len(sequences) / len(integrated_sequences)
            
            # Lower threshold for single sequences (exploratory analysis)
            threshold = 0.3 if is_single_sequence else self.recipe_probability_threshold
            
            if recipe_probability < threshold:
                continue
            
            # Extract event sequence pattern
            event_sequence = self._extract_common_event_pattern(sequences)
            
            # Calculate timing distribution
            timing_distribution = self._calculate_timing_distribution(sequences)
            
            # Calculate confidence interval
            success_scores = [seq['sequence_metrics']['sequence_success_score'] 
                            for seq in sequences if 'sequence_metrics' in seq]
            if success_scores:
                ci_lower = np.percentile(success_scores, 2.5)
                ci_upper = np.percentile(success_scores, 97.5)
                confidence_interval = (ci_lower, ci_upper)
            else:
                confidence_interval = (0.0, 1.0)
            
            # Identify success criteria and risk factors
            success_criteria = self._identify_success_criteria(sequences)
            risk_factors = self._identify_risk_factors(sequences, cross_validations)
            
            # Adjust names for single sequences
            recipe_name = f"{pattern_name}_exploratory" if is_single_sequence else pattern_name
            
            recipe = CascadeRecipe(
                recipe_name=recipe_name,
                trigger_condition="40% archaeological zone interaction",
                event_sequence=event_sequence,
                overall_probability=recipe_probability,
                timing_distribution=timing_distribution,
                confidence_interval=confidence_interval,
                success_criteria=success_criteria,
                risk_factors=risk_factors
            )
            
            recipes.append(recipe)
        
        return recipes
    
    def _group_similar_sequences(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar cascade sequences to identify common patterns"""
        
        # Simple grouping based on event counts - can be enhanced with clustering
        groups = {
            'liquidity_dominant': [],
            'fvg_dominant': [],
            'mixed_pattern': [],
            'displacement_dominant': []
        }
        
        for seq in sequences:
            metrics = seq.get('sequence_metrics', {}) or {}
            liquidity_count = metrics.get('liquidity_events_count') or 0
            fvg_count = metrics.get('fvg_events_count') or 0
            displacement_count = metrics.get('displacement_events_count') or 0
            
            # Classify based on dominant event type
            if liquidity_count > fvg_count and liquidity_count > displacement_count:
                groups['liquidity_dominant'].append(seq)
            elif fvg_count > liquidity_count and fvg_count > displacement_count:
                groups['fvg_dominant'].append(seq)
            elif displacement_count > liquidity_count and displacement_count > fvg_count:
                groups['displacement_dominant'].append(seq)
            else:
                groups['mixed_pattern'].append(seq)
        
        return groups
    
    def _extract_common_event_pattern(self, sequences: List[Dict]) -> List[Dict]:
        """Extract common event sequence pattern from grouped sequences"""
        
        
        if not sequences:
            return []
            
        # Extract event frequency analysis across sequences
        event_frequency = {}
        timing_patterns = {}
        
        for seq in sequences:
            metrics = seq.get('sequence_metrics', {}) or {}
            duration = metrics.get('total_duration_minutes', 0)
            
            # Count event types from sequence metrics
            event_counts = {
                'liquidity_events': metrics.get('liquidity_events_count', 0) or 0,
                'fvg_events': metrics.get('fvg_events_count', 0) or 0,
                'displacement_events': metrics.get('displacement_events_count', 0) or 0
            }
            
            # Build frequency distribution
            for event_type, count in event_counts.items():
                if event_type not in event_frequency:
                    event_frequency[event_type] = []
                event_frequency[event_type].append(count)
                
                # Track timing patterns
                if event_type not in timing_patterns:
                    timing_patterns[event_type] = []
                if duration > 0 and count > 0:
                    # Estimate average timing spread
                    timing_patterns[event_type].append(duration / max(count, 1))
        
        # Generate common pattern based on most frequent events
        common_pattern = []
        position = 1
        
        # Always start with initial touch
        common_pattern.append({
            'position': position,
            'event_type': 'archaeological_touch',
            'timing_minutes': 0,
            'probability': 1.0
        })
        position += 1
        
        # Add events in order of frequency
        for event_type, counts in event_frequency.items():
            if counts and sum(counts) > 0:
                avg_count = sum(counts) / len(counts)
                probability = min(avg_count / 10.0, 1.0)  # Normalize probability
                
                # Estimate timing from patterns
                avg_timing = 0
                if event_type in timing_patterns and timing_patterns[event_type]:
                    avg_timing = sum(timing_patterns[event_type]) / len(timing_patterns[event_type])
                
                common_pattern.append({
                    'position': position,
                    'event_type': event_type.replace('_events', ''),
                    'timing_minutes': round(avg_timing, 1),
                    'probability': round(probability, 2)
                })
                position += 1
        
        return common_pattern
    
    def _calculate_timing_distribution(self, sequences: List[Dict]) -> Dict:
        """Calculate timing distribution statistics for cascade events"""
        
        all_durations = []
        event_timings = {}
        
        for seq in sequences:
            duration = seq.get('sequence_metrics', {}).get('total_duration_minutes', 0)
            if duration > 0:
                all_durations.append(duration)
            
            # Collect event timing data
            for event in seq.get('classified_events', []):
                event_type = event.get('primary_classification', 'unknown')
                timing = event.get('minutes_after_touch', 0)
                
                if event_type not in event_timings:
                    event_timings[event_type] = []
                event_timings[event_type].append(timing)
        
        timing_stats = {}
        if all_durations:
            timing_stats['cascade_duration'] = {
                'mean': np.mean(all_durations),
                'std': np.std(all_durations),
                'min': min(all_durations),
                'max': max(all_durations),
                'percentile_25': np.percentile(all_durations, 25),
                'percentile_75': np.percentile(all_durations, 75)
            }
        
        # Add event-specific timing statistics
        for event_type, timings in event_timings.items():
            if timings:
                timing_stats[f'{event_type}_timing'] = {
                    'mean': np.mean(timings),
                    'std': np.std(timings),
                    'earliest': min(timings),
                    'latest': max(timings)
                }
        
        return timing_stats
    
    def _identify_success_criteria(self, sequences: List[Dict]) -> Dict:
        """Identify criteria that correlate with cascade success"""
        
        # Analyze successful vs less successful sequences
        success_scores = []
        for seq in sequences:
            score = seq.get('sequence_metrics', {}).get('sequence_success_score', 0.5)
            success_scores.append(score)
        
        if not success_scores:
            return {'criteria': 'insufficient_data'}
        
        high_success_threshold = np.percentile(success_scores, 75)
        
        return {
            'minimum_duration_minutes': 5,
            'minimum_events_count': 2,
            'high_success_threshold': high_success_threshold,
            'key_indicators': [
                'rapid_initial_displacement',
                'sustained_directional_bias',
                'multiple_event_types_present'
            ]
        }
    
    def _identify_risk_factors(self, sequences: List[Dict], cross_validations: Dict) -> List[str]:
        """Identify factors that may reduce cascade reliability"""
        
        risk_factors = []
        
        # Check for low cross-validation scores
        low_validation_count = sum(1 for cv in cross_validations.values() 
                                 if cv.cross_validation_score < 0.6)
        if low_validation_count > len(cross_validations) * 0.3:
            risk_factors.append('inconsistent_agent_validation')
        
        # Check for high variability in success scores
        success_scores = [seq.get('sequence_metrics', {}).get('sequence_success_score', 0.5) 
                         for seq in sequences]
        if success_scores and np.std(success_scores) > 0.25:
            risk_factors.append('high_success_score_variability')
        
        # Check for conflicting signals
        for cv in cross_validations.values():
            if cv.conflicting_signals:
                risk_factors.extend(cv.conflicting_signals)
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _get_methodology_summary(self) -> Dict:
        """Get summary of orchestration methodology"""
        
        return {
            'integration_approach': 'cross_agent_validation_with_confidence_weighting',
            'recipe_synthesis': 'pattern_grouping_with_statistical_validation',
            'thresholds': {
                'minimum_agreement': self.minimum_agreement_threshold,
                'high_confidence': self.high_confidence_threshold,
                'recipe_probability': self.recipe_probability_threshold
            },
            'quality_metrics': [
                'cross_validation_score',
                'confidence_alignment', 
                'coverage_ratio',
                'integration_score'
            ]
        }
    
    def export_results(self, orchestrated_results: Dict) -> str:
        """Export orchestrated results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_output_path / f"cascade_event_discovery_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(orchestrated_results, f, indent=2, default=str)
        
        print(f"✅ Results exported to: {output_file}")
        return str(output_file)

def main():
    """Execute cascade event orchestration"""
    
    print("🎼 CASCADE EVENT ORCHESTRATOR AGENT")
    print("=" * 40)
    
    # Import and run the actual agents
    import sys
    sys.path.append('/Users/jack/IRONFORGE')
    
    from cascade_sequence_discovery_agent import main as sequence_main
    from cascade_pattern_classification_agent import main as classification_main
    
    print("🔄 Running Sequence Discovery Agent...")
    sequence_results = sequence_main()
    
    print("🔄 Running Pattern Classification Agent...")  
    classification_results = classification_main()
    
    orchestrator = CascadeEventOrchestratorAgent(
        results_output_path="/Users/jack/IRONFORGE/cascade_discovery_results"
    )
    
    results = orchestrator.integrate_agent_findings(sequence_results, classification_results)
    
    # Display results
    summary = results['integration_summary']
    print(f"\n🤝 INTEGRATION SUMMARY:")
    print(f"   📊 Cross-Validation Score: {summary['cross_validation_score']:.2f}")
    print(f"   🔗 Integrated Sequences: {len(results.get('integrated_sequences', []))}")
    print(f"   📋 Cascade Recipes: {len(results.get('cascade_recipes', []))}")
    
    # Export results
    output_file = orchestrator.export_results(results)
    print(f"📄 Complete results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()