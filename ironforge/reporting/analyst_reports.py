#!/usr/bin/env python3
"""
IRONFORGE Analyst Reports - Sprint 2 Visibility Layer
====================================================

Comprehensive visibility into Sprint 2 structural intelligence features.
Provides actionable insights for analysts through detailed reporting.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class SessionAnalysis:
    """Container for session analysis results"""
    session_name: str
    edge_type_distribution: Dict[str, int]
    pattern_analysis: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    precursor_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    timestamp: str

class AnalystReports:
    """
    Comprehensive reporting system for IRONFORGE Sprint 2 features
    Provides visibility into structural intelligence and regime analysis
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates and formatting
        self.edge_type_names = {
            'temporal': 'Temporal Edges',
            'scale': 'Scale Edges', 
            'structural_context': 'Structural Context Edges',
            'discovered': 'Discovered Edges'
        }
        
    def generate_session_report(self, session_results: Dict[str, Any]) -> SessionAnalysis:
        """
        Generate comprehensive session analysis report
        
        Args:
            session_results: Combined results from IRONFORGE processing
            
        Returns:
            SessionAnalysis object with comprehensive insights
        """
        
        # Extract core data
        processing_results = session_results.get('processing_results', {})
        performance_report = session_results.get('performance_report', {})
        patterns = processing_results.get('patterns', [])
        metadata = processing_results.get('metadata', {})
        
        session_name = metadata.get('session_name', 'unknown_session')
        
        # Analyze edge type distribution
        edge_distribution = self._analyze_edge_type_distribution(metadata)
        
        # Analyze patterns
        pattern_analysis = self._analyze_patterns(patterns)
        
        # Analyze regime information
        regime_analysis = self._analyze_regime_distribution(patterns)
        
        # Analyze precursor events
        precursor_analysis = self._analyze_precursor_events(session_results)
        
        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics(performance_report)
        
        return SessionAnalysis(
            session_name=session_name,
            edge_type_distribution=edge_distribution,
            pattern_analysis=pattern_analysis,
            regime_analysis=regime_analysis,
            precursor_analysis=precursor_analysis,
            performance_metrics=performance_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_edge_type_distribution(self, metadata: Dict[str, Any]) -> Dict[str, int]:
        """Analyze distribution of 4 edge types"""
        
        edge_type_counts = metadata.get('edge_type_counts', {})
        
        # Ensure all 4 edge types are represented
        distribution = {}
        for edge_type in ['temporal', 'scale', 'structural_context', 'discovered']:
            distribution[edge_type] = edge_type_counts.get(edge_type, 0)
        
        return distribution
    
    def _analyze_patterns(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze discovered patterns for insights"""
        
        if not patterns:
            return {
                'total_patterns': 0,
                'pattern_types': {},
                'confidence_distribution': {},
                'top_contributing_features': []
            }
        
        # Pattern type analysis
        pattern_types = {}
        confidence_scores = []
        all_features = []
        
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            confidence = pattern.get('confidence', 0.5)
            confidence_scores.append(confidence)
            
            # Extract contributing features
            if 'nodes' in pattern and isinstance(pattern['nodes'], list):
                all_features.extend(pattern['nodes'])
        
        # Confidence distribution analysis
        confidence_distribution = self._analyze_confidence_distribution(confidence_scores)
        
        # Top contributing features analysis
        top_features = self._identify_top_contributing_features(patterns)
        
        return {
            'total_patterns': len(patterns),
            'pattern_types': pattern_types,
            'confidence_distribution': confidence_distribution,
            'top_contributing_features': top_features,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'high_confidence_patterns': len([c for c in confidence_scores if c > 0.8])
        }
    
    def _analyze_regime_distribution(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze regime distribution across patterns"""
        
        regime_labels = []
        regime_characteristics = []
        
        for pattern in patterns:
            regime_label = pattern.get('regime_label')
            if regime_label:
                regime_labels.append(regime_label)
            
            regime_chars = pattern.get('regime_characteristics')
            if regime_chars:
                regime_characteristics.append(regime_chars)
        
        if not regime_labels:
            return {
                'total_regimes': 0,
                'regime_distribution': {},
                'dominant_regime': None
            }
        
        # Count regime occurrences
        regime_counts = {}
        for label in regime_labels:
            regime_counts[label] = regime_counts.get(label, 0) + 1
        
        # Find dominant regime
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1]) if regime_counts else None
        
        # Analyze regime characteristics
        temporal_dominance = {}
        structural_dominance = {}
        
        for char in regime_characteristics:
            temp_dom = char.get('temporal_dominance', 'unknown')
            struct_dom = char.get('structural_dominance', 'unknown')
            
            temporal_dominance[temp_dom] = temporal_dominance.get(temp_dom, 0) + 1
            structural_dominance[struct_dom] = structural_dominance.get(struct_dom, 0) + 1
        
        return {
            'total_regimes': len(set(regime_labels)),
            'regime_distribution': regime_counts,
            'dominant_regime': dominant_regime[0] if dominant_regime else None,
            'dominant_regime_count': dominant_regime[1] if dominant_regime else 0,
            'temporal_dominance_distribution': temporal_dominance,
            'structural_dominance_distribution': structural_dominance
        }
    
    def _analyze_precursor_events(self, session_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze precursor event detection results"""
        
        # Look for precursor analysis in results
        precursor_data = None
        
        # Check multiple possible locations for precursor data
        processing_results = session_results.get('processing_results', {})
        if 'precursor_analysis' in processing_results:
            precursor_data = processing_results['precursor_analysis']
        elif 'precursor_index' in processing_results:
            precursor_data = {'precursor_index': processing_results['precursor_index']}
        
        if not precursor_data:
            return {
                'precursors_detected': False,
                'event_probabilities': {},
                'highest_probability_event': None
            }
        
        precursor_index = precursor_data.get('precursor_index', {})
        detected_precursors = precursor_data.get('detected_precursors', {})
        
        # Extract event probabilities
        event_probabilities = {}
        for key, value in precursor_index.items():
            if 'probability' in key:
                event_type = key.replace('_probability', '')
                event_probabilities[event_type] = value
        
        # Find highest probability event
        highest_prob_event = None
        if event_probabilities:
            highest_prob_event = max(event_probabilities.items(), key=lambda x: x[1])
        
        return {
            'precursors_detected': len(detected_precursors) > 0,
            'event_probabilities': event_probabilities,
            'highest_probability_event': highest_prob_event[0] if highest_prob_event else None,
            'highest_probability_value': highest_prob_event[1] if highest_prob_event else 0.0,
            'total_precursor_events': len(detected_precursors),
            'overall_activity': precursor_index.get('overall_precursor_activity', 0.0)
        }
    
    def _extract_performance_metrics(self, performance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance metrics for reporting"""
        
        session_metrics = performance_report.get('session_metrics', {})
        sprint2_analysis = performance_report.get('sprint2_analysis', {})
        
        return {
            'processing_time_ms': session_metrics.get('processing_time_ms', 0),
            'memory_usage_mb': session_metrics.get('memory_usage_mb', 0),
            'pattern_count': session_metrics.get('pattern_count', 0),
            'node_count': session_metrics.get('node_count', 0),
            'edge_count': session_metrics.get('edge_count', 0),
            'feature_dimensions': session_metrics.get('feature_dimensions', 0),
            'edge_types_count': session_metrics.get('edge_types_count', 0),
            'sprint2_feature_status': sprint2_analysis.get('feature_architecture_status', 'unknown'),
            'sprint2_edge_status': sprint2_analysis.get('edge_types_status', 'unknown')
        }
    
    def _analyze_confidence_distribution(self, confidence_scores: List[float]) -> Dict[str, Any]:
        """Analyze confidence score distribution"""
        
        if not confidence_scores:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high_conf = len([c for c in confidence_scores if c > 0.8])
        med_conf = len([c for c in confidence_scores if 0.5 < c <= 0.8])
        low_conf = len([c for c in confidence_scores if c <= 0.5])
        
        return {
            'high': high_conf,
            'medium': med_conf,
            'low': low_conf,
            'avg_confidence': np.mean(confidence_scores),
            'max_confidence': max(confidence_scores),
            'min_confidence': min(confidence_scores)
        }
    
    def _identify_top_contributing_features(self, patterns: List[Dict]) -> List[Dict[str, Any]]:
        """Identify top contributing features across patterns"""
        
        feature_contributions = {}
        
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            confidence = pattern.get('confidence', 0.0)
            
            # Extract feature information from different pattern fields
            contributing_factors = pattern.get('contributing_factors', {})
            
            for factor, value in contributing_factors.items():
                if factor not in feature_contributions:
                    feature_contributions[factor] = {
                        'total_contribution': 0.0,
                        'pattern_count': 0,
                        'pattern_types': set(),
                        'avg_confidence': 0.0
                    }
                
                feature_contributions[factor]['total_contribution'] += value
                feature_contributions[factor]['pattern_count'] += 1
                feature_contributions[factor]['pattern_types'].add(pattern_type)
                feature_contributions[factor]['avg_confidence'] += confidence
        
        # Calculate averages and sort by contribution
        top_features = []
        for feature, data in feature_contributions.items():
            if data['pattern_count'] > 0:
                avg_contribution = data['total_contribution'] / data['pattern_count']
                avg_confidence = data['avg_confidence'] / data['pattern_count']
                
                top_features.append({
                    'feature_name': feature,
                    'avg_contribution': avg_contribution,
                    'pattern_count': data['pattern_count'],
                    'avg_confidence': avg_confidence,
                    'pattern_types': list(data['pattern_types'])
                })
        
        # Sort by average contribution
        top_features.sort(key=lambda x: x['avg_contribution'], reverse=True)
        
        return top_features[:10]  # Top 10 contributing features
    
    def export_json_report(self, session_analysis: SessionAnalysis, filename: str = None) -> str:
        """Export session analysis as JSON"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_analysis_{session_analysis.session_name}_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        # Convert SessionAnalysis to dict for JSON serialization
        report_data = {
            'session_name': session_analysis.session_name,
            'timestamp': session_analysis.timestamp,
            'edge_type_distribution': session_analysis.edge_type_distribution,
            'pattern_analysis': session_analysis.pattern_analysis,
            'regime_analysis': session_analysis.regime_analysis,
            'precursor_analysis': session_analysis.precursor_analysis,
            'performance_metrics': session_analysis.performance_metrics
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report exported to {output_file}")
        return str(output_file)
    
    def export_human_readable_report(self, session_analysis: SessionAnalysis, filename: str = None) -> str:
        """Export session analysis as human-readable report"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_report_{session_analysis.session_name}_{timestamp}.txt"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            self._write_human_readable_report(f, session_analysis)
        
        self.logger.info(f"Human-readable report exported to {output_file}")
        return str(output_file)
    
    def _write_human_readable_report(self, f, analysis: SessionAnalysis):
        """Write human-readable report content"""
        
        f.write("=" * 80 + "\n")
        f.write("IRONFORGE SESSION ANALYSIS REPORT - SPRINT 2 ENHANCED\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Session: {analysis.session_name}\n")
        f.write(f"Timestamp: {analysis.timestamp}\n\n")
        
        # Edge Type Distribution
        f.write("🔗 EDGE TYPE DISTRIBUTION (4 Types)\n")
        f.write("-" * 40 + "\n")
        total_edges = sum(analysis.edge_type_distribution.values())
        
        for edge_type, count in analysis.edge_type_distribution.items():
            percentage = (count / total_edges * 100) if total_edges > 0 else 0
            f.write(f"{self.edge_type_names.get(edge_type, edge_type):<25}: {count:>5} ({percentage:>5.1f}%)\n")
        f.write(f"{'Total Edges':<25}: {total_edges:>5}\n\n")
        
        # Pattern Analysis
        f.write("🔍 PATTERN ANALYSIS\n")
        f.write("-" * 40 + "\n")
        pattern_analysis = analysis.pattern_analysis
        f.write(f"Total Patterns: {pattern_analysis['total_patterns']}\n")
        f.write(f"Average Confidence: {pattern_analysis.get('avg_confidence', 0):.3f}\n")
        f.write(f"High Confidence Patterns: {pattern_analysis.get('high_confidence_patterns', 0)}\n\n")
        
        # Pattern Types
        if pattern_analysis.get('pattern_types'):
            f.write("Pattern Types:\n")
            for pattern_type, count in pattern_analysis['pattern_types'].items():
                f.write(f"  • {pattern_type}: {count}\n")
            f.write("\n")
        
        # Top Contributing Features
        top_features = pattern_analysis.get('top_contributing_features', [])
        if top_features:
            f.write("Top Contributing Features:\n")
            for i, feature in enumerate(top_features[:5], 1):
                f.write(f"  {i}. {feature['feature_name']}: {feature['avg_contribution']:.3f} "
                       f"({feature['pattern_count']} patterns)\n")
            f.write("\n")
        
        # Regime Analysis
        f.write("🏛️ REGIME ANALYSIS\n")
        f.write("-" * 40 + "\n")
        regime_analysis = analysis.regime_analysis
        f.write(f"Total Regimes: {regime_analysis['total_regimes']}\n")
        
        dominant_regime = regime_analysis.get('dominant_regime')
        if dominant_regime:
            count = regime_analysis.get('dominant_regime_count', 0)
            f.write(f"Dominant Regime: {dominant_regime} ({count} patterns)\n")
        
        # Regime Distribution
        regime_dist = regime_analysis.get('regime_distribution', {})
        if regime_dist:
            f.write("\nRegime Distribution:\n")
            for regime, count in regime_dist.items():
                f.write(f"  • {regime}: {count}\n")
        f.write("\n")
        
        # Precursor Analysis
        f.write("⚡ PRECURSOR EVENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        precursor_analysis = analysis.precursor_analysis
        
        if precursor_analysis['precursors_detected']:
            f.write("✅ Precursor events detected\n")
            
            highest_event = precursor_analysis.get('highest_probability_event')
            highest_prob = precursor_analysis.get('highest_probability_value', 0)
            
            if highest_event:
                f.write(f"Highest Probability Event: {highest_event} ({highest_prob:.3f})\n")
            
            # Event probabilities
            event_probs = precursor_analysis.get('event_probabilities', {})
            if event_probs:
                f.write("\nEvent Probabilities:\n")
                for event, prob in event_probs.items():
                    f.write(f"  • {event}: {prob:.3f}\n")
            
            overall_activity = precursor_analysis.get('overall_activity', 0)
            f.write(f"\nOverall Precursor Activity: {overall_activity:.3f}\n")
        else:
            f.write("⚪ No precursor events detected\n")
        
        f.write("\n")
        
        # Performance Metrics
        f.write("📊 PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        perf = analysis.performance_metrics
        f.write(f"Processing Time: {perf.get('processing_time_ms', 0):.1f} ms\n")
        f.write(f"Memory Usage: {perf.get('memory_usage_mb', 0):.1f} MB\n")
        f.write(f"Nodes: {perf.get('node_count', 0)}\n")
        f.write(f"Edges: {perf.get('edge_count', 0)}\n")
        f.write(f"Feature Dimensions: {perf.get('feature_dimensions', 0)}\n")
        f.write(f"Edge Types: {perf.get('edge_types_count', 0)}\n")
        
        # Sprint 2 Status
        feature_status = perf.get('sprint2_feature_status', 'unknown')
        edge_status = perf.get('sprint2_edge_status', 'unknown')
        f.write(f"\nSprint 2 Status:\n")
        f.write(f"  Feature Architecture: {feature_status}\n")
        f.write(f"  Edge Types: {edge_status}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")

def main():
    """Command-line interface for analyst reports"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate IRONFORGE analyst reports")
    parser.add_argument('results_file', help="JSON file containing session results")
    parser.add_argument('--output-dir', '-o', default='reports', help="Output directory for reports")
    parser.add_argument('--format', choices=['json', 'text', 'both'], default='both', 
                       help="Report format")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load session results
    try:
        with open(args.results_file, 'r') as f:
            session_results = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results file: {e}")
        return 1
    
    # Generate analysis
    try:
        reporter = AnalystReports(output_dir=args.output_dir)
        analysis = reporter.generate_session_report(session_results)
        
        print(f"📊 Generated analysis for session: {analysis.session_name}")
        print(f"   Edge types: {analysis.edge_type_distribution}")
        print(f"   Patterns: {analysis.pattern_analysis['total_patterns']}")
        print(f"   Regimes: {analysis.regime_analysis['total_regimes']}")
        
        # Export reports
        if args.format in ['json', 'both']:
            json_file = reporter.export_json_report(analysis)
            print(f"📄 JSON report: {json_file}")
        
        if args.format in ['text', 'both']:
            text_file = reporter.export_human_readable_report(analysis)
            print(f"📄 Text report: {text_file}")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())