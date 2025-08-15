#!/usr/bin/env python3
"""
IRONFORGE Phase 2: Feature Authenticity Validation Framework
=========================================================

Validates that decontamination successfully removed artificial default values
and replaced them with authentic market-derived calculations.

Author: Iron-Data-Scientist
Date: 2025-08-14
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureAuthenticityValidator:
    """
    Validates feature authenticity and provides comprehensive contamination analysis
    before and after Phase 2 enhancement.
    """
    
    def __init__(self):
        self.base_path = Path("/Users/jack/IRONPULSE/data/sessions/level_1")
        self.enhanced_path = Path("/Users/jack/IRONPULSE/IRONFORGE/enhanced_sessions")
        self.quality_assessment_path = Path("/Users/jack/IRONPULSE/IRONFORGE/data_quality_assessment.json")
        
        # Load quality assessment
        with open(self.quality_assessment_path, 'r') as f:
            self.quality_data = json.load(f)
    
    def analyze_feature_distributions(self, session_files: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of feature values to identify contamination patterns.
        
        Contaminated features will show clustering around default values (0.3, 0.5, empty arrays).
        Authentic features will show diverse, non-uniform distributions.
        """
        distributions = {
            'htf_carryover_strengths': [],
            'energy_densities': [],
            'liquidity_event_counts': [],
            'sessions_analyzed': 0,
            'default_value_counts': {
                'htf_carryover_0.3': 0,
                'energy_density_0.5': 0,
                'empty_liquidity_events': 0
            }
        }
        
        for session_file in session_files:
            try:
                session_data = self.load_session_data(session_file)
                if not session_data:
                    continue
                
                distributions['sessions_analyzed'] += 1
                
                # HTF carryover strength
                htf_strength = self._extract_htf_carryover_strength(session_data)
                if htf_strength is not None:
                    distributions['htf_carryover_strengths'].append(htf_strength)
                    if htf_strength == 0.3:
                        distributions['default_value_counts']['htf_carryover_0.3'] += 1
                
                # Energy density
                energy_density = self._extract_energy_density(session_data)
                if energy_density is not None:
                    distributions['energy_densities'].append(energy_density)
                    if energy_density == 0.5:
                        distributions['default_value_counts']['energy_density_0.5'] += 1
                
                # Liquidity event counts
                event_count = len(session_data.get('session_liquidity_events', []))
                distributions['liquidity_event_counts'].append(event_count)
                if event_count == 0:
                    distributions['default_value_counts']['empty_liquidity_events'] += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing {session_file}: {e}")
        
        # Calculate statistics
        for feature_name in ['htf_carryover_strengths', 'energy_densities', 'liquidity_event_counts']:
            values = distributions[feature_name]
            if values:
                distributions[f'{feature_name}_stats'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'unique_values': len(set(values)),
                    'total_values': len(values)
                }
        
        return distributions
    
    def _extract_htf_carryover_strength(self, session_data: Dict) -> Optional[float]:
        """Extract HTF carryover strength from session data."""
        if 'contamination_analysis' in session_data:
            return session_data['contamination_analysis']['htf_contamination'].get('htf_carryover_strength')
        return None
    
    def _extract_energy_density(self, session_data: Dict) -> Optional[float]:
        """Extract energy density from session data."""
        if 'energy_state' in session_data:
            return session_data['energy_state'].get('energy_density')
        return None
    
    def load_session_data(self, session_filename: str, enhanced: bool = False) -> Optional[Dict]:
        """Load session data from file."""
        if enhanced:
            session_path = self.enhanced_path / f"enhanced_{session_filename}"
        else:
            # Find in original location
            session_path = None
            for year_month_dir in self.base_path.glob("2025_*"):
                potential_path = year_month_dir / session_filename
                if potential_path.exists():
                    session_path = potential_path
                    break
        
        if not session_path or not session_path.exists():
            return None
        
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {session_path}: {e}")
            return None
    
    def compare_before_after_enhancement(self, session_filenames: List[str]) -> Dict[str, Any]:
        """
        Compare feature distributions before and after enhancement to validate decontamination.
        """
        logger.info(f"Comparing before/after enhancement for {len(session_filenames)} sessions")
        
        # Analyze original sessions
        before_distributions = self.analyze_feature_distributions(session_filenames)
        
        # Analyze enhanced sessions (if they exist)
        enhanced_filenames = []
        for filename in session_filenames:
            enhanced_path = self.enhanced_path / f"enhanced_{filename}"
            if enhanced_path.exists():
                enhanced_filenames.append(filename)
        
        logger.info(f"Found {len(enhanced_filenames)} enhanced sessions")
        
        # Load enhanced sessions for comparison
        after_distributions = {
            'htf_carryover_strengths': [],
            'energy_densities': [],
            'liquidity_event_counts': [],
            'sessions_analyzed': 0,
            'default_value_counts': {
                'htf_carryover_0.3': 0,
                'energy_density_0.5': 0,
                'empty_liquidity_events': 0
            }
        }
        
        for session_file in enhanced_filenames:
            session_data = self.load_session_data(session_file, enhanced=True)
            if not session_data:
                continue
            
            after_distributions['sessions_analyzed'] += 1
            
            # HTF carryover strength
            htf_strength = self._extract_htf_carryover_strength(session_data)
            if htf_strength is not None:
                after_distributions['htf_carryover_strengths'].append(htf_strength)
                if htf_strength == 0.3:
                    after_distributions['default_value_counts']['htf_carryover_0.3'] += 1
            
            # Energy density
            energy_density = self._extract_energy_density(session_data)
            if energy_density is not None:
                after_distributions['energy_densities'].append(energy_density)
                if energy_density == 0.5:
                    after_distributions['default_value_counts']['energy_density_0.5'] += 1
            
            # Liquidity event counts
            event_count = len(session_data.get('session_liquidity_events', []))
            after_distributions['liquidity_event_counts'].append(event_count)
            if event_count == 0:
                after_distributions['default_value_counts']['empty_liquidity_events'] += 1
        
        # Calculate after statistics
        for feature_name in ['htf_carryover_strengths', 'energy_densities', 'liquidity_event_counts']:
            values = after_distributions[feature_name]
            if values:
                after_distributions[f'{feature_name}_stats'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'unique_values': len(set(values)),
                    'total_values': len(values)
                }
        
        # Calculate improvement metrics
        comparison_results = {
            'before_analysis': before_distributions,
            'after_analysis': after_distributions,
            'improvement_metrics': self._calculate_improvement_metrics(before_distributions, after_distributions),
            'decontamination_success': True,  # Will be determined by metrics
            'sessions_compared': len(enhanced_filenames)
        }
        
        # Determine overall decontamination success
        improvements = comparison_results['improvement_metrics']
        success_criteria = [
            improvements['default_contamination_reduction'] > 50,  # >50% reduction in defaults
            improvements['feature_diversity_improvement'] > 25,   # >25% increase in unique values
            improvements['distribution_variance_improvement'] > 0  # Increased variance (more diverse)
        ]
        
        comparison_results['decontamination_success'] = sum(success_criteria) >= 2  # At least 2/3 criteria
        
        return comparison_results
    
    def _calculate_improvement_metrics(self, before: Dict, after: Dict) -> Dict[str, float]:
        """Calculate improvement metrics between before/after distributions."""
        metrics = {}
        
        # Default value contamination reduction
        before_defaults = sum(before['default_value_counts'].values())
        after_defaults = sum(after['default_value_counts'].values())
        before_total = before['sessions_analyzed'] * 3  # 3 features per session
        after_total = after['sessions_analyzed'] * 3
        
        before_contamination_rate = (before_defaults / before_total * 100) if before_total > 0 else 0
        after_contamination_rate = (after_defaults / after_total * 100) if after_total > 0 else 0
        
        metrics['before_contamination_rate'] = before_contamination_rate
        metrics['after_contamination_rate'] = after_contamination_rate
        metrics['default_contamination_reduction'] = before_contamination_rate - after_contamination_rate
        
        # Feature diversity improvement (unique values)
        feature_diversity_improvements = []
        for feature_name in ['htf_carryover_strengths', 'energy_densities', 'liquidity_event_counts']:
            before_stats = before.get(f'{feature_name}_stats', {})
            after_stats = after.get(f'{feature_name}_stats', {})
            
            if before_stats and after_stats:
                before_unique = before_stats.get('unique_values', 1)
                after_unique = after_stats.get('unique_values', 1)
                diversity_improvement = ((after_unique - before_unique) / before_unique) * 100
                feature_diversity_improvements.append(diversity_improvement)
        
        metrics['feature_diversity_improvement'] = np.mean(feature_diversity_improvements) if feature_diversity_improvements else 0
        
        # Distribution variance improvement (indicates more natural variation)
        variance_improvements = []
        for feature_name in ['htf_carryover_strengths', 'energy_densities']:
            before_stats = before.get(f'{feature_name}_stats', {})
            after_stats = after.get(f'{feature_name}_stats', {})
            
            if before_stats and after_stats:
                before_std = before_stats.get('std', 0)
                after_std = after_stats.get('std', 0)
                if before_std > 0:
                    variance_improvement = ((after_std - before_std) / before_std) * 100
                    variance_improvements.append(variance_improvement)
        
        metrics['distribution_variance_improvement'] = np.mean(variance_improvements) if variance_improvements else 0
        
        return metrics
    
    def generate_decontamination_report(self, target_sessions: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive decontamination validation report.
        """
        logger.info("Generating comprehensive decontamination validation report")
        
        # Compare before/after distributions
        comparison_results = self.compare_before_after_enhancement(target_sessions)
        
        # Calculate pattern diversity metrics
        pattern_diversity = self._calculate_pattern_diversity(target_sessions)
        
        # Generate final report
        report = {
            'validation_timestamp': f"{pd.Timestamp.now()}",
            'validation_summary': {
                'total_sessions_analyzed': len(target_sessions),
                'enhanced_sessions_found': comparison_results['sessions_compared'],
                'decontamination_success': comparison_results['decontamination_success'],
                'overall_quality_score': self._calculate_overall_quality_score(comparison_results)
            },
            'contamination_analysis': comparison_results,
            'pattern_diversity_analysis': pattern_diversity,
            'recommendations': self._generate_recommendations(comparison_results),
            'next_steps': [
                "Apply enhancements to remaining high-quality sessions",
                "Validate TGAT model discovery quality on enhanced sessions",
                "Monitor pattern diversity in future discoveries",
                "Implement feature authenticity gates in data pipeline"
            ]
        }
        
        return report
    
    def _calculate_pattern_diversity(self, target_sessions: List[str]) -> Dict[str, Any]:
        """Calculate pattern diversity metrics to detect artificial duplication."""
        # This would analyze pattern discovery output if available
        # For now, return structure for future implementation
        return {
            'methodology': 'tgat_pattern_output_analysis',
            'status': 'pending_enhanced_discovery_run',
            'expected_improvements': [
                'Reduced pattern duplication from 96.8% to <20%',
                'Increased pattern description diversity',
                'More varied temporal relationships'
            ]
        }
    
    def _calculate_overall_quality_score(self, comparison_results: Dict) -> float:
        """Calculate overall decontamination quality score."""
        metrics = comparison_results['improvement_metrics']
        
        # Weight the different improvement metrics
        contamination_score = min(100, max(0, metrics['default_contamination_reduction'])) * 0.5
        diversity_score = min(100, max(0, metrics['feature_diversity_improvement'])) * 0.3
        variance_score = min(100, max(0, metrics['distribution_variance_improvement'] + 50)) * 0.2  # +50 to normalize
        
        overall_score = contamination_score + diversity_score + variance_score
        return round(overall_score, 1)
    
    def _generate_recommendations(self, comparison_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        improvements = comparison_results['improvement_metrics']
        
        if improvements['default_contamination_reduction'] < 50:
            recommendations.append("Increase contamination detection sensitivity - some default values may remain")
        
        if improvements['feature_diversity_improvement'] < 25:
            recommendations.append("Enhance feature calculation algorithms for more natural variation")
        
        if improvements['distribution_variance_improvement'] < 0:
            recommendations.append("Review authentic calculation methods - variance should increase with real market data")
        
        if comparison_results['decontamination_success']:
            recommendations.append("Proceed with TGAT model retraining on enhanced sessions")
            recommendations.append("Expand enhancement to remaining TGAT-ready sessions")
        else:
            recommendations.append("Address identified issues before proceeding to Phase 3")
        
        return recommendations


def main():
    """Main validation execution."""
    import pandas as pd
    
    logger.info("Starting Phase 2 Feature Authenticity Validation")
    
    validator = FeatureAuthenticityValidator()
    
    # Get high-quality session list for validation
    tgat_ready_sessions = [
        session['file'] for session in validator.quality_data['session_assessments'] 
        if session['tgat_readiness'] and session['quality_score'] >= 80
    ]
    
    # Run comprehensive validation
    validation_report = validator.generate_decontamination_report(tgat_ready_sessions)
    
    # Save validation report
    report_path = validator.enhanced_path / f"phase2_validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Display results
    print("\n" + "="*80)
    print("PHASE 2 FEATURE AUTHENTICITY VALIDATION REPORT")
    print("="*80)
    
    summary = validation_report['validation_summary']
    print(f"Sessions Analyzed: {summary['total_sessions_analyzed']}")
    print(f"Enhanced Sessions Found: {summary['enhanced_sessions_found']}")
    print(f"Decontamination Success: {summary['decontamination_success']}")
    print(f"Overall Quality Score: {summary['overall_quality_score']}/100")
    
    if 'improvement_metrics' in validation_report['contamination_analysis']:
        improvements = validation_report['contamination_analysis']['improvement_metrics']
        print(f"\nImprovement Metrics:")
        print(f"  Default Contamination Reduction: {improvements['default_contamination_reduction']:.1f}%")
        print(f"  Feature Diversity Improvement: {improvements['feature_diversity_improvement']:.1f}%")
        print(f"  Distribution Variance Improvement: {improvements['distribution_variance_improvement']:.1f}%")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(validation_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nValidation report saved to: {report_path}")
    
    return validation_report


if __name__ == "__main__":
    main()