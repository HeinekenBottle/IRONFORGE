#!/usr/bin/env python3
"""
Theory B Temporal Non-Locality Validation Framework
Rigorous statistical backtesting for archaeological zone precision claims

TQE Validation Specialist - Comprehensive validation of the 7.55 vs 30.80 point accuracy (4X improvement)
across all available sessions with statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from datetime import datetime
import json
from pathlib import Path
import logging

from archaeological_zone_calculator import ArchaeologicalZoneCalculator, ZoneEvent

@dataclass
class ValidationResult:
    """Statistical validation result for Theory B hypothesis"""
    session_id: str
    event_price: float
    event_time: str
    final_range_distance: float
    current_range_distance: float
    improvement_ratio: float
    meets_theory_b_threshold: bool
    precision_score: float
    session_progress_pct: float
    zone_type: str

@dataclass
class StatisticalSummary:
    """Comprehensive statistical summary of validation results"""
    total_events: int
    theory_b_events: int
    theory_b_rate: float
    mean_improvement_ratio: float
    median_improvement_ratio: float
    std_improvement_ratio: float
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    power: float

class TheoryBValidationFramework:
    """
    Comprehensive validation framework for Theory B temporal non-locality hypothesis
    
    Key Hypothesis: Events position with 7.55-point precision to final session ranges
    before the session range is fully established (4X more accurate than current positioning)
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.calculator = ArchaeologicalZoneCalculator()
        self.significance_level = significance_level
        self.validation_results = []
        
        # Theory B validation constants
        self.PRECISION_THRESHOLD = 7.55  # Points
        self.EXPECTED_IMPROVEMENT_RATIO = 4.0  # 4X improvement claim
        self.MIN_EVENTS_FOR_SIGNIFICANCE = 30  # Minimum sample size
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_session_data(self, data_path: str) -> Dict[str, Any]:
        """Load and parse session data from various IRONFORGE data sources"""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load session data from {data_path}: {e}")
            return {}
    
    def extract_events_from_session(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract archaeological zone events from session data
        
        Handles multiple data formats from IRONFORGE infrastructure
        """
        events = []
        
        # Handle enhanced session format
        if 'events' in session_data:
            events.extend(session_data['events'])
        
        # Handle FPFVG format from gauntlet analysis
        if 'fpfvg_events' in session_data:
            for fpfvg in session_data['fpfvg_events']:
                if 'archaeological_zones' in fpfvg:
                    events.extend(fpfvg['archaeological_zones'])
        
        # Handle archaeological confluence format
        if 'archaeological_confluences' in session_data:
            events.extend(session_data['archaeological_confluences'])
        
        # Filter for events with required fields
        valid_events = []
        for event in events:
            if all(key in event for key in ['price', 'timestamp']):
                # Ensure we have session progress information
                if 'session_progress_pct' not in event:
                    # Calculate from timestamp if session bounds available
                    event['session_progress_pct'] = self._estimate_session_progress(
                        event['timestamp'], session_data.get('session_start'), session_data.get('session_end')
                    )
                valid_events.append(event)
        
        return valid_events
    
    def _estimate_session_progress(self, event_time: str, session_start: Optional[str], session_end: Optional[str]) -> float:
        """Estimate session progress percentage from timestamp"""
        if not session_start or not session_end:
            return 50.0  # Default to mid-session if bounds unknown
        
        try:
            event_dt = datetime.strptime(event_time, "%H:%M:%S")
            start_dt = datetime.strptime(session_start, "%H:%M:%S") 
            end_dt = datetime.strptime(session_end, "%H:%M:%S")
            
            total_duration = (end_dt - start_dt).total_seconds()
            elapsed = (event_dt - start_dt).total_seconds()
            
            return max(0, min(100, (elapsed / total_duration) * 100))
        except:
            return 50.0
    
    def validate_single_session(self, session_data: Dict[str, Any], session_id: str) -> List[ValidationResult]:
        """
        Validate Theory B temporal non-locality for a single session
        
        Args:
            session_data: Complete session data including events and final statistics
            session_id: Unique identifier for the session
            
        Returns:
            List of validation results for events in this session
        """
        results = []
        
        # Extract session statistics
        session_stats = session_data.get('session_stats', {})
        if not session_stats or 'session_high' not in session_stats:
            self.logger.warning(f"Session {session_id}: Missing session statistics")
            return results
        
        # Extract events
        events = self.extract_events_from_session(session_data)
        if not events:
            self.logger.warning(f"Session {session_id}: No valid events found")
            return results
        
        session_high = session_stats['session_high']
        session_low = session_stats['session_low']
        session_range = session_high - session_low
        
        if session_range <= 0:
            self.logger.warning(f"Session {session_id}: Invalid session range")
            return results
        
        self.logger.info(f"Validating {len(events)} events in session {session_id}")
        
        for event in events:
            # TODO(human): Implement the core Theory B validation logic for each event
            # Calculate both final range positioning and current range positioning
            # Compare precision: distance to final 40% zone vs distance to current 40% zone
            # This is where we test the 7.55 vs 30.80 point accuracy claim
            pass
        
        return results
    
    def run_comprehensive_validation(self, data_directory: str) -> StatisticalSummary:
        """
        Run comprehensive validation across all available session data
        
        Args:
            data_directory: Directory containing session data files
            
        Returns:
            Comprehensive statistical summary of Theory B validation
        """
        self.logger.info("Starting comprehensive Theory B validation")
        
        data_path = Path(data_directory)
        session_files = list(data_path.rglob("*.json"))
        
        self.logger.info(f"Found {len(session_files)} potential data files")
        
        all_results = []
        sessions_processed = 0
        
        for file_path in session_files:
            try:
                session_data = self.load_session_data(str(file_path))
                if not session_data:
                    continue
                
                session_id = f"{file_path.stem}_{sessions_processed}"
                session_results = self.validate_single_session(session_data, session_id)
                
                if session_results:
                    all_results.extend(session_results)
                    sessions_processed += 1
                    
                    if sessions_processed % 10 == 0:
                        self.logger.info(f"Processed {sessions_processed} sessions, {len(all_results)} total events")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        self.validation_results = all_results
        self.logger.info(f"Validation complete: {sessions_processed} sessions, {len(all_results)} events")
        
        if len(all_results) < self.MIN_EVENTS_FOR_SIGNIFICANCE:
            self.logger.warning(f"Insufficient events for statistical significance: {len(all_results)} < {self.MIN_EVENTS_FOR_SIGNIFICANCE}")
        
        return self.calculate_statistical_summary(all_results)
    
    def calculate_statistical_summary(self, results: List[ValidationResult]) -> StatisticalSummary:
        """
        Calculate comprehensive statistical summary with hypothesis testing
        
        Args:
            results: List of validation results from all sessions
            
        Returns:
            Statistical summary with significance tests
        """
        if not results:
            return StatisticalSummary(0, 0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 1.0, 0.0, 0.0)
        
        # Extract key metrics
        improvement_ratios = [r.improvement_ratio for r in results]
        theory_b_events = [r for r in results if r.meets_theory_b_threshold]
        
        # Basic statistics
        total_events = len(results)
        theory_b_count = len(theory_b_events)
        theory_b_rate = theory_b_count / total_events
        
        mean_ratio = np.mean(improvement_ratios)
        median_ratio = np.median(improvement_ratios)
        std_ratio = np.std(improvement_ratios, ddof=1)
        
        # Confidence interval
        confidence_interval = stats.t.interval(
            1 - self.significance_level,
            len(improvement_ratios) - 1,
            loc=mean_ratio,
            scale=stats.sem(improvement_ratios)
        )
        
        # Hypothesis test: H0: mean_ratio <= 1.0, H1: mean_ratio > expected_improvement
        t_stat, p_value = stats.ttest_1samp(improvement_ratios, 1.0, alternative='greater')
        
        # Effect size (Cohen's d)
        effect_size = (mean_ratio - 1.0) / std_ratio if std_ratio > 0 else 0.0
        
        # Statistical power (approximate)
        power = self._calculate_statistical_power(improvement_ratios, self.EXPECTED_IMPROVEMENT_RATIO)
        
        return StatisticalSummary(
            total_events=total_events,
            theory_b_events=theory_b_count,
            theory_b_rate=theory_b_rate,
            mean_improvement_ratio=mean_ratio,
            median_improvement_ratio=median_ratio,
            std_improvement_ratio=std_ratio,
            confidence_interval=confidence_interval,
            p_value=p_value,
            effect_size=effect_size,
            power=power
        )
    
    def _calculate_statistical_power(self, data: List[float], expected_effect: float) -> float:
        """Calculate statistical power for the hypothesis test"""
        try:
            n = len(data)
            std_err = np.std(data, ddof=1) / np.sqrt(n)
            z_alpha = stats.norm.ppf(1 - self.significance_level)
            z_beta = (expected_effect - 1.0) / std_err - z_alpha
            power = stats.norm.cdf(z_beta)
            return max(0.0, min(1.0, power))
        except:
            return 0.0
    
    def generate_validation_report(self, output_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            output_path: Path to save the validation report
            
        Returns:
            Complete validation report dictionary
        """
        if not self.validation_results:
            self.logger.error("No validation results available. Run validation first.")
            return {}
        
        summary = self.calculate_statistical_summary(self.validation_results)
        
        # Theory B specific metrics
        theory_b_events = [r for r in self.validation_results if r.meets_theory_b_threshold]
        precision_scores = [r.precision_score for r in theory_b_events]
        
        # Session progress analysis
        early_events = [r for r in theory_b_events if r.session_progress_pct < 50]
        late_events = [r for r in theory_b_events if r.session_progress_pct >= 50]
        
        # Zone type breakdown
        zone_breakdown = {}
        for result in theory_b_events:
            zone = result.zone_type
            if zone not in zone_breakdown:
                zone_breakdown[zone] = 0
            zone_breakdown[zone] += 1
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "framework_version": "1.0",
            "hypothesis": "Theory B Temporal Non-Locality: Events position with 7.55-point precision to final session ranges",
            "statistical_summary": {
                "total_events_analyzed": summary.total_events,
                "theory_b_events_found": summary.theory_b_events,
                "theory_b_success_rate": round(summary.theory_b_rate * 100, 2),
                "mean_improvement_ratio": round(summary.mean_improvement_ratio, 3),
                "median_improvement_ratio": round(summary.median_improvement_ratio, 3),
                "confidence_interval_95pct": [round(summary.confidence_interval[0], 3), round(summary.confidence_interval[1], 3)],
                "p_value": round(summary.p_value, 6),
                "effect_size_cohens_d": round(summary.effect_size, 3),
                "statistical_power": round(summary.power, 3)
            },
            "theory_b_analysis": {
                "precision_threshold_met": summary.theory_b_events > 0,
                "expected_vs_actual_ratio": f"{self.EXPECTED_IMPROVEMENT_RATIO}x expected, {summary.mean_improvement_ratio:.2f}x observed",
                "statistical_significance": "Significant" if summary.p_value < self.significance_level else "Not Significant",
                "practical_significance": "High" if summary.effect_size > 0.8 else "Medium" if summary.effect_size > 0.5 else "Low",
                "average_precision_score": round(np.mean(precision_scores), 3) if precision_scores else 0.0,
                "best_precision_event": max(theory_b_events, key=lambda x: x.precision_score) if theory_b_events else None
            },
            "temporal_analysis": {
                "early_session_events": len(early_events),
                "late_session_events": len(late_events),
                "early_session_rate": round(len(early_events) / len(theory_b_events) * 100, 1) if theory_b_events else 0.0,
                "temporal_non_locality_confirmed": len(early_events) > len(late_events)
            },
            "zone_type_breakdown": zone_breakdown,
            "recommendations": self._generate_recommendations(summary),
            "raw_data_sample": [
                {
                    "session": r.session_id,
                    "improvement_ratio": round(r.improvement_ratio, 2),
                    "precision_score": round(r.precision_score, 3),
                    "zone_type": r.zone_type
                }
                for r in theory_b_events[:10]  # Top 10 examples
            ]
        }
        
        # Save report
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Validation report saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
        
        return report
    
    def _generate_recommendations(self, summary: StatisticalSummary) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        if summary.p_value < self.significance_level:
            if summary.mean_improvement_ratio >= self.EXPECTED_IMPROVEMENT_RATIO:
                recommendations.append("✅ Theory B hypothesis CONFIRMED with statistical significance")
                recommendations.append(f"Observed improvement ratio ({summary.mean_improvement_ratio:.2f}x) meets or exceeds expected ({self.EXPECTED_IMPROVEMENT_RATIO}x)")
            else:
                recommendations.append("⚠️ Statistical significance achieved but improvement ratio below expected")
                recommendations.append("Consider refining Theory B parameters or investigating session-specific factors")
        else:
            recommendations.append("❌ Theory B hypothesis NOT CONFIRMED - insufficient statistical evidence")
            recommendations.append(f"Need larger sample size (current: {summary.total_events}, minimum: {self.MIN_EVENTS_FOR_SIGNIFICANCE})")
        
        if summary.effect_size > 0.8:
            recommendations.append("💪 Large effect size indicates strong practical significance")
        elif summary.effect_size < 0.3:
            recommendations.append("⚠️ Small effect size - practical significance questionable")
        
        if summary.power < 0.8:
            recommendations.append("⚠️ Low statistical power - consider increasing sample size")
        
        return recommendations

def main():
    """Demonstrate the validation framework"""
    print("🔬 Theory B Temporal Non-Locality Validation Framework")
    print("=" * 70)
    
    framework = TheoryBValidationFramework()
    
    # Run validation on available data
    data_directory = "/Users/jack/IRONFORGE/data"
    summary = framework.run_comprehensive_validation(data_directory)
    
    # Generate report
    report_path = f"/Users/jack/IRONFORGE/theory_b_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = framework.generate_validation_report(report_path)
    
    print(f"\n📊 VALIDATION RESULTS SUMMARY")
    print(f"Total Events Analyzed: {summary.total_events}")
    print(f"Theory B Events Found: {summary.theory_b_events}")
    print(f"Success Rate: {summary.theory_b_rate * 100:.1f}%")
    print(f"Mean Improvement Ratio: {summary.mean_improvement_ratio:.2f}x")
    print(f"Statistical Significance: p = {summary.p_value:.6f}")
    print(f"Effect Size (Cohen's d): {summary.effect_size:.3f}")
    print(f"Statistical Power: {summary.power:.3f}")
    
    if report:
        print(f"\n📋 Full report saved to: {report_path}")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")

if __name__ == "__main__":
    main()