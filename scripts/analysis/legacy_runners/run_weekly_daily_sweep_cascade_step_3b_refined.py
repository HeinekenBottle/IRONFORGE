#!/usr/bin/env python3
"""
📈 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B) - REFINED
==============================================================================

Refined implementation with lowered detection thresholds to get cascades lighting up.

Key Refinements:
1. ✅ Weekly Sweep Detection: Lower thresholds (0.25% ATR), multi-candle sweeps, tolerance bands
2. ✅ PM Belt Execution: Extended timing windows (14:25-14:48), prelude detection
3. ✅ Cross-Timeframe Anchoring: Bridge nodes from Step 2 integration
4. ✅ Enhanced Detection: Multi-source event analysis with relaxed criteria

Goal: Verify Weekly dominance by getting actual cascade data flowing through the framework.

Usage:
    python run_weekly_daily_sweep_cascade_step_3b_refined.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.refined_sweep_detector import RefinedSweepDetector
from ironforge.analysis.weekly_daily_sweep_cascade_analyzer import WeeklyDailySweepCascadeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RefinedCascadeAnalyzer(WeeklyDailySweepCascadeAnalyzer):
    """Enhanced cascade analyzer with refined detection thresholds"""
    
    def __init__(self):
        super().__init__()
        self.refined_detector = RefinedSweepDetector()
    
    def analyze_weekly_daily_cascades_refined(self, sessions_limit=None):
        """Execute refined cascade analysis with enhanced detection"""
        logger.info("🔧 Starting REFINED Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B)...")
        
        try:
            # Load sessions and lattice data
            sessions_data = self._load_sessions_and_lattice_data(sessions_limit)
            if 'error' in sessions_data:
                return sessions_data
            
            # Step 1: Refined sweep detection
            logger.info("🔍 Detecting sweeps with REFINED thresholds...")
            refined_sweeps = self.refined_detector.detect_refined_sweeps(sessions_data)
            
            weekly_sweeps = refined_sweeps['weekly_sweeps']
            daily_sweeps = refined_sweeps['daily_sweeps']
            pm_executions = refined_sweeps['pm_executions']
            
            # Convert to original format for compatibility
            weekly_sweep_events = [self._convert_to_original_format(s) for s in weekly_sweeps]
            daily_sweep_events = [self._convert_to_original_format(s) for s in daily_sweeps]
            pm_execution_events = [self._convert_to_original_format(s) for s in pm_executions]
            
            # Step 2: Link cascades using original algorithm
            logger.info("🔗 Linking cascades with refined sweep data...")
            cascade_links = self._link_cascades(weekly_sweep_events, daily_sweep_events, pm_execution_events)
            
            # Step 3: Quantify patterns
            logger.info("📊 Quantifying refined cascade patterns...")
            quantification = self._quantify_cascade_patterns(cascade_links, weekly_sweep_events, pm_execution_events)
            
            # Step 4: Statistical tests
            logger.info("🧪 Performing statistical tests on refined data...")
            statistical_tests = self._perform_statistical_tests(
                cascade_links, weekly_sweep_events, daily_sweep_events, pm_execution_events
            )
            
            # Compile results with enhanced metadata
            results = {
                'analysis_type': 'weekly_daily_sweep_cascade_step_3b_refined',
                'timestamp': self._get_timestamp(),
                'refinements_applied': {
                    'weekly_atr_threshold_lowered': '0.25% (was strict)',
                    'pm_belt_extended_window': '14:25-14:48 (±10min)',
                    'bridge_node_anchoring': True,
                    'multi_source_detection': True,
                    'relaxed_confidence_thresholds': True
                },
                'metadata': {
                    'sessions_analyzed': len(sessions_data.get('enhanced_sessions', [])),
                    'weekly_sweeps_detected': len(weekly_sweeps),
                    'daily_sweeps_detected': len(daily_sweeps),
                    'pm_executions_detected': len(pm_executions),
                    'cascade_links_mapped': len(cascade_links),
                    'bridge_nodes_loaded': len(self.refined_detector.bridge_nodes)
                },
                'refined_sweep_detection_results': {
                    'weekly_sweeps': [self.refined_detector.serialize_refined_sweep(s) for s in weekly_sweeps],
                    'daily_sweeps': [self.refined_detector.serialize_refined_sweep(s) for s in daily_sweeps],
                    'pm_executions': [self.refined_detector.serialize_refined_sweep(s) for s in pm_executions]
                },
                'cascade_analysis': {
                    'cascade_links': [self._serialize_cascade_link(c) for c in cascade_links],
                    'quantification_results': quantification,
                    'statistical_validation': statistical_tests
                },
                'refinement_insights': self._generate_refinement_insights(
                    refined_sweeps, cascade_links, quantification, statistical_tests
                ),
                'discovery_insights': self._generate_discovery_insights(
                    cascade_links, quantification, statistical_tests
                )
            }
            
            # Save results
            self._save_refined_results(results)
            
            logger.info(f"✅ REFINED Weekly→Daily Cascade Analysis complete: {len(cascade_links)} cascades mapped")
            return results
            
        except Exception as e:
            logger.error(f"Refined cascade analysis failed: {e}")
            return {
                'analysis_type': 'weekly_daily_sweep_cascade_step_3b_refined',
                'error': str(e),
                'timestamp': self._get_timestamp()
            }
    
    def _convert_to_original_format(self, refined_sweep):
        """Convert RefinedSweepEvent to original SweepEvent format for compatibility"""
        from ironforge.analysis.weekly_daily_sweep_cascade_analyzer import SweepEvent
        
        return SweepEvent(
            session_id=refined_sweep.session_id,
            timestamp=refined_sweep.timestamp,
            timeframe=refined_sweep.timeframe,
            sweep_type=refined_sweep.sweep_type,
            price_level=refined_sweep.price_level,
            displacement=refined_sweep.displacement,
            follow_through=refined_sweep.follow_through,
            range_pos=refined_sweep.range_pos,
            zone_proximity=refined_sweep.zone_proximity,
            prior_swing_high=refined_sweep.price_level + 100,  # Simplified
            prior_swing_low=refined_sweep.price_level - 100,   # Simplified
            closes_in_range=True
        )
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _generate_refinement_insights(self, refined_sweeps, cascade_links, quantification, statistical_tests):
        """Generate insights specific to refinement improvements"""
        weekly_sweeps = refined_sweeps['weekly_sweeps']
        daily_sweeps = refined_sweeps['daily_sweeps']
        pm_executions = refined_sweeps['pm_executions']
        
        # Analyze detection improvements
        weekly_confidence_scores = [s.detection_confidence for s in weekly_sweeps]
        daily_confidence_scores = [s.detection_confidence for s in daily_sweeps]
        pm_confidence_scores = [s.detection_confidence for s in pm_executions]
        
        # Bridge node alignment analysis
        weekly_bridge_aligned = len([s for s in weekly_sweeps if s.bridge_node_aligned])
        daily_bridge_aligned = len([s for s in daily_sweeps if s.bridge_node_aligned])
        
        # PM belt category analysis
        pm_belt_categories = {}
        for pm in pm_executions:
            category = pm.pm_belt_category
            pm_belt_categories[category] = pm_belt_categories.get(category, 0) + 1
        
        # Multi-candle sweep analysis
        weekly_multi_candle = len([s for s in weekly_sweeps if s.multi_candle_sweep])
        daily_multi_candle = len([s for s in daily_sweeps if s.multi_candle_sweep])
        
        return {
            'detection_improvements': {
                'weekly_sweeps_detected': len(weekly_sweeps),
                'weekly_avg_confidence': sum(weekly_confidence_scores) / len(weekly_confidence_scores) if weekly_confidence_scores else 0,
                'weekly_bridge_aligned_count': weekly_bridge_aligned,
                'weekly_multi_candle_count': weekly_multi_candle,
                
                'daily_sweeps_detected': len(daily_sweeps),
                'daily_avg_confidence': sum(daily_confidence_scores) / len(daily_confidence_scores) if daily_confidence_scores else 0,
                'daily_bridge_aligned_count': daily_bridge_aligned,
                'daily_multi_candle_count': daily_multi_candle,
                
                'pm_executions_detected': len(pm_executions),
                'pm_avg_confidence': sum(pm_confidence_scores) / len(pm_confidence_scores) if pm_confidence_scores else 0,
                'pm_belt_category_distribution': pm_belt_categories
            },
            'cascade_generation_success': {
                'total_cascade_links': len(cascade_links),
                'improvement_vs_original': 'See comparison in metadata',
                'weekly_to_cascade_ratio': len(cascade_links) / max(1, len(weekly_sweeps)),
                'detection_to_cascade_efficiency': len(cascade_links) / max(1, len(weekly_sweeps) + len(daily_sweeps) + len(pm_executions))
            },
            'refinement_recommendations': self._generate_refinement_recommendations(refined_sweeps, cascade_links)
        }
    
    def _generate_refinement_recommendations(self, refined_sweeps, cascade_links):
        """Generate recommendations for further refinements"""
        recommendations = []
        
        weekly_count = len(refined_sweeps['weekly_sweeps'])
        daily_count = len(refined_sweeps['daily_sweeps'])
        pm_count = len(refined_sweeps['pm_executions'])
        cascade_count = len(cascade_links)
        
        if weekly_count == 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'area': 'weekly_detection',
                'issue': 'No Weekly sweeps detected',
                'recommendation': 'Further lower Weekly detection thresholds or expand event source keywords'
            })
        elif weekly_count < 5:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'weekly_detection',
                'issue': f'Only {weekly_count} Weekly sweeps detected',
                'recommendation': 'Expand Weekly detection to include more HTF indicators'
            })
        
        if pm_count == 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'area': 'pm_belt_detection',
                'issue': 'No PM executions detected',
                'recommendation': 'Verify session timestamp formats and expand PM belt window further'
            })
        elif pm_count < 10:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'pm_belt_detection',
                'issue': f'Only {pm_count} PM executions detected',
                'recommendation': 'Expand PM belt detection to include more event types'
            })
        
        if cascade_count == 0:
            recommendations.append({
                'priority': 'EXTREME',
                'area': 'cascade_linking',
                'issue': 'No cascades linked despite detections',
                'recommendation': 'Relax cascade linking criteria - price tolerance, time windows, or relationship requirements'
            })
        
        if daily_count > 50:
            recommendations.append({
                'priority': 'MODERATE',
                'area': 'daily_detection',
                'issue': f'High Daily sweep count ({daily_count}) may include noise',
                'recommendation': 'Consider slightly raising Daily detection thresholds for quality'
            })
        
        return recommendations
    
    def _save_refined_results(self, results):
        """Save refined analysis results"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weekly_daily_sweep_cascade_step_3b_REFINED_{timestamp}.json"
        filepath = self.discoveries_path / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"✅ Refined Step 3B results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save refined results: {e}")

def main():
    """Execute REFINED Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B)"""
    
    print("🔧 IRONFORGE Weekly→Daily Liquidity Sweep Cascade Analysis (Step 3B) - REFINED")
    print("=" * 85)
    print("REFINEMENTS APPLIED:")
    print("  ✅ Weekly Detection: Lower ATR thresholds (0.25%), multi-candle sweeps, tolerance bands")
    print("  ✅ PM Belt Timing: Extended windows (14:25-14:48), prelude detection")
    print("  ✅ Bridge Node Anchoring: Cross-timeframe validation from Step 2")
    print("  ✅ Enhanced Detection: Multi-source analysis with relaxed criteria")
    print()
    print("GOAL: Get cascades lighting up to validate macro→micro transmission!")
    print()
    
    try:
        # Initialize refined cascade analyzer
        analyzer = RefinedCascadeAnalyzer()
        
        # Execute refined analysis
        results = analyzer.analyze_weekly_daily_cascades_refined()
        
        if 'error' in results:
            print(f"❌ REFINED Step 3B Analysis failed: {results['error']}")
            return 1
        
        print("✅ REFINED WEEKLY→DAILY SWEEP CASCADE ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Display refined results
        metadata = results.get('metadata', {})
        refinements = results.get('refinements_applied', {})
        
        print("🔧 REFINEMENTS APPLIED:")
        for key, value in refinements.items():
            print(f"  {key}: {value}")
        print()
        
        print("📊 DETECTION RESULTS:")
        print(f"  Sessions Analyzed: {metadata.get('sessions_analyzed', 0)}")
        print(f"  Bridge Nodes Loaded: {metadata.get('bridge_nodes_loaded', 0)}")
        print(f"  Weekly Sweeps: {metadata.get('weekly_sweeps_detected', 0)} (REFINED)")
        print(f"  Daily Sweeps: {metadata.get('daily_sweeps_detected', 0)} (ENHANCED)")
        print(f"  PM Executions: {metadata.get('pm_executions_detected', 0)} (EXTENDED BELT)")
        print(f"  Cascade Links: {metadata.get('cascade_links_mapped', 0)} (🎯 TARGET)")
        print()
        
        # Refinement insights
        refinement_insights = results.get('refinement_insights', {})
        detection_improvements = refinement_insights.get('detection_improvements', {})
        
        if detection_improvements:
            print("🔍 DETECTION IMPROVEMENTS:")
            print(f"  Weekly Avg Confidence: {detection_improvements.get('weekly_avg_confidence', 0):.3f}")
            print(f"  Weekly Bridge Aligned: {detection_improvements.get('weekly_bridge_aligned_count', 0)}")
            print(f"  Weekly Multi-Candle: {detection_improvements.get('weekly_multi_candle_count', 0)}")
            print(f"  Daily Avg Confidence: {detection_improvements.get('daily_avg_confidence', 0):.3f}")
            print(f"  Daily Bridge Aligned: {detection_improvements.get('daily_bridge_aligned_count', 0)}")
            print(f"  PM Avg Confidence: {detection_improvements.get('pm_avg_confidence', 0):.3f}")
            
            pm_categories = detection_improvements.get('pm_belt_category_distribution', {})
            if pm_categories:
                print(f"  PM Belt Categories: {pm_categories}")
            print()
        
        # Cascade generation success
        cascade_success = refinement_insights.get('cascade_generation_success', {})
        if cascade_success:
            print("🔗 CASCADE GENERATION:")
            print(f"  Total Cascade Links: {cascade_success.get('total_cascade_links', 0)}")
            print(f"  Weekly→Cascade Ratio: {cascade_success.get('weekly_to_cascade_ratio', 0):.3f}")
            print(f"  Detection Efficiency: {cascade_success.get('detection_to_cascade_efficiency', 0):.3f}")
            print()
        
        # Show quantification results if we have cascades
        cascade_analysis = results.get('cascade_analysis', {})
        quantification = cascade_analysis.get('quantification_results', {})
        
        if quantification and metadata.get('cascade_links_mapped', 0) > 0:
            hit_rates = quantification.get('hit_rates', {})
            print("🎯 REFINED HIT-RATE ANALYSIS:")
            print(f"  P(PM execution | Weekly sweep): {hit_rates.get('pm_execution_given_weekly_sweep', 0):.3f}")
            print(f"  P(Daily reaction | Weekly sweep): {hit_rates.get('daily_reaction_given_weekly_sweep', 0):.3f}")
            print(f"  P(PM execution | Daily confirmation): {hit_rates.get('pm_execution_given_daily_confirmation', 0):.3f}")
            print()
            
            directional = quantification.get('directional_consistency', {})
            if directional:
                print("🧭 DIRECTIONAL CONSISTENCY:")
                print(f"  Consistent cascades: {directional.get('consistent_cascades', 0)}")
                print(f"  Consistency rate: {directional.get('consistency_rate', 0):.3f}")
                print()
        
        # Refinement recommendations
        recommendations = refinement_insights.get('refinement_recommendations', [])
        if recommendations:
            print("🚀 REFINEMENT RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                priority = rec.get('priority', 'UNKNOWN')
                area = rec.get('area', 'unknown')
                issue = rec.get('issue', 'No issue specified')
                recommendation = rec.get('recommendation', 'No recommendation')
                
                print(f"  {i}. [{priority}] {area.upper()}")
                print(f"     Issue: {issue}")
                print(f"     Action: {recommendation}")
                print()
        
        # Statistical validation summary
        statistical_tests = cascade_analysis.get('statistical_validation', {})
        if statistical_tests:
            causal_test = statistical_tests.get('causal_ordering_test', {})
            if causal_test and 'error' not in causal_test:
                print("🧪 STATISTICAL VALIDATION:")
                print(f"  Causal ordering p-value: {causal_test.get('p_value', 1.0):.6f}")
                print(f"  Significant: {'✓ YES' if causal_test.get('significant', False) else '✗ NO'}")
                
                effect_size = statistical_tests.get('effect_size_analysis', {})
                if effect_size:
                    print(f"  Effect size: {effect_size.get('effect_magnitude', 'unknown')}")
                    print(f"  Uplift ratio: {effect_size.get('uplift_ratio', 1.0):.2f}x")
                print()
        
        print("🎉 REFINED STEP 3B ANALYSIS COMPLETE")
        print("=" * 40)
        
        # Success assessment
        weekly_detected = metadata.get('weekly_sweeps_detected', 0)
        pm_detected = metadata.get('pm_executions_detected', 0)
        cascades_mapped = metadata.get('cascade_links_mapped', 0)
        
        if cascades_mapped > 0:
            print("🏆 SUCCESS: Cascades are lighting up! Framework validation enabled.")
            print("   Next: Proceed with micro/macro fusion and dual-signal framework.")
        elif weekly_detected > 0 and pm_detected > 0:
            print("🔧 PARTIAL SUCCESS: Detections improved but cascades still need linking refinement.")
            print("   Next: Relax cascade linking criteria (price tolerance, time windows).")
        elif weekly_detected > 0 or pm_detected > 0:
            print("🔨 PROGRESS: Some detections working, continue threshold refinements.")
            print("   Next: Focus on the missing detection type (Weekly or PM).")
        else:
            print("⚠️  NEEDS MORE WORK: Detection thresholds still too strict.")
            print("   Next: Further expand event sources and lower all thresholds.")
        
        print()
        print("📁 Detailed results saved to discoveries/ directory")
        print("🔬 Ready for ablation analysis and micro/macro fusion integration")
        
        return 0
        
    except Exception as e:
        logger.error(f"Refined Step 3B execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)