#!/usr/bin/env python3
"""
🔄 IRONFORGE FPFVG Network Analysis Execution (Step 3A)
======================================================

Micro Mechanism Analysis: Prove FVGs form networks whose re-deliveries align with Theory B zones and PM belt timing.

Key Tests:
1. Zone enrichment: odds ratio of redeliveries in Theory B zones vs baseline
2. PM-belt interaction: P(redelivery hits 14:35-:38 | prior FVG) vs baseline  
3. Reproducibility: per-session bootstrap analysis with confidence intervals
4. Latency: time-to-redelivery survival curves (belt vs non-belt strata)

Statistical Framework:
- Fisher exact test for zone enrichment
- Chi-square test for PM belt interaction
- Bootstrap analysis for reproducibility
- Mann-Whitney U test for latency comparison

Usage:
    python run_fpfvg_network_analysis.py
"""

import logging
import sys
from pathlib import Path

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from ironforge.analysis.fpfvg_network_analyzer import FPFVGNetworkAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Execute FPFVG Network Analysis (Step 3A)"""
    
    print("🔄 IRONFORGE FPFVG Network Analysis (Step 3A)")
    print("=" * 60)
    print("Micro Mechanism: FVG redelivery networks align with Theory B zones")
    print("Focus: Network construction, zone enrichment, PM belt interaction")
    print()
    
    try:
        # Initialize FPFVG network analyzer
        analyzer = FPFVGNetworkAnalyzer()
        
        # Execute comprehensive network analysis
        logger.info("Starting FPFVG Network Analysis...")
        analysis_results = analyzer.analyze_fpfvg_network()
        
        # Display results summary
        if 'error' not in analysis_results:
            print("✅ FPFVG NETWORK ANALYSIS COMPLETE")
            print("=" * 45)
            
            # Extract metadata
            metadata = analysis_results.get('analysis_metadata', {})
            print(f"🔬 Analysis Step: {metadata.get('step', 'unknown')}")
            print(f"🎯 Focus: {metadata.get('focus', 'unknown')}")
            print(f"⏰ PM Belt Window: {metadata.get('pm_belt_window', 'unknown')}")
            print(f"📊 Theory B Zones: {metadata.get('theory_b_zones', [])}")
            print()
            
            # FPFVG Candidates Summary
            candidates_info = analysis_results.get('fpfvg_candidates', {})
            total_candidates = candidates_info.get('total_candidates', 0)
            candidate_stats = candidates_info.get('summary_stats', {})
            
            print(f"📈 Total FPFVG Candidates: {total_candidates}")
            
            if candidate_stats:
                by_type = candidate_stats.get('by_event_type', {})
                pm_belt_count = candidate_stats.get('pm_belt_count', 0)
                in_zone_count = candidate_stats.get('in_zone_count', 0)
                
                print("Event type distribution:")
                for event_type, count in by_type.items():
                    print(f"  {event_type.title():>12}: {count:>3} candidates")
                
                print(f"PM belt candidates: {pm_belt_count}")
                print(f"In Theory B zones: {in_zone_count}")
                
                if total_candidates > 0:
                    print(f"PM belt rate: {pm_belt_count/total_candidates:.3f}")
                    print(f"Zone enrichment rate: {in_zone_count/total_candidates:.3f}")
                print()
            
            # Network Construction Results
            network_info = analysis_results.get('network_construction', {})
            if network_info:
                print("🕸️  NETWORK CONSTRUCTION")
                print("-" * 20)
                
                nodes = network_info.get('nodes', 0)
                edges = network_info.get('edges', 0)
                density = network_info.get('network_density', 0)
                motifs = network_info.get('network_motifs', {})
                
                print(f"Network nodes: {nodes}")
                print(f"Network edges: {edges}")
                print(f"Network density: {density:.4f}")
                
                if motifs:
                    chains = motifs.get('chains', [])
                    convergences = motifs.get('convergences', [])
                    pm_touchpoints = motifs.get('pm_belt_touchpoints', [])
                    formation_paths = motifs.get('formation_to_redelivery_paths', [])
                    
                    print("Network motifs detected:")
                    print(f"  Chains (≥3 nodes): {len(chains)}")
                    print(f"  Convergences (k-in ≥2): {len(convergences)}")
                    print(f"  PM belt touchpoints: {len(pm_touchpoints)}")
                    print(f"  Formation→Redelivery paths: {len(formation_paths)}")
                print()
            
            # Redelivery Scoring Results
            scoring_info = analysis_results.get('redelivery_scoring', {})
            if scoring_info:
                print("💪 REDELIVERY STRENGTH SCORING")
                print("-" * 30)
                
                total_scored = scoring_info.get('total_scored_edges', 0)
                high_strength = scoring_info.get('high_strength_redeliveries', 0)
                score_dist = scoring_info.get('score_distribution', {})
                
                print(f"Total scored edges: {total_scored}")
                print(f"High strength redeliveries (>0.7): {high_strength}")
                
                if score_dist:
                    mean_score = score_dist.get('mean', 0)
                    median_score = score_dist.get('median', 0)
                    categories = score_dist.get('strength_categories', {})
                    
                    print("Score distribution:")
                    print(f"  Mean: {mean_score:.3f}")
                    print(f"  Median: {median_score:.3f}")
                    
                    if categories:
                        for category, count in categories.items():
                            print(f"  {category.replace('_', ' ').title():>15}: {count:>3} edges")
                print()
            
            # Statistical Test Results
            
            # Zone Enrichment Test
            zone_test = analysis_results.get('zone_enrichment_test', {})
            if zone_test and 'error' not in zone_test:
                print("📊 ZONE ENRICHMENT TEST")
                print("-" * 23)
                
                observed_in = zone_test.get('observed_in_zones', 0)
                observed_out = zone_test.get('observed_outside_zones', 0)
                expected_in = zone_test.get('expected_in_zones', 0)
                odds_ratio = zone_test.get('odds_ratio', 1.0)
                p_value = zone_test.get('p_value', 1.0)
                significant = zone_test.get('significant', False)
                enrichment_factor = zone_test.get('enrichment_factor', 1.0)
                
                print(f"Observed in zones: {observed_in}")
                print(f"Expected in zones: {expected_in:.1f}")
                print(f"Enrichment factor: {enrichment_factor:.3f}")
                print(f"Odds ratio: {odds_ratio:.3f}")
                print(f"P-value: {p_value:.6f}")
                print(f"Significant (α=0.05): {'✓ YES' if significant else '✗ NO'}")
                
                if significant:
                    print("🎯 ZONE ENRICHMENT DETECTED - Theory B validation!")
                print()
            
            # PM Belt Interaction Test
            pm_test = analysis_results.get('pm_belt_interaction_test', {})
            if pm_test and 'error' not in pm_test:
                print("⏰ PM BELT INTERACTION TEST")
                print("-" * 27)
                
                sessions_fvg = pm_test.get('sessions_with_fvg', 0)
                sessions_pm = pm_test.get('sessions_with_pm_belt_redelivery', 0)
                sessions_both = pm_test.get('sessions_with_both', 0)
                total_sessions = pm_test.get('total_sessions', 0)
                p_pm_given_fvg = pm_test.get('p_pm_given_fvg', 0)
                p_pm_baseline = pm_test.get('p_pm_baseline', 0)
                relative_risk = pm_test.get('relative_risk', 1.0)
                p_value = pm_test.get('p_value', 1.0)
                significant = pm_test.get('significant', False)
                
                print(f"Sessions with FVG: {sessions_fvg}")
                print(f"Sessions with PM belt redelivery: {sessions_pm}")
                print(f"Sessions with both: {sessions_both}")
                print(f"Total sessions: {total_sessions}")
                print(f"P(PM belt | FVG): {p_pm_given_fvg:.3f}")
                print(f"P(PM belt baseline): {p_pm_baseline:.3f}")
                print(f"Relative risk: {relative_risk:.3f}")
                print(f"P-value: {p_value:.6f}")
                print(f"Significant (α=0.05): {'✓ YES' if significant else '✗ NO'}")
                
                if significant:
                    print("🎯 PM BELT INTERACTION DETECTED - Timing validation!")
                print()
            
            # Reproducibility Test
            repro_test = analysis_results.get('reproducibility_test', {})
            if repro_test and 'error' not in repro_test:
                print("🔄 REPRODUCIBILITY ANALYSIS")
                print("-" * 26)
                
                sessions_analyzed = repro_test.get('sessions_analyzed', 0)
                reproducible = repro_test.get('reproducible', False)
                
                print(f"Sessions analyzed: {sessions_analyzed}")
                print(f"Reproducible: {'✓ YES' if reproducible else '✗ NO'}")
                
                # Bootstrap results
                for metric in ['zone_enrichment_rates', 'pm_belt_interaction_rates']:
                    bootstrap_key = f'{metric}_bootstrap'
                    if bootstrap_key in repro_test:
                        bootstrap_data = repro_test[bootstrap_key]
                        mean_val = bootstrap_data.get('mean', 0)
                        ci_lower = bootstrap_data.get('ci_lower', 0)
                        ci_upper = bootstrap_data.get('ci_upper', 0)
                        
                        metric_name = metric.replace('_', ' ').title()
                        print(f"{metric_name}:")
                        print(f"  Mean: {mean_val:.3f}")
                        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print()
            
            # Latency Analysis
            latency_test = analysis_results.get('latency_analysis', {})
            if latency_test and 'error' not in latency_test:
                print("⚡ REDELIVERY LATENCY ANALYSIS")
                print("-" * 30)
                
                total_pairs = latency_test.get('total_formation_redelivery_pairs', 0)
                belt_pairs = latency_test.get('pm_belt_pairs', 0)
                non_belt_pairs = latency_test.get('non_belt_pairs', 0)
                
                print(f"Formation→Redelivery pairs: {total_pairs}")
                print(f"PM belt pairs: {belt_pairs}")
                print(f"Non-belt pairs: {non_belt_pairs}")
                
                # Belt latencies
                belt_latencies = latency_test.get('pm_belt_latencies', {})
                if belt_latencies:
                    print("PM belt latencies:")
                    print(f"  Mean: {belt_latencies.get('mean', 0):.1f} minutes")
                    print(f"  Median: {belt_latencies.get('median', 0):.1f} minutes")
                
                # Non-belt latencies
                non_belt_latencies = latency_test.get('non_belt_latencies', {})
                if non_belt_latencies:
                    print("Non-belt latencies:")
                    print(f"  Mean: {non_belt_latencies.get('mean', 0):.1f} minutes")
                    print(f"  Median: {non_belt_latencies.get('median', 0):.1f} minutes")
                
                # Statistical comparison
                stat_comp = latency_test.get('statistical_comparison', {})
                if stat_comp and 'error' not in stat_comp:
                    p_value = stat_comp.get('p_value', 1.0)
                    significant = stat_comp.get('significant', False)
                    difference = stat_comp.get('belt_vs_non_belt_difference', 0)
                    
                    print("Statistical comparison:")
                    print(f"  Difference (belt - non-belt): {difference:.1f} minutes")
                    print(f"  P-value: {p_value:.6f}")
                    print(f"  Significant: {'✓ YES' if significant else '✗ NO'}")
                print()
            
            # Summary Insights
            insights = analysis_results.get('summary_insights', {})
            if insights:
                print("💡 SUMMARY INSIGHTS")
                print("-" * 16)
                
                # Key findings
                key_findings = insights.get('key_findings', [])
                if key_findings:
                    print("Key findings:")
                    for i, finding in enumerate(key_findings, 1):
                        print(f"  {i}. {finding.get('finding', 'Unknown finding')}")
                        if 'p_value' in finding:
                            print(f"     P-value: {finding['p_value']:.6f}")
                        if 'odds_ratio' in finding:
                            print(f"     Odds ratio: {finding['odds_ratio']:.3f}")
                        if 'relative_risk' in finding:
                            print(f"     Relative risk: {finding['relative_risk']:.3f}")
                
                # Theory B validation
                theory_b = insights.get('theory_b_validation', {})
                if theory_b:
                    validation_status = theory_b.get('validation_status', 'unknown')
                    enrichment_strength = theory_b.get('enrichment_strength', 1.0)
                    
                    print(f"Theory B validation: {validation_status}")
                    print(f"Enrichment strength: {enrichment_strength:.3f}")
                
                # PM belt evidence
                pm_evidence = insights.get('pm_belt_evidence', {})
                if pm_evidence:
                    evidence_strength = pm_evidence.get('evidence_strength', 'unknown')
                    risk_elevation = pm_evidence.get('risk_elevation', 1.0)
                    
                    print(f"PM belt evidence: {evidence_strength}")
                    print(f"Risk elevation: {risk_elevation:.3f}")
                
                print()
                
                # Recommendations
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    print("🚀 RECOMMENDATIONS")
                    print("-" * 15)
                    for i, rec in enumerate(recommendations, 1):
                        priority = rec.get('priority', 'unknown')
                        description = rec.get('description', 'No description')
                        action = rec.get('action', 'No action specified')
                        
                        print(f"  {i}. [{priority}] {description}")
                        print(f"     Action: {action}")
                    print()
            
            print("🔄 FPFVG Network Analysis (Step 3A) complete")
            print("Statistical validation of micro mechanism Theory B alignment")
            print("Deliverables: network summary, statistics, analysis results")
            
        else:
            print("❌ FPFVG NETWORK ANALYSIS FAILED")
            print(f"Error: {analysis_results.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"FPFVG network analysis execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)