#!/usr/bin/env python3
"""
Phase 5: Archaeological Discovery Validation
==========================================
Test TGAT model pattern discovery capability on authentic enhanced features 
after Phase 2 decontamination to validate restoration of genuine archaeological 
discovery vs previous 96.8% duplication artifacts.

Critical Success Metrics:
- Pattern Duplication Rate: Target <20% (vs 96.8% with contaminated features)
- Unique Pattern Descriptions: Target >50 unique (vs 13 with contaminated)
- Time Span Validation: Patterns with >0.0 hour spans and realistic temporal distributions
- Cross-Session Relationships: Properly identified and linked sessions

Success Threshold:
If duplication drops significantly (<50%) and time spans become realistic,
proceed with full validation. If still producing 96.8% duplication-like artifacts,
recommend Phase 4 model retraining.
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# Add IRONFORGE to path for component access
ironforge_root = Path(__file__).parent
sys.path.insert(0, str(ironforge_root))

# Iron-core imports for shared infrastructure
ironpulse_root = ironforge_root.parent
iron_core_path = ironpulse_root / 'iron_core'
if str(ironpulse_root) not in sys.path:
    sys.path.insert(0, str(ironpulse_root))

from iron_core.performance.container import get_container


class Phase5ArchaeologicalValidator:
    """
    Phase 5 TGAT Archaeological Discovery Validation
    Tests pattern discovery on authentic enhanced features
    """
    
    def __init__(self):
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'test_sessions': [],
            'discovery_results': {},
            'quality_metrics': {},
            'comparative_analysis': {},
            'authenticity_assessment': {}
        }
        
        # Top 5 highest quality enhanced sessions for testing
        self.test_sessions = [
            'enhanced_NY_PM_Lvl-1_2025_07_29.json',
            'enhanced_ASIA_Lvl-1_2025_07_30.json', 
            'enhanced_NY_AM_Lvl-1_2025_07_25.json',
            'enhanced_LONDON_Lvl-1_2025_07_28.json',
            'enhanced_LONDON_Lvl-1_2025_07_25.json'
        ]
        
        self.enhanced_sessions_path = ironforge_root / 'enhanced_sessions'
        
    def load_enhanced_session(self, session_filename: str) -> dict[str, Any]:
        """Load and validate enhanced session data"""
        session_path = self.enhanced_sessions_path / session_filename
        
        if not session_path.exists():
            raise FileNotFoundError(f"Enhanced session not found: {session_path}")
            
        with open(session_path) as f:
            session_data = json.load(f)
            
        # Validate this is an authentic enhanced session
        if 'phase2_enhancement' not in session_data:
            raise ValueError(f"Session {session_filename} lacks Phase 2 enhancement metadata")
            
        enhancement_info = session_data['phase2_enhancement']
        if enhancement_info.get('post_enhancement_score', 0) != 100.0:
            print(f"⚠️ Warning: {session_filename} has quality score {enhancement_info.get('post_enhancement_score')}")
            
        return session_data
        
    def validate_feature_authenticity(self, session_data: dict[str, Any]) -> dict[str, Any]:
        """Validate that critical features have been decontaminated"""
        validation = {
            'htf_carryover_authentic': False,
            'energy_density_authentic': False, 
            'liquidity_events_authentic': False,
            'overall_authentic': False
        }
        
        # Check HTF carryover strength (should not be 0.3 default)
        contamination = session_data.get('contamination_analysis', {})
        htf_contamination = contamination.get('htf_contamination', {})
        htf_strength = htf_contamination.get('htf_carryover_strength', 0.3)
        
        validation['htf_carryover_authentic'] = htf_strength != 0.3
        validation['htf_carryover_value'] = htf_strength
        
        # Check energy density (should not be 0.5 default)
        energy_state = session_data.get('energy_state', {})
        energy_density = energy_state.get('energy_density', 0.5)
        
        validation['energy_density_authentic'] = energy_density != 0.5
        validation['energy_density_value'] = energy_density
        
        # Check session liquidity events (should not be empty)
        liquidity_events = session_data.get('session_liquidity_events', [])
        
        validation['liquidity_events_authentic'] = len(liquidity_events) > 0
        validation['liquidity_events_count'] = len(liquidity_events)
        
        # Overall authenticity assessment
        validation['overall_authentic'] = all([
            validation['htf_carryover_authentic'],
            validation['energy_density_authentic'], 
            validation['liquidity_events_authentic']
        ])
        
        return validation
        
    def run_tgat_discovery(self, session_data: dict[str, Any], session_name: str) -> dict[str, Any]:
        """Run TGAT archaeological discovery on enhanced session"""
        print(f"🔍 Running TGAT discovery on {session_name}...")
        
        try:
            # Initialize IRONFORGE container for TGAT access
            container = get_container()
            
            # Get TGAT discovery engine
            tgat_discovery = container.get_mathematical_component('tgat_discovery')
            
            # Run pattern discovery (no retraining, existing sophisticated model)
            patterns = tgat_discovery.discover_session_patterns(session_data)
            
            discovery_result = {
                'session_name': session_name,
                'discovery_success': True,
                'patterns_found': len(patterns) if patterns else 0,
                'patterns': patterns or [],
                'discovery_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ TGAT discovery completed: {len(patterns or [])} patterns found")
            return discovery_result
            
        except Exception as e:
            print(f"❌ TGAT discovery failed for {session_name}: {str(e)}")
            return {
                'session_name': session_name,
                'discovery_success': False,
                'error': str(e),
                'patterns_found': 0,
                'patterns': [],
                'discovery_timestamp': datetime.now().isoformat()
            }
    
    def analyze_pattern_quality(self, all_patterns: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze pattern quality metrics for archaeological authenticity"""
        if not all_patterns:
            return {
                'total_patterns': 0,
                'duplication_rate': 100.0,
                'unique_descriptions': 0,
                'time_spans_analysis': {},
                'temporal_coherence': False
            }
            
        total_patterns = len(all_patterns)
        
        # Pattern duplication analysis
        descriptions = []
        time_spans = []
        sessions_identified = set()
        
        for pattern in all_patterns:
            description = pattern.get('description', 'unknown')
            descriptions.append(description)
            
            # Time span analysis
            time_span = pattern.get('time_span_hours', 0.0)
            time_spans.append(time_span)
            
            # Session identification
            session_ref = pattern.get('session', 'unknown')
            sessions_identified.add(session_ref)
            
        # Calculate duplication rate
        unique_descriptions = len(set(descriptions))
        duplication_rate = ((total_patterns - unique_descriptions) / total_patterns) * 100.0
        
        # Time span analysis
        non_zero_spans = [span for span in time_spans if span > 0.0]
        time_spans_analysis = {
            'total_patterns': total_patterns,
            'zero_time_spans': total_patterns - len(non_zero_spans),
            'non_zero_time_spans': len(non_zero_spans),
            'zero_span_percentage': ((total_patterns - len(non_zero_spans)) / total_patterns) * 100.0,
            'average_time_span': sum(time_spans) / len(time_spans) if time_spans else 0.0,
            'max_time_span': max(time_spans) if time_spans else 0.0,
            'min_time_span': min(time_spans) if time_spans else 0.0
        }
        
        # Most common descriptions (artifact detection)
        description_counts = Counter(descriptions)
        
        return {
            'total_patterns': total_patterns,
            'unique_descriptions': unique_descriptions,
            'duplication_rate': duplication_rate,
            'time_spans_analysis': time_spans_analysis,
            'sessions_identified': len(sessions_identified),
            'sessions_list': list(sessions_identified),
            'description_frequency': dict(description_counts.most_common(10)),
            'temporal_coherence': len(non_zero_spans) > 0,
            'archaeological_authenticity_score': self.calculate_authenticity_score(
                duplication_rate, time_spans_analysis, len(sessions_identified)
            )
        }
        
    def calculate_authenticity_score(self, duplication_rate: float, 
                                   time_spans: dict, sessions_count: int) -> float:
        """Calculate archaeological authenticity score (0-100)"""
        # Penalize high duplication (96.8% = very poor, <20% = excellent)
        duplication_penalty = max(0, duplication_rate - 20) / 80.0  # 0-1 scale
        duplication_score = (1 - duplication_penalty) * 40  # 0-40 points
        
        # Reward realistic time spans (>0 hours shows temporal relationships)
        non_zero_percentage = 100 - time_spans.get('zero_span_percentage', 100)
        time_span_score = (non_zero_percentage / 100.0) * 30  # 0-30 points
        
        # Reward cross-session discovery capability
        session_score = min(sessions_count / 5.0, 1.0) * 30  # 0-30 points (5+ sessions = full)
        
        total_score = duplication_score + time_span_score + session_score
        return min(100.0, max(0.0, total_score))
        
    def run_comparative_analysis(self, current_results: dict[str, Any]) -> dict[str, Any]:
        """Compare enhanced results vs previous contaminated baseline"""
        
        # Previous contaminated baseline (from Phase 1 assessment)
        contaminated_baseline = {
            'total_patterns': 4840,
            'unique_descriptions': 13,
            'duplication_rate': 96.8,
            'zero_time_spans': 4840,  # All patterns had 0.0 time spans
            'sessions_identified': 1,  # All marked as "unknown"
            'authenticity_score': 2.1  # Very poor archaeological authenticity
        }
        
        current = current_results
        
        improvement_analysis = {
            'duplication_improvement': contaminated_baseline['duplication_rate'] - current.get('duplication_rate', 100),
            'unique_descriptions_improvement': current.get('unique_descriptions', 0) - contaminated_baseline['unique_descriptions'],
            'time_span_improvement': current.get('time_spans_analysis', {}).get('non_zero_time_spans', 0) - 0,
            'session_discovery_improvement': current.get('sessions_identified', 0) - contaminated_baseline['sessions_identified'],
            'authenticity_improvement': current.get('archaeological_authenticity_score', 0) - contaminated_baseline['authenticity_score']
        }
        
        return {
            'contaminated_baseline': contaminated_baseline,
            'enhanced_results': current,
            'improvements': improvement_analysis,
            'success_threshold_met': improvement_analysis['duplication_improvement'] > 46.8  # >50% reduction from 96.8%
        }
        
    def run_validation(self) -> dict[str, Any]:
        """Execute complete Phase 5 archaeological discovery validation"""
        print("🏛️ PHASE 5: ARCHAEOLOGICAL DISCOVERY VALIDATION")
        print("=" * 60)
        print(f"Testing TGAT model on {len(self.test_sessions)} enhanced sessions")
        print("Target: <20% duplication (vs 96.8% contaminated baseline)")
        print()
        
        # Load and validate enhanced sessions
        validated_sessions = []
        for session_name in self.test_sessions:
            try:
                session_data = self.load_enhanced_session(session_name)
                feature_validation = self.validate_feature_authenticity(session_data)
                
                print(f"📋 {session_name}:")
                print(f"   HTF Carryover: {feature_validation['htf_carryover_value']:.3f} ({'✅ Authentic' if feature_validation['htf_carryover_authentic'] else '❌ Default'})")
                print(f"   Energy Density: {feature_validation['energy_density_value']:.3f} ({'✅ Authentic' if feature_validation['energy_density_authentic'] else '❌ Default'})")
                print(f"   Liquidity Events: {feature_validation['liquidity_events_count']} ({'✅ Rich' if feature_validation['liquidity_events_authentic'] else '❌ Empty'})")
                print(f"   Overall Status: {'✅ AUTHENTIC' if feature_validation['overall_authentic'] else '❌ CONTAMINATED'}")
                print()
                
                validated_sessions.append((session_name, session_data, feature_validation))
                
            except Exception as e:
                print(f"❌ Failed to load {session_name}: {e}")
                continue
                
        if not validated_sessions:
            raise RuntimeError("No valid enhanced sessions found for testing")
            
        print(f"✅ {len(validated_sessions)} enhanced sessions validated for TGAT testing")
        print()
        
        # Run TGAT discovery on each validated session
        all_patterns = []
        discovery_results = {}
        
        for session_name, session_data, feature_validation in validated_sessions:
            discovery_result = self.run_tgat_discovery(session_data, session_name)
            discovery_results[session_name] = discovery_result
            
            if discovery_result['discovery_success']:
                all_patterns.extend(discovery_result['patterns'])
                
        print()
        print("🔍 TGAT Discovery Results:")
        print(f"   Sessions Processed: {len(discovery_results)}")
        print(f"   Successful Discoveries: {sum(1 for r in discovery_results.values() if r['discovery_success'])}")
        print(f"   Total Patterns Found: {len(all_patterns)}")
        print()
        
        # Analyze pattern quality for archaeological authenticity
        print("📊 PATTERN QUALITY ANALYSIS:")
        print("-" * 30)
        
        quality_metrics = self.analyze_pattern_quality(all_patterns)
        
        print(f"Total Patterns: {quality_metrics['total_patterns']}")
        print(f"Unique Descriptions: {quality_metrics['unique_descriptions']}")
        print(f"Duplication Rate: {quality_metrics['duplication_rate']:.1f}%")
        print(f"Non-Zero Time Spans: {quality_metrics['time_spans_analysis']['non_zero_time_spans']}/{quality_metrics['total_patterns']}")
        print(f"Sessions Identified: {quality_metrics['sessions_identified']}")
        print(f"Archaeological Authenticity Score: {quality_metrics['archaeological_authenticity_score']:.1f}/100")
        print()
        
        # Comparative analysis vs contaminated baseline
        print("🔄 COMPARATIVE ANALYSIS:")
        print("-" * 25)
        
        comparative_analysis = self.run_comparative_analysis(quality_metrics)
        
        baseline = comparative_analysis['contaminated_baseline']
        improvements = comparative_analysis['improvements']
        
        print("CONTAMINATED BASELINE (Phase 1):")
        print(f"   Duplication Rate: {baseline['duplication_rate']:.1f}%")
        print(f"   Unique Descriptions: {baseline['unique_descriptions']}")
        print(f"   Time Spans >0: {baseline['zero_time_spans']}/{baseline['total_patterns']} (0%)")
        print(f"   Authenticity Score: {baseline['authenticity_score']:.1f}/100")
        print()
        
        print("ENHANCED RESULTS (Phase 5):")
        print(f"   Duplication Rate: {quality_metrics['duplication_rate']:.1f}%")
        print(f"   Unique Descriptions: {quality_metrics['unique_descriptions']}")
        print(f"   Time Spans >0: {quality_metrics['time_spans_analysis']['non_zero_time_spans']}/{quality_metrics['total_patterns']} ({100-quality_metrics['time_spans_analysis']['zero_span_percentage']:.1f}%)")
        print(f"   Authenticity Score: {quality_metrics['archaeological_authenticity_score']:.1f}/100")
        print()
        
        print("IMPROVEMENTS:")
        print(f"   Duplication Reduction: {improvements['duplication_improvement']:.1f} percentage points")
        print(f"   Additional Unique Patterns: +{improvements['unique_descriptions_improvement']}")
        print(f"   New Temporal Relationships: +{improvements['time_span_improvement']}")
        print(f"   Enhanced Session Discovery: +{improvements['session_discovery_improvement']}")
        print(f"   Authenticity Improvement: +{improvements['authenticity_improvement']:.1f} points")
        print()
        
        # Final assessment
        success_threshold_met = comparative_analysis['success_threshold_met']
        duplication_rate = quality_metrics['duplication_rate']
        
        print("🎯 PHASE 5 ASSESSMENT:")
        print("=" * 22)
        
        if duplication_rate < 20:
            assessment = "✅ EXCELLENT: Archaeological discovery capability fully restored"
            recommendation = "Proceed with full production deployment"
        elif duplication_rate < 50:
            assessment = "✅ SUCCESS: Significant improvement, archaeological capability restored"  
            recommendation = "Proceed with full validation across all 33 enhanced sessions"
        elif improvements['duplication_improvement'] > 20:
            assessment = "🔶 PARTIAL SUCCESS: Notable improvement but more enhancement needed"
            recommendation = "Continue with additional feature enhancement or consider Phase 4 retraining"
        else:
            assessment = "❌ INSUFFICIENT: Still producing template artifacts"
            recommendation = "Proceed to Phase 4: TGAT model retraining required"
            
        print(f"Assessment: {assessment}")
        print(f"Recommendation: {recommendation}")
        print()
        print(f"Success Threshold (>50% duplication reduction): {'✅ MET' if success_threshold_met else '❌ NOT MET'}")
        
        # Store complete results
        self.results.update({
            'test_sessions': [name for name, _, _ in validated_sessions],
            'discovery_results': discovery_results,
            'quality_metrics': quality_metrics,
            'comparative_analysis': comparative_analysis,
            'authenticity_assessment': {
                'assessment': assessment,
                'recommendation': recommendation,
                'success_threshold_met': success_threshold_met,
                'duplication_rate': duplication_rate,
                'authenticity_score': quality_metrics['archaeological_authenticity_score']
            }
        })
        
        return self.results

def main():
    """Execute Phase 5 Archaeological Discovery Validation"""
    try:
        validator = Phase5ArchaeologicalValidator()
        results = validator.run_validation()
        
        # Save detailed results
        results_path = Path(__file__).parent / 'phase5_archaeological_discovery_validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"📋 Detailed results saved: {results_path}")
        
        # Save executive summary
        summary_path = Path(__file__).parent / 'phase5_executive_summary.md'
        with open(summary_path, 'w') as f:
            assessment = results['authenticity_assessment']
            quality = results['quality_metrics']
            results['comparative_analysis']
            
            f.write("# Phase 5: Archaeological Discovery Validation - Executive Summary\n\n")
            f.write(f"**Validation Date**: {results['validation_timestamp']}\n")
            f.write(f"**Test Sessions**: {len(results['test_sessions'])} enhanced sessions\n\n")
            
            f.write("## Key Results\n\n")
            f.write(f"- **Duplication Rate**: {quality['duplication_rate']:.1f}% (vs 96.8% contaminated baseline)\n")
            f.write(f"- **Unique Descriptions**: {quality['unique_descriptions']} (vs 13 contaminated baseline)\n")
            f.write(f"- **Temporal Relationships**: {quality['time_spans_analysis']['non_zero_time_spans']} patterns with >0 time spans\n")
            f.write(f"- **Archaeological Authenticity**: {quality['archaeological_authenticity_score']:.1f}/100\n\n")
            
            f.write("## Assessment\n\n")
            f.write(f"**Status**: {assessment['assessment']}\n\n")
            f.write(f"**Recommendation**: {assessment['recommendation']}\n\n")
            
            f.write("## Success Criteria\n\n")
            f.write(f"- Target <20% duplication: {'✅ ACHIEVED' if quality['duplication_rate'] < 20 else '❌ NOT ACHIEVED' if quality['duplication_rate'] >= 50 else '🔶 PARTIAL'}\n")
            f.write(f"- Target >50 unique patterns: {'✅ ACHIEVED' if quality['unique_descriptions'] > 50 else '❌ NOT ACHIEVED'}\n")
            f.write(f"- Realistic time spans: {'✅ ACHIEVED' if quality['time_spans_analysis']['non_zero_time_spans'] > 0 else '❌ NOT ACHIEVED'}\n")
            f.write(f"- Success threshold (>50% improvement): {'✅ MET' if assessment['success_threshold_met'] else '❌ NOT MET'}\n")
            
        print(f"📋 Executive summary saved: {summary_path}")
        return results
        
    except Exception as e:
        print(f"❌ Phase 5 validation failed: {e}")
        raise

if __name__ == "__main__":
    main()