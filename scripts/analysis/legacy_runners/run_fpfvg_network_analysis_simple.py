#!/usr/bin/env python3
"""
🔄 IRONFORGE FPFVG Network Analysis (Step 3A) - Simplified
==========================================================

Focused implementation to prove FVG redelivery alignment with Theory B zones and PM belt timing.

Key Tests:
1. Zone enrichment: odds ratio analysis
2. PM-belt interaction: conditional probability analysis
3. Basic network statistics

Usage:
    python run_fpfvg_network_analysis_simple.py
"""

import json
import logging
import sys
from datetime import time
from pathlib import Path
from typing import Any, Dict, List

from scipy.stats import chi2_contingency, fisher_exact

# Add IRONFORGE to path
sys.path.append(str(Path(__file__).parent))

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleFPFVGAnalyzer:
    """Simplified FPFVG network analyzer focused on key statistical tests"""
    
    def __init__(self):
        self.config = get_config()
        self.discoveries_path = Path(self.config.get_discoveries_path())
        
        # Theory B zones and PM belt
        self.theory_b_zones = [0.20, 0.40, 0.50, 0.618, 0.80]
        self.zone_tolerance = 0.05  # ±5% tolerance
        self.pm_belt_start = time(14, 35, 0)
        self.pm_belt_end = time(14, 38, 59)
    
    def analyze(self) -> Dict[str, Any]:
        """Execute simplified FPFVG analysis"""
        logger.info("Starting simplified FPFVG analysis...")
        
        # Load FPFVG data
        candidates = self._load_fpfvg_candidates()
        if not candidates:
            return {'error': 'No FPFVG candidates found'}
        
        results = {
            'analysis_type': 'simplified_fpfvg_network',
            'total_candidates': len(candidates),
            'candidate_stats': self._analyze_candidates(candidates),
            'zone_enrichment_test': self._test_zone_enrichment(candidates),
            'pm_belt_test': self._test_pm_belt_interaction(candidates),
            'summary': self._generate_summary(candidates)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _load_fpfvg_candidates(self) -> List[Dict[str, Any]]:
        """Load FPFVG candidates from lattice data"""
        candidates = []
        
        # Find FPFVG lattice files
        fpfvg_files = list(self.discoveries_path.glob("fpfvg_redelivery_lattice_*.json"))
        if not fpfvg_files:
            return candidates
        
        # Use most recent file
        latest_file = sorted(fpfvg_files, key=lambda x: x.stat().st_mtime)[-1]
        
        try:
            with open(latest_file, 'r') as f:
                fpfvg_data = json.load(f)
            
            networks = fpfvg_data.get('fvg_networks', [])
            
            for network in networks:
                session_id = network.get('session_name', 'unknown')
                
                # Process formations
                for formation in network.get('fvg_formations', []):
                    candidate = self._create_candidate(formation, session_id, 'formation')
                    if candidate:
                        candidates.append(candidate)
                
                # Process redeliveries
                for redelivery in network.get('fvg_redeliveries', []):
                    candidate = self._create_candidate(redelivery, session_id, 'redelivery')
                    if candidate:
                        candidates.append(candidate)
                        
        except Exception as e:
            logger.error(f"Failed to load FPFVG data: {e}")
        
        logger.info(f"Loaded {len(candidates)} FPFVG candidates")
        return candidates
    
    def _create_candidate(self, event_data: Dict[str, Any], session_id: str, event_type: str) -> Dict[str, Any]:
        """Create simplified candidate record"""
        timestamp = event_data.get('timestamp', '')
        price = self._safe_float(event_data.get('price_level', event_data.get('target_price', 0)))
        
        # Calculate range position (simplified)
        range_pos = self._estimate_range_position(price, session_id)
        
        # Check PM belt timing
        in_pm_belt = self._is_in_pm_belt(timestamp)
        
        # Check zone proximity
        in_zone = self._is_in_theory_b_zone(range_pos)
        closest_zone = self._get_closest_zone(range_pos)
        
        return {
            'session_id': session_id,
            'event_type': event_type,
            'timestamp': timestamp,
            'price_level': price,
            'range_pos': range_pos,
            'in_pm_belt': in_pm_belt,
            'in_theory_b_zone': in_zone,
            'closest_zone': closest_zone,
            'zone_distance': self._get_zone_distance(range_pos, closest_zone)
        }
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert to float"""
        try:
            return float(value) if value is not None else 0.0
        except:
            return 0.0
    
    def _estimate_range_position(self, price: float, session_id: str) -> float:
        """Estimate range position (simplified)"""
        if price == 0:
            return 0.5
        
        # Rough estimation based on typical ranges
        if 'NY_PM' in session_id:
            low, high = 23000, 23500
        elif 'LONDON' in session_id:
            low, high = 23100, 23400
        else:
            low, high = 23050, 23450
        
        if high > low:
            return max(0.0, min(1.0, (price - low) / (high - low)))
        return 0.5
    
    def _is_in_pm_belt(self, timestamp: str) -> bool:
        """Check if timestamp is in PM belt"""
        try:
            if ':' in timestamp:
                time_part = timestamp.split(' ')[-1] if ' ' in timestamp else timestamp
                hour, minute = map(int, time_part.split(':')[:2])
                event_time = time(hour, minute)
                return self.pm_belt_start <= event_time <= self.pm_belt_end
        except:
            pass
        return False
    
    def _is_in_theory_b_zone(self, range_pos: float) -> bool:
        """Check if range position is near any Theory B zone"""
        for zone in self.theory_b_zones:
            if abs(range_pos - zone) <= self.zone_tolerance:
                return True
        return False
    
    def _get_closest_zone(self, range_pos: float) -> float:
        """Get closest Theory B zone"""
        return min(self.theory_b_zones, key=lambda z: abs(range_pos - z))
    
    def _get_zone_distance(self, range_pos: float, zone: float) -> float:
        """Get distance to specific zone"""
        return abs(range_pos - zone)
    
    def _analyze_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate basic candidate statistics"""
        stats = {
            'total': len(candidates),
            'by_type': {},
            'pm_belt_count': 0,
            'theory_b_zone_count': 0,
            'price_stats': {}
        }
        
        # Count by type
        for candidate in candidates:
            event_type = candidate['event_type']
            stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1
            
            if candidate['in_pm_belt']:
                stats['pm_belt_count'] += 1
            
            if candidate['in_theory_b_zone']:
                stats['theory_b_zone_count'] += 1
        
        # Price statistics
        prices = [c['price_level'] for c in candidates if c['price_level'] > 0]
        if prices:
            stats['price_stats'] = {
                'count': len(prices),
                'min': min(prices),
                'max': max(prices),
                'mean': sum(prices) / len(prices)
            }
        
        return stats
    
    def _test_zone_enrichment(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test if redeliveries are enriched in Theory B zones"""
        redeliveries = [c for c in candidates if c['event_type'] == 'redelivery']
        
        if not redeliveries:
            return {'error': 'No redeliveries found'}
        
        # Count redeliveries in zones vs outside
        in_zones = len([r for r in redeliveries if r['in_theory_b_zone']])
        outside_zones = len(redeliveries) - in_zones
        
        # Calculate expected based on zone coverage
        total_zone_coverage = len(self.theory_b_zones) * self.zone_tolerance * 2
        expected_in = len(redeliveries) * min(1.0, total_zone_coverage)
        expected_out = len(redeliveries) - expected_in
        
        # Fisher exact test
        try:
            odds_ratio, p_value = fisher_exact([
                [in_zones, outside_zones],
                [max(1, int(expected_in)), max(1, int(expected_out))]
            ])
            
            return {
                'test_type': 'zone_enrichment_fisher_exact',
                'redeliveries_total': len(redeliveries),
                'observed_in_zones': in_zones,
                'observed_outside_zones': outside_zones,
                'expected_in_zones': expected_in,
                'expected_outside_zones': expected_out,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'enrichment_factor': in_zones / max(1, expected_in)
            }
            
        except Exception as e:
            return {'error': f'Statistical test failed: {e}'}
    
    def _test_pm_belt_interaction(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test PM belt interaction patterns"""
        # Group by session
        sessions = {}
        for candidate in candidates:
            session_id = candidate['session_id']
            if session_id not in sessions:
                sessions[session_id] = {'formations': [], 'redeliveries': [], 'pm_belt_events': []}
            
            if candidate['event_type'] == 'formation':
                sessions[session_id]['formations'].append(candidate)
            else:
                sessions[session_id]['redeliveries'].append(candidate)
            
            if candidate['in_pm_belt']:
                sessions[session_id]['pm_belt_events'].append(candidate)
        
        # Calculate conditional probabilities
        sessions_with_formations = 0
        sessions_with_pm_redeliveries = 0
        sessions_with_both = 0
        total_sessions = len(sessions)
        
        for session_id, session_data in sessions.items():
            has_formations = len(session_data['formations']) > 0
            has_pm_redeliveries = len([r for r in session_data['redeliveries'] if r['in_pm_belt']]) > 0
            
            if has_formations:
                sessions_with_formations += 1
            
            if has_pm_redeliveries:
                sessions_with_pm_redeliveries += 1
            
            if has_formations and has_pm_redeliveries:
                sessions_with_both += 1
        
        # P(PM redelivery | Formation)
        p_pm_given_formation = sessions_with_both / max(1, sessions_with_formations)
        
        # P(PM redelivery) baseline
        p_pm_baseline = sessions_with_pm_redeliveries / max(1, total_sessions)
        
        # Chi-square test
        try:
            # Contingency table
            contingency = [
                [sessions_with_both, sessions_with_formations - sessions_with_both],
                [sessions_with_pm_redeliveries - sessions_with_both, 
                 total_sessions - sessions_with_formations - sessions_with_pm_redeliveries + sessions_with_both]
            ]
            
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            return {
                'test_type': 'pm_belt_interaction_chi2',
                'total_sessions': total_sessions,
                'sessions_with_formations': sessions_with_formations,
                'sessions_with_pm_redeliveries': sessions_with_pm_redeliveries,
                'sessions_with_both': sessions_with_both,
                'p_pm_given_formation': p_pm_given_formation,
                'p_pm_baseline': p_pm_baseline,
                'relative_risk': p_pm_given_formation / max(0.001, p_pm_baseline),
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'contingency_table': contingency
            }
            
        except Exception as e:
            return {'error': f'Statistical test failed: {e}'}
    
    def _generate_summary(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis summary"""
        formations = [c for c in candidates if c['event_type'] == 'formation']
        redeliveries = [c for c in candidates if c['event_type'] == 'redelivery']
        
        pm_belt_formations = [c for c in formations if c['in_pm_belt']]
        pm_belt_redeliveries = [c for c in redeliveries if c['in_pm_belt']]
        
        zone_formations = [c for c in formations if c['in_theory_b_zone']]
        zone_redeliveries = [c for c in redeliveries if c['in_theory_b_zone']]
        
        return {
            'total_candidates': len(candidates),
            'formations': len(formations),
            'redeliveries': len(redeliveries),
            'pm_belt_rate': {
                'formations': len(pm_belt_formations) / max(1, len(formations)),
                'redeliveries': len(pm_belt_redeliveries) / max(1, len(redeliveries))
            },
            'theory_b_zone_rate': {
                'formations': len(zone_formations) / max(1, len(formations)),
                'redeliveries': len(zone_redeliveries) / max(1, len(redeliveries))
            },
            'key_insight': 'Statistical validation of FPFVG redelivery alignment with Theory B zones and PM belt timing'
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fpfvg_network_analysis_simple_{timestamp}.json"
        filepath = self.discoveries_path / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Execute simplified FPFVG network analysis"""
    print("🔄 IRONFORGE FPFVG Network Analysis (Step 3A) - Simplified")
    print("=" * 65)
    print("Key Tests: Zone enrichment, PM belt interaction, statistical validation")
    print()
    
    try:
        analyzer = SimpleFPFVGAnalyzer()
        results = analyzer.analyze()
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return 1
        
        print("✅ FPFVG NETWORK ANALYSIS COMPLETE")
        print("=" * 40)
        
        # Display results
        print(f"📈 Total Candidates: {results['total_candidates']}")
        
        candidate_stats = results.get('candidate_stats', {})
        by_type = candidate_stats.get('by_type', {})
        pm_belt_count = candidate_stats.get('pm_belt_count', 0)
        zone_count = candidate_stats.get('theory_b_zone_count', 0)
        
        print(f"Event types: {by_type}")
        print(f"PM belt events: {pm_belt_count}")
        print(f"Theory B zone events: {zone_count}")
        print()
        
        # Zone enrichment test
        zone_test = results.get('zone_enrichment_test', {})
        if 'error' not in zone_test:
            print("📊 ZONE ENRICHMENT TEST")
            print("-" * 23)
            print(f"Redeliveries in zones: {zone_test.get('observed_in_zones', 0)}")
            print(f"Expected in zones: {zone_test.get('expected_in_zones', 0):.1f}")
            print(f"Enrichment factor: {zone_test.get('enrichment_factor', 1.0):.3f}")
            print(f"Odds ratio: {zone_test.get('odds_ratio', 1.0):.3f}")
            print(f"P-value: {zone_test.get('p_value', 1.0):.6f}")
            print(f"Significant: {'✓ YES' if zone_test.get('significant', False) else '✗ NO'}")
            print()
        
        # PM belt interaction test
        pm_test = results.get('pm_belt_test', {})
        if 'error' not in pm_test:
            print("⏰ PM BELT INTERACTION TEST")
            print("-" * 27)
            print(f"P(PM redelivery | Formation): {pm_test.get('p_pm_given_formation', 0):.3f}")
            print(f"P(PM redelivery baseline): {pm_test.get('p_pm_baseline', 0):.3f}")
            print(f"Relative risk: {pm_test.get('relative_risk', 1.0):.3f}")
            print(f"Chi2 statistic: {pm_test.get('chi2_statistic', 0):.3f}")
            print(f"P-value: {pm_test.get('p_value', 1.0):.6f}")
            print(f"Significant: {'✓ YES' if pm_test.get('significant', False) else '✗ NO'}")
            print()
        
        # Summary
        summary = results.get('summary', {})
        print("💡 SUMMARY")
        print("-" * 9)
        print(f"PM belt formation rate: {summary.get('pm_belt_rate', {}).get('formations', 0):.3f}")
        print(f"PM belt redelivery rate: {summary.get('pm_belt_rate', {}).get('redeliveries', 0):.3f}")
        print(f"Zone formation rate: {summary.get('theory_b_zone_rate', {}).get('formations', 0):.3f}")
        print(f"Zone redelivery rate: {summary.get('theory_b_zone_rate', {}).get('redeliveries', 0):.3f}")
        print()
        
        print("🔄 FPFVG Network Analysis (Step 3A) complete")
        print("Statistical validation of micro mechanism completed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis execution failed: {e}")
        print(f"❌ Execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)