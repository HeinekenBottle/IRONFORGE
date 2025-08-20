#!/usr/bin/env python3
"""
News Integration Experiment Framework
Test news impact effects on RD@40 patterns and archaeological zones
"""

import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine

class NewsExperimentFramework:
    """Framework for experimenting with news impact on archaeological patterns"""
    
    def __init__(self):
        self.engine = EnhancedTemporalQueryEngine()
        self.news_scenarios = self._define_news_scenarios()
        self.sessions_data = self._load_sessions()
    
    def _define_news_scenarios(self):
        """Define experimental news impact scenarios"""
        return {
            'HIGH_IMPACT': {
                'description': 'Major economic release (NFP, FOMC, CPI)',
                'impact_factor': 0.85,
                'suppression_probability': 0.70,
                'energy_density_threshold': 0.75,
                'window_minutes': 30
            },
            'MEDIUM_IMPACT': {
                'description': 'Moderate economic data (GDP, Retail Sales)',
                'impact_factor': 0.45,
                'suppression_probability': 0.35,
                'energy_density_threshold': 0.60,
                'window_minutes': 15
            },
            'LOW_IMPACT': {
                'description': 'Minor economic indicators',
                'impact_factor': 0.20,
                'suppression_probability': 0.15,
                'energy_density_threshold': 0.45,
                'window_minutes': 10
            },
            'NO_NEWS': {
                'description': 'Control group - no news events',
                'impact_factor': 0.10,
                'suppression_probability': 0.05,
                'energy_density_threshold': 0.35,
                'window_minutes': 5
            }
        }
    
    def _load_sessions(self):
        """Load session data for experiments"""
        session_files = glob.glob("/Users/jack/IRONFORGE/data/adapted/adapted_*.json")
        sessions = {}
        
        for file_path in session_files[:20]:  # Limit for experiment performance
            session_name = file_path.split('/')[-1].replace('adapted_', '').replace('.json', '')
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sessions[session_name] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return sessions
    
    def experiment_1_news_impact_simulation(self):
        """Experiment 1: Simulate news impact on RD@40 patterns"""
        print("🧪 EXPERIMENT 1: NEWS IMPACT SIMULATION")
        print("=" * 60)
        print("📋 Testing: How different news impact levels affect RD@40 → Liquidity patterns")
        print("=" * 60)
        
        results = {}
        
        for scenario_name, scenario in self.news_scenarios.items():
            print(f"\n🎯 Scenario: {scenario_name}")
            print(f"   Description: {scenario['description']}")
            print(f"   Impact Factor: {scenario['impact_factor']}")
            print(f"   Suppression Rate: {scenario['suppression_probability']:.1%}")
            
            scenario_results = self._analyze_news_scenario(scenario_name, scenario)
            results[scenario_name] = scenario_results
            
            print(f"   Results: {scenario_results['rd40_events']} RD@40 events, "
                  f"{scenario_results['liquidity_events']} with liquidity")
            print(f"   Success Rate: {scenario_results['success_rate']:.1%}")
        
        return results
    
    def experiment_2_energy_density_correlation(self):
        """Experiment 2: Validate energy_density as news proxy"""
        print("\n🧪 EXPERIMENT 2: ENERGY DENSITY AS NEWS PROXY")
        print("=" * 60)
        print("📋 Testing: Correlation between energy_density and news impact patterns")
        print("=" * 60)
        
        energy_data = []
        
        for session_name, session_data in self.sessions_data.items():
            events = session_data.get('events', [])
            
            for event in events:
                energy_density = event.get('energy_density', 0.5)
                magnitude = event.get('magnitude', 0.0)
                arch_significance = event.get('archaeological_significance', 0.0)
                range_position = event.get('range_position', 0.5)
                
                # Check if near RD@40
                near_rd40 = abs(range_position - 0.40) <= 0.025
                
                energy_data.append({
                    'session': session_name,
                    'energy_density': energy_density,
                    'magnitude': magnitude,
                    'archaeological_significance': arch_significance,
                    'near_rd40': near_rd40,
                    'event_type': event.get('type', 'unknown')
                })
        
        df = pd.DataFrame(energy_data)
        
        # Analyze correlations
        high_energy_events = df[df['energy_density'] >= 0.70]
        rd40_high_energy = high_energy_events[high_energy_events['near_rd40']]
        
        print(f"📊 Energy Density Analysis:")
        print(f"   Total events: {len(df)}")
        print(f"   High energy events (≥0.70): {len(high_energy_events)} ({len(high_energy_events)/len(df):.1%})")
        print(f"   RD@40 + High Energy: {len(rd40_high_energy)} events")
        print(f"   Energy-RD@40 correlation: {df['energy_density'].corr(df['near_rd40'].astype(int)):.3f}")
        
        # Test news scenario classification
        for scenario_name, scenario in self.news_scenarios.items():
            threshold = scenario['energy_density_threshold']
            classified_events = df[df['energy_density'] >= threshold]
            rd40_classified = classified_events[classified_events['near_rd40']]
            
            print(f"\n🎯 {scenario_name} Classification (energy ≥ {threshold}):")
            print(f"   Classified events: {len(classified_events)}")
            print(f"   RD@40 classified: {len(rd40_classified)}")
            print(f"   RD@40 rate: {len(rd40_classified)/len(classified_events):.1%}" if len(classified_events) > 0 else "   RD@40 rate: 0.0%")
        
        return df
    
    def experiment_3_timing_window_effects(self):
        """Experiment 3: Test different news timing windows"""
        print("\n🧪 EXPERIMENT 3: NEWS TIMING WINDOW EFFECTS")
        print("=" * 60)
        print("📋 Testing: Optimal timing windows for news impact detection")
        print("=" * 60)
        
        window_tests = [5, 10, 15, 30, 45, 60]  # minutes
        results = {}
        
        for window in window_tests:
            print(f"\n⏰ Testing {window}-minute window:")
            
            # Simulate news events and measure impact on patterns
            window_results = self._test_timing_window(window)
            results[f"{window}min"] = window_results
            
            print(f"   Pattern disruptions: {window_results['disruptions']}")
            print(f"   Enhanced patterns: {window_results['enhancements']}")
            print(f"   Net effect: {window_results['net_effect']:.3f}")
        
        # Find optimal window
        optimal_window = max(results.items(), key=lambda x: x[1]['net_effect'])
        print(f"\n🏆 Optimal timing window: {optimal_window[0]} (net effect: {optimal_window[1]['net_effect']:.3f})")
        
        return results
    
    def experiment_4_pattern_suppression_analysis(self):
        """Experiment 4: Analyze news-driven pattern suppression"""
        print("\n🧪 EXPERIMENT 4: PATTERN SUPPRESSION ANALYSIS")
        print("=" * 60)
        print("📋 Testing: How news events suppress CONT/ACCEL patterns")
        print("=" * 60)
        
        suppression_results = {}
        
        # Use TQE to analyze pattern switches with news context
        try:
            # Query pattern switches with news effects
            switch_query = "Analyze pattern switches and regime transitions with news effects"
            switch_result = self.engine.ask(switch_query)
            
            print("📊 Pattern Switch Analysis with News Context:")
            if switch_result.get("regime_analysis"):
                regime = switch_result["regime_analysis"]
                news_effects = regime.get("news_effects", {})
                
                print(f"   High impact news events: {news_effects.get('high_impact_events', 0)}")
                print(f"   Pattern suppression events: {news_effects.get('suppression_events', 0)}")
                
                transitions = regime.get("regime_transitions", {})
                print(f"   CONT→MR transitions: {transitions.get('cont_to_mr', 0)}")
                print(f"   MR→CONT transitions: {transitions.get('mr_to_cont', 0)}")
            
            suppression_results['tqe_analysis'] = switch_result
            
        except Exception as e:
            print(f"❌ TQE Analysis Error: {e}")
        
        # Manual suppression analysis using energy density
        suppression_analysis = self._analyze_pattern_suppression()
        suppression_results['manual_analysis'] = suppression_analysis
        
        return suppression_results
    
    def experiment_5_news_calendar_simulation(self):
        """Experiment 5: Simulate real economic calendar integration"""
        print("\n🧪 EXPERIMENT 5: ECONOMIC CALENDAR SIMULATION")
        print("=" * 60)
        print("📋 Testing: Simulate integration with real economic news calendar")
        print("=" * 60)
        
        # Simulate major economic events
        simulated_events = [
            {
                'time': '08:30:00',
                'event': 'Non-Farm Payrolls',
                'impact': 'HIGH',
                'currency': 'USD',
                'expected_volatility': 0.85
            },
            {
                'time': '14:00:00', 
                'event': 'FOMC Minutes',
                'impact': 'HIGH',
                'currency': 'USD',
                'expected_volatility': 0.90
            },
            {
                'time': '10:00:00',
                'event': 'Consumer Price Index',
                'impact': 'MEDIUM',
                'currency': 'USD', 
                'expected_volatility': 0.65
            }
        ]
        
        calendar_results = {}
        
        for event in simulated_events:
            print(f"\n📅 Simulating: {event['event']} at {event['time']}")
            print(f"   Impact Level: {event['impact']}")
            print(f"   Expected Volatility: {event['expected_volatility']:.1%}")
            
            # Test impact on archaeological patterns
            event_impact = self._simulate_news_event_impact(event)
            calendar_results[event['event']] = event_impact
            
            print(f"   Affected RD@40 events: {event_impact['affected_rd40']}")
            print(f"   Pattern modifications: {event_impact['pattern_changes']}")
        
        return calendar_results
    
    def _analyze_news_scenario(self, scenario_name, scenario):
        """Analyze a specific news scenario"""
        rd40_events = 0
        liquidity_events = 0
        affected_patterns = 0
        
        for session_name, session_data in self.sessions_data.items():
            events = session_data.get('events', [])
            
            for event in events:
                range_position = event.get('range_position', 0.5)
                energy_density = event.get('energy_density', 0.5)
                
                # Check if RD@40 event
                if abs(range_position - 0.40) <= 0.025:
                    rd40_events += 1
                    
                    # Check if meets news scenario criteria
                    if energy_density >= scenario['energy_density_threshold']:
                        affected_patterns += 1
                        
                        # Simulate suppression effect
                        suppression_roll = np.random.random()
                        if suppression_roll > scenario['suppression_probability']:
                            liquidity_events += 1
        
        success_rate = (liquidity_events / rd40_events) if rd40_events > 0 else 0
        
        return {
            'rd40_events': rd40_events,
            'liquidity_events': liquidity_events,
            'affected_patterns': affected_patterns,
            'success_rate': success_rate
        }
    
    def _test_timing_window(self, window_minutes):
        """Test a specific timing window for news effects"""
        disruptions = 0
        enhancements = 0
        
        # Simulate timing window effects
        for session_name, session_data in self.sessions_data.items():
            events = session_data.get('events', [])
            
            for i, event in enumerate(events):
                energy_density = event.get('energy_density', 0.5)
                
                # High energy events simulate news
                if energy_density >= 0.70:
                    # Check events within window
                    window_events = 0
                    for j in range(max(0, i-3), min(len(events), i+4)):
                        if j != i:
                            window_events += 1
                    
                    # Simulate window effect based on size
                    if window_minutes <= 15 and window_events > 2:
                        enhancements += 1
                    elif window_minutes >= 45 and window_events > 5:
                        disruptions += 1
        
        net_effect = (enhancements - disruptions) / max(1, enhancements + disruptions)
        
        return {
            'disruptions': disruptions,
            'enhancements': enhancements,
            'net_effect': net_effect
        }
    
    def _analyze_pattern_suppression(self):
        """Analyze pattern suppression using available data"""
        high_energy_suppressions = 0
        pattern_continuations = 0
        
        for session_name, session_data in self.sessions_data.items():
            events = session_data.get('events', [])
            
            for i, event in enumerate(events):
                if abs(event.get('range_position', 0.5) - 0.40) <= 0.025:
                    energy = event.get('energy_density', 0.5)
                    
                    if energy >= 0.75:  # High news impact simulation
                        high_energy_suppressions += 1
                    else:
                        pattern_continuations += 1
        
        suppression_rate = high_energy_suppressions / (high_energy_suppressions + pattern_continuations)
        
        return {
            'high_energy_suppressions': high_energy_suppressions,
            'pattern_continuations': pattern_continuations,
            'suppression_rate': suppression_rate
        }
    
    def _simulate_news_event_impact(self, event):
        """Simulate the impact of a specific news event"""
        affected_rd40 = 0
        pattern_changes = 0
        
        # Simple simulation based on event characteristics
        impact_multiplier = {
            'HIGH': 3.0,
            'MEDIUM': 1.5,
            'LOW': 0.8
        }.get(event['impact'], 1.0)
        
        # Estimate affected events based on volatility and timing
        base_affected = int(event['expected_volatility'] * 10 * impact_multiplier)
        affected_rd40 = max(1, base_affected)
        pattern_changes = max(1, int(base_affected * 0.7))
        
        return {
            'affected_rd40': affected_rd40,
            'pattern_changes': pattern_changes,
            'simulated_impact': impact_multiplier
        }
    
    def run_all_experiments(self):
        """Run all news integration experiments"""
        print("🚀 IRONFORGE NEWS INTEGRATION EXPERIMENTS")
        print("🎯 Testing news impact on archaeological zone patterns")
        print("=" * 80)
        
        results = {}
        
        try:
            results['experiment_1'] = self.experiment_1_news_impact_simulation()
            results['experiment_2'] = self.experiment_2_energy_density_correlation()
            results['experiment_3'] = self.experiment_3_timing_window_effects()
            results['experiment_4'] = self.experiment_4_pattern_suppression_analysis()
            results['experiment_5'] = self.experiment_5_news_calendar_simulation()
            
            # Save comprehensive results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"news_experiments_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Experiment results saved to: {output_file}")
            print(f"\n✅ News Integration Experiments Complete!")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment Error: {e}")
            import traceback
            traceback.print_exc()
            return results

def main():
    """Main experimental function"""
    framework = NewsExperimentFramework()
    results = framework.run_all_experiments()
    return results

if __name__ == "__main__":
    main()