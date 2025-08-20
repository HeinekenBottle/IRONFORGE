#!/usr/bin/env python3
"""
IRONFORGE Theory B + f8 Spike Correlation Analyzer
Advanced correlation analysis between Theory B precision events and f8 liquidity spikes
with 5-minute temporal windows, gap age filtering, and session type analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from session_time_manager import SessionTimeManager
from archaeological_zone_calculator import ArchaeologicalZoneCalculator
import json

class TheoryBF8CorrelationAnalyzer:
    """
    Analyze correlations between Theory B precision events and f8 liquidity spikes
    with advanced filtering by temporal proximity, gap age, and session characteristics
    """
    
    def __init__(self):
        self.engine = EnhancedTemporalQueryEngine()
        self.session_manager = SessionTimeManager()
        self.zone_calculator = ArchaeologicalZoneCalculator()
        
        # Analysis parameters
        self.TIME_WINDOW_MINUTES = 5
        self.F8_SPIKE_THRESHOLD_PERCENTILE = 95  # f8 values above 95th percentile
        self.THEORY_B_PRECISION_THRESHOLD = 7.55  # Points
        
    def find_correlated_events(self, 
                             gap_age_filter: Optional[str] = None,
                             session_types: Optional[List[str]] = None,
                             min_theory_b_score: float = 0.5) -> Dict[str, Any]:
        """
        Find Theory B precision events within 5 minutes of f8 spikes
        
        Args:
            gap_age_filter: Filter by gap age ('young', 'mature', 'aged', or None)
            session_types: List of session types to include (e.g., ['NYAM', 'NYPM'])
            min_theory_b_score: Minimum Theory B precision score (0.0-1.0)
            
        Returns:
            Detailed correlation analysis with matched events
        """
        print(f"🔍 Analyzing Theory B + f8 Correlations")
        print(f"   Time Window: ±{self.TIME_WINDOW_MINUTES} minutes")
        print(f"   Gap Age Filter: {gap_age_filter or 'All'}")
        print(f"   Session Types: {session_types or 'All'}")
        print(f"   Min Theory B Score: {min_theory_b_score}")
        
        # Get all f8 spikes above threshold
        f8_spikes = self._identify_f8_spikes()
        print(f"   Found {len(f8_spikes)} f8 spikes above {self.F8_SPIKE_THRESHOLD_PERCENTILE}th percentile")
        
        # Get all Theory B precision events
        theory_b_events = self._identify_theory_b_events(min_theory_b_score)
        print(f"   Found {len(theory_b_events)} Theory B events (score >= {min_theory_b_score})")
        
        # Apply filters
        filtered_f8 = self._apply_gap_age_filter(f8_spikes, gap_age_filter)
        filtered_f8 = self._apply_session_type_filter(filtered_f8, session_types)
        
        filtered_theory_b = self._apply_session_type_filter(theory_b_events, session_types)
        
        print(f"   After filtering: {len(filtered_f8)} f8 spikes, {len(filtered_theory_b)} Theory B events")
        
        # Find correlations within time window
        correlations = self._find_temporal_correlations(filtered_f8, filtered_theory_b)
        
        # Analyze correlation patterns
        analysis = self._analyze_correlation_patterns(correlations)
        
        return {
            "query_parameters": {
                "time_window_minutes": self.TIME_WINDOW_MINUTES,
                "gap_age_filter": gap_age_filter,
                "session_types": session_types,
                "min_theory_b_score": min_theory_b_score
            },
            "data_summary": {
                "total_f8_spikes": len(f8_spikes),
                "total_theory_b_events": len(theory_b_events),
                "filtered_f8_spikes": len(filtered_f8),
                "filtered_theory_b_events": len(filtered_theory_b)
            },
            "correlations": correlations,
            "analysis": analysis,
            "insights": self._generate_insights(correlations, analysis)
        }
    
    def _identify_f8_spikes(self) -> List[Dict[str, Any]]:
        """Identify f8 liquidity spikes above threshold"""
        spikes = []
        
        # Collect all f8 values to calculate percentile
        all_f8_values = []
        for session_id, session_data in self.engine.sessions.items():
            if 'f8' in session_data.columns:
                all_f8_values.extend(session_data['f8'].dropna().tolist())
        
        if not all_f8_values:
            return spikes
            
        f8_threshold = np.percentile(all_f8_values, self.F8_SPIKE_THRESHOLD_PERCENTILE)
        
        # Find spikes above threshold
        for session_id, session_data in self.engine.sessions.items():
            if 'f8' not in session_data.columns:
                continue
                
            spike_events = session_data[session_data['f8'] > f8_threshold]
            
            for _, event in spike_events.iterrows():
                spike = {
                    "session_id": session_id,
                    "timestamp": self._convert_to_time_string(event['t']),
                    "price": event['price'],
                    "f8_value": event['f8'],
                    "f8_percentile": (event['f8'] >= np.percentile(all_f8_values, 
                                    list(range(0, 101, 5)))).sum() * 5,
                    "session_type": session_id.split('_')[0] if '_' in session_id else 'UNKNOWN',
                    "gap_age": self._determine_gap_age(event, session_data),
                    "node_id": event.get('node_id', 'unknown')
                }
                spikes.append(spike)
        
        return sorted(spikes, key=lambda x: (x['session_id'], x['timestamp']))
    
    def _identify_theory_b_events(self, min_score: float) -> List[Dict[str, Any]]:
        """Identify Theory B precision events meeting score threshold"""
        events = []
        
        for session_id, session_data in self.engine.sessions.items():
            if session_id not in self.engine.session_stats:
                continue
                
            session_stats = self.engine.session_stats[session_id]
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Analyze each price event for Theory B characteristics
            for _, event in session_data.iterrows():
                if pd.isna(event['price']):
                    continue
                    
                timestamp = self._convert_to_time_string(event['t'])
                
                try:
                    analysis = self.zone_calculator.analyze_event_positioning(
                        event_price=event['price'],
                        event_time=timestamp,
                        session_type=session_type,
                        final_session_stats=session_stats
                    )
                    
                    theory_b = analysis.get('theory_b_analysis', {})
                    precision_score = theory_b.get('precision_score', 0)
                    
                    if (precision_score >= min_score and 
                        theory_b.get('meets_theory_b_precision', False)):
                        
                        theory_b_event = {
                            "session_id": session_id,
                            "timestamp": timestamp,
                            "price": event['price'],
                            "precision_score": precision_score,
                            "distance_to_final_40pct": theory_b.get('distance_to_final_40pct', 0),
                            "session_type": session_type,
                            "archaeological_zone": analysis.get('dimensional_relationship', 'unknown'),
                            "session_progress": analysis.get('temporal_context', {}).get('session_progress_pct', 0),
                            "is_dimensional_destiny": analysis.get('closest_zone', {}).get('is_dimensional_destiny', False),
                            "node_id": event.get('node_id', 'unknown')
                        }
                        events.append(theory_b_event)
                        
                except Exception as e:
                    # Skip events that can't be analyzed
                    continue
        
        return sorted(events, key=lambda x: x['precision_score'], reverse=True)
    
    def _determine_gap_age(self, event: pd.Series, session_data: pd.DataFrame) -> str:
        """Determine gap age based on event characteristics"""
        # Simple gap age classification based on available data
        # This could be enhanced with actual gap analysis
        
        if 'f40' in event and not pd.isna(event['f40']):
            if event['f40'] > 0.8:
                return 'young'
            elif event['f40'] > 0.4:
                return 'mature' 
            else:
                return 'aged'
        
        # Fallback: use position in session
        event_time = event['t']
        session_start = session_data['t'].min()
        session_duration = session_data['t'].max() - session_start
        
        if session_duration > 0:
            progress = (event_time - session_start) / session_duration
            if progress < 0.3:
                return 'young'
            elif progress < 0.7:
                return 'mature'
            else:
                return 'aged'
        
        return 'unknown'
    
    def _apply_gap_age_filter(self, events: List[Dict], gap_age_filter: Optional[str]) -> List[Dict]:
        """Filter events by gap age"""
        if not gap_age_filter:
            return events
        
        return [event for event in events if event.get('gap_age') == gap_age_filter]
    
    def _apply_session_type_filter(self, events: List[Dict], session_types: Optional[List[str]]) -> List[Dict]:
        """Filter events by session type"""
        if not session_types:
            return events
        
        return [event for event in events if event.get('session_type') in session_types]
    
    def _convert_to_time_string(self, timestamp_ms: int) -> str:
        """Convert millisecond timestamp to time string (proper epoch conversion)"""
        # Check if this is epoch time (milliseconds since 1970) or relative time
        if timestamp_ms > 1000000000000:  # Epoch time (after year 2001)
            # Convert epoch milliseconds to datetime
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            return dt.strftime("%H:%M:%S")
        else:
            # Relative time since session start
            seconds = timestamp_ms // 1000
            hours = (seconds // 3600) % 24
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _find_temporal_correlations(self, f8_spikes: List[Dict], theory_b_events: List[Dict]) -> List[Dict]:
        """Find Theory B events within time window of f8 spikes"""
        correlations = []
        time_window_ms = self.TIME_WINDOW_MINUTES * 60 * 1000
        
        for f8_spike in f8_spikes:
            f8_session = f8_spike['session_id']
            f8_time_str = f8_spike['timestamp']
            
            # Find Theory B events in same session within time window
            for theory_b_event in theory_b_events:
                if theory_b_event['session_id'] != f8_session:
                    continue
                
                # Calculate time difference (simplified - assumes same day)
                time_diff_minutes = self._calculate_time_difference_minutes(
                    f8_time_str, theory_b_event['timestamp']
                )
                
                if abs(time_diff_minutes) <= self.TIME_WINDOW_MINUTES:
                    correlation = {
                        "session_id": f8_session,
                        "f8_spike": f8_spike,
                        "theory_b_event": theory_b_event,
                        "time_difference_minutes": time_diff_minutes,
                        "temporal_relationship": self._classify_temporal_relationship(time_diff_minutes),
                        "combined_score": self._calculate_combined_score(f8_spike, theory_b_event),
                        "archaeological_context": theory_b_event['archaeological_zone'],
                        "session_phase": self._determine_session_phase(theory_b_event['session_progress'])
                    }
                    correlations.append(correlation)
        
        return sorted(correlations, key=lambda x: x['combined_score'], reverse=True)
    
    def _calculate_time_difference_minutes(self, time1: str, time2: str) -> float:
        """Calculate time difference in minutes between two time strings"""
        try:
            # Parse time strings (HH:MM:SS format)
            t1_parts = list(map(int, time1.split(':')))
            t2_parts = list(map(int, time2.split(':')))
            
            t1_minutes = t1_parts[0] * 60 + t1_parts[1] + t1_parts[2] / 60
            t2_minutes = t2_parts[0] * 60 + t2_parts[1] + t2_parts[2] / 60
            
            return t2_minutes - t1_minutes
        except:
            return 0
    
    def _classify_temporal_relationship(self, time_diff_minutes: float) -> str:
        """Classify temporal relationship between events"""
        if time_diff_minutes < -2:
            return "theory_b_precedes_f8"
        elif time_diff_minutes > 2:
            return "f8_precedes_theory_b"
        else:
            return "simultaneous"
    
    def _calculate_combined_score(self, f8_spike: Dict, theory_b_event: Dict) -> float:
        """Calculate combined significance score"""
        # Normalize f8 percentile to 0-1
        f8_score = f8_spike.get('f8_percentile', 50) / 100
        
        # Theory B precision score already 0-1
        theory_b_score = theory_b_event.get('precision_score', 0)
        
        # Weight based on dimensional destiny
        destiny_multiplier = 1.5 if theory_b_event.get('is_dimensional_destiny', False) else 1.0
        
        return (f8_score * 0.4 + theory_b_score * 0.6) * destiny_multiplier
    
    def _determine_session_phase(self, session_progress: float) -> str:
        """Determine session phase from progress percentage"""
        if session_progress < 20:
            return "opening"
        elif session_progress < 40:
            return "early"
        elif session_progress < 60:
            return "mid"
        elif session_progress < 80:
            return "late"
        else:
            return "closing"
    
    def _analyze_correlation_patterns(self, correlations: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in the correlations"""
        if not correlations:
            return {"total_correlations": 0}
        
        # Session type distribution
        session_type_dist = {}
        gap_age_dist = {}
        temporal_relationship_dist = {}
        archaeological_zone_dist = {}
        session_phase_dist = {}
        
        for correlation in correlations:
            # Session type
            session_type = correlation['f8_spike'].get('session_type', 'unknown')
            session_type_dist[session_type] = session_type_dist.get(session_type, 0) + 1
            
            # Gap age
            gap_age = correlation['f8_spike'].get('gap_age', 'unknown')
            gap_age_dist[gap_age] = gap_age_dist.get(gap_age, 0) + 1
            
            # Temporal relationship
            temp_rel = correlation['temporal_relationship']
            temporal_relationship_dist[temp_rel] = temporal_relationship_dist.get(temp_rel, 0) + 1
            
            # Archaeological zone
            arch_zone = correlation['archaeological_context']
            archaeological_zone_dist[arch_zone] = archaeological_zone_dist.get(arch_zone, 0) + 1
            
            # Session phase
            phase = correlation['session_phase']
            session_phase_dist[phase] = session_phase_dist.get(phase, 0) + 1
        
        # Calculate statistics
        combined_scores = [c['combined_score'] for c in correlations]
        time_differences = [abs(c['time_difference_minutes']) for c in correlations]
        theory_b_scores = [c['theory_b_event']['precision_score'] for c in correlations]
        f8_percentiles = [c['f8_spike'].get('f8_percentile', 50) for c in correlations]
        
        return {
            "total_correlations": len(correlations),
            "session_type_distribution": session_type_dist,
            "gap_age_distribution": gap_age_dist,
            "temporal_relationship_distribution": temporal_relationship_dist,
            "archaeological_zone_distribution": archaeological_zone_dist,
            "session_phase_distribution": session_phase_dist,
            "statistics": {
                "avg_combined_score": np.mean(combined_scores),
                "max_combined_score": np.max(combined_scores),
                "avg_time_difference": np.mean(time_differences),
                "avg_theory_b_score": np.mean(theory_b_scores),
                "avg_f8_percentile": np.mean(f8_percentiles),
                "dimensional_destiny_percentage": sum(1 for c in correlations 
                                                   if c['theory_b_event'].get('is_dimensional_destiny', False)) / len(correlations) * 100
            }
        }
    
    def _generate_insights(self, correlations: List[Dict], analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        if not correlations:
            insights.append("No correlations found within the specified criteria")
            return insights
        
        total = len(correlations)
        stats = analysis.get('statistics', {})
        
        insights.append(f"Found {total} Theory B + f8 correlations within {self.TIME_WINDOW_MINUTES}-minute windows")
        
        # Session type insights
        session_dist = analysis.get('session_type_distribution', {})
        if session_dist:
            dominant_session = max(session_dist.keys(), key=lambda k: session_dist[k])
            insights.append(f"Most correlations occur in {dominant_session} sessions ({session_dist[dominant_session]}/{total})")
        
        # Temporal relationship insights
        temp_rel_dist = analysis.get('temporal_relationship_distribution', {})
        if temp_rel_dist:
            dominant_relationship = max(temp_rel_dist.keys(), key=lambda k: temp_rel_dist[k])
            percentage = temp_rel_dist[dominant_relationship] / total * 100
            insights.append(f"{percentage:.1f}% show '{dominant_relationship.replace('_', ' ')}' timing pattern")
        
        # Archaeological zone insights
        zone_dist = analysis.get('archaeological_zone_distribution', {})
        if zone_dist:
            top_zone = max(zone_dist.keys(), key=lambda k: zone_dist[k])
            zone_percentage = zone_dist[top_zone] / total * 100
            insights.append(f"{zone_percentage:.1f}% occur in {top_zone.replace('_', ' ')} zones")
        
        # Quality insights
        if stats:
            avg_score = stats.get('avg_combined_score', 0)
            destiny_pct = stats.get('dimensional_destiny_percentage', 0)
            
            insights.append(f"Average correlation strength: {avg_score:.3f}")
            insights.append(f"{destiny_pct:.1f}% involve dimensional destiny zones")
            
            if avg_score > 0.7:
                insights.append("🔥 High-quality correlations detected - strong predictive potential")
            elif destiny_pct > 50:
                insights.append("⭐ Significant dimensional destiny involvement - Theory B validation")
        
        # Gap age insights
        gap_dist = analysis.get('gap_age_distribution', {})
        if gap_dist and 'unknown' not in gap_dist:
            dominant_gap_age = max(gap_dist.keys(), key=lambda k: gap_dist[k])
            gap_percentage = gap_dist[dominant_gap_age] / total * 100
            insights.append(f"Gap age pattern: {gap_percentage:.1f}% involve '{dominant_gap_age}' gaps")
        
        return insights


def demo_theory_b_f8_correlation_analysis():
    """Demonstrate Theory B + f8 correlation analysis with various filters"""
    print("🚀 IRONFORGE Theory B + f8 Spike Correlation Analysis")
    print("=" * 70)
    
    analyzer = TheoryBF8CorrelationAnalyzer()
    
    # Test different filter combinations
    test_scenarios = [
        {
            "name": "All Sessions - High Precision Theory B Events",
            "gap_age_filter": None,
            "session_types": None,
            "min_theory_b_score": 0.7
        },
        {
            "name": "NYAM Sessions Only - Young Gaps",
            "gap_age_filter": "young",
            "session_types": ["NYAM"],
            "min_theory_b_score": 0.5
        },
        {
            "name": "NY Sessions - Mature/Aged Gaps",
            "gap_age_filter": "mature",
            "session_types": ["NYAM", "NYPM"],
            "min_theory_b_score": 0.6
        },
        {
            "name": "All Sessions - Maximum Precision",
            "gap_age_filter": None,
            "session_types": None,
            "min_theory_b_score": 0.8
        }
    ]
    
    results = {}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 50)
        
        result = analyzer.find_correlated_events(
            gap_age_filter=scenario['gap_age_filter'],
            session_types=scenario['session_types'],
            min_theory_b_score=scenario['min_theory_b_score']
        )
        
        results[scenario['name']] = result
        
        # Display summary
        data_summary = result['data_summary']
        analysis = result['analysis']
        
        print(f"📊 Data Summary:")
        print(f"   Filtered f8 spikes: {data_summary['filtered_f8_spikes']}")
        print(f"   Filtered Theory B events: {data_summary['filtered_theory_b_events']}")
        print(f"   Correlations found: {analysis['total_correlations']}")
        
        if analysis['total_correlations'] > 0:
            stats = analysis.get('statistics', {})
            print(f"   Average correlation strength: {stats.get('avg_combined_score', 0):.3f}")
            print(f"   Dimensional destiny involvement: {stats.get('dimensional_destiny_percentage', 0):.1f}%")
            
            # Show top correlation
            correlations = result['correlations']
            if correlations:
                top_correlation = correlations[0]
                f8_spike = top_correlation['f8_spike']
                theory_b = top_correlation['theory_b_event']
                
                print(f"\n🎯 Top Correlation:")
                print(f"   Session: {top_correlation['session_id']}")
                print(f"   f8 spike: {f8_spike['timestamp']} ({f8_spike.get('f8_percentile', 0):.1f}th percentile)")
                print(f"   Theory B: {theory_b['timestamp']} (precision: {theory_b['precision_score']:.3f})")
                print(f"   Time difference: {top_correlation['time_difference_minutes']:.1f} minutes")
                print(f"   Archaeological zone: {theory_b['archaeological_zone']}")
                print(f"   Combined score: {top_correlation['combined_score']:.3f}")
        
        # Show key insights
        insights = result['insights']
        if insights:
            print(f"\n💡 Key Insights:")
            for insight in insights[:3]:  # Show top 3 insights
                print(f"   • {insight}")
    
    # Cross-scenario comparison
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO COMPARISON")
    print("=" * 70)
    
    print(f"\n📈 Correlation Counts by Scenario:")
    for scenario_name, result in results.items():
        count = result['analysis']['total_correlations']
        avg_score = result['analysis'].get('statistics', {}).get('avg_combined_score', 0)
        print(f"   {scenario_name}: {count} correlations (avg score: {avg_score:.3f})")
    
    # Find best performing scenario
    best_scenario = max(results.items(), 
                       key=lambda x: x[1]['analysis']['total_correlations'])
    
    print(f"\n🏆 Best Performing Scenario: {best_scenario[0]}")
    print(f"   {best_scenario[1]['analysis']['total_correlations']} correlations found")
    
    # Session type analysis across all scenarios
    all_session_types = set()
    for result in results.values():
        session_dist = result['analysis'].get('session_type_distribution', {})
        all_session_types.update(session_dist.keys())
    
    if all_session_types:
        print(f"\n📊 Session Types with Correlations: {', '.join(sorted(all_session_types))}")
    
    return results


if __name__ == "__main__":
    correlation_results = demo_theory_b_f8_correlation_analysis()
    print(f"\n✅ Theory B + f8 correlation analysis complete!")
    print(f"   Advanced filtering and correlation detection operational")