#!/usr/bin/env python3
"""
Enhanced Theory B + f8 Correlation Analyzer
Refined analysis focusing on actual trading sessions with improved gap age detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from enhanced_temporal_query_engine import EnhancedTemporalQueryEngine
from session_time_manager import SessionTimeManager
from archaeological_zone_calculator import ArchaeologicalZoneCalculator

class EnhancedTheoryBF8Analyzer:
    """
    Enhanced analyzer with improved session filtering and gap age detection
    """
    
    def __init__(self):
        self.engine = EnhancedTemporalQueryEngine()
        self.session_manager = SessionTimeManager()
        self.zone_calculator = ArchaeologicalZoneCalculator()
        
        # Analysis parameters
        self.TIME_WINDOW_MINUTES = 5
        self.F8_SPIKE_THRESHOLD_PERCENTILE = 90  # Lowered to find more events
        
    def analyze_correlations_comprehensive(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of Theory B + f8 correlations
        """
        print("🔍 Enhanced Theory B + f8 Correlation Analysis")
        print("=" * 60)
        
        # First, let's examine what sessions we have
        session_analysis = self._analyze_available_sessions()
        print(f"📊 Session Analysis:")
        for session_type, count in session_analysis['session_type_counts'].items():
            print(f"   {session_type}: {count} sessions")
        
        # Get f8 spikes with better thresholding
        f8_spikes = self._get_enhanced_f8_spikes()
        print(f"\n🚀 Found {len(f8_spikes)} f8 spikes across all sessions")
        
        # Get Theory B events with various precision levels
        theory_b_events = self._get_enhanced_theory_b_events()
        print(f"⚡ Found {len(theory_b_events)} Theory B precision events")
        
        # Find correlations
        correlations = self._find_enhanced_correlations(f8_spikes, theory_b_events)
        print(f"🔗 Found {len(correlations)} correlations within {self.TIME_WINDOW_MINUTES}-minute windows")
        
        # Analyze by session type
        session_analysis = self._analyze_by_session_type(correlations)
        
        # Analyze by gap characteristics  
        gap_analysis = self._analyze_by_gap_characteristics(correlations)
        
        # Generate detailed insights
        insights = self._generate_enhanced_insights(correlations, session_analysis, gap_analysis)
        
        return {
            "total_f8_spikes": len(f8_spikes),
            "total_theory_b_events": len(theory_b_events),
            "total_correlations": len(correlations),
            "correlations": correlations,
            "session_analysis": session_analysis,
            "gap_analysis": gap_analysis,
            "insights": insights,
            "best_correlations": sorted(correlations, key=lambda x: x['combined_score'], reverse=True)[:10]
        }
    
    def _analyze_available_sessions(self) -> Dict[str, Any]:
        """Analyze what sessions we have available"""
        session_type_counts = {}
        trading_sessions = []
        
        for session_id in self.engine.sessions.keys():
            # Extract session type
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            session_type_counts[session_type] = session_type_counts.get(session_type, 0) + 1
            
            # Check if it's a trading session
            if session_type in ['NYAM', 'NYPM', 'LONDON', 'ASIA', 'TOKYO']:
                trading_sessions.append(session_id)
        
        return {
            "session_type_counts": session_type_counts,
            "trading_sessions": trading_sessions,
            "total_sessions": len(self.engine.sessions)
        }
    
    def _get_enhanced_f8_spikes(self) -> List[Dict[str, Any]]:
        """Get f8 spikes with enhanced detection"""
        spikes = []
        
        # Collect all f8 values for percentile calculation
        all_f8_values = []
        session_f8_stats = {}
        
        for session_id, session_data in self.engine.sessions.items():
            if 'f8' in session_data.columns:
                f8_values = session_data['f8'].dropna()
                if len(f8_values) > 0:
                    all_f8_values.extend(f8_values.tolist())
                    session_f8_stats[session_id] = {
                        'mean': f8_values.mean(),
                        'std': f8_values.std(),
                        'max': f8_values.max(),
                        'percentile_95': f8_values.quantile(0.95)
                    }
        
        if not all_f8_values:
            return spikes
        
        global_f8_threshold = np.percentile(all_f8_values, self.F8_SPIKE_THRESHOLD_PERCENTILE)
        
        # Find spikes using both global and local thresholds
        for session_id, session_data in self.engine.sessions.items():
            if 'f8' not in session_data.columns or session_id not in session_f8_stats:
                continue
            
            session_stats = session_f8_stats[session_id]
            local_threshold = session_stats['mean'] + 2 * session_stats['std']
            
            # Use the lower of global or local threshold to capture more events
            threshold = min(global_f8_threshold, local_threshold) if local_threshold > 0 else global_f8_threshold
            
            spike_events = session_data[session_data['f8'] > threshold]
            
            for _, event in spike_events.iterrows():
                # Enhanced gap age detection
                gap_age = self._determine_enhanced_gap_age(event, session_data)
                
                spike = {
                    "session_id": session_id,
                    "session_type": session_id.split('_')[0] if '_' in session_id else 'UNKNOWN',
                    "timestamp": self._convert_ms_to_time(event['t']),
                    "timestamp_ms": event['t'],
                    "price": event['price'],
                    "f8_value": event['f8'],
                    "f8_percentile": self._calculate_percentile_rank(event['f8'], all_f8_values),
                    "gap_age": gap_age,
                    "session_progress": self._calculate_session_progress(event['t'], session_data),
                    "node_id": event.get('node_id', 'unknown'),
                    "relative_strength": event['f8'] / session_stats['mean'] if session_stats['mean'] > 0 else 1.0
                }
                spikes.append(spike)
        
        return sorted(spikes, key=lambda x: x['f8_percentile'], reverse=True)
    
    def _determine_enhanced_gap_age(self, event: pd.Series, session_data: pd.DataFrame) -> str:
        """Enhanced gap age determination using multiple indicators"""
        
        # Method 1: Use f40 if available (gap fill indicator)
        if 'f40' in event and not pd.isna(event['f40']):
            f40_val = event['f40']
            if f40_val > 0.7:
                return 'young'
            elif f40_val > 0.3:
                return 'mature'
            else:
                return 'aged'
        
        # Method 2: Use price volatility in surrounding period
        event_time = event['t']
        time_window = 300000  # 5 minutes in ms
        
        nearby_data = session_data[
            (session_data['t'] >= event_time - time_window) & 
            (session_data['t'] <= event_time + time_window)
        ]
        
        if len(nearby_data) > 1:
            price_volatility = nearby_data['price'].std()
            session_volatility = session_data['price'].std()
            
            if price_volatility > session_volatility * 1.5:
                return 'young'  # High volatility suggests fresh activity
            elif price_volatility > session_volatility * 0.8:
                return 'mature'
            else:
                return 'aged'  # Lower volatility suggests filled/aged gaps
        
        # Method 3: Position in session as fallback
        session_progress = self._calculate_session_progress(event_time, session_data)
        
        if session_progress < 30:
            return 'young'
        elif session_progress < 70:
            return 'mature'
        else:
            return 'aged'
    
    def _get_enhanced_theory_b_events(self) -> List[Dict[str, Any]]:
        """Get Theory B events with enhanced analysis"""
        events = []
        
        for session_id, session_data in self.engine.sessions.items():
            if session_id not in self.engine.session_stats:
                continue
            
            session_stats = self.engine.session_stats[session_id]
            session_type = session_id.split('_')[0] if '_' in session_id else 'UNKNOWN'
            
            # Sample events for analysis (every 10th event to manage performance)
            sampled_data = session_data.iloc[::10] if len(session_data) > 50 else session_data
            
            for _, event in sampled_data.iterrows():
                if pd.isna(event['price']):
                    continue
                
                timestamp = self._convert_ms_to_time(event['t'])
                
                try:
                    analysis = self.zone_calculator.analyze_event_positioning(
                        event_price=event['price'],
                        event_time=timestamp,
                        session_type=session_type,
                        final_session_stats=session_stats
                    )
                    
                    theory_b = analysis.get('theory_b_analysis', {})
                    precision_score = theory_b.get('precision_score', 0)
                    
                    # Lower threshold to capture more events
                    if precision_score >= 0.3 and theory_b.get('meets_theory_b_precision', False):
                        theory_b_event = {
                            "session_id": session_id,
                            "session_type": session_type,
                            "timestamp": timestamp,
                            "timestamp_ms": event['t'],
                            "price": event['price'],
                            "precision_score": precision_score,
                            "distance_to_final_40pct": theory_b.get('distance_to_final_40pct', 0),
                            "archaeological_zone": analysis.get('dimensional_relationship', 'unknown'),
                            "session_progress": self._calculate_session_progress(event['t'], session_data),
                            "is_dimensional_destiny": analysis.get('closest_zone', {}).get('is_dimensional_destiny', False),
                            "node_id": event.get('node_id', 'unknown'),
                            "zone_distance": analysis.get('closest_zone', {}).get('distance', 999)
                        }
                        events.append(theory_b_event)
                        
                except Exception as e:
                    continue
        
        return sorted(events, key=lambda x: x['precision_score'], reverse=True)
    
    def _find_enhanced_correlations(self, f8_spikes: List[Dict], theory_b_events: List[Dict]) -> List[Dict]:
        """Find correlations with enhanced matching"""
        correlations = []
        time_window_ms = self.TIME_WINDOW_MINUTES * 60 * 1000
        
        for f8_spike in f8_spikes:
            f8_session = f8_spike['session_id']
            f8_time_ms = f8_spike['timestamp_ms']
            
            for theory_b_event in theory_b_events:
                if theory_b_event['session_id'] != f8_session:
                    continue
                
                time_diff_ms = theory_b_event['timestamp_ms'] - f8_time_ms
                time_diff_minutes = time_diff_ms / (60 * 1000)
                
                if abs(time_diff_minutes) <= self.TIME_WINDOW_MINUTES:
                    correlation = {
                        "session_id": f8_session,
                        "session_type": f8_spike['session_type'],
                        "f8_spike": f8_spike,
                        "theory_b_event": theory_b_event,
                        "time_difference_minutes": time_diff_minutes,
                        "temporal_relationship": self._classify_temporal_relationship(time_diff_minutes),
                        "combined_score": self._calculate_enhanced_combined_score(f8_spike, theory_b_event),
                        "gap_age": f8_spike['gap_age'],
                        "archaeological_zone": theory_b_event['archaeological_zone'],
                        "session_phase": self._determine_session_phase(theory_b_event['session_progress'])
                    }
                    correlations.append(correlation)
        
        return correlations
    
    def _calculate_enhanced_combined_score(self, f8_spike: Dict, theory_b_event: Dict) -> float:
        """Enhanced combined score calculation"""
        # f8 strength (percentile + relative strength)
        f8_percentile_score = f8_spike.get('f8_percentile', 50) / 100
        f8_relative_score = min(f8_spike.get('relative_strength', 1.0) / 3.0, 1.0)
        f8_score = (f8_percentile_score * 0.7 + f8_relative_score * 0.3)
        
        # Theory B precision score
        theory_b_score = theory_b_event.get('precision_score', 0)
        
        # Zone quality multiplier
        zone_distance = theory_b_event.get('zone_distance', 999)
        zone_multiplier = max(0.5, (20 - zone_distance) / 20) if zone_distance < 20 else 0.5
        
        # Dimensional destiny bonus
        destiny_bonus = 0.3 if theory_b_event.get('is_dimensional_destiny', False) else 0
        
        # Gap age bonus (young gaps might be more significant)
        gap_age_bonus = 0.2 if f8_spike.get('gap_age') == 'young' else 0.1 if f8_spike.get('gap_age') == 'mature' else 0
        
        base_score = (f8_score * 0.4 + theory_b_score * 0.6) * zone_multiplier
        return base_score + destiny_bonus + gap_age_bonus
    
    def _analyze_by_session_type(self, correlations: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations by session type"""
        session_analysis = {}
        
        for correlation in correlations:
            session_type = correlation['session_type']
            if session_type not in session_analysis:
                session_analysis[session_type] = {
                    'count': 0,
                    'scores': [],
                    'gap_ages': [],
                    'zones': [],
                    'temporal_relationships': []
                }
            
            session_analysis[session_type]['count'] += 1
            session_analysis[session_type]['scores'].append(correlation['combined_score'])
            session_analysis[session_type]['gap_ages'].append(correlation['gap_age'])
            session_analysis[session_type]['zones'].append(correlation['archaeological_zone'])
            session_analysis[session_type]['temporal_relationships'].append(correlation['temporal_relationship'])
        
        # Calculate statistics for each session type
        for session_type, data in session_analysis.items():
            if data['scores']:
                data['avg_score'] = np.mean(data['scores'])
                data['max_score'] = np.max(data['scores'])
                data['dominant_gap_age'] = max(set(data['gap_ages']), key=data['gap_ages'].count)
                data['dominant_zone'] = max(set(data['zones']), key=data['zones'].count)
        
        return session_analysis
    
    def _analyze_by_gap_characteristics(self, correlations: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations by gap characteristics"""
        gap_analysis = {
            'young': {'count': 0, 'scores': [], 'session_types': []},
            'mature': {'count': 0, 'scores': [], 'session_types': []},
            'aged': {'count': 0, 'scores': [], 'session_types': []},
            'unknown': {'count': 0, 'scores': [], 'session_types': []}
        }
        
        for correlation in correlations:
            gap_age = correlation['gap_age']
            if gap_age in gap_analysis:
                gap_analysis[gap_age]['count'] += 1
                gap_analysis[gap_age]['scores'].append(correlation['combined_score'])
                gap_analysis[gap_age]['session_types'].append(correlation['session_type'])
        
        # Calculate statistics
        for gap_age, data in gap_analysis.items():
            if data['scores']:
                data['avg_score'] = np.mean(data['scores'])
                data['max_score'] = np.max(data['scores'])
                if data['session_types']:
                    data['dominant_session_type'] = max(set(data['session_types']), key=data['session_types'].count)
        
        return gap_analysis
    
    def _convert_ms_to_time(self, timestamp_ms: int) -> str:
        """Convert milliseconds to readable time (proper epoch conversion)"""
        # Check if this is epoch time (milliseconds since 1970) or relative time
        if timestamp_ms > 1000000000000:  # Epoch time (after year 2001)
            # Convert epoch milliseconds to datetime
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp_ms / 1000)
            return dt.strftime("%H:%M:%S")
        else:
            # Relative time since session start - convert to approximate display time
            seconds = timestamp_ms // 1000
            hours = (seconds // 3600) % 24
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            
            # For relative time, assume session starts around market open for display
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _calculate_session_progress(self, event_time_ms: int, session_data: pd.DataFrame) -> float:
        """Calculate session progress percentage"""
        session_start = session_data['t'].min()
        session_end = session_data['t'].max()
        session_duration = session_end - session_start
        
        if session_duration > 0:
            progress = (event_time_ms - session_start) / session_duration * 100
            return max(0, min(100, progress))
        return 0
    
    def _calculate_percentile_rank(self, value: float, all_values: List[float]) -> float:
        """Calculate percentile rank of a value"""
        return (sum(1 for v in all_values if v <= value) / len(all_values)) * 100
    
    def _classify_temporal_relationship(self, time_diff_minutes: float) -> str:
        """Classify temporal relationship"""
        if time_diff_minutes < -1:
            return "theory_b_precedes_f8"
        elif time_diff_minutes > 1:
            return "f8_precedes_theory_b"
        else:
            return "simultaneous"
    
    def _determine_session_phase(self, session_progress: float) -> str:
        """Determine session phase"""
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
    
    def _generate_enhanced_insights(self, correlations: List[Dict], 
                                  session_analysis: Dict, gap_analysis: Dict) -> List[str]:
        """Generate enhanced insights"""
        insights = []
        
        if not correlations:
            insights.append("No correlations found - consider adjusting filter criteria")
            return insights
        
        total = len(correlations)
        insights.append(f"Found {total} Theory B + f8 correlations within 5-minute windows")
        
        # Session type insights
        if session_analysis:
            best_session = max(session_analysis.keys(), key=lambda k: session_analysis[k]['count'])
            best_count = session_analysis[best_session]['count']
            best_avg_score = session_analysis[best_session].get('avg_score', 0)
            
            insights.append(f"Most active session type: {best_session} ({best_count} correlations, avg score: {best_avg_score:.3f})")
            
            # Trading session analysis
            trading_sessions = {k: v for k, v in session_analysis.items() if k in ['NYAM', 'NYPM', 'LONDON', 'ASIA']}
            if trading_sessions:
                trading_total = sum(data['count'] for data in trading_sessions.values())
                insights.append(f"Trading sessions account for {trading_total} correlations ({trading_total/total*100:.1f}%)")
        
        # Gap age insights
        if gap_analysis:
            gap_counts = {k: v['count'] for k, v in gap_analysis.items() if v['count'] > 0}
            if gap_counts:
                dominant_gap = max(gap_counts.keys(), key=lambda k: gap_counts[k])
                dominant_count = gap_counts[dominant_gap]
                dominant_score = gap_analysis[dominant_gap].get('avg_score', 0)
                
                insights.append(f"Gap age pattern: {dominant_gap} gaps dominate ({dominant_count}/{total}, avg score: {dominant_score:.3f})")
        
        # Quality insights
        scores = [c['combined_score'] for c in correlations]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        insights.append(f"Quality metrics: avg score {avg_score:.3f}, max score {max_score:.3f}")
        
        high_quality = sum(1 for s in scores if s > 0.8)
        if high_quality > 0:
            insights.append(f"{high_quality} high-quality correlations (score > 0.8) detected")
        
        # Temporal relationship insights
        temp_relationships = [c['temporal_relationship'] for c in correlations]
        if temp_relationships:
            dominant_timing = max(set(temp_relationships), key=temp_relationships.count)
            timing_count = temp_relationships.count(dominant_timing)
            insights.append(f"Temporal pattern: {timing_count}/{total} show '{dominant_timing.replace('_', ' ')}'")
        
        return insights


def run_enhanced_analysis():
    """Run enhanced Theory B + f8 correlation analysis"""
    analyzer = EnhancedTheoryBF8Analyzer()
    results = analyzer.analyze_correlations_comprehensive()
    
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS RESULTS")
    print("=" * 60)
    
    # Display key findings
    insights = results['insights']
    print(f"\n💡 Key Insights:")
    for insight in insights:
        print(f"   • {insight}")
    
    # Display session analysis
    session_analysis = results['session_analysis']
    if session_analysis:
        print(f"\n📊 Session Type Analysis:")
        for session_type, data in session_analysis.items():
            print(f"   {session_type}: {data['count']} correlations (avg: {data.get('avg_score', 0):.3f})")
    
    # Display gap analysis
    gap_analysis = results['gap_analysis']
    if any(data['count'] > 0 for data in gap_analysis.values()):
        print(f"\n🔍 Gap Age Analysis:")
        for gap_age, data in gap_analysis.items():
            if data['count'] > 0:
                print(f"   {gap_age}: {data['count']} correlations (avg: {data.get('avg_score', 0):.3f})")
    
    # Display best correlations
    best_correlations = results['best_correlations'][:5]
    if best_correlations:
        print(f"\n🎯 Top 5 Correlations:")
        for i, corr in enumerate(best_correlations, 1):
            f8 = corr['f8_spike']
            tb = corr['theory_b_event']
            print(f"   {i}. {corr['session_type']} - Score: {corr['combined_score']:.3f}")
            print(f"      f8: {f8['timestamp']} ({f8.get('f8_percentile', 0):.1f}%ile)")
            print(f"      Theory B: {tb['timestamp']} (precision: {tb['precision_score']:.3f})")
            print(f"      Gap: {corr['gap_age']}, Zone: {tb['archaeological_zone']}")
    
    return results


if __name__ == "__main__":
    results = run_enhanced_analysis()
    print(f"\n✅ Enhanced Theory B + f8 analysis complete!")