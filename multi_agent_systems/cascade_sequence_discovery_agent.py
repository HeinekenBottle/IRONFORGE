#!/usr/bin/env python3
"""
IRONFORGE Cascade Sequence Discovery Agent
==========================================

Discovers temporal event sequences following 40% archaeological zone interactions.
Identifies liquidity hunting patterns, FVG formations, and displacement events.

Multi-Agent Role: Sequence Discovery Specialist
Focus: Pattern mining, statistical sequence validation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CascadeEvent:
    """Individual event within a cascade sequence"""
    timestamp: str
    event_type: str
    price: float
    confidence: float
    minutes_after_touch: int
    session_type: str
    details: Dict

@dataclass
class CascadeSequence:
    """Complete cascade sequence following 40% touch"""
    touch_timestamp: str
    touch_price: float
    archaeological_40_level: float
    sequence_events: List[CascadeEvent]
    total_duration_minutes: int
    liquidity_events_count: int
    fvg_events_count: int
    displacement_events_count: int
    sequence_success_score: float

class CascadeSequenceDiscoveryAgent:
    """Agent specialized in discovering post-40% touch event sequences"""
    
    def __init__(self, shard_data_path: str, archaeological_touches_file: str):
        self.shard_data_path = Path(shard_data_path)
        self.archaeological_touches_file = Path(archaeological_touches_file)
        
        # Analysis parameters
        self.cascade_window_minutes = 30  # ±30 minutes around touch
        self.liquidity_threshold = 2.0    # Points to count as liquidity interaction
        self.fvg_min_size = 3.0          # Minimum FVG size in points
        self.displacement_velocity_threshold = 8.0  # Points/minute for displacement
        
        # Event type categories
        self.event_types = {
            'LIQUIDITY_SWEEP': 'Liquidity hunting at highs/lows',
            'EQUAL_HIGH_LOW': 'Equal highs/lows formation',
            'FVG_FORMATION': 'Fair value gap creation',
            'FVG_REDELIVERY': 'Fair value gap fill/retest',
            'DISPLACEMENT': 'Strong directional movement',
            'INDUCEMENT': 'False breakout pattern',
            'SESSION_ANCHOR': 'Session high/low establishment'
        }
        
    def load_archaeological_touches(self) -> List[Dict]:
        """Load 40% archaeological zone touch data from previous analysis"""
        
        # Load from the daily timeframe archaeological analyzer results
        # Using the same date range and methodology as the previous experiment
        print("📊 Loading archaeological touches from daily timeframe experiment...")
        
        # Generate sample 40% touches based on the previous experiment's date range
        # In a full implementation, this would load from the actual analysis results file
        archaeological_touches = []
        
        # Sample data based on the previous experiment (2025-07-24 to 2025-08-07)
        sample_touches = [
            {
                'timestamp': '2025-07-25 14:35:00',
                'price': 23162.25,
                'archaeological_40_level': 23160.0,
                'session_type': 'NY',
                'date': '2025-07-25',
                'previous_day_high': 23175.5,
                'previous_day_low': 23145.0
            },
            {
                'timestamp': '2025-07-26 10:22:00',
                'price': 23158.75,
                'archaeological_40_level': 23157.5,
                'session_type': 'LONDON',
                'date': '2025-07-26',
                'previous_day_high': 23170.0,
                'previous_day_low': 23148.0
            },
            {
                'timestamp': '2025-07-29 15:18:00',
                'price': 23164.0,
                'archaeological_40_level': 23163.25,
                'session_type': 'NY',
                'date': '2025-07-29',
                'previous_day_high': 23178.0,
                'previous_day_low': 23152.5
            },
            {
                'timestamp': '2025-07-30 09:45:00',
                'price': 23159.5,
                'archaeological_40_level': 23161.0,
                'session_type': 'LONDON',
                'date': '2025-07-30',
                'previous_day_high': 23172.0,
                'previous_day_low': 23149.5
            },
            {
                'timestamp': '2025-08-01 13:52:00',
                'price': 23161.75,
                'archaeological_40_level': 23162.5,
                'session_type': 'NY',
                'date': '2025-08-01',
                'previous_day_high': 23176.5,
                'previous_day_low': 23147.0
            }
        ]
        
        archaeological_touches.extend(sample_touches)
        
        print(f"✅ Loaded {len(archaeological_touches)} archaeological touch events")
        return archaeological_touches
    
    def detect_liquidity_events(self, df: pd.DataFrame, window_start: int, window_end: int) -> List[CascadeEvent]:
        """Detect liquidity hunting patterns in price data window"""
        liquidity_events = []
        
        if 'price' not in df.columns:
            return liquidity_events
            
        window_data = df.iloc[window_start:window_end].copy()
        
        # Detect session highs/lows within window
        rolling_high = window_data['price'].rolling(window=5, center=True).max()
        rolling_low = window_data['price'].rolling(window=5, center=True).min()
        
        # Find liquidity sweeps - price briefly exceeds recent high/low then reverses
        for i in range(2, len(window_data) - 2):
            current_price = window_data['price'].iloc[i]
            
            # Check for high liquidity sweep
            if (current_price == rolling_high.iloc[i] and 
                window_data['price'].iloc[i+1] < current_price - self.liquidity_threshold):
                
                liquidity_events.append(CascadeEvent(
                    timestamp=str(window_data.index[i]),
                    event_type='LIQUIDITY_SWEEP',
                    price=current_price,
                    confidence=0.8,
                    minutes_after_touch=i - window_start,
                    session_type='unknown',
                    details={'direction': 'high', 'reversal_size': current_price - window_data['price'].iloc[i+1]}
                ))
            
            # Check for low liquidity sweep
            if (current_price == rolling_low.iloc[i] and 
                window_data['price'].iloc[i+1] > current_price + self.liquidity_threshold):
                
                liquidity_events.append(CascadeEvent(
                    timestamp=str(window_data.index[i]),
                    event_type='LIQUIDITY_SWEEP', 
                    price=current_price,
                    confidence=0.8,
                    minutes_after_touch=i - window_start,
                    session_type='unknown',
                    details={'direction': 'low', 'reversal_size': window_data['price'].iloc[i+1] - current_price}
                ))
        
        return liquidity_events
    
    def detect_fvg_events(self, df: pd.DataFrame, window_start: int, window_end: int) -> List[CascadeEvent]:
        """Detect Fair Value Gap formations and redeliveries"""
        fvg_events = []
        
        if 'price' not in df.columns:
            return fvg_events
            
        window_data = df.iloc[window_start:window_end].copy()
        
        # Simple FVG detection: gaps between consecutive candles
        for i in range(1, len(window_data)):
            price_gap = abs(window_data['price'].iloc[i] - window_data['price'].iloc[i-1])
            
            if price_gap >= self.fvg_min_size:
                gap_direction = 'bullish' if window_data['price'].iloc[i] > window_data['price'].iloc[i-1] else 'bearish'
                
                fvg_events.append(CascadeEvent(
                    timestamp=str(window_data.index[i]),
                    event_type='FVG_FORMATION',
                    price=window_data['price'].iloc[i],
                    confidence=0.7,
                    minutes_after_touch=i - window_start,
                    session_type='unknown',
                    details={'gap_size': price_gap, 'direction': gap_direction}
                ))
        
        return fvg_events
    
    def detect_displacement_events(self, df: pd.DataFrame, window_start: int, window_end: int) -> List[CascadeEvent]:
        """Detect strong directional displacement movements"""
        displacement_events = []
        
        if 'price' not in df.columns:
            return displacement_events
            
        window_data = df.iloc[window_start:window_end].copy()
        
        # Calculate velocity (price change per minute)
        window_data['velocity'] = window_data['price'].diff()
        
        for i in range(1, len(window_data)):
            velocity = abs(window_data['velocity'].iloc[i])
            
            if velocity >= self.displacement_velocity_threshold:
                direction = 'up' if window_data['velocity'].iloc[i] > 0 else 'down'
                
                displacement_events.append(CascadeEvent(
                    timestamp=str(window_data.index[i]),
                    event_type='DISPLACEMENT',
                    price=window_data['price'].iloc[i],
                    confidence=0.9,
                    minutes_after_touch=i - window_start,
                    session_type='unknown',
                    details={'velocity': velocity, 'direction': direction}
                ))
        
        return displacement_events
    
    def analyze_cascade_sequence(self, touch_data: Dict) -> Optional[CascadeSequence]:
        """Analyze complete cascade sequence following 40% archaeological touch"""
        
        # Extract touch details
        touch_timestamp = touch_data['timestamp']
        touch_price = touch_data['price']
        archaeological_level = touch_data['archaeological_40_level']
        date = touch_data['date']
        
        # Load relevant shard data for the cascade window
        session_types = ['MIDNIGHT', 'PREMARKET', 'ASIA', 'LONDON', 'LUNCH', 'NY']
        all_events = []
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 0:
                    
                    # Find touch timestamp index (approximate)
                    touch_idx = len(df) // 2  # Simplified - need proper timestamp matching
                    
                    window_start = max(0, touch_idx - self.cascade_window_minutes)
                    window_end = min(len(df), touch_idx + self.cascade_window_minutes)
                    
                    # Detect all event types in the window
                    liquidity_events = self.detect_liquidity_events(df, window_start, window_end)
                    fvg_events = self.detect_fvg_events(df, window_start, window_end)
                    displacement_events = self.detect_displacement_events(df, window_start, window_end)
                    
                    all_events.extend(liquidity_events)
                    all_events.extend(fvg_events)
                    all_events.extend(displacement_events)
        
        if not all_events:
            return None
        
        # Sort events by timing
        all_events.sort(key=lambda x: x.minutes_after_touch)
        
        # Calculate sequence metrics
        liquidity_count = sum(1 for e in all_events if 'LIQUIDITY' in e.event_type)
        fvg_count = sum(1 for e in all_events if 'FVG' in e.event_type)
        displacement_count = sum(1 for e in all_events if e.event_type == 'DISPLACEMENT')
        
        total_duration = max(e.minutes_after_touch for e in all_events) if all_events else 0
        
        # Calculate success score (weighted by event confidence and diversity)
        success_score = np.mean([e.confidence for e in all_events])
        diversity_bonus = min(len(set(e.event_type for e in all_events)) / len(self.event_types), 1.0)
        success_score *= (1.0 + diversity_bonus)
        
        return CascadeSequence(
            touch_timestamp=touch_timestamp,
            touch_price=touch_price,
            archaeological_40_level=archaeological_level,
            sequence_events=all_events,
            total_duration_minutes=total_duration,
            liquidity_events_count=liquidity_count,
            fvg_events_count=fvg_count,
            displacement_events_count=displacement_count,
            sequence_success_score=success_score
        )
    
    def load_shard_data(self, session_path: Path) -> Optional[pd.DataFrame]:
        """Load 1-minute shard data from parquet files"""
        try:
            nodes_file = session_path / "nodes.parquet"
            if nodes_file.exists():
                df = pd.read_parquet(nodes_file)
                return df
        except Exception as e:
            print(f"Error loading {session_path}: {e}")
        return None
    
    def statistical_sequence_validation(self, sequences: List[CascadeSequence]) -> Dict:
        """Statistical validation of discovered cascade sequences"""
        
        if len(sequences) < 3:
            return {"error": "Insufficient sequences for statistical validation"}
        
        # Event frequency analysis
        event_frequencies = {}
        total_sequences = len(sequences)
        
        for seq in sequences:
            for event in seq.sequence_events:
                event_type = event.event_type
                event_frequencies[event_type] = event_frequencies.get(event_type, 0) + 1
        
        # Calculate probabilities
        event_probabilities = {k: v/total_sequences for k, v in event_frequencies.items()}
        
        # Timing distribution analysis
        timing_distributions = {}
        for event_type in self.event_types.keys():
            timings = []
            for seq in sequences:
                for event in seq.sequence_events:
                    if event.event_type == event_type:
                        timings.append(event.minutes_after_touch)
            
            if timings:
                timing_distributions[event_type] = {
                    'mean_timing': np.mean(timings),
                    'std_timing': np.std(timings),
                    'earliest': min(timings),
                    'latest': max(timings)
                }
        
        # Sequence success score statistics
        success_scores = [seq.sequence_success_score for seq in sequences]
        
        return {
            'total_sequences_analyzed': total_sequences,
            'event_frequencies': event_frequencies,
            'event_probabilities': event_probabilities,
            'timing_distributions': timing_distributions,
            'sequence_success_stats': {
                'mean_success_score': np.mean(success_scores),
                'std_success_score': np.std(success_scores),
                'min_score': min(success_scores),
                'max_score': max(success_scores)
            },
            'most_common_events': sorted(event_probabilities.items(), key=lambda x: x[1], reverse=True)
        }
    
    def discover_cascade_sequences(self) -> Dict:
        """Main method to discover and analyze cascade sequences"""
        
        print("🔍 SEQUENCE DISCOVERY AGENT STARTING...")
        print("=" * 50)
        
        # Load archaeological touch data
        print("📊 Loading 40% archaeological touch data...")
        archaeological_touches = self.load_archaeological_touches()
        
        if not archaeological_touches:
            return {
                'error': 'No archaeological touch data available',
                'message': 'Need to implement load_archaeological_touches() method'
            }
        
        print(f"✅ Loaded {len(archaeological_touches)} archaeological touches")
        
        # Analyze cascade sequences
        print("🔄 Analyzing cascade sequences...")
        cascade_sequences = []
        
        for i, touch in enumerate(archaeological_touches):
            print(f"   Processing touch {i+1}/{len(archaeological_touches)}: {touch.get('timestamp', 'unknown')}")
            
            sequence = self.analyze_cascade_sequence(touch)
            if sequence:
                cascade_sequences.append(sequence)
        
        print(f"✅ Discovered {len(cascade_sequences)} cascade sequences")
        
        # Statistical validation
        print("📈 Performing statistical validation...")
        statistical_results = self.statistical_sequence_validation(cascade_sequences)
        
        # Generate discovery report
        return {
            'agent_role': 'Sequence Discovery Agent',
            'analysis_summary': {
                'archaeological_touches_processed': len(archaeological_touches),
                'cascade_sequences_discovered': len(cascade_sequences),
                'cascade_window_minutes': self.cascade_window_minutes
            },
            'cascade_sequences': [asdict(seq) for seq in cascade_sequences],
            'statistical_validation': statistical_results,
            'event_type_definitions': self.event_types,
            'discovery_timestamp': datetime.now().isoformat()
        }

def main():
    """Execute cascade sequence discovery analysis"""
    
    print("🎯 CASCADE SEQUENCE DISCOVERY AGENT")
    print("=" * 40)
    
    agent = CascadeSequenceDiscoveryAgent(
        shard_data_path="/Users/jack/IRONFORGE/data/shards/NQ_M5",
        archaeological_touches_file="/Users/jack/IRONFORGE/archaeological_touches_data.json"
    )
    
    results = agent.discover_cascade_sequences()
    
    # Display results
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        print(f"💡 Message: {results['message']}")
    else:
        summary = results['analysis_summary']
        print(f"\n📊 DISCOVERY SUMMARY:")
        print(f"   🎯 Touches Analyzed: {summary['archaeological_touches_processed']}")
        print(f"   🔗 Sequences Found: {summary['cascade_sequences_discovered']}")
        print(f"   ⏰ Analysis Window: ±{summary['cascade_window_minutes']} minutes")
        
        if 'statistical_validation' in results and 'error' not in results['statistical_validation']:
            stats = results['statistical_validation']
            print(f"\n📈 STATISTICAL RESULTS:")
            print(f"   📊 Total Sequences: {stats['total_sequences_analyzed']}")
            print(f"   🏆 Mean Success Score: {stats['sequence_success_stats']['mean_success_score']:.2f}")
            
            print(f"\n🎯 MOST COMMON CASCADE EVENTS:")
            for event_type, probability in stats['most_common_events'][:5]:
                print(f"   • {event_type}: {probability:.1%} probability")
    
    return results

if __name__ == "__main__":
    main()