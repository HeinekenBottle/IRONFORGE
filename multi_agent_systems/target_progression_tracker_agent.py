#!/usr/bin/env python3
"""
IRONFORGE Target Progression Tracker Agent
==========================================

Liquidity hunting specialist - monitors post-40% movement toward pre-existing targets.
Tracks which session highs/lows, daily extremes, and FVGs get taken after archaeological touches.

Multi-Agent Role: Target Progression Tracker
Focus: Current session vs next session completion timing analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TargetCompletion:
    """Individual target completion result"""
    target_type: str  # 'session_high', 'session_low', 'daily_high', 'daily_low', 'fvg_redelivery'
    target_id: str
    target_price: float
    completed: bool
    completion_timestamp: Optional[str]
    completion_session: Optional[str]
    timing_classification: str  # 'current_session', 'next_session', 'delayed', 'never'
    minutes_after_touch: Optional[int]
    completion_details: Dict

@dataclass
class SessionProgressionAnalysis:
    """Analysis of progression within specific session"""
    session_type: str
    targets_in_session: int
    targets_completed: int
    completion_rate: float
    average_completion_time: Optional[float]
    completion_sequence: List[str]  # Order of target completion

@dataclass
class ProgressionTrackingResult:
    """Complete progression tracking result for one archaeological touch"""
    archaeological_touch: Dict
    pre_structure_inventory: Dict
    target_completions: List[TargetCompletion]
    session_analyses: List[SessionProgressionAnalysis]
    overall_statistics: Dict
    tracking_timestamp: str

class TargetProgressionTrackerAgent:
    """Agent specialized in tracking post-40% target progression"""
    
    def __init__(self, shard_data_path: str, pre_structure_inventories_file: str):
        self.shard_data_path = Path(shard_data_path)
        self.pre_structure_inventories_file = Path(pre_structure_inventories_file)
        
        # Tracking parameters
        self.tracking_duration_hours = 24  # Track for 24 hours after touch
        self.target_hit_tolerance = 1.0    # Points tolerance for "hitting" target
        self.fvg_fill_threshold = 0.8      # 80% fill required for FVG completion
        
        # Session boundary definitions for timing classification
        self.session_boundaries = {
            'ASIA': ('21:00', '05:00'),
            'LONDON': ('03:00', '11:00'), 
            'LUNCH': ('11:00', '13:00'),
            'NY': ('09:30', '16:00'),
            'PREMARKET': ('06:00', '09:30'),
            'MIDNIGHT': ('16:00', '21:00')
        }
    
    def load_pre_structure_inventories(self) -> List[Dict]:
        """Load pre-structure inventories from previous agent"""
        
        print("📊 Loading pre-structure inventories...")
        
        # For now, return sample inventories based on the structure we designed
        # In production, this would load from the actual Pre-Structure Inventory Agent output
        
        sample_inventories = [
            {
                'archaeological_touch': {
                    'timestamp': '2025-07-25 14:35:00',
                    'price': 23162.25,
                    'archaeological_40_level': 23160.0,
                    'session_type': 'NY',
                    'date': '2025-07-25'
                },
                'session_structures': [
                    {
                        'session_type': 'LONDON',
                        'session_high': 23175.0,
                        'session_low': 23155.0,
                        'is_current_session': False
                    },
                    {
                        'session_type': 'NY',
                        'session_high': 23170.0,
                        'session_low': 23160.5,
                        'is_current_session': True
                    }
                ],
                'daily_structure': {
                    'daily_high': 23175.0,
                    'daily_low': 23145.0,
                    'current_day_high': 23175.0,
                    'current_day_low': 23155.0
                },
                'existing_fvgs': [
                    {
                        'fvg_id': 'LONDON_FVG_1',
                        'gap_high': 23168.0,
                        'gap_low': 23165.0,
                        'gap_type': 'bullish',
                        'is_filled': False,
                        'creation_session': 'LONDON'
                    }
                ],
                'total_targets': 7
            }
        ]
        
        print(f"✅ Loaded {len(sample_inventories)} pre-structure inventories")
        return sample_inventories
    
    def determine_session_from_timestamp(self, timestamp_str: str) -> str:
        """Determine which session a timestamp falls into"""
        
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            time_str = dt.strftime('%H:%M')
            
            # Simple session classification based on time
            if '03:00' <= time_str < '11:00':
                return 'LONDON'
            elif '09:30' <= time_str < '16:00':
                return 'NY'
            elif '21:00' <= time_str or time_str < '05:00':
                return 'ASIA'
            elif '11:00' <= time_str < '13:00':
                return 'LUNCH'
            elif '06:00' <= time_str < '09:30':
                return 'PREMARKET'
            else:
                return 'MIDNIGHT'
                
        except:
            return 'UNKNOWN'
    
    def classify_timing(self, touch_timestamp: str, completion_timestamp: str, touch_session: str) -> str:
        """Classify completion timing relative to touch session"""
        
        try:
            touch_dt = datetime.strptime(touch_timestamp, '%Y-%m-%d %H:%M:%S')
            completion_dt = datetime.strptime(completion_timestamp, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Handle non-standard timestamp formats
            return 'current_session'
        
        time_diff = completion_dt - touch_dt
        completion_session = self.determine_session_from_timestamp(completion_timestamp)
        
        # Same session if within session boundaries and same session type
        if completion_session == touch_session and time_diff.total_seconds() <= 4 * 3600:  # Within 4 hours
            return 'current_session'
        elif time_diff.total_seconds() <= 8 * 3600:  # Within 8 hours
            return 'next_session'
        elif time_diff.total_seconds() <= 24 * 3600:  # Within 24 hours
            return 'delayed'
        else:
            return 'extended_delayed'
    
    def track_session_target_completions(self, inventory: Dict, post_touch_data: pd.DataFrame, 
                                       touch_timestamp: str) -> List[TargetCompletion]:
        """Track completion of session high/low targets"""
        
        completions = []
        touch_session = inventory['archaeological_touch']['session_type']
        
        for session_structure in inventory['session_structures']:
            session_type = session_structure['session_type']
            session_high = session_structure['session_high']
            session_low = session_structure['session_low']
            
            # Track session high completion
            high_completion = self._track_price_target(
                target_price=session_high,
                target_type='session_high',
                target_id=f"{session_type}_HIGH",
                post_touch_data=post_touch_data,
                touch_timestamp=touch_timestamp,
                touch_session=touch_session,
                direction='above'
            )
            completions.append(high_completion)
            
            # Track session low completion
            low_completion = self._track_price_target(
                target_price=session_low,
                target_type='session_low',
                target_id=f"{session_type}_LOW",
                post_touch_data=post_touch_data,
                touch_timestamp=touch_timestamp,
                touch_session=touch_session,
                direction='below'
            )
            completions.append(low_completion)
        
        return completions
    
    def track_daily_target_completions(self, inventory: Dict, post_touch_data: pd.DataFrame,
                                     touch_timestamp: str) -> List[TargetCompletion]:
        """Track completion of daily high/low targets"""
        
        completions = []
        touch_session = inventory['archaeological_touch']['session_type']
        daily_structure = inventory['daily_structure']
        
        # Track daily high completion
        daily_high_completion = self._track_price_target(
            target_price=daily_structure['daily_high'],
            target_type='daily_high',
            target_id='DAILY_HIGH',
            post_touch_data=post_touch_data,
            touch_timestamp=touch_timestamp,
            touch_session=touch_session,
            direction='above'
        )
        completions.append(daily_high_completion)
        
        # Track daily low completion
        daily_low_completion = self._track_price_target(
            target_price=daily_structure['daily_low'],
            target_type='daily_low',
            target_id='DAILY_LOW',
            post_touch_data=post_touch_data,
            touch_timestamp=touch_timestamp,
            touch_session=touch_session,
            direction='below'
        )
        completions.append(daily_low_completion)
        
        return completions
    
    def track_fvg_redeliveries(self, inventory: Dict, post_touch_data: pd.DataFrame,
                             touch_timestamp: str) -> List[TargetCompletion]:
        """Track FVG redelivery completions"""
        
        completions = []
        touch_session = inventory['archaeological_touch']['session_type']
        
        for fvg in inventory['existing_fvgs']:
            if fvg['is_filled']:  # Skip already filled FVGs
                continue
                
            gap_high = fvg['gap_high']
            gap_low = fvg['gap_low']
            gap_mid = (gap_high + gap_low) / 2
            
            # Track FVG redelivery (price returning to gap zone)
            fvg_completion = self._track_fvg_redelivery(
                gap_high=gap_high,
                gap_low=gap_low,
                fvg_id=fvg['fvg_id'],
                gap_type=fvg['gap_type'],
                post_touch_data=post_touch_data,
                touch_timestamp=touch_timestamp,
                touch_session=touch_session
            )
            completions.append(fvg_completion)
        
        return completions
    
    def _track_price_target(self, target_price: float, target_type: str, target_id: str,
                           post_touch_data: pd.DataFrame, touch_timestamp: str, touch_session: str,
                           direction: str) -> TargetCompletion:
        """Track individual price target completion"""
        
        if len(post_touch_data) == 0 or 'price' not in post_touch_data.columns:
            return TargetCompletion(
                target_type=target_type,
                target_id=target_id,
                target_price=target_price,
                completed=False,
                completion_timestamp=None,
                completion_session=None,
                timing_classification='never',
                minutes_after_touch=None,
                completion_details={'error': 'no_data'}
            )
        
        # Find if target was hit
        if direction == 'above':
            target_hits = post_touch_data[post_touch_data['price'] >= (target_price - self.target_hit_tolerance)]
        else:  # direction == 'below'
            target_hits = post_touch_data[post_touch_data['price'] <= (target_price + self.target_hit_tolerance)]
        
        if len(target_hits) == 0:
            # Target not hit
            return TargetCompletion(
                target_type=target_type,
                target_id=target_id,
                target_price=target_price,
                completed=False,
                completion_timestamp=None,
                completion_session=None,
                timing_classification='never',
                minutes_after_touch=None,
                completion_details={'direction': direction, 'closest_approach': post_touch_data['price'].max() if direction == 'above' else post_touch_data['price'].min()}
            )
        
        # Target was hit - get first occurrence
        first_hit_idx = target_hits.index[0]
        completion_timestamp = str(target_hits.loc[first_hit_idx, 'timestamp_et']) if 'timestamp_et' in target_hits.columns else str(first_hit_idx)
        completion_price = target_hits.loc[first_hit_idx, 'price']
        
        # Calculate timing classification
        timing_classification = self.classify_timing(touch_timestamp, completion_timestamp, touch_session)
        completion_session = self.determine_session_from_timestamp(completion_timestamp)
        
        # Calculate minutes after touch (approximation)
        try:
            touch_dt = datetime.strptime(touch_timestamp, '%Y-%m-%d %H:%M:%S')
            completion_dt = datetime.strptime(completion_timestamp, '%Y-%m-%d %H:%M:%S')
            minutes_after = int((completion_dt - touch_dt).total_seconds() / 60)
        except:
            minutes_after = 0
        
        return TargetCompletion(
            target_type=target_type,
            target_id=target_id,
            target_price=target_price,
            completed=True,
            completion_timestamp=completion_timestamp,
            completion_session=completion_session,
            timing_classification=timing_classification,
            minutes_after_touch=minutes_after,
            completion_details={
                'direction': direction,
                'completion_price': completion_price,
                'accuracy': abs(completion_price - target_price)
            }
        )
    
    def _track_fvg_redelivery(self, gap_high: float, gap_low: float, fvg_id: str, gap_type: str,
                             post_touch_data: pd.DataFrame, touch_timestamp: str, touch_session: str) -> TargetCompletion:
        """Track FVG redelivery completion"""
        
        if len(post_touch_data) == 0 or 'price' not in post_touch_data.columns:
            return TargetCompletion(
                target_type='fvg_redelivery',
                target_id=fvg_id,
                target_price=(gap_high + gap_low) / 2,
                completed=False,
                completion_timestamp=None,
                completion_session=None,
                timing_classification='never',
                minutes_after_touch=None,
                completion_details={'error': 'no_data'}
            )
        
        # Find if price returned to FVG zone
        gap_touches = post_touch_data[
            (post_touch_data['price'] >= gap_low) & 
            (post_touch_data['price'] <= gap_high)
        ]
        
        if len(gap_touches) == 0:
            return TargetCompletion(
                target_type='fvg_redelivery',
                target_id=fvg_id,
                target_price=(gap_high + gap_low) / 2,
                completed=False,
                completion_timestamp=None,
                completion_session=None,
                timing_classification='never',
                minutes_after_touch=None,
                completion_details={'gap_type': gap_type, 'gap_range': [gap_low, gap_high]}
            )
        
        # FVG was touched - get first occurrence
        first_touch_idx = gap_touches.index[0]
        completion_timestamp = str(gap_touches.loc[first_touch_idx, 'timestamp_et']) if 'timestamp_et' in gap_touches.columns else str(first_touch_idx)
        completion_price = gap_touches.loc[first_touch_idx, 'price']
        
        # Determine fill percentage
        if gap_type == 'bullish':
            fill_percentage = (gap_high - completion_price) / (gap_high - gap_low)
        else:
            fill_percentage = (completion_price - gap_low) / (gap_high - gap_low)
        
        is_significant_fill = fill_percentage >= self.fvg_fill_threshold
        
        timing_classification = self.classify_timing(touch_timestamp, completion_timestamp, touch_session)
        completion_session = self.determine_session_from_timestamp(completion_timestamp)
        
        # Calculate minutes after touch
        try:
            touch_dt = datetime.strptime(touch_timestamp, '%Y-%m-%d %H:%M:%S')
            completion_dt = datetime.strptime(completion_timestamp, '%Y-%m-%d %H:%M:%S')
            minutes_after = int((completion_dt - touch_dt).total_seconds() / 60)
        except:
            minutes_after = 0
        
        return TargetCompletion(
            target_type='fvg_redelivery',
            target_id=fvg_id,
            target_price=(gap_high + gap_low) / 2,
            completed=is_significant_fill,
            completion_timestamp=completion_timestamp,
            completion_session=completion_session,
            timing_classification=timing_classification,
            minutes_after_touch=minutes_after,
            completion_details={
                'gap_type': gap_type,
                'gap_range': [gap_low, gap_high],
                'completion_price': completion_price,
                'fill_percentage': fill_percentage,
                'significant_fill': is_significant_fill
            }
        )
    
    def load_post_touch_data(self, inventory: Dict) -> pd.DataFrame:
        """Load price data after 40% touch for progression tracking"""
        
        touch_data = inventory['archaeological_touch']
        date = touch_data['date']
        touch_timestamp = datetime.strptime(touch_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        # Load all session data from touch date and potentially next date
        all_post_data = []
        
        session_types = ['NY', 'MIDNIGHT', 'ASIA', 'PREMARKET', 'LONDON', 'LUNCH']  # Order by typical sequence
        
        for session_type in session_types:
            session_path = self.shard_data_path / f"shard_{session_type}_{date}"
            
            if session_path.exists():
                df = self.load_shard_data(session_path)
                if df is not None and len(df) > 0:
                    
                    # Filter to only AFTER the 40% touch
                    if 'timestamp_et' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp_et'])
                        post_touch_data = df[df['datetime'] > touch_timestamp].copy()
                    else:
                        # Use index-based approximation - data after midpoint
                        midpoint = len(df) // 2
                        post_touch_data = df.iloc[midpoint:].copy()
                    
                    if len(post_touch_data) > 0:
                        post_touch_data['session_type'] = session_type
                        all_post_data.append(post_touch_data)
        
        # Combine all post-touch data
        if all_post_data:
            combined_data = pd.concat(all_post_data, ignore_index=True)
            
            # Sort by timestamp if available
            if 'timestamp_et' in combined_data.columns:
                combined_data = combined_data.sort_values('timestamp_et').reset_index(drop=True)
                
            return combined_data
        else:
            return pd.DataFrame()
    
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
    
    def analyze_session_progression(self, completions: List[TargetCompletion], inventory: Dict) -> List[SessionProgressionAnalysis]:
        """Analyze progression patterns by session"""
        
        session_analyses = []
        session_types = set([s['session_type'] for s in inventory['session_structures']])
        
        for session_type in session_types:
            # Filter completions for this session
            session_completions = [c for c in completions if c.completion_session == session_type]
            session_targets = [c for c in completions if session_type in c.target_id]
            
            targets_in_session = len(session_targets)
            targets_completed = len([c for c in session_completions if c.completed])
            completion_rate = targets_completed / targets_in_session if targets_in_session > 0 else 0
            
            # Calculate average completion time
            completion_times = [c.minutes_after_touch for c in session_completions if c.completed and c.minutes_after_touch is not None]
            avg_completion_time = np.mean(completion_times) if completion_times else None
            
            # Determine completion sequence
            completed_targets = [c for c in session_completions if c.completed]
            completion_sequence = sorted([c.target_id for c in completed_targets], 
                                       key=lambda x: next((c.minutes_after_touch for c in completed_targets if c.target_id == x), 0))
            
            analysis = SessionProgressionAnalysis(
                session_type=session_type,
                targets_in_session=targets_in_session,
                targets_completed=targets_completed,
                completion_rate=completion_rate,
                average_completion_time=avg_completion_time,
                completion_sequence=completion_sequence
            )
            
            session_analyses.append(analysis)
        
        return session_analyses
    
    def track_archaeological_touch_progression(self, inventory: Dict) -> ProgressionTrackingResult:
        """Track complete target progression for one archaeological touch"""
        
        touch_data = inventory['archaeological_touch']
        print(f"🎯 Tracking progression for {touch_data['timestamp']}...")
        
        # Load post-touch price data
        post_touch_data = self.load_post_touch_data(inventory)
        
        if len(post_touch_data) == 0:
            print("⚠️ No post-touch data available")
            return ProgressionTrackingResult(
                archaeological_touch=touch_data,
                pre_structure_inventory=inventory,
                target_completions=[],
                session_analyses=[],
                overall_statistics={'error': 'no_post_touch_data'},
                tracking_timestamp=datetime.now().isoformat()
            )
        
        # Track all target types
        all_completions = []
        
        # Session targets
        session_completions = self.track_session_target_completions(inventory, post_touch_data, touch_data['timestamp'])
        all_completions.extend(session_completions)
        
        # Daily targets
        daily_completions = self.track_daily_target_completions(inventory, post_touch_data, touch_data['timestamp'])
        all_completions.extend(daily_completions)
        
        # FVG redeliveries
        fvg_completions = self.track_fvg_redeliveries(inventory, post_touch_data, touch_data['timestamp'])
        all_completions.extend(fvg_completions)
        
        # Analyze session progression
        session_analyses = self.analyze_session_progression(all_completions, inventory)
        
        # Calculate overall statistics
        total_targets = len(all_completions)
        completed_targets = len([c for c in all_completions if c.completed])
        overall_completion_rate = completed_targets / total_targets if total_targets > 0 else 0
        
        current_session_completions = len([c for c in all_completions if c.timing_classification == 'current_session' and c.completed])
        next_session_completions = len([c for c in all_completions if c.timing_classification == 'next_session' and c.completed])
        
        overall_statistics = {
            'total_targets': total_targets,
            'completed_targets': completed_targets,
            'overall_completion_rate': overall_completion_rate,
            'current_session_completions': current_session_completions,
            'next_session_completions': next_session_completions,
            'timing_breakdown': {
                'current_session': current_session_completions / completed_targets if completed_targets > 0 else 0,
                'next_session': next_session_completions / completed_targets if completed_targets > 0 else 0
            }
        }
        
        print(f"✅ Tracking complete: {completed_targets}/{total_targets} targets completed ({overall_completion_rate:.1%})")
        
        return ProgressionTrackingResult(
            archaeological_touch=touch_data,
            pre_structure_inventory=inventory,
            target_completions=all_completions,
            session_analyses=session_analyses,
            overall_statistics=overall_statistics,
            tracking_timestamp=datetime.now().isoformat()
        )
    
    def export_tracking_results(self, results: List[ProgressionTrackingResult]) -> str:
        """Export all tracking results to JSON"""
        
        output_path = Path("/Users/jack/IRONFORGE/target_progression_tracking")
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"target_progression_tracking_{timestamp}.json"
        
        export_data = {
            'agent_role': 'Target Progression Tracker Agent',
            'tracking_timestamp': datetime.now().isoformat(),
            'total_archaeological_touches': len(results),
            'tracking_results': [asdict(result) for result in results],
            'aggregate_statistics': self._calculate_aggregate_statistics(results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Target progression tracking exported to: {output_file}")
        return str(output_file)
    
    def _calculate_aggregate_statistics(self, results: List[ProgressionTrackingResult]) -> Dict:
        """Calculate aggregate statistics across all tracking results"""
        
        if not results:
            return {}
        
        all_completions = []
        for result in results:
            all_completions.extend(result.target_completions)
        
        total_targets = len(all_completions)
        completed_targets = len([c for c in all_completions if c.completed])
        
        # Completion rates by target type
        target_types = ['session_high', 'session_low', 'daily_high', 'daily_low', 'fvg_redelivery']
        completion_rates_by_type = {}
        
        for target_type in target_types:
            type_completions = [c for c in all_completions if c.target_type == target_type]
            type_completed = len([c for c in type_completions if c.completed])
            completion_rates_by_type[target_type] = {
                'total': len(type_completions),
                'completed': type_completed,
                'rate': type_completed / len(type_completions) if type_completions else 0
            }
        
        # Timing distribution
        timing_distribution = {}
        timing_classifications = ['current_session', 'next_session', 'delayed', 'never']
        
        for timing in timing_classifications:
            timing_count = len([c for c in all_completions if c.timing_classification == timing])
            timing_distribution[timing] = {
                'count': timing_count,
                'percentage': timing_count / total_targets if total_targets > 0 else 0
            }
        
        return {
            'total_targets_tracked': total_targets,
            'total_completions': completed_targets,
            'overall_completion_rate': completed_targets / total_targets if total_targets > 0 else 0,
            'completion_rates_by_target_type': completion_rates_by_type,
            'timing_distribution': timing_distribution,
            'average_completion_rate_per_touch': np.mean([r.overall_statistics['overall_completion_rate'] for r in results])
        }
    
    def run_target_progression_tracking(self) -> Dict:
        """Execute complete target progression tracking analysis"""
        
        print("🎯 TARGET PROGRESSION TRACKER AGENT STARTING...")
        print("=" * 55)
        
        # Load pre-structure inventories
        inventories = self.load_pre_structure_inventories()
        
        # Track progression for each inventory
        all_tracking_results = []
        
        for i, inventory in enumerate(inventories):
            print(f"\n📊 Processing inventory {i+1}/{len(inventories)}")
            tracking_result = self.track_archaeological_touch_progression(inventory)
            all_tracking_results.append(tracking_result)
        
        # Export results
        export_file = self.export_tracking_results(all_tracking_results)
        
        return {
            'agent_role': 'Target Progression Tracker Agent',
            'tracking_results_created': len(all_tracking_results),
            'export_file': export_file,
            'total_targets_tracked': sum(len(r.target_completions) for r in all_tracking_results),
            'ready_for_statistical_analysis': True
        }

def main():
    """Execute target progression tracking analysis"""
    
    print("🎯 TARGET PROGRESSION TRACKER AGENT")
    print("=" * 45)
    
    agent = TargetProgressionTrackerAgent(
        shard_data_path="/Users/jack/IRONFORGE/data/shards/NQ_M5",
        pre_structure_inventories_file="/Users/jack/IRONFORGE/pre_structure_inventories/latest.json"
    )
    
    results = agent.run_target_progression_tracking()
    
    # Display results
    print(f"\n🎯 TARGET PROGRESSION TRACKING COMPLETE:")
    print(f"   📊 Tracking Results: {results['tracking_results_created']}")
    print(f"   🎯 Total Targets Tracked: {results['total_targets_tracked']}")
    print(f"   📁 Export File: {results['export_file']}")
    print(f"   ✅ Ready for Statistical Analysis: {results['ready_for_statistical_analysis']}")
    
    return results

if __name__ == "__main__":
    main()