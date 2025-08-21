#!/usr/bin/env python3
"""
Experiment F: Gauntlet-Archaeological Integration (Minimal Spec)
Drop-in implementation focusing on Theory B 40% zones with proper statistical controls
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExperimentFConfig:
    """Configuration for Experiment F implementation"""
    
    # Time windows (±5m halo)
    time_windows = {
        'w_open_935_1000': {'start': time(9, 35), 'end': time(10, 0), 'halo_min': 5},
        'w_macro_50_10': {'pattern': ':50-:10', 'halo_min': 5},  # XX:50-XX:10 each hour
        'w_half_20_40': {'pattern': ':20-:40', 'halo_min': 5},   # XX:20-XX:40 each hour
        'w_close_350_410': {'start': time(15, 50), 'end': time(16, 10), 'halo_min': 5}
    }
    
    # Gauntlet parameters
    gauntlet_config = {
        'enabled': True,
        'timeout_min': 12,
        'tp1_handles': 10,
        'tp2_handles': 25,
        'sl_pad_ticks': 2,
        'displacement_atr_multiplier': 1.5
    }
    
    # Post-open inversion FVG parameters
    post_open_inv_config = {
        'enabled': True,
        'timeout_min': 12,
        'tp1_handles': 10,
        'tp2_handles': 25
    }
    
    # Theory B integration (40% zones only)
    theory_b_config = {
        'precision_threshold': 7.55,  # Points
        'zone_percentage': 0.40,      # Only 40% zones
        'confidence_multiplier': 2.0
    }
    
    # News hazard zones (static until calendar fixed)
    news_hazards = {
        'static_buckets_et': [
            {'start': time(8, 25), 'end': time(8, 35)},   # NFP window
            {'start': time(9, 55), 'end': time(10, 5)}    # ISM window
        ]
    }

class ExperimentFProcessor:
    """
    Minimal Gauntlet + Theory B 40% zone processor
    Focuses on AM sessions with proper statistical controls
    """
    
    def __init__(self, config: ExperimentFConfig):
        self.config = config
        self.session_cache = {}
        
    def load_session_data(self, session_path: Path) -> Optional[pd.DataFrame]:
        """Load session data with proper ET timezone handling"""
        try:
            nodes_file = session_path / "nodes.parquet"
            if not nodes_file.exists():
                return None
                
            df = pd.read_parquet(nodes_file)
            
            # Convert timestamps to ET
            df['utc_ts'] = pd.to_datetime(df['t'], unit='ms')
            df['et_ts'] = df['utc_ts'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            df['hour'] = df['et_ts'].dt.hour
            df['minute'] = df['et_ts'].dt.minute
            df['minute_index'] = df['hour'] * 60 + df['minute']
            
            # Extract session info
            session_name = session_path.name.replace('shard_', '')
            session_parts = session_name.split('_')
            df['session_id'] = session_name
            df['session_type'] = session_parts[0] if len(session_parts) > 0 else 'unknown'
            df['session_date'] = session_parts[1] if len(session_parts) > 1 else 'unknown'
            
            return df
            
        except Exception as e:
            print(f"Error loading {session_path}: {e}")
            return None
    
    def calculate_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time window flags with ±5m halo"""
        df = df.copy()
        
        # Initialize window flags
        for window_name in self.config.time_windows.keys():
            df[window_name] = False
        
        # Calculate specific windows
        for _, row in df.iterrows():
            et_time = row['et_ts'].time()
            hour = row['hour']
            minute = row['minute']
            
            # w_open_935_1000: 09:35-10:00 ET
            if time(9, 30) <= et_time <= time(10, 5):  # ±5m halo
                df.loc[row.name, 'w_open_935_1000'] = True
            
            # w_macro_50_10: XX:50-XX:10 each hour (±5m halo)
            if (45 <= minute <= 59) or (0 <= minute <= 15):
                df.loc[row.name, 'w_macro_50_10'] = True
            
            # w_half_20_40: XX:20-XX:40 each hour (±5m halo)
            if 15 <= minute <= 45:
                df.loc[row.name, 'w_half_20_40'] = True
            
            # w_close_350_410: 15:50-16:10 ET
            if time(15, 45) <= et_time <= time(16, 15):  # ±5m halo
                df.loc[row.name, 'w_close_350_410'] = True
        
        return df
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect session-aware liquidity sweeps"""
        df = df.copy()
        
        # Initialize sweep flags
        df['swept_session_low'] = False
        df['swept_session_high'] = False
        df['sweep_ts'] = pd.NaT
        
        if len(df) < 5:
            return df
        
        # Calculate session extremes
        session_high = df['price'].max()
        session_low = df['price'].min()
        
        # Detect sweeps (simplified)
        for i in range(5, len(df)):
            current_price = df.iloc[i]['price']
            recent_low = df.iloc[i-5:i]['price'].min()
            recent_close = df.iloc[i]['price']
            
            # Long sweep: swept below session low then recovered
            if (recent_low < session_low * 0.9999 and  # Swept below (with small buffer)
                recent_close > session_low):              # Recovered above
                df.iloc[i, df.columns.get_loc('swept_session_low')] = True
                df.iloc[i, df.columns.get_loc('sweep_ts')] = df.iloc[i]['et_ts']
        
        return df
    
    def detect_fvgs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps (3-bar pattern)"""
        df = df.copy()
        
        # Initialize FVG columns
        df['fvg_id'] = None
        df['fvg_side'] = None
        df['fvg_birth_ts'] = pd.NaT
        df['fvg_upper'] = np.nan
        df['fvg_lower'] = np.nan
        df['fvg_ce'] = np.nan
        df['is_first_in_session'] = False
        df['is_post_open_inv'] = False
        
        if len(df) < 3:
            return df
        
        # Assume OHLC from price (simplified for single price series)
        df['high'] = df['price']
        df['low'] = df['price']
        
        fvg_counter = 0
        first_bull_fvg_found = False
        
        for i in range(2, len(df)):
            # Bull FVG: low[t] > high[t-2]
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                fvg_counter += 1
                upper = df.iloc[i]['low']
                lower = df.iloc[i-2]['high']
                ce = (upper + lower) / 2
                
                df.iloc[i, df.columns.get_loc('fvg_id')] = f"fvg_{fvg_counter}"
                df.iloc[i, df.columns.get_loc('fvg_side')] = 'bull'
                df.iloc[i, df.columns.get_loc('fvg_birth_ts')] = df.iloc[i]['et_ts']
                df.iloc[i, df.columns.get_loc('fvg_upper')] = upper
                df.iloc[i, df.columns.get_loc('fvg_lower')] = lower
                df.iloc[i, df.columns.get_loc('fvg_ce')] = ce
                
                # Mark first bull FVG in session
                if not first_bull_fvg_found:
                    df.iloc[i, df.columns.get_loc('is_first_in_session')] = True
                    first_bull_fvg_found = True
                    
                    # Check if this is post-open inversion FVG
                    if df.iloc[i]['et_ts'].time() > time(9, 30):
                        # Check for recent sweep
                        recent_sweep = df.iloc[max(0, i-12):i]['swept_session_low'].any()  # Last 12 bars
                        if recent_sweep:
                            df.iloc[i, df.columns.get_loc('is_post_open_inv')] = True
        
        return df
    
    def detect_gauntlet_setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Gauntlet FVG and CE reclaim setups"""
        df = df.copy()
        
        # Initialize Gauntlet columns
        df['breaker_run_id'] = None
        df['gauntlet_fvg_id'] = None
        df['gauntlet_ce'] = np.nan
        df['gauntlet_reclaim_long'] = False
        df['trigger_ts'] = pd.NaT
        
        # Simplified Gauntlet detection
        for i in range(len(df)):
            if df.iloc[i]['swept_session_low'] and not pd.isna(df.iloc[i]['fvg_ce']):
                # This is a potential Gauntlet setup
                df.iloc[i, df.columns.get_loc('gauntlet_fvg_id')] = df.iloc[i]['fvg_id']
                df.iloc[i, df.columns.get_loc('gauntlet_ce')] = df.iloc[i]['fvg_ce']
                
                # Check for CE reclaim in next few bars
                for j in range(i+1, min(i+6, len(df))):
                    if df.iloc[j]['price'] >= df.iloc[i]['fvg_ce']:
                        df.iloc[j, df.columns.get_loc('gauntlet_reclaim_long')] = True
                        df.iloc[j, df.columns.get_loc('trigger_ts')] = df.iloc[j]['et_ts']
                        break
        
        return df
    
    def calculate_theory_b_zones(self, df: pd.DataFrame, prior_session_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate Theory B 40% zones from prior session completion"""
        df = df.copy()
        
        # Initialize Theory B columns
        df['theory_b_40_level'] = np.nan
        df['theory_b_40_precision'] = np.nan
        df['theory_b_40_active'] = False
        
        if prior_session_data is None or len(prior_session_data) == 0:
            return df
        
        # Calculate 40% zone from prior session
        prior_high = prior_session_data['price'].max()
        prior_low = prior_session_data['price'].min()
        prior_range = prior_high - prior_low
        
        if prior_range <= 0:
            return df
        
        # 40% zone level
        zone_40_level = prior_low + (prior_range * self.config.theory_b_config['zone_percentage'])
        
        # Add zone info to all rows
        df['theory_b_40_level'] = zone_40_level
        
        # Calculate precision for each price point
        df['theory_b_40_precision'] = abs(df['price'] - zone_40_level)
        
        # Mark active when within Theory B threshold
        precision_threshold = self.config.theory_b_config['precision_threshold']
        df['theory_b_40_active'] = df['theory_b_40_precision'] <= precision_threshold
        
        return df
    
    def generate_motif_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate gauntlet_v1_long and post_open_inv_fvg_v1_long signals"""
        df = df.copy()
        
        # Initialize signal columns
        df['gauntlet_v1_long_signal'] = False
        df['post_open_inv_fvg_v1_long_signal'] = False
        df['entry_price'] = np.nan
        df['confidence_multiplier'] = 1.0
        
        # News hazard check
        df['news_hazard_830_1000'] = False
        for _, row in df.iterrows():
            et_time = row['et_ts'].time()
            for hazard in self.config.news_hazards['static_buckets_et']:
                if hazard['start'] <= et_time <= hazard['end']:
                    df.loc[row.name, 'news_hazard_830_1000'] = True
        
        # Gauntlet V1 Long signals
        gauntlet_conditions = (
            (df['swept_session_low'] == True) &
            (df['gauntlet_reclaim_long'] == True) &
            ((df['w_open_935_1000'] == True) | (df['w_macro_50_10'] == True)) &
            (df['news_hazard_830_1000'] == False)
        )
        
        df.loc[gauntlet_conditions, 'gauntlet_v1_long_signal'] = True
        df.loc[gauntlet_conditions, 'entry_price'] = df.loc[gauntlet_conditions, 'gauntlet_ce']
        
        # Theory B confidence multiplier
        theory_b_confluence = gauntlet_conditions & (df['theory_b_40_active'] == True)
        df.loc[theory_b_confluence, 'confidence_multiplier'] = self.config.theory_b_config['confidence_multiplier']
        
        # Post-open inversion FVG signals
        post_open_conditions = (
            (df['is_post_open_inv'] == True) &
            (df['w_open_935_1000'] == True) &
            (df['news_hazard_830_1000'] == False)
        )
        
        df.loc[post_open_conditions, 'post_open_inv_fvg_v1_long_signal'] = True
        df.loc[post_open_conditions, 'entry_price'] = df.loc[post_open_conditions, 'fvg_ce']
        
        return df
    
    def process_session(self, session_path: Path, prior_session_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Process a single session through the complete Experiment F pipeline"""
        
        # Load session data
        df = self.load_session_data(session_path)
        if df is None:
            return None
        
        print(f"Processing session: {session_path.name}")
        
        # Process through pipeline
        df = self.calculate_time_windows(df)
        df = self.detect_liquidity_sweeps(df)
        df = self.detect_fvgs(df)
        df = self.detect_gauntlet_setup(df)
        df = self.calculate_theory_b_zones(df, prior_session_data)
        df = self.generate_motif_signals(df)
        
        return df
    
    def run_experiment_f(self, data_dir: str = "/Users/jack/IRONFORGE/data/shards/NQ_M5") -> Dict[str, Any]:
        """Run Experiment F on available AM sessions"""
        
        data_path = Path(data_dir)
        session_dirs = sorted([d for d in data_path.glob("shard_*") if d.is_dir()])
        
        # Focus on AM sessions (NYAM, PREMARKET)
        am_sessions = [d for d in session_dirs if any(am_type in d.name for am_type in ['NYAM', 'PREMARKET', 'NY_'])]
        
        print(f"Found {len(am_sessions)} AM sessions for Experiment F")
        
        results = {
            'processed_sessions': [],
            'gauntlet_signals': 0,
            'post_open_signals': 0,
            'theory_b_confluences': 0,
            'am_session_count': len(am_sessions),
            'session_summaries': []
        }
        
        prior_session = None
        
        for session_dir in am_sessions[:10]:  # Limit for testing
            try:
                processed_df = self.process_session(session_dir, prior_session)
                
                if processed_df is not None:
                    # Count signals
                    gauntlet_count = processed_df['gauntlet_v1_long_signal'].sum()
                    post_open_count = processed_df['post_open_inv_fvg_v1_long_signal'].sum()
                    theory_b_count = (processed_df['confidence_multiplier'] > 1.0).sum()
                    
                    results['gauntlet_signals'] += gauntlet_count
                    results['post_open_signals'] += post_open_count
                    results['theory_b_confluences'] += theory_b_count
                    
                    session_summary = {
                        'session_name': session_dir.name,
                        'gauntlet_signals': int(gauntlet_count),
                        'post_open_signals': int(post_open_count),
                        'theory_b_confluences': int(theory_b_count),
                        'total_bars': len(processed_df),
                        'theory_b_zones_mapped': not processed_df['theory_b_40_level'].isna().all()
                    }
                    
                    results['session_summaries'].append(session_summary)
                    results['processed_sessions'].append(processed_df)
                    
                    # Use this session as prior for next iteration
                    prior_session = processed_df
                    
                    print(f"  {session_dir.name}: {gauntlet_count} gauntlet, {post_open_count} post-open, {theory_b_count} theory-b")
                
            except Exception as e:
                print(f"  Error processing {session_dir.name}: {e}")
        
        return results

def main():
    """Run Experiment F demonstration"""
    print("🎯 Experiment F: Gauntlet-Archaeological Integration (Minimal)")
    print("=" * 65)
    
    # Initialize configuration
    config = ExperimentFConfig()
    processor = ExperimentFProcessor(config)
    
    # Run experiment
    results = processor.run_experiment_f()
    
    # Display results
    print(f"\n📊 EXPERIMENT F RESULTS:")
    print(f"   AM sessions processed: {len(results['processed_sessions'])}/{results['am_session_count']}")
    print(f"   Gauntlet signals: {results['gauntlet_signals']}")
    print(f"   Post-open inversion signals: {results['post_open_signals']}")
    print(f"   Theory B confluences: {results['theory_b_confluences']}")
    
    if results['theory_b_confluences'] > 0:
        confluence_rate = (results['theory_b_confluences'] / max(results['gauntlet_signals'], 1)) * 100
        print(f"   Theory B confluence rate: {confluence_rate:.1f}%")
    
    print(f"\n📋 SESSION BREAKDOWN:")
    for summary in results['session_summaries']:
        print(f"   {summary['session_name']}: G={summary['gauntlet_signals']}, P={summary['post_open_signals']}, TB={summary['theory_b_confluences']}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"experiment_f_results_{timestamp}.json"
    
    # Convert DataFrames to dict for JSON serialization
    json_results = {k: v for k, v in results.items() if k != 'processed_sessions'}
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()