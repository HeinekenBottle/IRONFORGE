#!/usr/bin/env python3
"""
Experiment F Adapted: Realistic Gauntlet Detection for IRONFORGE Data
Adapted for actual market microstructure with proper sweep and FVG detection
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
class AdaptedConfig:
    """Adapted configuration for realistic Gauntlet detection"""
    
    # Dynamic volatility-based thresholds
    sweep_detection = {
        'lookback_bars': 20,           # Look for sweeps of recent 20-bar lows
        'volatility_multiplier': 0.3,  # Sweep threshold as % of recent range
        'recovery_bars': 5,            # Must recover within 5 bars
        'min_sweep_points': 2.0        # Minimum sweep distance in points
    }
    
    # Displacement and FVG parameters
    displacement_detection = {
        'atr_multiplier': 1.5,         # Displacement > 1.5x recent ATR
        'min_displacement_points': 15,  # Minimum displacement in points
        'displacement_bars': 3         # Must occur within 3 bars
    }
    
    # FVG detection (using synthetic OHLC)
    fvg_detection = {
        'min_gap_points': 5,           # Minimum gap size in points
        'window_size': 3,              # Rolling window for synthetic OHLC
        'gap_persistence_bars': 2      # Gap must persist for at least 2 bars
    }
    
    # Time windows (adapted for real market hours)
    time_windows = {
        'am_core': {'start': time(9, 30), 'end': time(11, 0)},
        'am_open': {'start': time(9, 30), 'end': time(10, 0)},
        'macro_9_50': {'start': time(9, 50), 'end': time(10, 10)},
        'macro_10_50': {'start': time(10, 50), 'end': time(11, 10)},
        'premarket': {'start': time(8, 0), 'end': time(9, 30)}
    }
    
    # Theory B (40% only, dynamic thresholds)
    theory_b = {
        'zone_percentage': 0.40,
        'base_precision_threshold': 7.55,
        'volatility_adjustment': True,  # Adjust threshold based on session volatility
        'max_precision_threshold': 15.0  # Maximum threshold for high volatility sessions
    }

class AdaptedGauntletProcessor:
    """
    Adapted processor for realistic Gauntlet detection in IRONFORGE data
    Focuses on actual market microstructure patterns
    """
    
    def __init__(self, config: AdaptedConfig):
        self.config = config
        
    def load_and_prepare_session(self, session_path: Path) -> Optional[pd.DataFrame]:
        """Load session data and create synthetic OHLC structure"""
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
            
            # Sort by timestamp
            df = df.sort_values('et_ts').reset_index(drop=True)
            
            # Create synthetic OHLC from price points using rolling windows
            window_size = self.config.fvg_detection['window_size']
            
            df['open'] = df['price']  # Use current price as open
            df['high'] = df['price'].rolling(window=window_size, min_periods=1).max()
            df['low'] = df['price'].rolling(window=window_size, min_periods=1).min()
            df['close'] = df['price']  # Use current price as close
            
            # Calculate additional metrics
            df['bar_range'] = df['high'] - df['low']
            df['atr'] = df['bar_range'].rolling(window=14, min_periods=1).mean()
            
            # Extract session info
            session_name = session_path.name.replace('shard_', '')
            df['session_id'] = session_name
            df['session_type'] = session_name.split('_')[0]
            df['session_date'] = session_name.split('_')[1] if '_' in session_name else 'unknown'
            
            return df
            
        except Exception as e:
            print(f"Error loading {session_path}: {e}")
            return None
    
    def filter_am_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to actual AM trading hours with data"""
        if df is None or len(df) == 0:
            return df
            
        # Check if session actually contains AM hours
        am_start = time(9, 30)
        am_end = time(11, 0)
        
        am_mask = (df['et_ts'].dt.time >= am_start) & (df['et_ts'].dt.time <= am_end)
        am_data = df[am_mask].copy()
        
        if len(am_data) < 5:  # Need minimum bars for analysis
            return pd.DataFrame()  # Return empty if insufficient AM data
            
        return am_data
    
    def calculate_time_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time window flags for AM hours"""
        df = df.copy()
        
        # Initialize all window flags
        for window_name in self.config.time_windows.keys():
            df[f'w_{window_name}'] = False
        
        for idx, row in df.iterrows():
            et_time = row['et_ts'].time()
            
            # Check each time window
            for window_name, window_config in self.config.time_windows.items():
                if window_config['start'] <= et_time <= window_config['end']:
                    df.loc[idx, f'w_{window_name}'] = True
        
        return df
    
    def detect_swing_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect swing lows using rolling windows"""
        df = df.copy()
        
        lookback = self.config.sweep_detection['lookback_bars']
        
        # Calculate rolling lows
        df['swing_low_level'] = df['low'].rolling(window=lookback, min_periods=5).min()
        df['is_swing_low'] = df['low'] == df['swing_low_level']
        
        return df
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect realistic micro-sweeps of swing lows"""
        df = df.copy()
        
        # Initialize sweep columns
        df['swept_swing_low'] = False
        df['sweep_level'] = np.nan
        df['sweep_ts'] = pd.NaT
        df['sweep_recovery'] = False
        
        if len(df) < 10:
            return df
        
        # More aggressive detection for high volatility data
        for i in range(10, len(df) - 3):  # Reduced lookback, recovery window
            # Get recent 10-bar low (more responsive)
            recent_data = df.iloc[i-10:i]
            if len(recent_data) == 0:
                continue
                
            swing_low = recent_data['low'].min()
            current_price = df.iloc[i]['close']  # Use close instead of low
            
            # More lenient sweep detection: any break below recent low
            if current_price < swing_low:
                # Check for recovery within next 3 bars
                recovery_data = df.iloc[i:i+4]
                if len(recovery_data) > 0:
                    recovery_price = recovery_data['close'].max()
                    
                    # Recovery: price moves back above the swing low
                    if recovery_price > swing_low:
                        # Mark the sweep
                        df.iloc[i, df.columns.get_loc('swept_swing_low')] = True
                        df.iloc[i, df.columns.get_loc('sweep_level')] = swing_low
                        df.iloc[i, df.columns.get_loc('sweep_ts')] = df.iloc[i]['et_ts']
                        
                        # Mark recovery point
                        recovery_indices = recovery_data[recovery_data['close'] > swing_low].index
                        if len(recovery_indices) > 0:
                            recovery_idx = recovery_indices[0]
                            df.iloc[recovery_idx, df.columns.get_loc('sweep_recovery')] = True
        
        return df
    
    def detect_displacement_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect displacement moves following sweeps"""
        df = df.copy()
        
        # Initialize displacement columns
        df['displacement_bar'] = False
        df['displacement_size'] = 0.0
        
        if len(df) < 5:
            return df
        
        atr_mult = self.config.displacement_detection['atr_multiplier']
        min_disp = self.config.displacement_detection['min_displacement_points']
        disp_bars = self.config.displacement_detection['displacement_bars']
        
        for i in range(len(df) - disp_bars):
            if df.iloc[i]['swept_swing_low']:
                # Look for displacement in next few bars
                for j in range(i+1, min(i+disp_bars+1, len(df))):
                    current_atr = df.iloc[j]['atr']
                    bar_size = df.iloc[j]['high'] - df.iloc[j]['low']
                    
                    # Check for displacement (large bar relative to ATR)
                    atr_threshold = current_atr * atr_mult
                    displacement_threshold = max(min_disp, atr_threshold)
                    
                    if bar_size >= displacement_threshold:
                        df.iloc[j, df.columns.get_loc('displacement_bar')] = True
                        df.iloc[j, df.columns.get_loc('displacement_size')] = bar_size
                        break
        
        return df
    
    def detect_fvgs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Fair Value Gaps using synthetic OHLC"""
        df = df.copy()
        
        # Initialize FVG columns
        df['fvg_id'] = None
        df['fvg_type'] = None
        df['fvg_upper'] = np.nan
        df['fvg_lower'] = np.nan
        df['fvg_ce'] = np.nan
        df['is_post_sweep_fvg'] = False
        
        if len(df) < 3:
            return df
        
        min_gap = self.config.fvg_detection['min_gap_points']
        fvg_counter = 0
        
        for i in range(2, len(df)):
            # Bullish FVG: low[i] > high[i-2] (gap up)
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                
                if gap_size >= min_gap:
                    fvg_counter += 1
                    upper = df.iloc[i]['low']
                    lower = df.iloc[i-2]['high']
                    ce = (upper + lower) / 2
                    
                    df.iloc[i, df.columns.get_loc('fvg_id')] = f"fvg_{fvg_counter}"
                    df.iloc[i, df.columns.get_loc('fvg_type')] = 'bullish'
                    df.iloc[i, df.columns.get_loc('fvg_upper')] = upper
                    df.iloc[i, df.columns.get_loc('fvg_lower')] = lower
                    df.iloc[i, df.columns.get_loc('fvg_ce')] = ce
                    
                    # Check if this FVG formed after a recent sweep
                    recent_sweep = df.iloc[max(0, i-10):i]['swept_swing_low'].any()
                    if recent_sweep:
                        df.iloc[i, df.columns.get_loc('is_post_sweep_fvg')] = True
        
        return df
    
    def detect_gauntlet_setups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect complete Gauntlet setups: sweep → displacement → FVG → CE reclaim"""
        df = df.copy()
        
        # Initialize Gauntlet columns
        df['gauntlet_setup'] = False
        df['gauntlet_fvg_id'] = None
        df['gauntlet_ce'] = np.nan
        df['gauntlet_entry_signal'] = False
        df['gauntlet_entry_price'] = np.nan
        
        if len(df) < 10:
            return df
        
        # Look for complete sequence
        for i in range(len(df)):
            if (df.iloc[i]['is_post_sweep_fvg'] and 
                not pd.isna(df.iloc[i]['fvg_ce']) and
                df.iloc[i]['fvg_type'] == 'bullish'):
                
                # This is a potential Gauntlet FVG
                df.iloc[i, df.columns.get_loc('gauntlet_setup')] = True
                df.iloc[i, df.columns.get_loc('gauntlet_fvg_id')] = df.iloc[i]['fvg_id']
                df.iloc[i, df.columns.get_loc('gauntlet_ce')] = df.iloc[i]['fvg_ce']
                
                # Look for CE reclaim in subsequent bars
                ce_level = df.iloc[i]['fvg_ce']
                for j in range(i+1, min(i+8, len(df))):
                    if df.iloc[j]['close'] >= ce_level:
                        # CE reclaimed - this is the entry signal
                        df.iloc[j, df.columns.get_loc('gauntlet_entry_signal')] = True
                        df.iloc[j, df.columns.get_loc('gauntlet_entry_price')] = ce_level
                        break
        
        return df
    
    def calculate_theory_b_zones(self, df: pd.DataFrame, prior_session_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate dynamic Theory B 40% zones"""
        df = df.copy()
        
        # Initialize Theory B columns
        df['theory_b_40_level'] = np.nan
        df['theory_b_precision'] = np.nan
        df['theory_b_active'] = False
        df['theory_b_threshold'] = self.config.theory_b['base_precision_threshold']
        
        if prior_session_data is None or len(prior_session_data) == 0:
            return df
        
        # Calculate 40% zone from prior session
        prior_high = prior_session_data['high'].max()
        prior_low = prior_session_data['low'].min()
        prior_range = prior_high - prior_low
        
        if prior_range <= 0:
            return df
        
        # Calculate zone level
        zone_40_level = prior_low + (prior_range * self.config.theory_b['zone_percentage'])
        
        # Dynamic precision threshold based on current session volatility
        if self.config.theory_b['volatility_adjustment'] and len(df) > 0:
            current_atr = df['atr'].mean()
            volatility_factor = min(2.0, current_atr / 20.0)  # Scale based on ATR
            dynamic_threshold = self.config.theory_b['base_precision_threshold'] * volatility_factor
            dynamic_threshold = min(dynamic_threshold, self.config.theory_b['max_precision_threshold'])
        else:
            dynamic_threshold = self.config.theory_b['base_precision_threshold']
        
        # Apply to all rows
        df['theory_b_40_level'] = zone_40_level
        df['theory_b_threshold'] = dynamic_threshold
        df['theory_b_precision'] = abs(df['close'] - zone_40_level)
        df['theory_b_active'] = df['theory_b_precision'] <= dynamic_threshold
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate final trading signals with Theory B confluence"""
        df = df.copy()
        
        # Initialize signal columns
        df['gauntlet_long_signal'] = False
        df['signal_confidence'] = 1.0
        df['signal_notes'] = ''
        
        # Gauntlet long signals: entry signal + AM time window
        gauntlet_conditions = (
            df['gauntlet_entry_signal'] &
            (df['w_am_core'] | df['w_macro_9_50'] | df['w_macro_10_50'])
        )
        
        df.loc[gauntlet_conditions, 'gauntlet_long_signal'] = True
        
        # Theory B confluence (40% zone proximity)
        theory_b_confluence = gauntlet_conditions & df['theory_b_active']
        df.loc[theory_b_confluence, 'signal_confidence'] = 2.0
        df.loc[theory_b_confluence, 'signal_notes'] = 'Theory B 40% confluence'
        
        # Standard signals without Theory B
        standard_signals = gauntlet_conditions & ~df['theory_b_active']
        df.loc[standard_signals, 'signal_notes'] = 'Standard Gauntlet setup'
        
        return df
    
    def process_session(self, session_path: Path, prior_session_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Process complete session through adapted pipeline"""
        
        # Load and prepare data
        df = self.load_and_prepare_session(session_path)
        if df is None:
            return None
        
        # Filter to AM hours
        am_df = self.filter_am_hours(df)
        if len(am_df) == 0:
            return {'session_name': session_path.name, 'am_data': False, 'signals': 0}
        
        print(f"Processing AM session: {session_path.name} ({len(am_df)} bars)")
        
        # Process through pipeline
        am_df = self.calculate_time_windows(am_df)
        am_df = self.detect_swing_lows(am_df)
        am_df = self.detect_liquidity_sweeps(am_df)
        am_df = self.detect_displacement_moves(am_df)
        am_df = self.detect_fvgs(am_df)
        am_df = self.detect_gauntlet_setups(am_df)
        am_df = self.calculate_theory_b_zones(am_df, prior_session_data)
        am_df = self.generate_trading_signals(am_df)
        
        # Calculate session statistics
        session_stats = {
            'session_name': session_path.name,
            'session_type': am_df['session_type'].iloc[0] if len(am_df) > 0 else 'unknown',
            'am_data': True,
            'total_bars': len(am_df),
            'time_range': f"{am_df['et_ts'].min().strftime('%H:%M')}-{am_df['et_ts'].max().strftime('%H:%M')}",
            'price_range': am_df['high'].max() - am_df['low'].min(),
            'sweeps_detected': am_df['swept_swing_low'].sum(),
            'fvgs_detected': am_df['fvg_id'].notna().sum(),
            'gauntlet_setups': am_df['gauntlet_setup'].sum(),
            'entry_signals': am_df['gauntlet_entry_signal'].sum(),
            'long_signals': am_df['gauntlet_long_signal'].sum(),
            'theory_b_confluences': (am_df['signal_confidence'] > 1.0).sum(),
            'theory_b_zones_available': not am_df['theory_b_40_level'].isna().all(),
            'avg_signal_confidence': am_df[am_df['gauntlet_long_signal']]['signal_confidence'].mean() if am_df['gauntlet_long_signal'].any() else 0,
            'processed_data': am_df
        }
        
        return session_stats

def run_adapted_experiment_f():
    """Run the adapted Experiment F on AM sessions"""
    print("🎯 Experiment F Adapted: Realistic Gauntlet Detection")
    print("=" * 55)
    
    config = AdaptedConfig()
    processor = AdaptedGauntletProcessor(config)
    
    # Load AM sessions
    data_dir = Path("/Users/jack/IRONFORGE/data/shards/NQ_M5")
    session_dirs = sorted([d for d in data_dir.glob("shard_*") if d.is_dir()])
    
    # Focus on sessions most likely to have AM data
    am_sessions = [d for d in session_dirs if any(am_type in d.name for am_type in ['NYAM', 'PREMARKET', 'NY_2025'])]
    
    print(f"Found {len(am_sessions)} potential AM sessions")
    
    results = {
        'sessions_processed': 0,
        'sessions_with_am_data': 0,
        'total_signals': 0,
        'total_theory_b_confluences': 0,
        'session_summaries': []
    }
    
    prior_session = None
    
    # Process sessions
    for session_dir in am_sessions[:15]:  # Limit for testing
        try:
            session_result = processor.process_session(session_dir, prior_session)
            
            if session_result:
                results['sessions_processed'] += 1
                
                if session_result['am_data']:
                    results['sessions_with_am_data'] += 1
                    results['total_signals'] += session_result.get('long_signals', 0)
                    results['total_theory_b_confluences'] += session_result.get('theory_b_confluences', 0)
                    
                    # Use processed data as prior session for next iteration
                    if 'processed_data' in session_result:
                        prior_session = session_result['processed_data']
                
                # Store summary (without full dataframe)
                summary = {k: v for k, v in session_result.items() if k != 'processed_data'}
                results['session_summaries'].append(summary)
                
                print(f"  {session_result['session_name']}: {session_result.get('long_signals', 0)} signals, {session_result.get('theory_b_confluences', 0)} TB confluences")
        
        except Exception as e:
            print(f"  Error processing {session_dir.name}: {e}")
    
    # Display results
    print(f"\n📊 ADAPTED EXPERIMENT F RESULTS:")
    print(f"   Sessions processed: {results['sessions_processed']}")
    print(f"   Sessions with AM data: {results['sessions_with_am_data']}")
    print(f"   Total Gauntlet signals: {results['total_signals']}")
    print(f"   Theory B confluences: {results['total_theory_b_confluences']}")
    
    if results['total_signals'] > 0:
        confluence_rate = (results['total_theory_b_confluences'] / results['total_signals']) * 100
        print(f"   Theory B confluence rate: {confluence_rate:.1f}%")
        signals_per_session = results['total_signals'] / max(results['sessions_with_am_data'], 1)
        print(f"   Average signals per AM session: {signals_per_session:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"experiment_f_adapted_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    run_adapted_experiment_f()