"""
M1 Sparse Event Layer
Detects micro-timeframe events from 1-minute OHLCV data
Creates sparse event nodes to supplement M5 bars for multi-scale analysis
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class M1Event:
    """M1 event data structure"""
    event_id: str
    session_id: str
    timestamp_ms: int
    parent_m5_seq_idx: int
    event_kind: str
    price: float
    volume: float
    features: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class M1EventConfig:
    """Configuration for M1 event detection"""
    # FVG detection parameters
    fvg_min_gap_pips: float = 2.0
    fvg_volume_threshold: float = 100.0
    
    # Sweep detection parameters
    sweep_wick_ratio: float = 0.3
    sweep_volume_multiplier: float = 1.5
    
    # Impulse detection parameters
    impulse_z_threshold: float = 2.0
    impulse_min_range_pips: float = 5.0
    
    # VWAP parameters
    vwap_touch_threshold_pips: float = 1.0
    vwap_period: int = 20
    
    # Imbalance detection
    imbalance_ratio_threshold: float = 2.0
    imbalance_min_volume: float = 50.0
    
    # Wick extreme detection
    wick_extreme_percentile: float = 95.0
    wick_min_ratio: float = 0.5
    
    # General parameters
    lookback_period: int = 20
    min_confidence: float = 0.3


class M1EventDetector:
    """
    M1 Event Detector for Sparse Event Layer
    
    Detects significant micro-timeframe events from M1 OHLCV data:
    - micro_fvg_fill: Fair Value Gap interactions at M1 level
    - micro_sweep: Liquidity sweeps and stop hunts
    - micro_impulse: Strong directional moves (z-score based)
    - vwap_touch: VWAP interaction events
    - imbalance_burst: Ask/bid imbalance spikes
    - wick_extreme: Exceptional wick formations
    """
    
    def __init__(self, config: M1EventConfig = None):
        self.config = config or M1EventConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"M1 Event Detector initialized: {self.config}")
        
    def detect_events(
        self, 
        m1_ohlcv: pd.DataFrame, 
        session_id: str,
        m5_bars: pd.DataFrame = None
    ) -> List[M1Event]:
        """
        Detect all M1 events from OHLCV data
        
        Args:
            m1_ohlcv: DataFrame with columns [timestamp, open, high, low, close, volume]
            session_id: Session identifier
            m5_bars: Optional M5 bars for cross-scale mapping
            
        Returns:
            List of detected M1Event objects
        """
        if m1_ohlcv.empty:
            return []
            
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in m1_ohlcv.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return []
            
        # Prepare data
        df = m1_ohlcv.copy().sort_values('timestamp')
        df = self._prepare_technical_indicators(df)
        
        # Create M5 mapping if provided
        m5_mapping = self._create_m5_mapping(df, m5_bars) if m5_bars is not None else {}
        
        all_events = []
        
        # Detect each event type
        event_detectors = [
            ('micro_fvg_fill', self._detect_micro_fvg_events),
            ('micro_sweep', self._detect_micro_sweep_events), 
            ('micro_impulse', self._detect_micro_impulse_events),
            ('vwap_touch', self._detect_vwap_touch_events),
            ('imbalance_burst', self._detect_imbalance_burst_events),
            ('wick_extreme', self._detect_wick_extreme_events)
        ]
        
        for event_type, detector_func in event_detectors:
            try:
                events = detector_func(df, session_id, m5_mapping)
                all_events.extend(events)
                self.logger.debug(f"Detected {len(events)} {event_type} events")
            except Exception as e:
                self.logger.warning(f"Failed to detect {event_type} events: {e}")
                
        # Filter by confidence and deduplicate
        all_events = [e for e in all_events if e.confidence >= self.config.min_confidence]
        all_events = self._deduplicate_events(all_events)
        
        self.logger.info(f"Detected {len(all_events)} total M1 events for session {session_id}")
        
        return all_events
        
    def _prepare_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for event detection"""
        df = df.copy()
        
        # Price-based indicators
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['range_pips'] = (df['high'] - df['low']) * 10000  # Assuming 4-decimal pricing
        df['body_pips'] = abs(df['close'] - df['open']) * 10000
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 1e-8)
        
        # VWAP calculation
        df['vwap'] = (df['typical_price'] * df['volume']).rolling(self.config.vwap_period).sum() / \
                     df['volume'].rolling(self.config.vwap_period).sum()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(self.config.lookback_period).mean()
        df['volume_std'] = df['volume'].rolling(self.config.lookback_period).std()
        df['volume_z'] = (df['volume'] - df['volume_ma']) / (df['volume_std'] + 1e-8)
        
        # Price change indicators  
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open'] * 100
        df['range_z'] = (df['range_pips'] - df['range_pips'].rolling(self.config.lookback_period).mean()) / \
                        (df['range_pips'].rolling(self.config.lookback_period).std() + 1e-8)
        
        # Momentum indicators
        df['momentum_3'] = df['close'] - df['close'].shift(3)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        
        # Bid/Ask proxy (using high/low as approximation)
        df['ask_proxy'] = df['high']
        df['bid_proxy'] = df['low']
        df['spread_proxy'] = df['ask_proxy'] - df['bid_proxy']
        df['imbalance_proxy'] = (df['ask_proxy'] - df['bid_proxy']) / (df['ask_proxy'] + df['bid_proxy'] + 1e-8)
        
        return df
        
    def _create_m5_mapping(self, m1_df: pd.DataFrame, m5_bars: pd.DataFrame) -> Dict[int, int]:
        """Create mapping from M1 timestamps to M5 bar indices"""
        if m5_bars.empty:
            return {}
            
        mapping = {}
        
        for i, row in m1_df.iterrows():
            m1_timestamp = row['timestamp']
            
            # Find which M5 bar contains this M1 bar
            for j, m5_row in m5_bars.iterrows():
                m5_start = m5_row['timestamp']
                m5_end = m5_start + 300000  # 5 minutes in milliseconds
                
                if m5_start <= m1_timestamp < m5_end:
                    mapping[i] = j
                    break
                    
        return mapping
        
    def _detect_micro_fvg_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect micro Fair Value Gap events"""
        events = []
        
        for i in range(2, len(df) - 1):
            # Look for 3-bar FVG pattern
            bar1 = df.iloc[i-1]
            bar2 = df.iloc[i]    # Gap bar
            bar3 = df.iloc[i+1]
            
            # Check for bullish FVG (gap up)
            if bar1['high'] < bar3['low']:
                gap_size = (bar3['low'] - bar1['high']) * 10000
                
                if gap_size >= self.config.fvg_min_gap_pips:
                    # Check if next bar fills the gap
                    fill_bar = df.iloc[i+2] if i+2 < len(df) else None
                    is_filled = fill_bar is not None and fill_bar['low'] <= bar1['high']
                    
                    confidence = min(1.0, gap_size / 10.0)  # Larger gaps = higher confidence
                    
                    event = M1Event(
                        event_id=f"{session_id}_fvg_{i}",
                        session_id=session_id,
                        timestamp_ms=int(bar2['timestamp']),
                        parent_m5_seq_idx=m5_mapping.get(i, -1),
                        event_kind='micro_fvg_fill' if is_filled else 'micro_fvg_formation',
                        price=(bar1['high'] + bar3['low']) / 2,  # Mid-gap price
                        volume=bar2['volume'],
                        features={
                            'gap_size_pips': gap_size,
                            'gap_direction': 1.0,  # Bullish
                            'fill_speed': 1.0 if is_filled else 0.0,
                            'volume_strength': bar2['volume_z'] if 'volume_z' in df.columns else 0.0
                        },
                        confidence=confidence,
                        metadata={'pattern_type': 'bullish_fvg', 'bars_used': 3}
                    )
                    events.append(event)
                    
            # Check for bearish FVG (gap down)
            elif bar1['low'] > bar3['high']:
                gap_size = (bar1['low'] - bar3['high']) * 10000
                
                if gap_size >= self.config.fvg_min_gap_pips:
                    # Check if next bar fills the gap
                    fill_bar = df.iloc[i+2] if i+2 < len(df) else None
                    is_filled = fill_bar is not None and fill_bar['high'] >= bar1['low']
                    
                    confidence = min(1.0, gap_size / 10.0)
                    
                    event = M1Event(
                        event_id=f"{session_id}_fvg_{i}",
                        session_id=session_id,
                        timestamp_ms=int(bar2['timestamp']),
                        parent_m5_seq_idx=m5_mapping.get(i, -1),
                        event_kind='micro_fvg_fill' if is_filled else 'micro_fvg_formation',
                        price=(bar1['low'] + bar3['high']) / 2,  # Mid-gap price
                        volume=bar2['volume'],
                        features={
                            'gap_size_pips': gap_size,
                            'gap_direction': -1.0,  # Bearish
                            'fill_speed': 1.0 if is_filled else 0.0,
                            'volume_strength': bar2['volume_z'] if 'volume_z' in df.columns else 0.0
                        },
                        confidence=confidence,
                        metadata={'pattern_type': 'bearish_fvg', 'bars_used': 3}
                    )
                    events.append(event)
                    
        return events
        
    def _detect_micro_sweep_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect micro liquidity sweep events"""
        events = []
        
        for i in range(self.config.lookback_period, len(df)):
            current_bar = df.iloc[i]
            
            # Look for wick spikes indicating liquidity sweeps
            if current_bar['wick_ratio'] >= self.config.sweep_wick_ratio:
                # Check volume confirmation
                volume_multiplier = current_bar['volume'] / current_bar['volume_ma'] if current_bar['volume_ma'] > 0 else 1.0
                
                if volume_multiplier >= self.config.sweep_volume_multiplier:
                    # Determine sweep direction based on which wick is larger
                    upper_wick_size = current_bar['upper_wick'] * 10000
                    lower_wick_size = current_bar['lower_wick'] * 10000
                    
                    if upper_wick_size > lower_wick_size:
                        # Upper wick sweep (sell-side liquidity)
                        sweep_direction = 1.0
                        swept_price = current_bar['high']
                        wick_size = upper_wick_size
                    else:
                        # Lower wick sweep (buy-side liquidity)
                        sweep_direction = -1.0
                        swept_price = current_bar['low']
                        wick_size = lower_wick_size
                        
                    confidence = min(1.0, (wick_size / 5.0) * (volume_multiplier / 2.0))
                    
                    event = M1Event(
                        event_id=f"{session_id}_sweep_{i}",
                        session_id=session_id,
                        timestamp_ms=int(current_bar['timestamp']),
                        parent_m5_seq_idx=m5_mapping.get(i, -1),
                        event_kind='micro_sweep',
                        price=swept_price,
                        volume=current_bar['volume'],
                        features={
                            'sweep_direction': sweep_direction,
                            'wick_size_pips': wick_size,
                            'volume_multiplier': volume_multiplier,
                            'wick_ratio': current_bar['wick_ratio']
                        },
                        confidence=confidence,
                        metadata={'sweep_type': 'upper' if sweep_direction > 0 else 'lower'}
                    )
                    events.append(event)
                    
        return events
        
    def _detect_micro_impulse_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect micro impulse move events"""
        events = []
        
        for i in range(self.config.lookback_period, len(df)):
            current_bar = df.iloc[i]
            
            # Check for strong range expansion
            if abs(current_bar['range_z']) >= self.config.impulse_z_threshold:
                if current_bar['range_pips'] >= self.config.impulse_min_range_pips:
                    
                    # Determine impulse direction
                    price_direction = 1.0 if current_bar['close'] > current_bar['open'] else -1.0
                    
                    # Calculate impulse strength
                    impulse_strength = abs(current_bar['range_z'])
                    momentum_confirmation = abs(current_bar.get('momentum_3', 0)) * 10000
                    
                    confidence = min(1.0, (impulse_strength / 3.0) + (momentum_confirmation / 10.0))
                    
                    event = M1Event(
                        event_id=f"{session_id}_impulse_{i}",
                        session_id=session_id,
                        timestamp_ms=int(current_bar['timestamp']),
                        parent_m5_seq_idx=m5_mapping.get(i, -1),
                        event_kind='micro_impulse',
                        price=current_bar['typical_price'],
                        volume=current_bar['volume'],
                        features={
                            'impulse_direction': price_direction,
                            'range_z_score': current_bar['range_z'],
                            'range_pips': current_bar['range_pips'],
                            'momentum_3': momentum_confirmation,
                            'body_ratio': current_bar['body_pips'] / current_bar['range_pips']
                        },
                        confidence=confidence,
                        metadata={'impulse_type': 'strong_range_expansion'}
                    )
                    events.append(event)
                    
        return events
        
    def _detect_vwap_touch_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect VWAP interaction events"""
        events = []
        
        for i in range(self.config.vwap_period, len(df)):
            current_bar = df.iloc[i]
            
            if pd.isna(current_bar['vwap']):
                continue
                
            # Check if price touches VWAP
            vwap_distance = abs(current_bar['typical_price'] - current_bar['vwap']) * 10000
            
            if vwap_distance <= self.config.vwap_touch_threshold_pips:
                # Determine touch type
                if current_bar['low'] <= current_bar['vwap'] <= current_bar['high']:
                    touch_type = 'cross'
                elif current_bar['close'] > current_bar['vwap']:
                    touch_type = 'bounce_up'
                else:
                    touch_type = 'bounce_down'
                    
                # Calculate confidence based on volume and precision
                volume_factor = current_bar.get('volume_z', 0)
                precision_factor = 1.0 - (vwap_distance / self.config.vwap_touch_threshold_pips)
                
                confidence = min(1.0, 0.5 + 0.3 * abs(volume_factor) + 0.2 * precision_factor)
                
                event = M1Event(
                    event_id=f"{session_id}_vwap_{i}",
                    session_id=session_id,
                    timestamp_ms=int(current_bar['timestamp']),
                    parent_m5_seq_idx=m5_mapping.get(i, -1),
                    event_kind='vwap_touch',
                    price=current_bar['vwap'],
                    volume=current_bar['volume'],
                    features={
                        'vwap_distance_pips': vwap_distance,
                        'touch_type': {'cross': 0, 'bounce_up': 1, 'bounce_down': -1}[touch_type],
                        'volume_z': volume_factor,
                        'price_vs_vwap': (current_bar['close'] - current_bar['vwap']) / current_bar['vwap']
                    },
                    confidence=confidence,
                    metadata={'touch_type': touch_type, 'vwap_period': self.config.vwap_period}
                )
                events.append(event)
                
        return events
        
    def _detect_imbalance_burst_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect order flow imbalance burst events"""
        events = []
        
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            
            # Use spread proxy as imbalance indicator
            if current_bar['spread_proxy'] > 0 and current_bar['volume'] >= self.config.imbalance_min_volume:
                
                # Calculate imbalance ratio using high/low bias
                if current_bar['close'] > current_bar['open']:
                    # Bullish bias - assume ask-side pressure
                    imbalance_strength = (current_bar['high'] - current_bar['open']) / current_bar['spread_proxy']
                    imbalance_direction = 1.0
                else:
                    # Bearish bias - assume bid-side pressure  
                    imbalance_strength = (current_bar['open'] - current_bar['low']) / current_bar['spread_proxy']
                    imbalance_direction = -1.0
                    
                if imbalance_strength >= self.config.imbalance_ratio_threshold:
                    
                    # Volume confirmation
                    volume_factor = current_bar.get('volume_z', 0)
                    
                    confidence = min(1.0, (imbalance_strength / 5.0) + abs(volume_factor) * 0.2)
                    
                    event = M1Event(
                        event_id=f"{session_id}_imbalance_{i}",
                        session_id=session_id,
                        timestamp_ms=int(current_bar['timestamp']),
                        parent_m5_seq_idx=m5_mapping.get(i, -1),
                        event_kind='imbalance_burst',
                        price=current_bar['typical_price'],
                        volume=current_bar['volume'],
                        features={
                            'imbalance_direction': imbalance_direction,
                            'imbalance_strength': imbalance_strength,
                            'spread_proxy': current_bar['spread_proxy'] * 10000,
                            'volume_z': volume_factor
                        },
                        confidence=confidence,
                        metadata={'imbalance_type': 'ask_pressure' if imbalance_direction > 0 else 'bid_pressure'}
                    )
                    events.append(event)
                    
        return events
        
    def _detect_wick_extreme_events(
        self, 
        df: pd.DataFrame, 
        session_id: str, 
        m5_mapping: Dict[int, int]
    ) -> List[M1Event]:
        """Detect exceptional wick formation events"""
        events = []
        
        # Calculate wick size percentiles for the session
        upper_wick_sizes = df['upper_wick'] * 10000
        lower_wick_sizes = df['lower_wick'] * 10000
        
        upper_threshold = np.percentile(upper_wick_sizes.dropna(), self.config.wick_extreme_percentile)
        lower_threshold = np.percentile(lower_wick_sizes.dropna(), self.config.wick_extreme_percentile)
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            upper_wick_pips = current_bar['upper_wick'] * 10000
            lower_wick_pips = current_bar['lower_wick'] * 10000
            
            # Check for extreme upper wick
            if upper_wick_pips >= upper_threshold and current_bar['wick_ratio'] >= self.config.wick_min_ratio:
                confidence = min(1.0, upper_wick_pips / (upper_threshold * 2))
                
                event = M1Event(
                    event_id=f"{session_id}_wick_upper_{i}",
                    session_id=session_id,
                    timestamp_ms=int(current_bar['timestamp']),
                    parent_m5_seq_idx=m5_mapping.get(i, -1),
                    event_kind='wick_extreme',
                    price=current_bar['high'],
                    volume=current_bar['volume'],
                    features={
                        'wick_direction': 1.0,  # Upper
                        'wick_size_pips': upper_wick_pips,
                        'wick_ratio': current_bar['wick_ratio'],
                        'percentile_rank': self.config.wick_extreme_percentile
                    },
                    confidence=confidence,
                    metadata={'wick_type': 'upper_extreme', 'threshold': upper_threshold}
                )
                events.append(event)
                
            # Check for extreme lower wick
            if lower_wick_pips >= lower_threshold and current_bar['wick_ratio'] >= self.config.wick_min_ratio:
                confidence = min(1.0, lower_wick_pips / (lower_threshold * 2))
                
                event = M1Event(
                    event_id=f"{session_id}_wick_lower_{i}",
                    session_id=session_id,
                    timestamp_ms=int(current_bar['timestamp']),
                    parent_m5_seq_idx=m5_mapping.get(i, -1),
                    event_kind='wick_extreme',
                    price=current_bar['low'],
                    volume=current_bar['volume'],
                    features={
                        'wick_direction': -1.0,  # Lower
                        'wick_size_pips': lower_wick_pips,
                        'wick_ratio': current_bar['wick_ratio'],
                        'percentile_rank': self.config.wick_extreme_percentile
                    },
                    confidence=confidence,
                    metadata={'wick_type': 'lower_extreme', 'threshold': lower_threshold}
                )
                events.append(event)
                
        return events
        
    def _deduplicate_events(self, events: List[M1Event]) -> List[M1Event]:
        """Remove duplicate events that occur at the same timestamp"""
        if not events:
            return events
            
        # Sort by timestamp and confidence
        events.sort(key=lambda x: (x.timestamp_ms, -x.confidence))
        
        deduplicated = []
        seen_timestamps = set()
        
        for event in events:
            # Allow multiple event types at the same timestamp, but not duplicates of the same type
            key = (event.timestamp_ms, event.event_kind)
            
            if key not in seen_timestamps:
                deduplicated.append(event)
                seen_timestamps.add(key)
                
        return deduplicated
        
    def save_events_parquet(self, events: List[M1Event], output_path: Path):
        """Save M1 events to parquet format"""
        if not events:
            self.logger.warning("No events to save")
            return
            
        # Convert events to DataFrame
        records = []
        
        for event in events:
            record = {
                'event_id': event.event_id,
                'session_id': event.session_id,
                'timestamp_ms': event.timestamp_ms,
                'parent_m5_seq_idx': event.parent_m5_seq_idx,
                'event_kind': event.event_kind,
                'price': event.price,
                'volume': event.volume,
                'confidence': event.confidence
            }
            
            # Add features as columns
            for feature_name, feature_value in event.features.items():
                record[f'feature_{feature_name}'] = feature_value
                
            # Add metadata as JSON string
            import json
            record['metadata'] = json.dumps(event.metadata)
            
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with ZSTD compression
        df.to_parquet(
            output_path,
            compression='zstd',
            row_group_size=5000,
            engine='pyarrow'
        )
        
        self.logger.info(f"Saved {len(events)} M1 events to {output_path}")


def process_session_m1_events(
    m1_ohlcv_path: Path,
    m5_bars_path: Path,
    session_id: str,
    output_dir: Path,
    config: M1EventConfig = None
) -> List[M1Event]:
    """
    Convenience function to process M1 events for a single session
    
    Args:
        m1_ohlcv_path: Path to M1 OHLCV data (CSV or parquet)
        m5_bars_path: Path to M5 bars data (optional)
        session_id: Session identifier
        output_dir: Output directory for results
        config: Event detection configuration
        
    Returns:
        List of detected M1Event objects
    """
    config = config or M1EventConfig()
    detector = M1EventDetector(config)
    
    # Load M1 data
    try:
        if m1_ohlcv_path.suffix.lower() == '.parquet':
            m1_data = pd.read_parquet(m1_ohlcv_path)
        else:
            m1_data = pd.read_csv(m1_ohlcv_path)
    except Exception as e:
        logger.error(f"Failed to load M1 data from {m1_ohlcv_path}: {e}")
        return []
        
    # Load M5 data if provided
    m5_data = None
    if m5_bars_path and m5_bars_path.exists():
        try:
            if m5_bars_path.suffix.lower() == '.parquet':
                m5_data = pd.read_parquet(m5_bars_path)
            else:
                m5_data = pd.read_csv(m5_bars_path)
        except Exception as e:
            logger.warning(f"Failed to load M5 data from {m5_bars_path}: {e}")
            
    # Detect events
    events = detector.detect_events(m1_data, session_id, m5_data)
    
    # Save results
    if events:
        output_path = output_dir / f"{session_id}_m1_events.parquet"
        detector.save_events_parquet(events, output_path)
        
    return events


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect M1 sparse events")
    parser.add_argument("--m1-data", required=True, help="Path to M1 OHLCV data")
    parser.add_argument("--m5-data", help="Path to M5 bars data (optional)")
    parser.add_argument("--session-id", required=True, help="Session identifier")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum event confidence")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure detection
    config = M1EventConfig(min_confidence=args.min_confidence)
    
    # Process events
    events = process_session_m1_events(
        Path(args.m1_data),
        Path(args.m5_data) if args.m5_data else None,
        args.session_id,
        Path(args.output_dir),
        config
    )
    
    print(f"Detected {len(events)} M1 events")
    
    # Print event summary
    event_counts = {}
    for event in events:
        event_counts[event.event_kind] = event_counts.get(event.event_kind, 0) + 1
        
    for event_type, count in sorted(event_counts.items()):
        print(f"- {event_type}: {count}")