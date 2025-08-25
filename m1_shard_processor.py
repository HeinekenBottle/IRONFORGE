#!/usr/bin/env python3
"""
M1 Shard Data Processor for ICT Gauntlet Pattern Discovery
Converts M5 shard data to M1 resolution and integrates with FPFVG detection
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from fpfvg_m1_detector import FPFVGM1Detector, FPFVGEvent, GauntletSequence

class M1ShardProcessor:
    """Process M5 shards to generate M1 resolution data for FPFVG detection"""
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5"):
        self.shard_dir = Path(shard_dir)
        self.fpfvg_detector = FPFVGM1Detector()
        
        # M1 interpolation parameters
        self.M1_INTERVALS_PER_M5 = 5  # 5 minutes = 5 one-minute intervals
        self.INTERPOLATION_METHOD = "linear"  # Linear interpolation for missing M1 data
        
        # Session timing (ET)
        self.NY_AM_START = "09:30:00"
        self.NY_AM_END = "11:59:00"
        self.MACRO_WINDOW = ("09:50:00", "10:10:00")
        
    def generate_m1_from_m5_shard(self, shard_path: str) -> pd.DataFrame:
        """
        Generate M1 OHLC data from M5 shard using interpolation
        
        Args:
            shard_path: Path to M5 shard directory
            
        Returns:
            DataFrame with M1 OHLC data
        """
        # Load M5 nodes data
        nodes_path = Path(shard_path) / "nodes.parquet"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        
        nodes_df = pd.read_parquet(nodes_path)
        
        # Extract price and timestamp data
        price_data = self._extract_price_data(nodes_df)
        
        # Interpolate to M1 resolution
        m1_data = self._interpolate_to_m1(price_data)
        
        return m1_data
    
    def _extract_price_data(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """Extract price and timing data from shard nodes"""
        # Create OHLC data structure from M5 nodes
        # Note: This is a simplified extraction - actual implementation 
        # would depend on your specific shard structure
        
        # Handle timestamp conversion more carefully
        # Check if timestamps are in Unix format or another format
        timestamp_col = nodes_df['t']
        
        # Convert timestamps - based on examination, they are in milliseconds format
        try:
            # The timestamp values are in milliseconds (confirmed from data analysis)
            timestamps = pd.to_datetime(timestamp_col, unit='ms')
        except (ValueError, OverflowError) as e:
            print(f"Timestamp conversion error: {e}")
            # Fallback to treating as already datetime or string
            timestamps = pd.to_datetime(timestamp_col)
        
        price_df = pd.DataFrame({
            'timestamp': timestamps,
            'price': nodes_df['price'],
            'node_id': nodes_df['node_id']  # Fixed column name
        })
        
        # Sort by timestamp
        price_df = price_df.sort_values('timestamp').reset_index(drop=True)
        
        # Create OHLC from price points (simplified approach)
        # In practice, you would need actual OHLC data from your source
        ohlc_data = []
        
        for i in range(0, len(price_df), 1):  # Group every 5 points for M5 OHLC
            group = price_df.iloc[i:i+5] if i+5 <= len(price_df) else price_df.iloc[i:]
            
            if len(group) > 0:
                ohlc_bar = {
                    'timestamp': group['timestamp'].iloc[0],
                    'open': group['price'].iloc[0],
                    'high': group['price'].max(),
                    'low': group['price'].min(),
                    'close': group['price'].iloc[-1],
                    'volume': len(group)  # Simplified volume metric
                }
                ohlc_data.append(ohlc_bar)
        
        return pd.DataFrame(ohlc_data)
    
    def _interpolate_to_m1(self, m5_ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate M5 OHLC data to M1 resolution
        
        This creates synthetic M1 data for gap detection.
        For production, you would use actual M1 source data.
        """
        m1_data = []
        
        for i in range(len(m5_ohlc) - 1):
            current_bar = m5_ohlc.iloc[i]
            next_bar = m5_ohlc.iloc[i + 1]
            
            # Add the current M5 bar as the first M1 bar
            m1_data.append({
                'timestamp': current_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close']
            })
            
            # Interpolate 4 additional M1 bars to reach the next M5 bar
            for j in range(1, self.M1_INTERVALS_PER_M5):
                interpolated_time = current_bar['timestamp'] + timedelta(minutes=j)
                
                # Linear interpolation between current and next bar
                progress = j / self.M1_INTERVALS_PER_M5
                interpolated_price = current_bar['close'] + (
                    (next_bar['open'] - current_bar['close']) * progress
                )
                
                # Create synthetic OHLC for interpolated bar
                # This is simplified - real M1 data would have proper OHLC
                price_variance = abs(next_bar['high'] - next_bar['low']) * 0.1
                
                m1_data.append({
                    'timestamp': interpolated_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': interpolated_price,
                    'high': interpolated_price + (price_variance * np.random.uniform(0, 1)),
                    'low': interpolated_price - (price_variance * np.random.uniform(0, 1)),
                    'close': interpolated_price + (price_variance * np.random.uniform(-0.5, 0.5))
                })
        
        # Add the final M5 bar
        if len(m5_ohlc) > 0:
            final_bar = m5_ohlc.iloc[-1]
            m1_data.append({
                'timestamp': final_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'open': final_bar['open'],
                'high': final_bar['high'],
                'low': final_bar['low'],
                'close': final_bar['close']
            })
        
        return pd.DataFrame(m1_data)
    
    def process_session_for_gauntlet(self, session_id: str) -> Dict[str, Any]:
        """
        Process a complete session for ICT Gauntlet pattern discovery
        
        Args:
            session_id: Session identifier (e.g., "NY_2025-07-29")
            
        Returns:
            Complete session analysis with FPFVG and Gauntlet patterns
        """
        # Construct shard path
        shard_path = self.shard_dir / f"shard_{session_id}"
        
        if not shard_path.exists():
            return {
                'error': f"Shard not found: {shard_path}",
                'session_id': session_id
            }
        
        # Load session metadata
        meta_path = shard_path / "meta.json"
        session_meta = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                session_meta = json.load(f)
        
        # Generate M1 data from M5 shard
        try:
            m1_ohlc_data = self.generate_m1_from_m5_shard(str(shard_path))
            
            if len(m1_ohlc_data) == 0:
                return {
                    'error': 'No M1 data generated',
                    'session_id': session_id
                }
            
            # Extract session type and date from session_id
            session_parts = session_id.split('_')
            session_type = session_parts[0].lower()
            session_date = session_parts[1] if len(session_parts) > 1 else "unknown"
            
            # Detect FPFVG events
            fpfvg_events = self.fpfvg_detector.detect_fpfvg_from_m1_data(
                m1_ohlc_data, f"{session_type}_am", session_date
            )
            
            # Calculate session statistics
            session_high = m1_ohlc_data['high'].max()
            session_low = m1_ohlc_data['low'].min()
            session_range = session_high - session_low
            
            # Detect complete Gauntlet sequences
            gauntlet_sequences = self.fpfvg_detector.detect_gauntlet_sequence(
                fpfvg_events, m1_ohlc_data, session_high, session_low
            )
            
            # Generate enhanced session format compatible with existing structure
            enhanced_session = self._generate_enhanced_session_format(
                session_id, session_meta, fpfvg_events, gauntlet_sequences,
                session_high, session_low, m1_ohlc_data
            )
            
            return enhanced_session
            
        except Exception as e:
            return {
                'error': f"Processing failed: {str(e)}",
                'session_id': session_id
            }
    
    def _generate_enhanced_session_format(self, session_id: str, session_meta: Dict[str, Any],
                                        fpfvg_events: List[FPFVGEvent], 
                                        gauntlet_sequences: List[GauntletSequence],
                                        session_high: float, session_low: float,
                                        m1_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced session format compatible with existing IRONFORGE structure"""
        
        # Convert session_id to session type and date
        session_parts = session_id.split('_')
        session_type = f"{session_parts[0].lower()}_am"
        session_date = session_parts[1] if len(session_parts) > 1 else "unknown"
        
        # Create session metadata
        session_metadata = {
            "session_type": session_type,
            "session_date": session_date,
            "session_start": self.NY_AM_START,
            "session_end": self.NY_AM_END,
            "session_duration": len(m1_data),
            "transcription_source": "m1_interpolated_from_m5_shard",
            "data_completeness": "complete_session",
            "timezone": "ET",
            "session_status": "processed_for_gauntlet",
            "m1_resolution": True,
            "fpfvg_detection_enabled": True
        }
        
        # Create FPFVG section
        session_fpfvg = {
            "fpfvg_present": len(fpfvg_events) > 0,
            "fpfvg_count": len(fpfvg_events),
            "fpfvg_events": []
        }
        
        # Add FPFVG events
        for fpfvg in fpfvg_events:
            fpfvg_data = {
                "formation_time": fpfvg.formation_time,
                "premium_high": fpfvg.premium_high,
                "discount_low": fpfvg.discount_low,
                "consequent_encroachment": fpfvg.consequent_encroachment,
                "gap_size": fpfvg.gap_size,
                "direction": fpfvg.session_context.get('direction', 'unknown'),
                "formation_phase": fpfvg.session_context.get('formation_phase', 'unknown'),
                "candle_sequence": fpfvg.candle_sequence,
                "archaeological_proximity": fpfvg.archaeological_proximity
            }
            session_fpfvg["fpfvg_events"].append(fpfvg_data)
        
        # Create Gauntlet sequences section
        gauntlet_analysis = {
            "gauntlet_sequences_detected": len(gauntlet_sequences),
            "complete_sequences": len([seq for seq in gauntlet_sequences if seq.completion_status == "complete"]),
            "partial_sequences": len([seq for seq in gauntlet_sequences if seq.completion_status != "complete"]),
            "sequences": []
        }
        
        for sequence in gauntlet_sequences:
            sequence_data = {
                "fpfvg_formation": sequence.fpfvg.formation_time,
                "liquidity_hunt_time": sequence.liquidity_hunt_time,
                "reversal_time": sequence.reversal_time,
                "ce_breach_time": sequence.ce_breach_time,
                "completion_status": sequence.completion_status,
                "archaeological_confluence": sequence.archaeological_confluence
            }
            gauntlet_analysis["sequences"].append(sequence_data)
        
        # Create price movements section
        price_movements = [
            {
                "timestamp": self.NY_AM_START,
                "price_level": m1_data['open'].iloc[0] if len(m1_data) > 0 else 0.0,
                "movement_type": "session_open"
            },
            {
                "timestamp": "N/A",
                "price_level": session_high,
                "movement_type": "session_high"
            },
            {
                "timestamp": "N/A", 
                "price_level": session_low,
                "movement_type": "session_low"
            },
            {
                "timestamp": self.NY_AM_END,
                "price_level": m1_data['close'].iloc[-1] if len(m1_data) > 0 else 0.0,
                "movement_type": "session_close"
            }
        ]
        
        # Assemble complete enhanced session
        enhanced_session = {
            "session_metadata": session_metadata,
            "session_fpfvg": session_fpfvg,
            "gauntlet_analysis": gauntlet_analysis,
            "price_movements": price_movements,
            "session_statistics": {
                "session_high": session_high,
                "session_low": session_low,
                "session_range": session_high - session_low,
                "m1_bars_processed": len(m1_data),
                "fpfvg_detection_success": len(fpfvg_events) > 0,
                "gauntlet_pattern_success": any(seq.completion_status == "complete" for seq in gauntlet_sequences)
            },
            "archaeological_integration": {
                "zone_analysis_enabled": True,
                "theory_b_validation": True,
                "confluence_detection": True
            }
        }
        
        return enhanced_session
    
    def batch_process_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Batch process multiple sessions for ICT Gauntlet patterns
        
        Args:
            session_ids: List of session identifiers to process
            
        Returns:
            Batch processing results with summary statistics
        """
        results = {
            'processed_sessions': {},
            'summary': {
                'total_sessions': len(session_ids),
                'successful_processing': 0,
                'failed_processing': 0,
                'total_fpfvg_events': 0,
                'total_gauntlet_sequences': 0,
                'complete_gauntlet_sequences': 0,
                'archaeological_confluences': 0
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        for session_id in session_ids:
            print(f"Processing session: {session_id}")
            
            session_result = self.process_session_for_gauntlet(session_id)
            results['processed_sessions'][session_id] = session_result
            
            if 'error' not in session_result:
                results['summary']['successful_processing'] += 1
                
                # Update summary statistics
                fpfvg_count = session_result.get('session_fpfvg', {}).get('fpfvg_count', 0)
                gauntlet_count = session_result.get('gauntlet_analysis', {}).get('gauntlet_sequences_detected', 0)
                complete_count = session_result.get('gauntlet_analysis', {}).get('complete_sequences', 0)
                
                results['summary']['total_fpfvg_events'] += fpfvg_count
                results['summary']['total_gauntlet_sequences'] += gauntlet_count
                results['summary']['complete_gauntlet_sequences'] += complete_count
                
                # Count archaeological confluences
                sequences = session_result.get('gauntlet_analysis', {}).get('sequences', [])
                for seq in sequences:
                    confluence = seq.get('archaeological_confluence', {})
                    if confluence.get('closest_zone', {}).get('confluence', False):
                        results['summary']['archaeological_confluences'] += 1
            else:
                results['summary']['failed_processing'] += 1
        
        return results

# Testing and demonstration
def demo_m1_processing():
    """Demonstrate M1 shard processing for ICT Gauntlet detection"""
    print("🔧 M1 Shard Processor Demo")
    print("=" * 50)
    
    processor = M1ShardProcessor()
    
    # Process a single session
    session_id = "NY_2025-07-29"
    print(f"\n📊 Processing Session: {session_id}")
    
    result = processor.process_session_for_gauntlet(session_id)
    
    if 'error' in result:
        print(f"❌ Processing failed: {result['error']}")
    else:
        print("✅ Processing successful!")
        print(f"   FPFVG Events: {result.get('session_fpfvg', {}).get('fpfvg_count', 0)}")
        print(f"   Gauntlet Sequences: {result.get('gauntlet_analysis', {}).get('gauntlet_sequences_detected', 0)}")
        print(f"   Complete Sequences: {result.get('gauntlet_analysis', {}).get('complete_sequences', 0)}")
        
        # Show session statistics
        stats = result.get('session_statistics', {})
        print(f"   Session Range: {stats.get('session_range', 0):.2f} points")
        print(f"   M1 Bars Processed: {stats.get('m1_bars_processed', 0)}")

if __name__ == "__main__":
    demo_m1_processing()