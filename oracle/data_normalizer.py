#!/usr/bin/env python3
"""
Oracle Data Normalizer - Convert Parquet shards to canonical training format
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OracleDataNormalizer:
    """Normalizes IRONFORGE Parquet shards to canonical Oracle training format"""
    
    def __init__(self, data_dir: str = "data/shards", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[NORMALIZER] {message}")
    
    def find_shards(self, symbol: str, timeframe: str) -> List[Path]:
        """Find all shard directories for a given symbol/timeframe"""
        
        # Handle both M5 and 5 timeframe formats
        tf_string = timeframe if timeframe.startswith('M') else f"M{timeframe}"
        
        shard_pattern = f"{symbol}_{tf_string}"
        shard_base = self.data_dir / shard_pattern
        
        if not shard_base.exists():
            self.log(f"No shard directory found: {shard_base}")
            return []
        
        # Find all shard_* subdirectories
        shards = [d for d in shard_base.iterdir() if d.is_dir() and d.name.startswith("shard_")]
        
        self.log(f"Found {len(shards)} shards in {shard_base}")
        return sorted(shards)
    
    def load_shard_metadata(self, shard_dir: Path) -> Optional[Dict]:
        """Load shard metadata from meta.json"""
        meta_file = shard_dir / "meta.json"
        
        if not meta_file.exists():
            self.log(f"No meta.json in {shard_dir}")
            return None
        
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Failed to load {meta_file}: {e}")
            return None
    
    def load_shard_nodes(self, shard_dir: Path) -> Optional[pd.DataFrame]:
        """Load shard nodes from nodes.parquet"""
        nodes_file = shard_dir / "nodes.parquet"
        
        if not nodes_file.exists():
            self.log(f"No nodes.parquet in {shard_dir}")
            return None
        
        try:
            df = pd.read_parquet(nodes_file)
            self.log(f"Loaded {len(df)} nodes from {nodes_file}")
            return df
        except Exception as e:
            self.log(f"Failed to load {nodes_file}: {e}")
            return None
    
    def parse_session_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse session timestamp from various formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        self.log(f"Failed to parse timestamp: {timestamp_str}")
        return None
    
    def compute_session_ohlc(self, nodes_df: pd.DataFrame) -> Tuple[float, float, float, float]:
        """Compute OHLC from node prices"""
        if len(nodes_df) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        prices = nodes_df['price'].values
        open_price = prices[0]
        high_price = prices.max()
        low_price = prices.min()
        close_price = prices[-1]
        
        return open_price, high_price, low_price, close_price
    
    def assess_data_quality(self, nodes_df: pd.DataFrame, meta: Dict) -> str:
        """Assess data quality: excellent/good/fair/poor"""
        
        if len(nodes_df) == 0:
            return "poor"
        
        n_events = len(nodes_df)
        
        # Check for required fields
        required_fields = ['price']
        for i in range(45):  # f0-f44 features
            required_fields.append(f'f{i}')
        
        missing_fields = [field for field in required_fields if field not in nodes_df.columns]
        
        if missing_fields:
            self.log(f"Missing fields: {missing_fields}")
            return "poor"
        
        # Quality thresholds
        if n_events >= 50:
            return "excellent"
        elif n_events >= 20:
            return "good"
        elif n_events >= 10:
            return "fair"
        else:
            return "poor"
    
    def normalize_shard(self, shard_dir: Path) -> Optional[Dict]:
        """Normalize a single shard to canonical format"""
        
        # Load metadata
        meta = self.load_shard_metadata(shard_dir)
        if not meta:
            return None
        
        # Load nodes
        nodes_df = self.load_shard_nodes(shard_dir)
        if nodes_df is None or len(nodes_df) == 0:
            self.log(f"No valid nodes in {shard_dir}")
            return None
        
        # Extract session info
        session_id = meta.get('session_id', shard_dir.name)
        symbol = meta.get('symbol', 'UNKNOWN')
        timeframe_str = meta.get('timeframe', 'M5')
        
        # Parse timeframe to numeric and standardize string format
        if timeframe_str.startswith('M'):
            tf_numeric = int(timeframe_str[1:])
            tf_string = timeframe_str  # Already in M5 format
        else:
            tf_numeric = int(timeframe_str)
            tf_string = f"M{timeframe_str}"  # Convert 5 -> M5
        
        # Parse session date
        session_date = meta.get('date', '1970-01-01')
        
        # Compute OHLC
        open_price, high_price, low_price, close_price = self.compute_session_ohlc(nodes_df)
        
        # Compute session metrics
        center = (high_price + low_price) / 2
        half_range = (high_price - low_price) / 2
        
        # Determine HTF mode from feature columns
        htf_mode = "51D" if 'f50' in nodes_df.columns else "45D"
        
        # Assess quality
        quality = self.assess_data_quality(nodes_df, meta)
        
        # Create canonical session row
        n_events = len(nodes_df)
        session_row = {
            'symbol': symbol,
            'tf': tf_string,  # Use normalized timeframe string (M5) not numeric (5)
            'session_date': session_date,
            'start_ts': f"{session_date}T09:30:00",  # Default session start
            'end_ts': f"{session_date}T16:00:00",    # Default session end
            'final_high': high_price,
            'final_low': low_price,
            'center': center,
            'half_range': half_range,
            'htf_mode': htf_mode,
            'n_events': n_events,
            'data_quality': quality,
            'session_id': session_id,
            'shard_path': str(shard_dir)
        }
        
        self.log(f"Normalized {session_id}: {n_events} events, range={half_range*2:.1f}, quality={quality}")
        
        return session_row
    
    def normalize_symbol_timeframe(
        self, 
        symbol: str, 
        timeframe: str,
        min_quality: str = "fair",
        min_events: int = 10
    ) -> pd.DataFrame:
        """Normalize all shards for a symbol/timeframe combination"""
        
        # Find all shards
        shards = self.find_shards(symbol, timeframe)
        
        if not shards:
            self.log(f"No shards found for {symbol}/{timeframe}")
            return pd.DataFrame()
        
        # Normalize each shard
        normalized_rows = []
        quality_levels = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
        min_quality_level = quality_levels.get(min_quality, 2)
        
        for shard_dir in shards:
            try:
                session_row = self.normalize_shard(shard_dir)
                if session_row:
                    # Apply quality and event count filters
                    quality_level = quality_levels.get(session_row['data_quality'], 1)
                    
                    if quality_level >= min_quality_level and session_row['n_events'] >= min_events:
                        normalized_rows.append(session_row)
                    else:
                        self.log(f"Filtered out {session_row['session_id']}: quality={session_row['data_quality']}, events={session_row['n_events']}")
                        
            except Exception as e:
                self.log(f"Failed to normalize {shard_dir}: {e}")
                continue
        
        if not normalized_rows:
            self.log(f"No valid sessions after filtering")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(normalized_rows)
        
        # Sort by session date
        df = df.sort_values('session_date').reset_index(drop=True)
        
        self.log(f"Normalized {len(df)} sessions for {symbol}/{timeframe}")
        return df
    
    def normalize_symbol_timeframe_with_dates(
        self,
        symbol: str,
        timeframe: str,
        from_date: str,
        to_date: str,
        min_quality: str = "fair",
        min_events: int = 10
    ) -> pd.DataFrame:
        """Normalize sessions with date range filtering"""
        from datetime import datetime
        
        # First get all sessions
        df = self.normalize_symbol_timeframe(
            symbol=symbol,
            timeframe=timeframe,
            min_quality=min_quality,
            min_events=min_events
        )
        
        if df.empty:
            return df
        
        # Parse date range
        try:
            start_date = datetime.strptime(from_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(to_date, "%Y-%m-%d").date()
        except ValueError as e:
            self.log(f"Invalid date format: {e}")
            return pd.DataFrame()
        
        # Filter by date range
        def in_date_range(date_str):
            try:
                session_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                return start_date <= session_date <= end_date
            except:
                return False
        
        filtered_df = df[df['session_date'].apply(in_date_range)].copy()
        
        if filtered_df.empty:
            self.log(f"No sessions found in date range {from_date} to {to_date}")
            return filtered_df
        
        self.log(f"Date filtered: {len(df)} → {len(filtered_df)} sessions in range {from_date} to {to_date}")
        return filtered_df


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Normalize IRONFORGE shards to Oracle training format")
    parser.add_argument("--data-dir", default="data/shards", help="Shard data directory")
    parser.add_argument("--symbol", required=True, help="Symbol to process (e.g., NQ)")
    parser.add_argument("--tf", required=True, help="Timeframe to process (e.g., 5 or M5)")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--quality", default="fair", choices=["excellent", "good", "fair", "poor"], 
                       help="Minimum data quality threshold")
    parser.add_argument("--min-events", type=int, default=10, help="Minimum events per session")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Initialize normalizer
    normalizer = OracleDataNormalizer(args.data_dir, args.verbose)
    
    # Normalize sessions
    print(f"🔄 Normalizing {args.symbol}/{args.tf} sessions...")
    df = normalizer.normalize_symbol_timeframe(
        symbol=args.symbol,
        timeframe=args.tf,
        min_quality=args.quality,
        min_events=args.min_events
    )
    
    if df.empty:
        print("❌ No sessions to normalize")
        return 1
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # Summary
    print(f"✅ Normalized {len(df)} sessions")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Quality distribution: {df['data_quality'].value_counts().to_dict()}")
    print(f"🎯 Average range: {df['half_range'].mean()*2:.1f} ± {df['half_range'].std()*2:.1f} points")
    
    return 0


if __name__ == "__main__":
    exit(main())