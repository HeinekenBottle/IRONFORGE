#!/usr/bin/env python3
"""
Session Data Normalizer for Oracle Training

Transforms diverse session formats into canonical training rows:

symbol, tf, session_date, start_ts, end_ts,
final_high, final_low, center, half_range,
htf_mode, n_events

Handles missing OHLC by computing from raw/shard data.
Enforces timezone-aware timestamps and validates data quality.

Usage:
    python scripts/normalize_sessions.py --data-dir data/enhanced --output training_sessions.parquet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import pytz
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NormalizedSession:
    """Canonical session record for Oracle training"""
    symbol: str
    tf: str  # timeframe
    session_date: str  # YYYY-MM-DD
    start_ts: str  # ISO timestamp
    end_ts: str  # ISO timestamp
    final_high: float
    final_low: float
    center: float
    half_range: float
    htf_mode: str  # "45D" or "51D"
    n_events: int
    
    # Quality metadata
    source_file: str
    data_quality: str  # "excellent", "good", "fair", "poor"
    computed_ohlc: bool  # True if OHLC was computed from events
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "tf": self.tf,
            "session_date": self.session_date,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "final_high": self.final_high,
            "final_low": self.final_low,
            "center": self.center,
            "half_range": self.half_range,
            "htf_mode": self.htf_mode,
            "n_events": self.n_events,
            "source_file": self.source_file,
            "data_quality": self.data_quality,
            "computed_ohlc": self.computed_ohlc
        }


class SessionNormalizer:
    """Normalize session data to canonical format"""
    
    def __init__(self, default_timezone: str = "US/Eastern"):
        self.default_tz = pytz.timezone(default_timezone)
        self.processed_count = 0
        self.error_count = 0
        self.quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
    def normalize_file(self, file_path: Path) -> List[NormalizedSession]:
        """Normalize single session file to canonical format"""
        try:
            logger.debug(f"Normalizing {file_path}")
            
            # Load session data
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix == '.parquet':
                data = pd.read_parquet(file_path).to_dict('records')
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []
            
            # Process data based on structure
            if isinstance(data, dict):
                sessions = [self._normalize_session_dict(data, file_path)]
            elif isinstance(data, list):
                sessions = [self._normalize_session_dict(item, file_path) for item in data]
            else:
                logger.warning(f"Unexpected data structure in {file_path}")
                return []
            
            # Filter out None results
            valid_sessions = [s for s in sessions if s is not None]
            self.processed_count += len(valid_sessions)
            
            return valid_sessions
            
        except Exception as e:
            logger.error(f"Failed to normalize {file_path}: {e}")
            self.error_count += 1
            return []
    
    def _normalize_session_dict(self, data: Dict[str, Any], file_path: Path) -> Optional[NormalizedSession]:
        """Normalize single session dictionary"""
        try:
            # Extract basic metadata
            symbol, tf = self._extract_symbol_tf(data, file_path)
            session_name = data.get("session_name", file_path.stem)
            
            # Extract events
            events = data.get("events", [])
            if not events:
                logger.debug(f"No events in session {session_name}")
                return None
                
            # Extract timestamps
            start_ts, end_ts, session_date = self._extract_timestamps(data, events)
            
            # Extract or compute OHLC
            ohlc_data = self._extract_or_compute_ohlc(data, events)
            if not ohlc_data:
                logger.debug(f"Could not determine OHLC for session {session_name}")
                return None
                
            final_high, final_low, computed_ohlc = ohlc_data
            
            # Calculate center and half_range
            center = (final_high + final_low) / 2
            half_range = (final_high - final_low) / 2
            
            # Determine HTF mode (45D vs 51D)
            htf_mode = self._determine_htf_mode(data, events)
            
            # Assess data quality
            data_quality = self._assess_data_quality(events, computed_ohlc, start_ts, end_ts)
            self.quality_stats[data_quality] += 1
            
            return NormalizedSession(
                symbol=symbol,
                tf=tf,
                session_date=session_date,
                start_ts=start_ts,
                end_ts=end_ts,
                final_high=final_high,
                final_low=final_low,
                center=center,
                half_range=half_range,
                htf_mode=htf_mode,
                n_events=len(events),
                source_file=file_path.name,
                data_quality=data_quality,
                computed_ohlc=computed_ohlc
            )
            
        except Exception as e:
            logger.error(f"Failed to normalize session in {file_path}: {e}")
            return None
    
    def _extract_symbol_tf(self, data: Dict[str, Any], file_path: Path) -> tuple[str, str]:
        """Extract symbol and timeframe from session data or filename"""
        # Try from data first
        symbol = data.get("symbol", "")
        tf = data.get("timeframe", data.get("tf", ""))
        
        # Try from filename if not in data
        if not symbol or not tf:
            filename = file_path.name.upper()
            
            # Common symbol patterns
            if "NQ" in filename:
                symbol = "NQ"
            elif "ES" in filename:
                symbol = "ES"
            elif not symbol:
                symbol = "UNKNOWN"
                
            # Common timeframe patterns  
            if "M5" in filename or "_5" in filename:
                tf = "M5"
            elif "M1" in filename or "_1" in filename:
                tf = "M1"
            elif not tf:
                tf = "M5"  # Default assumption
                
        return symbol, tf
    
    def _extract_timestamps(self, data: Dict[str, Any], events: List[Dict[str, Any]]) -> tuple[str, str, str]:
        """Extract and normalize start/end timestamps"""
        # Try session-level timestamp first
        session_ts = data.get("timestamp")
        
        # Try from first/last events
        start_event_ts = events[0].get("timestamp") if events else None
        end_event_ts = events[-1].get("timestamp") if events else None
        
        # Parse and normalize timestamps
        start_ts = self._parse_timestamp(start_event_ts or session_ts)
        end_ts = self._parse_timestamp(end_event_ts or session_ts) 
        
        # Extract session date (YYYY-MM-DD)
        if start_ts:
            session_date = pd.to_datetime(start_ts).strftime("%Y-%m-%d")
        else:
            # Fallback to today's date
            session_date = datetime.now().strftime("%Y-%m-%d")
            
        return start_ts, end_ts, session_date
    
    def _parse_timestamp(self, ts_value: Any) -> Optional[str]:
        """Parse timestamp to ISO format with timezone"""
        if not ts_value:
            return None
            
        try:
            # Try parsing as pandas timestamp
            dt = pd.to_datetime(ts_value)
            
            # Add timezone if naive
            if dt.tz is None:
                dt = self.default_tz.localize(dt)
            
            return dt.isoformat()
            
        except Exception as e:
            logger.debug(f"Could not parse timestamp {ts_value}: {e}")
            return None
    
    def _extract_or_compute_ohlc(self, data: Dict[str, Any], events: List[Dict[str, Any]]) -> Optional[tuple[float, float, bool]]:
        """Extract OHLC from metadata or compute from events"""
        computed = False
        
        # Try from metadata first
        metadata = data.get("metadata", {})
        high = metadata.get("high")
        low = metadata.get("low")
        
        if high is not None and low is not None:
            try:
                return float(high), float(low), computed
            except (ValueError, TypeError):
                pass
        
        # Compute from events
        if not events:
            return None
            
        prices = []
        for event in events:
            # Try different price field names
            price = None
            for price_key in ["price", "close", "price_close", "high", "low"]:
                if price_key in event:
                    try:
                        price = float(event[price_key])
                        break
                    except (ValueError, TypeError):
                        continue
                        
            if price is not None:
                prices.append(price)
        
        if not prices:
            logger.debug("No valid prices found in events")
            return None
            
        computed = True
        return max(prices), min(prices), computed
    
    def _determine_htf_mode(self, data: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
        """Determine if session uses 45D or 51D features"""
        # Check for HTF indicators in metadata
        metadata = data.get("metadata", {})
        if "htf_context" in metadata:
            return "51D" if metadata["htf_context"] else "45D"
        
        # Check filename patterns
        source_file = data.get("source_file", "")
        if "htf" in source_file.lower():
            return "51D"
            
        # Check for HTF features in events
        if events:
            sample_event = events[0]
            if "feature" in sample_event and isinstance(sample_event["feature"], list):
                feature_dim = len(sample_event["feature"])
                if feature_dim >= 51:
                    return "51D"
                elif feature_dim >= 45:
                    return "45D"
        
        # Default assumption
        return "45D"
    
    def _assess_data_quality(self, events: List[Dict[str, Any]], computed_ohlc: bool, 
                           start_ts: str, end_ts: str) -> str:
        """Assess overall data quality for training suitability"""
        score = 0
        max_score = 10
        
        # Event count (0-3 points)
        n_events = len(events)
        if n_events >= 50:
            score += 3
        elif n_events >= 20:
            score += 2
        elif n_events >= 10:
            score += 1
            
        # OHLC quality (0-2 points) 
        if not computed_ohlc:
            score += 2  # Metadata OHLC is better
        elif computed_ohlc:
            score += 1  # Computed is okay
            
        # Timestamp quality (0-2 points)
        if start_ts and end_ts:
            score += 2
        elif start_ts or end_ts:
            score += 1
            
        # Price data quality (0-2 points)
        has_price_fields = any("price" in event or "close" in event for event in events[:5])
        has_volume = any("volume" in event for event in events[:5])
        
        if has_price_fields and has_volume:
            score += 2
        elif has_price_fields:
            score += 1
            
        # Session range sanity check (0-1 point)
        if events:
            try:
                prices = [float(event.get("price", event.get("close", 0))) for event in events if event.get("price") or event.get("close")]
                if prices:
                    price_range = max(prices) - min(prices)
                    if 1 < price_range < 1000:  # Reasonable range for most instruments
                        score += 1
            except:
                pass
        
        # Quality mapping
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good" 
        elif score >= 4:
            return "fair"
        else:
            return "poor"
    
    def normalize_directory(self, data_dir: Path, symbol: str = None, 
                          timeframe: str = None, quality_filter: str = None) -> pd.DataFrame:
        """Normalize all sessions in directory to DataFrame"""
        logger.info(f"Normalizing sessions in {data_dir}")
        
        # Find session files
        patterns = ["*.json", "**/*.json", "*.parquet", "**/*.parquet"]
        all_files = []
        
        for pattern in patterns:
            all_files.extend(data_dir.glob(pattern))
        
        # Filter by symbol/timeframe
        if symbol or timeframe:
            filtered_files = []
            for f in all_files:
                name_lower = f.name.lower()
                if symbol and symbol.lower() not in name_lower:
                    continue
                if timeframe and timeframe.lower() not in name_lower:
                    continue
                filtered_files.append(f)
            all_files = filtered_files
        
        logger.info(f"Processing {len(all_files)} session files")
        
        # Normalize all files
        all_sessions = []
        for file_path in all_files:
            sessions = self.normalize_file(file_path)
            all_sessions.extend(sessions)
        
        # Convert to DataFrame
        if not all_sessions:
            logger.warning("No valid sessions found")
            return pd.DataFrame()
            
        df = pd.DataFrame([s.to_dict() for s in all_sessions])
        
        # Apply quality filter
        if quality_filter:
            quality_order = {"excellent": 3, "good": 2, "fair": 1, "poor": 0}
            min_quality = quality_order.get(quality_filter, 0)
            df = df[df["data_quality"].map(quality_order) >= min_quality]
        
        # Sort by session_date and start_ts
        df = df.sort_values(["session_date", "start_ts"])
        
        # Report statistics
        logger.info(f"Normalized {len(df)} sessions")
        logger.info(f"Quality distribution: {dict(df['data_quality'].value_counts())}")
        logger.info(f"Symbol distribution: {dict(df['symbol'].value_counts())}")
        logger.info(f"Timeframe distribution: {dict(df['tf'].value_counts())}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Normalize session data for Oracle training")
    parser.add_argument("--data-dir", default="data/enhanced", help="Directory containing session files")
    parser.add_argument("--symbol", help="Filter by symbol (e.g., NQ)")
    parser.add_argument("--tf", "--timeframe", help="Filter by timeframe (e.g., M5)")
    parser.add_argument("--quality", choices=["excellent", "good", "fair", "poor"],
                       help="Minimum quality threshold")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--timezone", default="US/Eastern", help="Default timezone for timestamps")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Run normalization
    normalizer = SessionNormalizer(default_timezone=args.timezone)
    df = normalizer.normalize_directory(
        Path(args.data_dir),
        symbol=args.symbol,
        timeframe=args.tf,
        quality_filter=args.quality
    )
    
    # Save results
    if not df.empty:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"✅ Normalized sessions written to: {args.output}")
        print(f"📊 Sessions: {len(df)}, Quality stats: {dict(df['data_quality'].value_counts())}")
    else:
        print("❌ No valid sessions found to normalize")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())