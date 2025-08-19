#!/usr/bin/env python3
"""
Session Schema Auditor for Oracle Training

Scans session data files and reports:
- Column schemas and data types
- Row counts and time spans  
- Data quality issues
- Missing required fields

Usage:
    python scripts/audit_sessions.py --data-dir data/enhanced --symbol NQ --tf M5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class SessionAuditor:
    """Audit session files for Oracle training compatibility"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.audit_results = []
        
    def audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit individual session file"""
        try:
            logger.debug(f"Auditing {file_path}")
            
            # Load session data
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_path.suffix == '.parquet':
                data = pd.read_parquet(file_path).to_dict('records')
            else:
                return {"file": str(file_path), "error": "Unsupported file format"}
            
            # Extract basic info
            result = {
                "file": str(file_path.name),
                "full_path": str(file_path),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "format": file_path.suffix[1:],
                "timestamp": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            # Analyze session structure
            if isinstance(data, dict):
                result.update(self._analyze_session_dict(data))
            elif isinstance(data, list):
                result.update(self._analyze_session_list(data))
            else:
                result["error"] = f"Unexpected data type: {type(data)}"
                
            return result
            
        except Exception as e:
            return {
                "file": str(file_path.name), 
                "error": str(e),
                "exception_type": type(e).__name__
            }
    
    def _analyze_session_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single session dictionary"""
        result = {"session_type": "dict"}
        
        # Core session fields
        result["has_session_name"] = "session_name" in data
        result["has_timestamp"] = "timestamp" in data
        result["has_events"] = "events" in data
        result["has_metadata"] = "metadata" in data
        
        # Events analysis
        events = data.get("events", [])
        result["event_count"] = len(events)
        
        if events:
            result.update(self._analyze_events(events))
            
        # Metadata analysis  
        metadata = data.get("metadata", {})
        if metadata:
            result["metadata_keys"] = list(metadata.keys())
            result["has_ohlc"] = all(k in metadata for k in ["high", "low"])
            if "high" in metadata and "low" in metadata:
                result["session_range"] = metadata["high"] - metadata["low"]
        
        # Time span analysis
        if "timestamp" in data:
            try:
                result["session_timestamp"] = data["timestamp"]
                result["timestamp_format"] = "iso" if "T" in str(data["timestamp"]) else "unknown"
            except:
                result["timestamp_error"] = "Could not parse timestamp"
                
        return result
    
    def _analyze_session_list(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze list of sessions or events"""
        result = {"session_type": "list", "item_count": len(data)}
        
        if not data:
            return result
            
        # Sample first item to understand structure
        sample = data[0]
        if isinstance(sample, dict):
            result["sample_keys"] = list(sample.keys())
            
            # Check if it's events or sessions
            if "price" in sample or "timestamp" in sample:
                result["appears_to_be"] = "events_list"
                result.update(self._analyze_events(data))
            elif "session_name" in sample or "events" in sample:
                result["appears_to_be"] = "sessions_list"
                result["session_count"] = len(data)
            else:
                result["appears_to_be"] = "unknown_list"
        
        return result
    
    def _analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze event structure for Oracle training compatibility"""
        if not events:
            return {"events_analysis": "empty"}
            
        result = {}
        
        # Sample event structure
        sample_event = events[0]
        result["event_sample_keys"] = list(sample_event.keys())
        
        # Check for required fields
        required_fields = ["price", "timestamp", "volume"]
        optional_fields = ["index", "high", "low", "close"]
        
        result["has_price_field"] = any(k in sample_event for k in ["price", "close", "price_close"])
        result["has_timestamp_field"] = "timestamp" in sample_event
        result["has_volume_field"] = "volume" in sample_event
        result["has_index_field"] = "index" in sample_event
        
        # Price analysis
        price_values = []
        timestamps = []
        
        for event in events[:min(100, len(events))]:  # Sample first 100 events
            # Extract price
            price = None
            for price_key in ["price", "close", "price_close"]:
                if price_key in event:
                    price = event[price_key]
                    break
                    
            if price is not None:
                try:
                    price_values.append(float(price))
                except:
                    pass
                    
            # Extract timestamp
            if "timestamp" in event:
                timestamps.append(event["timestamp"])
        
        if price_values:
            result["price_range"] = {"min": min(price_values), "max": max(price_values)}
            result["price_span"] = max(price_values) - min(price_values)
            
        if timestamps:
            result["timestamp_sample"] = timestamps[:3]
            result["unique_timestamps"] = len(set(timestamps))
            
        return result
    
    def audit_directory(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Audit all session files in directory"""
        logger.info(f"Starting audit of {self.data_dir}")
        
        # Find session files
        patterns = [
            "*.json",
            "**/*.json", 
            "*.parquet",
            "**/*.parquet"
        ]
        
        all_files = []
        for pattern in patterns:
            all_files.extend(self.data_dir.glob(pattern))
        
        # Filter by symbol/timeframe if specified
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
        
        logger.info(f"Found {len(all_files)} files to audit")
        
        # Audit each file
        file_results = []
        for file_path in all_files[:50]:  # Limit to prevent overwhelming output
            result = self.audit_file(file_path)
            file_results.append(result)
            
        # Generate summary
        summary = self._generate_summary(file_results)
        
        return {
            "audit_timestamp": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "filter_symbol": symbol,
            "filter_timeframe": timeframe,
            "total_files_found": len(all_files),
            "files_audited": len(file_results),
            "summary": summary,
            "files": file_results
        }
    
    def _generate_summary(self, file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate audit summary statistics"""
        summary = {
            "total_files": len(file_results),
            "successful_audits": 0,
            "errors": 0,
            "formats": {},
            "session_types": {},
            "total_events": 0,
            "files_with_ohlc": 0,
            "files_with_timestamps": 0,
            "files_with_price_data": 0,
            "quality_issues": []
        }
        
        for result in file_results:
            if "error" in result:
                summary["errors"] += 1
                continue
                
            summary["successful_audits"] += 1
            
            # Format tracking
            fmt = result.get("format", "unknown")
            summary["formats"][fmt] = summary["formats"].get(fmt, 0) + 1
            
            # Session type tracking
            session_type = result.get("session_type", "unknown")
            summary["session_types"][session_type] = summary["session_types"].get(session_type, 0) + 1
            
            # Event counting
            event_count = result.get("event_count", 0)
            summary["total_events"] += event_count
            
            # Quality checks
            if result.get("has_ohlc"):
                summary["files_with_ohlc"] += 1
                
            if result.get("has_timestamp_field"):
                summary["files_with_timestamps"] += 1
                
            if result.get("has_price_field"):
                summary["files_with_price_data"] += 1
            
            # Quality issues
            if event_count == 0:
                summary["quality_issues"].append(f"{result['file']}: No events")
            elif event_count < 10:
                summary["quality_issues"].append(f"{result['file']}: Very few events ({event_count})")
                
            if not result.get("has_price_field"):
                summary["quality_issues"].append(f"{result['file']}: No price field")
                
            if not result.get("has_timestamp_field"):
                summary["quality_issues"].append(f"{result['file']}: No timestamp field")
        
        # Calculate percentages
        if summary["successful_audits"] > 0:
            summary["ohlc_coverage"] = summary["files_with_ohlc"] / summary["successful_audits"]
            summary["timestamp_coverage"] = summary["files_with_timestamps"] / summary["successful_audits"] 
            summary["price_coverage"] = summary["files_with_price_data"] / summary["successful_audits"]
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Audit session files for Oracle training compatibility")
    parser.add_argument("--data-dir", default="data", help="Directory containing session files")
    parser.add_argument("--symbol", help="Filter files by symbol (e.g., NQ)")
    parser.add_argument("--tf", "--timeframe", help="Filter files by timeframe (e.g., M5)")
    parser.add_argument("--output", help="Output file for audit results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Run audit
    auditor = SessionAuditor(Path(args.data_dir))
    results = auditor.audit_directory(symbol=args.symbol, timeframe=args.tf)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Audit results written to: {args.output}")
    else:
        print(json.dumps(results["summary"], indent=2))
    
    # Print key findings
    summary = results["summary"]
    print(f"\n📊 Audit Summary:")
    print(f"  Files audited: {summary['successful_audits']}/{summary['total_files']}")
    print(f"  Total events: {summary['total_events']:,}")
    print(f"  Price data coverage: {summary.get('price_coverage', 0):.1%}")
    print(f"  Timestamp coverage: {summary.get('timestamp_coverage', 0):.1%}")
    print(f"  OHLC coverage: {summary.get('ohlc_coverage', 0):.1%}")
    
    if summary["quality_issues"]:
        print(f"\n⚠️  Quality Issues ({len(summary['quality_issues'])}):")
        for issue in summary["quality_issues"][:10]:  # Show first 10
            print(f"    {issue}")
        if len(summary["quality_issues"]) > 10:
            print(f"    ... and {len(summary['quality_issues']) - 10} more")
    
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())