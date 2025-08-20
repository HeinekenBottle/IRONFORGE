#!/usr/bin/env python3
"""
Economic Calendar Loader - Phase 5
Pluggable calendar system for real economic news integration
"""

import pandas as pd
import json
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pytz
from dataclasses import dataclass

@dataclass
class EconomicEvent:
    """Normalized economic event structure"""
    event: str
    impact: str  # low, medium, high
    event_time_utc: datetime
    source: str
    
    def to_dict(self) -> Dict:
        return {
            "event": self.event,
            "impact": self.impact,
            "event_time_utc": self.event_time_utc.isoformat(),
            "source": self.source
        }

class CalendarLoader:
    """Pluggable economic calendar loader supporting multiple formats"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'ics']
        self.impact_mappings = {
            # Common impact level variations
            'low': 'low',
            'medium': 'medium', 'med': 'medium', 'moderate': 'medium',
            'high': 'high', 'important': 'high', 'critical': 'high'
        }
        
    def load_calendar(self, file_path: Union[str, Path], format_type: str = 'csv') -> List[EconomicEvent]:
        """Load economic calendar from specified file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Calendar file not found: {file_path}")
            
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
            
        if format_type == 'csv':
            return self._load_csv(file_path)
        elif format_type == 'json':
            return self._load_json(file_path)
        elif format_type == 'ics':
            return self._load_ics(file_path)
    
    def _load_csv(self, file_path: Path) -> List[EconomicEvent]:
        """Load calendar from CSV format"""
        events = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Expected columns: event, impact, datetime, source (optional)
            required_cols = ['event', 'impact', 'datetime']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            for _, row in df.iterrows():
                try:
                    # Parse datetime and ensure UTC
                    event_time = self._parse_datetime(row['datetime'])
                    
                    # Normalize impact level
                    impact = self._normalize_impact(row['impact'])
                    
                    # Get source (default to filename)
                    source = row.get('source', file_path.stem)
                    
                    event = EconomicEvent(
                        event=str(row['event']).strip(),
                        impact=impact,
                        event_time_utc=event_time,
                        source=str(source).strip()
                    )
                    
                    events.append(event)
                    
                except Exception as e:
                    print(f"⚠️  Skipping malformed row: {e}")
                    continue
                    
        except Exception as e:
            raise ValueError(f"Failed to load CSV calendar: {e}")
            
        return events
    
    def _load_json(self, file_path: Path) -> List[EconomicEvent]:
        """Load calendar from JSON format (stub)"""
        # Stub implementation for future expansion
        print(f"📋 JSON loader stub - {file_path}")
        return []
    
    def _load_ics(self, file_path: Path) -> List[EconomicEvent]:
        """Load calendar from ICS format (stub)"""
        # Stub implementation for future expansion  
        print(f"📅 ICS loader stub - {file_path}")
        return []
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string and convert to UTC"""
        # Try common datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                
                # If no timezone info, assume ET and convert to UTC
                if dt.tzinfo is None:
                    et_tz = pytz.timezone('US/Eastern')
                    dt = et_tz.localize(dt)
                    
                # Convert to UTC
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
                
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse datetime: {dt_str}")
    
    def _normalize_impact(self, impact_str: str) -> str:
        """Normalize impact level to low/medium/high"""
        impact_clean = str(impact_str).lower().strip()
        
        return self.impact_mappings.get(impact_clean, 'medium')

class SessionTimeMapper:
    """Maps UTC events to NY session time and determines news buckets"""
    
    def __init__(self):
        self.ny_tz = pytz.timezone('US/Eastern')
        
    def map_event_to_session(self, event: EconomicEvent, session_start_et: datetime, 
                           session_end_et: datetime, session_id: str) -> Dict:
        """Map economic event to session context"""
        
        # Convert session times to UTC for comparison
        session_start_utc = self._et_to_utc(session_start_et)
        session_end_utc = self._et_to_utc(session_end_et)
        
        # Convert event time to ET for session context
        event_time_et = self._utc_to_et(event.event_time_utc)
        
        # Calculate distance from session midpoint
        session_mid_utc = session_start_utc + (session_end_utc - session_start_utc) / 2
        distance_mins = abs((event.event_time_utc - session_mid_utc).total_seconds() / 60)
        
        # Determine news bucket based on impact and distance
        news_bucket = self._get_news_bucket(event.impact, distance_mins)
        
        # Check if event overlaps with session
        session_overlap = (session_start_utc <= event.event_time_utc <= session_end_utc)
        
        return {
            "news_source": event.source,
            "event_time_utc": event.event_time_utc.isoformat(),
            "session_time_et": event_time_et.isoformat(),
            "news_bucket": news_bucket,
            "news_distance_mins": int(distance_mins),
            "session_overlap": session_overlap,
            "event_name": event.event,
            "impact_level": event.impact
        }
    
    def _et_to_utc(self, dt_et: datetime) -> datetime:
        """Convert ET datetime to UTC"""
        if dt_et.tzinfo is None:
            dt_et = self.ny_tz.localize(dt_et)
        return dt_et.astimezone(timezone.utc).replace(tzinfo=None)
    
    def _utc_to_et(self, dt_utc: datetime) -> datetime:
        """Convert UTC datetime to ET"""
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return dt_utc.astimezone(self.ny_tz).replace(tzinfo=None)
    
    def _get_news_bucket(self, impact: str, distance_mins: float) -> str:
        """Determine news bucket based on impact and distance"""
        if impact == 'high' and distance_mins <= 120:
            return 'high±120m'
        elif impact == 'medium' and distance_mins <= 60:
            return 'medium±60m'
        elif impact == 'low' and distance_mins <= 30:
            return 'low±30m'
        else:
            return 'quiet'

class EconomicCalendarIntegrator:
    """Integrates economic calendar with existing IRONFORGE session data"""
    
    def __init__(self):
        self.loader = CalendarLoader()
        self.mapper = SessionTimeMapper()
        
    def integrate_calendar_with_sessions(self, calendar_path: str, 
                                       enhanced_session_dir: str = "/Users/jack/IRONFORGE/data/day_news_enhanced",
                                       format_type: str = 'csv') -> Dict:
        """Integrate real economic calendar with enhanced session data"""
        
        print(f"📅 Loading economic calendar from {calendar_path}")
        
        # Load economic events
        try:
            events = self.loader.load_calendar(calendar_path, format_type)
            print(f"✅ Loaded {len(events)} economic events")
        except Exception as e:
            print(f"❌ Failed to load calendar: {e}")
            return {"error": str(e)}
        
        # Process each enhanced session file
        import glob
        enhanced_files = glob.glob(f"{enhanced_session_dir}/day_news_*.json")
        
        if not enhanced_files:
            return {"error": "No enhanced session files found"}
        
        integration_stats = {
            "processed_sessions": 0,
            "updated_events": 0,
            "news_bucket_distribution": {},
            "source_distribution": {},
            "errors": []
        }
        
        for file_path in enhanced_files:
            try:
                result = self._process_session_file(file_path, events)
                integration_stats["processed_sessions"] += result.get("processed_sessions", 0)
                integration_stats["updated_events"] += result.get("updated_events", 0)
                
                # Aggregate distributions
                for bucket, count in result.get("news_buckets", {}).items():
                    integration_stats["news_bucket_distribution"][bucket] = \
                        integration_stats["news_bucket_distribution"].get(bucket, 0) + count
                        
                for source, count in result.get("sources", {}).items():
                    integration_stats["source_distribution"][source] = \
                        integration_stats["source_distribution"].get(source, 0) + count
                
            except Exception as e:
                integration_stats["errors"].append(f"{Path(file_path).name}: {e}")
                
        return integration_stats
    
    def _process_session_file(self, file_path: str, events: List[EconomicEvent]) -> Dict:
        """Process single enhanced session file with economic events"""
        
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Extract session timing info
        session_info = session_data.get("session_info", {})
        session_start_str = session_info.get("start_time", "")
        session_end_str = session_info.get("end_time", "")
        
        if not session_start_str or not session_end_str:
            return {"error": "Missing session timing info"}
        
        # Parse session times (assume ET format)
        session_start_et = datetime.fromisoformat(session_start_str.replace('Z', ''))
        session_end_et = datetime.fromisoformat(session_end_str.replace('Z', ''))
        session_id = Path(file_path).stem
        
        stats = {
            "processed_sessions": 1,
            "updated_events": 0,
            "news_buckets": {},
            "sources": {}
        }
        
        # Process each event in the session
        events_list = session_data.get("events", [])
        for event in events_list:
            range_position = event.get('range_position', 0.5)
            
            # Only process RD@40 events
            if abs(range_position - 0.40) <= 0.025:
                # Find nearest economic event
                nearest_event, event_context = self._find_nearest_economic_event(
                    events, session_start_et, session_end_et, session_id
                )
                
                if nearest_event:
                    # Update event with real news context
                    event["real_news_context"] = event_context
                    
                    stats["updated_events"] += 1
                    bucket = event_context["news_bucket"]
                    source = event_context["news_source"]
                    
                    stats["news_buckets"][bucket] = stats["news_buckets"].get(bucket, 0) + 1
                    stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Save updated session data
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        return stats
    
    def _find_nearest_economic_event(self, events: List[EconomicEvent], 
                                   session_start_et: datetime, session_end_et: datetime,
                                   session_id: str) -> Tuple[Optional[EconomicEvent], Optional[Dict]]:
        """Find nearest economic event to session with de-duplication"""
        
        if not events:
            return None, None
        
        session_mid_et = session_start_et + (session_end_et - session_start_et) / 2
        session_mid_utc = self.mapper._et_to_utc(session_mid_et)
        
        # Find events within reasonable proximity (6 hours)
        nearby_events = []
        for event in events:
            distance_mins = abs((event.event_time_utc - session_mid_utc).total_seconds() / 60)
            if distance_mins <= 360:  # 6 hours
                nearby_events.append((event, distance_mins))
        
        if not nearby_events:
            return None, None
        
        # Sort by distance, then by impact (high > medium > low)
        impact_priority = {'high': 3, 'medium': 2, 'low': 1}
        nearby_events.sort(key=lambda x: (x[1], -impact_priority.get(x[0].impact, 0)))
        
        # Select nearest (with tie-break by highest impact)
        nearest_event = nearby_events[0][0]
        
        # Generate event context
        event_context = self.mapper.map_event_to_session(
            nearest_event, session_start_et, session_end_et, session_id
        )
        
        return nearest_event, event_context

def create_sample_economic_calendar():
    """Create a sample economic calendar for testing"""
    import os
    
    sample_data = [
        {
            "event": "NFP - Non-Farm Payrolls",
            "impact": "high",
            "datetime": "2025-08-01 08:30:00",
            "source": "BLS"
        },
        {
            "event": "FOMC Rate Decision", 
            "impact": "high",
            "datetime": "2025-08-01 14:00:00",
            "source": "Fed"
        },
        {
            "event": "Initial Jobless Claims",
            "impact": "medium",
            "datetime": "2025-08-01 08:30:00", 
            "source": "DOL"
        },
        {
            "event": "Consumer Confidence",
            "impact": "medium",
            "datetime": "2025-08-01 10:00:00",
            "source": "Conference Board"
        },
        {
            "event": "Building Permits",
            "impact": "low",
            "datetime": "2025-08-01 08:30:00",
            "source": "Census"
        }
    ]
    
    os.makedirs("/Users/jack/IRONFORGE/data/economic_calendar", exist_ok=True)
    
    with open("/Users/jack/IRONFORGE/data/economic_calendar/sample_calendar.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["event", "impact", "datetime", "source"])
        writer.writeheader()
        writer.writerows(sample_data)
    
    print("✅ Created sample economic calendar at /Users/jack/IRONFORGE/data/economic_calendar/sample_calendar.csv")

if __name__ == "__main__":
    # Create sample calendar for testing
    create_sample_economic_calendar()
    
    # Test calendar loading
    integrator = EconomicCalendarIntegrator()
    
    print("\n🧪 Testing Calendar Integration:")
    result = integrator.integrate_calendar_with_sessions(
        "/Users/jack/IRONFORGE/data/economic_calendar/sample_calendar.csv"
    )
    
    print(f"📊 Integration Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")