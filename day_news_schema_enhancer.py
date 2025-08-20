#!/usr/bin/env python3
"""
Day & News Schema Enhancer
Adds day-of-week profiles and economic news context to RD@40 analysis
"""

import json
import glob
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd

class DayNewsSchemaEnhancer:
    """Enhances session data with day profiles and news context for Experiment E"""
    
    def __init__(self):
        self.day_profiles = {
            'Monday': {
                'profile_name': 'gap_fill_bias',
                'description': 'Gap fill bias - expect more MR patterns',
                'expected_mr_bias': 0.15,  # +15% bias toward MR
                'expected_accel_bias': -0.10,  # -10% bias away from ACCEL
                'characteristics': ['gap_resolution', 'mean_reversion_tendency', 'low_momentum']
            },
            'Tuesday': {
                'profile_name': 'trend_continuation', 
                'description': 'Trend continuation - expect more ACCEL patterns',
                'expected_mr_bias': -0.10,
                'expected_accel_bias': 0.20,  # +20% bias toward ACCEL
                'characteristics': ['momentum_building', 'trend_following', 'breakout_tendency']
            },
            'Wednesday': {
                'profile_name': 'balanced',
                'description': 'Balanced distribution - roughly even MR/ACCEL',
                'expected_mr_bias': 0.0,
                'expected_accel_bias': 0.0,
                'characteristics': ['neutral_bias', 'mixed_signals', 'range_bound_tendency']
            },
            'Thursday': {
                'profile_name': 'reversal_setup',
                'description': 'Reversal setup - late-day MR patterns',
                'expected_mr_bias': 0.12,
                'expected_accel_bias': -0.08,
                'characteristics': ['position_unwinding', 'reversal_anticipation', 'late_day_mr']
            },
            'Friday': {
                'profile_name': 'profit_taking',
                'description': 'Profit taking - early ACCEL, later MR',
                'expected_mr_bias': 0.08,  # Mixed: early ACCEL bias, late MR bias
                'expected_accel_bias': 0.05,
                'characteristics': ['early_momentum', 'late_profit_taking', 'week_end_effects']
            }
        }
        
        self.news_impact_classifications = {
            'HIGH': {
                'window_minutes': 120,  # ±120 minutes
                'events': ['FOMC', 'NFP', 'CPI', 'Fed Speech', 'ECB Decision', 'BOE Decision'],
                'expected_accel_bias': 0.25,  # Strong bias toward ACCEL
                'expected_mr_bias': -0.15,
                'volatility_multiplier': 2.5
            },
            'MEDIUM': {
                'window_minutes': 60,   # ±60 minutes  
                'events': ['PMI', 'Housing Data', 'Retail Sales', 'GDP', 'Unemployment Claims'],
                'expected_accel_bias': 0.10,
                'expected_mr_bias': -0.05,
                'volatility_multiplier': 1.5
            },
            'LOW': {
                'window_minutes': 30,   # ±30 minutes
                'events': ['Minor Indicators', 'Regional Fed', 'Consumer Sentiment'],
                'expected_accel_bias': 0.05,
                'expected_mr_bias': 0.0,
                'volatility_multiplier': 1.1
            }
        }
        
        # Sample economic calendar for testing (real implementation would use external feed)
        self.sample_news_calendar = self._create_sample_calendar()
    
    def _create_sample_calendar(self) -> List[Dict]:
        """Create sample economic calendar for testing purposes"""
        # This simulates major economic events during our data period
        return [
            {
                'date': '2025-07-24',
                'time_et': '08:30:00',
                'event': 'Initial Jobless Claims',
                'impact': 'MEDIUM',
                'currency': 'USD',
                'session_context': 'PREMARKET'
            },
            {
                'date': '2025-07-25',
                'time_et': '14:00:00', 
                'event': 'FOMC Minutes',
                'impact': 'HIGH',
                'currency': 'USD',
                'session_context': 'NYPM'
            },
            {
                'date': '2025-07-29',
                'time_et': '08:30:00',
                'event': 'GDP Preliminary',
                'impact': 'MEDIUM', 
                'currency': 'USD',
                'session_context': 'PREMARKET'
            },
            {
                'date': '2025-07-30',
                'time_et': '14:00:00',
                'event': 'Fed Chair Powell Speech',
                'impact': 'HIGH',
                'currency': 'USD',
                'session_context': 'NYPM'
            },
            {
                'date': '2025-08-05',
                'time_et': '08:30:00',
                'event': 'Non-Farm Payrolls',
                'impact': 'HIGH',
                'currency': 'USD',
                'session_context': 'PREMARKET'
            },
            {
                'date': '2025-08-06',
                'time_et': '10:00:00',
                'event': 'Consumer Price Index',
                'impact': 'HIGH',
                'currency': 'USD',
                'session_context': 'NYAM'
            }
        ]
    
    def extract_session_date(self, filename: str) -> Optional[datetime]:
        """Extract trading date from session filename"""
        # Pattern: adapted_enhanced_rel_SESSION_Lvl-1_YYYY_MM_DD.json
        pattern = r'(\d{4})_(\d{2})_(\d{2})'
        match = re.search(pattern, filename)
        
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                return None
        return None
    
    def get_day_profile(self, session_date: datetime) -> Dict[str, Any]:
        """Get day profile for a given trading date"""
        day_name = session_date.strftime('%A')
        
        if day_name in self.day_profiles:
            profile = self.day_profiles[day_name].copy()
            profile['day_of_week'] = day_name
            profile['trading_week_position'] = self._get_week_position(day_name)
            profile['session_date'] = session_date.strftime('%Y-%m-%d')
            return profile
        
        # Default for weekends (shouldn't occur in trading data)
        return {
            'day_of_week': day_name,
            'profile_name': 'unknown',
            'description': 'Non-trading day',
            'expected_mr_bias': 0.0,
            'expected_accel_bias': 0.0,
            'characteristics': ['non_trading'],
            'trading_week_position': 'unknown',
            'session_date': session_date.strftime('%Y-%m-%d')
        }
    
    def _get_week_position(self, day_name: str) -> str:
        """Get position within trading week"""
        positions = {
            'Monday': 'first_day',
            'Tuesday': 'early_week', 
            'Wednesday': 'mid_week',
            'Thursday': 'late_week',
            'Friday': 'last_day'
        }
        return positions.get(day_name, 'unknown')
    
    def find_news_proximity(self, session_date: datetime, event_timestamp: str, session_name: str) -> Dict[str, Any]:
        """Find nearest news event and calculate proximity"""
        session_datetime_str = f"{session_date.strftime('%Y-%m-%d')} {event_timestamp}"
        
        try:
            event_datetime = datetime.strptime(session_datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Return quiet classification for unparseable timestamps
            return self._get_quiet_classification()
        
        # Find closest news event
        closest_news = None
        min_distance = float('inf')
        
        for news_event in self.sample_news_calendar:
            news_date = datetime.strptime(news_event['date'], '%Y-%m-%d')
            news_datetime_str = f"{news_event['date']} {news_event['time_et']}"
            news_datetime = datetime.strptime(news_datetime_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate time difference in minutes
            time_diff = abs((event_datetime - news_datetime).total_seconds() / 60)
            
            if time_diff < min_distance:
                min_distance = time_diff
                closest_news = news_event
        
        if closest_news is None:
            return self._get_quiet_classification()
        
        # Determine if within impact window
        impact_config = self.news_impact_classifications[closest_news['impact']]
        window_minutes = impact_config['window_minutes']
        
        if min_distance <= window_minutes:
            # Within impact window
            return {
                'news_bucket': f"{closest_news['impact'].lower()}±{window_minutes}m",
                'news_distance_mins': min_distance,
                'news_source': 'sample_calendar',
                'news_event': closest_news['event'],
                'news_impact': closest_news['impact'],
                'event_time_utc': news_datetime.isoformat() + 'Z',  # Assuming ET = UTC for simplicity
                'session_time_et': event_timestamp,
                'session_overlap': self._check_session_overlap(session_name, event_timestamp),
                'expected_accel_bias': impact_config['expected_accel_bias'],
                'expected_mr_bias': impact_config['expected_mr_bias'],
                'volatility_multiplier': impact_config['volatility_multiplier']
            }
        else:
            # Outside impact window - quiet period
            return self._get_quiet_classification()
    
    def _get_quiet_classification(self) -> Dict[str, Any]:
        """Get quiet period classification"""
        return {
            'news_bucket': 'quiet',
            'news_distance_mins': None,
            'news_source': 'sample_calendar',
            'news_event': 'No major news nearby',
            'news_impact': 'QUIET',
            'event_time_utc': None,
            'session_time_et': None,
            'session_overlap': False,
            'expected_accel_bias': 0.0,
            'expected_mr_bias': 0.05,  # Slight bias toward MR in quiet periods
            'volatility_multiplier': 0.8
        }
    
    def _check_session_overlap(self, session_name: str, timestamp: str) -> bool:
        """Check if event occurs during London↔NY overlap window"""
        # London-NY overlap typically 08:00-12:00 ET
        try:
            event_time = datetime.strptime(timestamp, '%H:%M:%S').time()
            overlap_start = datetime.strptime('08:00:00', '%H:%M:%S').time()
            overlap_end = datetime.strptime('12:00:00', '%H:%M:%S').time()
            
            return overlap_start <= event_time <= overlap_end
        except ValueError:
            return False
    
    def enhance_session_metadata(self, session_file_path: str) -> Dict[str, Any]:
        """Enhance session with day profile and news context metadata"""
        try:
            with open(session_file_path, 'r') as f:
                session_data = json.load(f)
            
            # Extract session date from filename
            session_date = self.extract_session_date(session_file_path)
            if session_date is None:
                return session_data  # Return unchanged if date parsing fails
            
            session_name = session_file_path.split('/')[-1].replace('adapted_', '').replace('.json', '')
            
            # Add day profile
            day_profile = self.get_day_profile(session_date)
            
            # Enhance events with news context for RD@40 candidates
            enhanced_events = []
            
            for event in session_data.get('events', []):
                enhanced_event = event.copy()
                
                # Check if event is near RD@40 zone (±2.5% tolerance)
                range_position = event.get('range_position', 0.5)
                if abs(range_position - 0.40) <= 0.025:
                    # This is an RD@40 candidate - add news context
                    timestamp = event.get('timestamp', '12:00:00')
                    news_context = self.find_news_proximity(session_date, timestamp, session_name)
                    enhanced_event['news_context'] = news_context
                
                enhanced_events.append(enhanced_event)
            
            # Create enhanced session data
            enhanced_session = {
                **session_data,
                'events': enhanced_events,
                'day_profile': day_profile,
                'session_metadata': {
                    'original_filename': session_file_path.split('/')[-1],
                    'session_name': session_name,
                    'trading_date': session_date.strftime('%Y-%m-%d'),
                    'enhancement_timestamp': datetime.now().isoformat(),
                    'enhancement_version': 'day_news_v1.0'
                }
            }
            
            return enhanced_session
            
        except Exception as e:
            print(f"Error enhancing {session_file_path}: {e}")
            # Return original data if enhancement fails
            try:
                with open(session_file_path, 'r') as f:
                    return json.load(f)
            except:
                return {'events': [], 'error': str(e)}
    
    def enhance_all_sessions(self, input_dir: str = "/Users/jack/IRONFORGE/data/adapted", 
                           output_dir: str = "/Users/jack/IRONFORGE/data/day_news_enhanced") -> Dict[str, Any]:
        """Enhance all session files with day and news context"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        session_files = glob.glob(f"{input_dir}/adapted_*.json")
        enhancement_summary = {
            'total_files': len(session_files),
            'enhanced_files': 0,
            'failed_files': 0,
            'rd40_events_enhanced': 0,
            'day_profile_distribution': defaultdict(int),
            'news_bucket_distribution': defaultdict(int),
            'errors': []
        }
        
        print(f"🔄 Enhancing {len(session_files)} session files with day & news context...")
        
        for session_file in session_files:
            try:
                enhanced_data = self.enhance_session_metadata(session_file)
                
                # Count enhancements
                if 'day_profile' in enhanced_data:
                    enhancement_summary['enhanced_files'] += 1
                    day_profile = enhanced_data['day_profile']['profile_name']
                    enhancement_summary['day_profile_distribution'][day_profile] += 1
                    
                    # Count RD@40 events with news context
                    for event in enhanced_data.get('events', []):
                        if 'news_context' in event:
                            enhancement_summary['rd40_events_enhanced'] += 1
                            news_bucket = event['news_context']['news_bucket']
                            enhancement_summary['news_bucket_distribution'][news_bucket] += 1
                
                # Save enhanced file
                output_filename = session_file.split('/')[-1].replace('adapted_', 'day_news_')
                output_path = f"{output_dir}/{output_filename}"
                
                with open(output_path, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                
            except Exception as e:
                enhancement_summary['failed_files'] += 1
                enhancement_summary['errors'].append(f"{session_file}: {str(e)}")
                print(f"❌ Failed to enhance {session_file}: {e}")
        
        # Print summary
        print(f"\n📊 ENHANCEMENT SUMMARY:")
        print(f"   Total files: {enhancement_summary['total_files']}")
        print(f"   Enhanced files: {enhancement_summary['enhanced_files']}")
        print(f"   Failed files: {enhancement_summary['failed_files']}")
        print(f"   RD@40 events enhanced: {enhancement_summary['rd40_events_enhanced']}")
        
        print(f"\n📅 DAY PROFILE DISTRIBUTION:")
        for profile, count in enhancement_summary['day_profile_distribution'].items():
            print(f"   {profile}: {count} sessions")
        
        print(f"\n📰 NEWS BUCKET DISTRIBUTION:")
        for bucket, count in enhancement_summary['news_bucket_distribution'].items():
            print(f"   {bucket}: {count} RD@40 events")
        
        # Save summary
        summary_path = f"{output_dir}/enhancement_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(dict(enhancement_summary), f, indent=2)
        
        print(f"\n💾 Enhancement summary saved to: {summary_path}")
        
        return dict(enhancement_summary)

def main():
    """Main enhancement function"""
    print("🚀 IRONFORGE: Day & News Schema Enhancement")
    print("🎯 Adding day profiles and economic news context to RD@40 analysis")
    print("=" * 80)
    
    enhancer = DayNewsSchemaEnhancer()
    summary = enhancer.enhance_all_sessions()
    
    print(f"\n✅ Schema Enhancement Complete!")
    print(f"Enhanced {summary['enhanced_files']} sessions with day & news context")
    
    return summary

if __name__ == "__main__":
    main()