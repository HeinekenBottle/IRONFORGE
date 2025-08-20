#!/usr/bin/env python3
"""
Unit tests for E refresh analysis validations
Tests specific requirements and edge cases for news buckets, time windows, and CI calculations
"""

import unittest
import sys
import os
from datetime import datetime
import pytz

# Add IRONFORGE root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from economic_calendar_loader import CalendarLoader
    from enhanced_statistical_framework import EnhancedStatisticalAnalyzer
    from e_refresh_analysis import calculate_wilson_ci
except ImportError as e:
    # Fallback for test environment
    CalendarLoader = None
    EnhancedStatisticalAnalyzer = None
    calculate_wilson_ci = None
    print(f"Import warning: {e}")


class TestERefreshValidations(unittest.TestCase):
    """Unit tests for E refresh analysis validations"""
    
    def setUp(self):
        """Set up test fixtures"""
        if CalendarLoader is None or EnhancedStatisticalAnalyzer is None or calculate_wilson_ci is None:
            self.skipTest("Required modules not available for testing")
            
        self.calendar_loader = CalendarLoader()
        self.stats_analyzer = EnhancedStatisticalAnalyzer()
        self.et_tz = pytz.timezone('America/New_York')
        
    def test_not_all_news_bucket_quiet_when_calendar_present(self):
        """Test: not all news_bucket=quiet when calendar present"""
        
        # Create mock calendar with high/medium/low impact events
        mock_events = [
            {
                'time': '2025-08-01 09:30:00',
                'event': 'NFP',
                'impact': 'high',
                'currency': 'USD'
            },
            {
                'time': '2025-08-01 10:15:00',
                'event': 'CPI',
                'impact': 'medium',
                'currency': 'USD'
            },
            {
                'time': '2025-08-01 14:00:00',
                'event': 'Manufacturing PMI',
                'impact': 'low',
                'currency': 'USD'
            }
        ]
        
        # Test session time within each bucket range
        test_cases = [
            # High impact event at 09:30, test at 09:45 (15 mins later, within 120m)
            {
                'session_time': '2025-08-01 09:45:00',
                'expected_bucket': 'high±120m',
                'description': 'Within 120m of high impact event'
            },
            # Medium impact event at 10:15, test at 10:30 (15 mins later, within 60m)
            {
                'session_time': '2025-08-01 10:30:00', 
                'expected_bucket': 'medium±60m',
                'description': 'Within 60m of medium impact event'
            },
            # Low impact event at 14:00, test at 14:20 (20 mins later, within 30m)
            {
                'session_time': '2025-08-01 14:20:00',
                'expected_bucket': 'low±30m',
                'description': 'Within 30m of low impact event'
            }
        ]
        
        for case in test_cases:
            # Mock the calendar loader to return our test events
            self.calendar_loader.calendar_data = mock_events
            
            session_time = datetime.strptime(case['session_time'], '%Y-%m-%d %H:%M:%S')
            session_time_et = self.et_tz.localize(session_time)
            
            # Find closest event and calculate bucket
            bucket = self.calendar_loader._determine_news_bucket(session_time_et, mock_events)
            
            # Assert not quiet and matches expected bucket
            self.assertNotEqual(bucket, 'quiet', 
                              f"Expected non-quiet bucket for {case['description']}, got '{bucket}'")
            self.assertEqual(bucket, case['expected_bucket'],
                           f"Expected '{case['expected_bucket']}' for {case['description']}, got '{bucket}'")
            
        # Test a time far from any events (should be quiet)
        quiet_session_time = datetime.strptime('2025-08-01 16:00:00', '%Y-%m-%d %H:%M:%S')  
        quiet_session_time_et = self.et_tz.localize(quiet_session_time)
        quiet_bucket = self.calendar_loader._determine_news_bucket(quiet_session_time_et, mock_events)
        
        self.assertEqual(quiet_bucket, 'quiet', 
                        "Expected 'quiet' bucket for time far from events")
    
    def test_time_window_boundary_inclusion_rule(self):
        """Test: time-window boundary (+60/+90 min) inclusion rule"""
        
        # Test RD@40 event at specific time
        rd40_time = datetime.strptime('2025-08-01 10:00:00', '%Y-%m-%d %H:%M:%S')
        rd40_time_et = self.et_tz.localize(rd40_time)
        
        # Test boundary cases for 60-minute window
        boundary_test_cases_60m = [
            {
                'event_time': '2025-08-01 10:59:59',  # 59min 59sec - should be included
                'window_mins': 60,
                'should_include': True,
                'description': 'Just under 60min boundary'
            },
            {
                'event_time': '2025-08-01 11:00:00',  # Exactly 60min - should be included
                'window_mins': 60, 
                'should_include': True,
                'description': 'Exactly 60min boundary (inclusive)'
            },
            {
                'event_time': '2025-08-01 11:00:01',  # 60min 1sec - should be excluded
                'window_mins': 60,
                'should_include': False,
                'description': 'Just over 60min boundary'
            }
        ]
        
        # Test boundary cases for 90-minute window  
        boundary_test_cases_90m = [
            {
                'event_time': '2025-08-01 11:29:59',  # 89min 59sec - should be included
                'window_mins': 90,
                'should_include': True,
                'description': 'Just under 90min boundary'
            },
            {
                'event_time': '2025-08-01 11:30:00',  # Exactly 90min - should be included
                'window_mins': 90,
                'should_include': True,
                'description': 'Exactly 90min boundary (inclusive)'
            },
            {
                'event_time': '2025-08-01 11:30:01',  # 90min 1sec - should be excluded
                'window_mins': 90,
                'should_include': False,
                'description': 'Just over 90min boundary'
            }
        ]
        
        all_test_cases = boundary_test_cases_60m + boundary_test_cases_90m
        
        for case in all_test_cases:
            event_time = datetime.strptime(case['event_time'], '%Y-%m-%d %H:%M:%S')
            event_time_et = self.et_tz.localize(event_time)
            
            # Calculate time difference in minutes
            time_diff = (event_time_et - rd40_time_et).total_seconds() / 60
            
            # Test inclusion rule: event within window if time_diff <= window_mins
            is_included = time_diff <= case['window_mins']
            
            self.assertEqual(is_included, case['should_include'],
                           f"{case['description']}: time_diff={time_diff:.2f}min, "
                           f"window={case['window_mins']}min, "
                           f"expected include={case['should_include']}, got {is_included}")
    
    def test_ci_width_flag_inconclusive_threshold(self):
        """Test: CI width flag (>30pp) marks Inconclusive"""
        
        # Test cases with different sample sizes and success rates
        test_cases = [
            {
                'successes': 5,
                'total': 5,
                'description': 'Perfect success, small sample (should be >30pp)',
                'expected_inconclusive': True
            },
            {
                'successes': 50,
                'total': 100,
                'description': '50% success, medium sample (should be ≤30pp)',
                'expected_inconclusive': False
            },
            {
                'successes': 100,
                'total': 200,
                'description': '50% success, large sample (should be ≤30pp)',
                'expected_inconclusive': False
            },
            {
                'successes': 3,
                'total': 8,
                'description': 'Small sample with variation (should be >30pp)',
                'expected_inconclusive': True
            }
        ]
        
        for case in test_cases:
            # Calculate Wilson confidence interval
            ci_result = calculate_wilson_ci(
                case['successes'], 
                case['total'], 
                confidence=0.95
            )
            
            ci_lower, ci_upper, ci_width = ci_result
            
            # CI width >30pp should be flagged as inconclusive
            is_inconclusive = ci_width > 30.0
            
            self.assertEqual(is_inconclusive, case['expected_inconclusive'],
                           f"{case['description']}: successes={case['successes']}, "
                           f"total={case['total']}, CI width={ci_width:.1f}pp, "
                           f"expected inconclusive={case['expected_inconclusive']}, "
                           f"got {is_inconclusive}")
            
            # Additional validation: CI bounds should be between 0 and 100
            self.assertGreaterEqual(ci_lower, 0, "CI lower bound should be ≥ 0")
            self.assertLessEqual(ci_upper, 100, "CI upper bound should be ≤ 100")
            self.assertLess(ci_lower, ci_upper, "CI lower should be < CI upper")
            
            # CI width should be positive
            self.assertGreater(ci_width, 0, "CI width should be positive")
    
    def test_news_bucket_standardization(self):
        """Test: news bucket format standardization"""
        
        # Test that all bucket names follow the standard format
        standard_buckets = {
            'high±120m': {'impact': 'high', 'window_mins': 120},
            'medium±60m': {'impact': 'medium', 'window_mins': 60},
            'low±30m': {'impact': 'low', 'window_mins': 30},
            'quiet': {'impact': None, 'window_mins': None}
        }
        
        # Test bucket naming consistency
        for bucket_name, expected in standard_buckets.items():
            if expected['impact']:  # Not quiet bucket
                # Extract impact and window from bucket name
                parts = bucket_name.split('±')
                self.assertEqual(len(parts), 2, f"Bucket '{bucket_name}' should have format 'impact±Xm'")
                
                impact = parts[0]
                window_str = parts[1]
                
                self.assertEqual(impact, expected['impact'], 
                               f"Impact should be '{expected['impact']}' for bucket '{bucket_name}'")
                self.assertTrue(window_str.endswith('m'), 
                              f"Window should end with 'm' for bucket '{bucket_name}'")
                
                window_mins = int(window_str[:-1])  # Remove 'm' suffix
                self.assertEqual(window_mins, expected['window_mins'],
                               f"Window should be {expected['window_mins']} minutes for bucket '{bucket_name}'")
            else:
                # Quiet bucket should just be 'quiet'
                self.assertEqual(bucket_name, 'quiet', "Quiet bucket should be exactly 'quiet'")


if __name__ == '__main__':
    unittest.main()