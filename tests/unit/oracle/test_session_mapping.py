#!/usr/bin/env python3
"""
Test suite for Oracle Session Mapping utilities
"""

import pytest
from pathlib import Path
from datetime import datetime, date
import tempfile
import os

from oracle.session_mapping import (
    SessionMapper, SessionMappingError, 
    normalize_timeframe, resolve_shard_path, is_in_date_range
)


class TestSessionMapper:
    """Test SessionMapper class functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.mapper = SessionMapper(base_shard_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_normalize_timeframe_valid(self):
        """Test valid timeframe normalization"""
        assert self.mapper.normalize_timeframe('5') == 'M5'
        assert self.mapper.normalize_timeframe('15') == 'M15'
        assert self.mapper.normalize_timeframe('M5') == 'M5'
        assert self.mapper.normalize_timeframe('M15') == 'M15'
        assert self.mapper.normalize_timeframe('m5') == 'M5'
        assert self.mapper.normalize_timeframe('m15') == 'M15'
    
    def test_normalize_timeframe_invalid(self):
        """Test invalid timeframe formats"""
        with pytest.raises(SessionMappingError):
            self.mapper.normalize_timeframe('')
        
        with pytest.raises(SessionMappingError):
            self.mapper.normalize_timeframe('abc')
        
        with pytest.raises(SessionMappingError):
            self.mapper.normalize_timeframe(None)
        
        with pytest.raises(SessionMappingError):
            self.mapper.normalize_timeframe('5M')
    
    def test_parse_session_id_valid(self):
        """Test valid session ID parsing"""
        # Standard format
        result = self.mapper.parse_session_id('MIDNIGHT_2025-08-07')
        assert result['session_type'] == 'MIDNIGHT'
        assert result['date'] == '2025-08-07'
        
        # Underscore date format
        result = self.mapper.parse_session_id('NY_AM_2025_08_07')
        assert result['session_type'] == 'NY_AM'
        assert result['date'] == '2025-08-07'
        
        # Other session types
        result = self.mapper.parse_session_id('LONDON_2025-07-24')
        assert result['session_type'] == 'LONDON'
        assert result['date'] == '2025-07-24'
    
    def test_parse_session_id_invalid(self):
        """Test invalid session ID formats"""
        with pytest.raises(SessionMappingError):
            self.mapper.parse_session_id('')
        
        with pytest.raises(SessionMappingError):
            self.mapper.parse_session_id('INVALID_FORMAT')
        
        with pytest.raises(SessionMappingError):
            self.mapper.parse_session_id('UNKNOWN_TYPE_2025-08-07')
        
        with pytest.raises(SessionMappingError):
            self.mapper.parse_session_id('MIDNIGHT_invalid-date')
        
        with pytest.raises(SessionMappingError):
            self.mapper.parse_session_id(None)
    
    def test_build_shard_name(self):
        """Test shard name building"""
        name = self.mapper.build_shard_name('MIDNIGHT', '2025-08-07')
        assert name == 'shard_MIDNIGHT_2025-08-07'
        
        name = self.mapper.build_shard_name('NY_AM', '2025-07-24')
        assert name == 'shard_NY_AM_2025-07-24'
    
    def test_build_shard_name_invalid(self):
        """Test invalid shard name building"""
        with pytest.raises(SessionMappingError):
            self.mapper.build_shard_name('UNKNOWN_TYPE', '2025-08-07')
        
        with pytest.raises(SessionMappingError):
            self.mapper.build_shard_name('MIDNIGHT', 'invalid-date')
    
    def test_resolve_shard_path(self):
        """Test shard path resolution"""
        path = self.mapper.resolve_shard_path('NQ', 'M5', 'MIDNIGHT', '2025-08-07')
        expected = Path(self.temp_dir) / 'NQ_M5' / 'shard_MIDNIGHT_2025-08-07'
        assert path == expected.resolve()
        
        # Test timeframe normalization
        path = self.mapper.resolve_shard_path('NQ', '5', 'MIDNIGHT', '2025-08-07')
        assert path == expected.resolve()
    
    def test_is_in_date_range(self):
        """Test date range checking"""
        assert self.mapper.is_in_date_range('2025-08-01', '2025-07-24', '2025-08-07')
        assert self.mapper.is_in_date_range('2025-07-24', '2025-07-24', '2025-08-07')
        assert self.mapper.is_in_date_range('2025-08-07', '2025-07-24', '2025-08-07')
        
        assert not self.mapper.is_in_date_range('2025-07-23', '2025-07-24', '2025-08-07')
        assert not self.mapper.is_in_date_range('2025-08-08', '2025-07-24', '2025-08-07')
    
    def test_is_in_date_range_invalid(self):
        """Test invalid date range inputs"""
        with pytest.raises(SessionMappingError):
            self.mapper.is_in_date_range('invalid-date', '2025-07-24', '2025-08-07')
        
        with pytest.raises(SessionMappingError):
            self.mapper.is_in_date_range('2025-08-01', 'invalid-date', '2025-08-07')
        
        with pytest.raises(SessionMappingError):
            self.mapper.is_in_date_range('2025-08-01', '2025-07-24', 'invalid-date')
    
    def create_test_shard_structure(self):
        """Create test shard directory structure"""
        base_dir = Path(self.temp_dir) / 'NQ_M5'
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test shard directories
        test_shards = [
            'shard_MIDNIGHT_2025-07-24',
            'shard_MIDNIGHT_2025-07-25',
            'shard_LONDON_2025-07-24',
            'shard_NY_AM_2025-07-24',
            'shard_LUNCH_2025-07-25',
            'shard_ASIA_2025-07-26',  # Outside date range
            'invalid_shard_name',     # Invalid format
        ]
        
        for shard_name in test_shards:
            shard_dir = base_dir / shard_name
            shard_dir.mkdir(exist_ok=True)
            
            # Create basic files to indicate valid shard
            (shard_dir / 'meta.json').touch()
            (shard_dir / 'nodes.parquet').touch()
            (shard_dir / 'edges.parquet').touch()
    
    def test_discover_sessions(self):
        """Test session discovery"""
        self.create_test_shard_structure()
        
        # Discover sessions in date range
        sessions = self.mapper.discover_sessions('NQ', 'M5', '2025-07-24', '2025-07-25')
        
        # Should find 5 valid sessions, excluding out-of-range and invalid
        assert len(sessions) == 5
        
        # Verify session details
        session_ids = {s['session_id'] for s in sessions}
        expected_ids = {
            'MIDNIGHT_2025-07-24', 'MIDNIGHT_2025-07-25',
            'LONDON_2025-07-24', 'NY_AM_2025-07-24',
            'LUNCH_2025-07-25'
        }
        assert session_ids == expected_ids
        
        # Verify sorting (by date then session type)
        dates = [s['date'] for s in sessions]
        assert dates == sorted(dates)
    
    def test_discover_sessions_nonexistent_dir(self):
        """Test session discovery with nonexistent directory"""
        sessions = self.mapper.discover_sessions('INVALID', 'M5', '2025-07-24', '2025-07-25')
        assert sessions == []
    
    def test_get_missing_sessions(self):
        """Test missing session analysis"""
        # Test with sufficient sessions
        found_sessions = [{'date': f'2025-07-{i:02d}'} for i in range(1, 60)]  # 59 sessions
        result = self.mapper.get_missing_sessions(found_sessions, target_count=57)
        assert result['missing_count'] == 0
        assert result['date_gaps'] == []
        
        # Test with insufficient sessions
        found_sessions = [
            {'date': '2025-07-24', 'session_type': 'MIDNIGHT'},
            {'date': '2025-07-25', 'session_type': 'LONDON'},
            {'date': '2025-07-27', 'session_type': 'NY_AM'},  # Gap at 07-26
        ]
        result = self.mapper.get_missing_sessions(found_sessions, target_count=57)
        assert result['missing_count'] == 54
        assert '2025-07-26' in result['date_gaps']
        assert len(result['remediation']) > 0
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty session list
        result = self.mapper.get_missing_sessions([], target_count=57)
        assert result['missing_count'] == 57
        
        # Single session
        sessions = [{'date': '2025-07-24', 'session_type': 'MIDNIGHT'}]
        result = self.mapper.get_missing_sessions(sessions, target_count=2)
        assert result['missing_count'] == 1
        
        # Leap year handling
        assert self.mapper.is_in_date_range('2024-02-29', '2024-02-28', '2024-03-01')
        
        # Year boundary
        assert self.mapper.is_in_date_range('2024-12-31', '2024-12-30', '2025-01-01')


class TestConvenienceFunctions:
    """Test module-level convenience functions"""
    
    def test_normalize_timeframe_function(self):
        """Test normalize_timeframe convenience function"""
        assert normalize_timeframe('5') == 'M5'
        assert normalize_timeframe('M15') == 'M15'
    
    def test_is_in_date_range_function(self):
        """Test is_in_date_range convenience function"""
        assert is_in_date_range('2025-08-01', '2025-07-24', '2025-08-07')
        assert not is_in_date_range('2025-08-08', '2025-07-24', '2025-08-07')
    
    def test_resolve_shard_path_function(self):
        """Test resolve_shard_path convenience function"""
        path = resolve_shard_path('NQ', '5', 'MIDNIGHT', '2025-08-07')
        assert 'NQ_M5' in str(path)
        assert 'shard_MIDNIGHT_2025-08-07' in str(path)


class TestSessionTypes:
    """Test session type validation"""
    
    def test_all_session_types(self):
        """Test all known session types are valid"""
        mapper = SessionMapper()
        
        for session_type in mapper.SESSION_TYPES:
            result = mapper.parse_session_id(f'{session_type}_2025-08-07')
            assert result['session_type'] == session_type
            assert result['date'] == '2025-08-07'
    
    def test_session_type_case_sensitivity(self):
        """Test session types are case sensitive"""
        mapper = SessionMapper()
        
        with pytest.raises(SessionMappingError):
            mapper.parse_session_id('midnight_2025-08-07')  # lowercase
        
        with pytest.raises(SessionMappingError):
            mapper.parse_session_id('Midnight_2025-08-07')  # mixed case


class TestTimeframePatterns:
    """Test timeframe pattern matching"""
    
    def test_numeric_patterns(self):
        """Test numeric timeframe patterns"""
        mapper = SessionMapper()
        
        assert mapper.normalize_timeframe('1') == 'M1'
        assert mapper.normalize_timeframe('5') == 'M5'
        assert mapper.normalize_timeframe('15') == 'M15'
        assert mapper.normalize_timeframe('60') == 'M60'
        assert mapper.normalize_timeframe('240') == 'M240'
    
    def test_m_prefix_patterns(self):
        """Test M-prefixed timeframe patterns"""
        mapper = SessionMapper()
        
        assert mapper.normalize_timeframe('M1') == 'M1'
        assert mapper.normalize_timeframe('M5') == 'M5'
        assert mapper.normalize_timeframe('M15') == 'M15'
        assert mapper.normalize_timeframe('m1') == 'M1'
        assert mapper.normalize_timeframe('m5') == 'M5'
    
    def test_whitespace_handling(self):
        """Test whitespace in timeframe strings"""
        mapper = SessionMapper()
        
        assert mapper.normalize_timeframe(' 5 ') == 'M5'
        assert mapper.normalize_timeframe(' M5 ') == 'M5'
        assert mapper.normalize_timeframe('\t15\n') == 'M15'


if __name__ == '__main__':
    pytest.main([__file__])