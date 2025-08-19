#!/usr/bin/env python3
"""
Session Mapping Utilities - Deterministic TF/timezone/session-ID handling

Provides standardized mapping between various session representations used across
the Oracle training pipeline.
"""

import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class SessionMappingError(Exception):
    """Raised when session mapping operations fail"""
    pass


class SessionMapper:
    """Deterministic session mapping and validation utilities"""
    
    # Standard session types found in IRONFORGE data
    SESSION_TYPES = {
        'MIDNIGHT', 'PREMARKET', 'LONDON', 'NY', 'NY_AM', 'NY_PM', 
        'LUNCH', 'ASIA', 'PREASIA', 'NYAM', 'NYPM'
    }
    
    # Timeframe normalization patterns
    TF_PATTERNS = {
        r'^\d+$': lambda x: f'M{x}',  # '5' -> 'M5'
        r'^M\d+$': lambda x: x,       # 'M5' -> 'M5'
        r'^m\d+$': lambda x: x.upper() # 'm5' -> 'M5'
    }
    
    def __init__(self, base_shard_dir: Union[str, Path] = "data/shards"):
        self.base_shard_dir = Path(base_shard_dir)
        
    def normalize_timeframe(self, timeframe: str) -> str:
        """
        Normalize timeframe to standard format (M5, M15, etc.)
        
        Args:
            timeframe: Input timeframe ('5', 'M5', 'm5', etc.)
            
        Returns:
            Normalized timeframe string (e.g., 'M5')
            
        Raises:
            SessionMappingError: If timeframe format is invalid
        """
        if not timeframe or not isinstance(timeframe, str):
            raise SessionMappingError(f"Invalid timeframe: {timeframe}")
        
        timeframe = timeframe.strip()
        
        for pattern, normalizer in self.TF_PATTERNS.items():
            if re.match(pattern, timeframe):
                return normalizer(timeframe)
        
        raise SessionMappingError(f"Unrecognized timeframe format: {timeframe}")
    
    def parse_session_id(self, session_id: str) -> Dict[str, str]:
        """
        Parse session ID into components
        
        Args:
            session_id: Session identifier (e.g., 'MIDNIGHT_2025-08-07')
            
        Returns:
            Dict with 'session_type', 'date' keys
            
        Raises:
            SessionMappingError: If session ID format is invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise SessionMappingError(f"Invalid session ID: {session_id}")
        
        # Handle various session ID formats
        patterns = [
            r'^([A-Z_]+)_(\d{4}-\d{2}-\d{2})$',  # MIDNIGHT_2025-08-07
            r'^([A-Z_]+)_(\d{4}_\d{2}_\d{2})$',  # MIDNIGHT_2025_08_07  
        ]
        
        for pattern in patterns:
            match = re.match(pattern, session_id)
            if match:
                session_type, date_str = match.groups()
                
                # Normalize date format to YYYY-MM-DD
                normalized_date = date_str.replace('_', '-')
                
                # Validate session type
                if session_type not in self.SESSION_TYPES:
                    raise SessionMappingError(f"Unknown session type: {session_type}")
                
                # Validate date format
                try:
                    datetime.strptime(normalized_date, '%Y-%m-%d')
                except ValueError:
                    raise SessionMappingError(f"Invalid date format in session ID: {date_str}")
                
                return {
                    'session_type': session_type,
                    'date': normalized_date
                }
        
        raise SessionMappingError(f"Unrecognized session ID format: {session_id}")
    
    def build_shard_name(self, session_type: str, session_date: str) -> str:
        """
        Build shard directory name from components
        
        Args:
            session_type: Session type (e.g., 'MIDNIGHT')
            session_date: Session date in YYYY-MM-DD format
            
        Returns:
            Shard directory name (e.g., 'shard_MIDNIGHT_2025-08-07')
        """
        if session_type not in self.SESSION_TYPES:
            raise SessionMappingError(f"Unknown session type: {session_type}")
        
        # Validate date format
        try:
            datetime.strptime(session_date, '%Y-%m-%d')
        except ValueError:
            raise SessionMappingError(f"Invalid date format: {session_date}")
        
        return f"shard_{session_type}_{session_date}"
    
    def resolve_shard_path(self, symbol: str, timeframe: str, 
                          session_type: str, session_date: str) -> Path:
        """
        Resolve full path to shard directory
        
        Args:
            symbol: Trading symbol (e.g., 'NQ')
            timeframe: Timeframe string (e.g., '5' or 'M5')
            session_type: Session type (e.g., 'MIDNIGHT')
            session_date: Session date in YYYY-MM-DD format
            
        Returns:
            Absolute path to shard directory
            
        Raises:
            SessionMappingError: If path components are invalid
        """
        # Normalize inputs
        normalized_tf = self.normalize_timeframe(timeframe)
        shard_name = self.build_shard_name(session_type, session_date)
        
        # Build path: data/shards/{SYMBOL}_{TF}/{shard_name}/
        symbol_tf_dir = f"{symbol}_{normalized_tf}"
        full_path = self.base_shard_dir / symbol_tf_dir / shard_name
        
        return full_path.resolve()
    
    def is_in_date_range(self, session_date: str, from_date: str, to_date: str) -> bool:
        """
        Check if session date falls within specified range (inclusive)
        
        Args:
            session_date: Session date in YYYY-MM-DD format
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            True if session_date is within [from_date, to_date]
            
        Raises:
            SessionMappingError: If any date format is invalid
        """
        try:
            session_dt = datetime.strptime(session_date, '%Y-%m-%d').date()
            from_dt = datetime.strptime(from_date, '%Y-%m-%d').date()
            to_dt = datetime.strptime(to_date, '%Y-%m-%d').date()
        except ValueError as e:
            raise SessionMappingError(f"Invalid date format: {e}")
        
        return from_dt <= session_dt <= to_dt
    
    def discover_sessions(self, symbol: str, timeframe: str, 
                         from_date: str, to_date: str) -> List[Dict[str, str]]:
        """
        Discover all available sessions matching criteria
        
        Args:
            symbol: Trading symbol (e.g., 'NQ')
            timeframe: Timeframe string (e.g., '5' or 'M5')
            from_date: Start date in YYYY-MM-DD format  
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of session info dicts with keys: session_id, session_type, 
            date, shard_path, shard_exists
        """
        normalized_tf = self.normalize_timeframe(timeframe)
        symbol_tf_dir = self.base_shard_dir / f"{symbol}_{normalized_tf}"
        
        sessions = []
        
        if not symbol_tf_dir.exists():
            return sessions
        
        # Find all shard_* directories
        for shard_dir in symbol_tf_dir.iterdir():
            if not shard_dir.is_dir() or not shard_dir.name.startswith('shard_'):
                continue
            
            try:
                # Extract session info from directory name
                # Format: shard_SESSIONTYPE_YYYY-MM-DD
                # Handle session types with underscores (e.g., NY_AM)
                name_parts = shard_dir.name.split('_')
                if len(name_parts) < 3:
                    continue
                
                # First part is always "shard"
                if name_parts[0] != 'shard':
                    continue
                
                # Last part is always the date (YYYY-MM-DD format)
                session_date = name_parts[-1]
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', session_date):
                    continue
                
                # Middle parts form the session type
                session_type = '_'.join(name_parts[1:-1])
                
                # Validate and filter by date range
                if not self.is_in_date_range(session_date, from_date, to_date):
                    continue
                
                session_id = f"{session_type}_{session_date}"
                
                sessions.append({
                    'session_id': session_id,
                    'session_type': session_type,
                    'date': session_date,
                    'shard_path': str(shard_dir),
                    'shard_exists': shard_dir.exists()
                })
                
            except (ValueError, SessionMappingError):
                # Skip malformed directory names
                continue
        
        # Sort by date then session type for deterministic ordering
        sessions.sort(key=lambda x: (x['date'], x['session_type']))
        
        return sessions
    
    def get_missing_sessions(self, found_sessions: List[Dict], target_count: int = 57) -> Dict[str, List[str]]:
        """
        Analyze missing sessions and provide remediation guidance
        
        Args:
            found_sessions: List of discovered sessions from discover_sessions()
            target_count: Target number of sessions (default 57)
            
        Returns:
            Dict with 'missing_count', 'date_gaps', 'remediation' keys
        """
        if len(found_sessions) >= target_count:
            return {'missing_count': 0, 'date_gaps': [], 'remediation': []}
        
        missing_count = target_count - len(found_sessions)
        
        # Analyze date gaps
        dates = sorted(set(s['date'] for s in found_sessions))
        date_gaps = []
        
        if dates:
            start_date = datetime.strptime(dates[0], '%Y-%m-%d').date()
            end_date = datetime.strptime(dates[-1], '%Y-%m-%d').date()
            
            # Check for gaps in date range
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in dates:
                    date_gaps.append(date_str)
                current_date = current_date.replace(day=current_date.day + 1)
        
        # Generate remediation guidance
        remediation = []
        
        if date_gaps:
            remediation.append(f"Fill date gaps: {', '.join(date_gaps[:5])}" + 
                             (f" (and {len(date_gaps)-5} more)" if len(date_gaps) > 5 else ""))
        
        if missing_count > len(date_gaps):
            remediation.append(f"Extend date range to capture {missing_count - len(date_gaps)} additional sessions")
        
        # Session type analysis
        session_types = set(s['session_type'] for s in found_sessions)
        missing_types = self.SESSION_TYPES - session_types
        if missing_types:
            remediation.append(f"Check for missing session types: {', '.join(sorted(missing_types))}")
        
        return {
            'missing_count': missing_count,
            'date_gaps': date_gaps,
            'remediation': remediation
        }


# Convenience functions for backward compatibility
def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe string"""
    mapper = SessionMapper()
    return mapper.normalize_timeframe(tf)


def resolve_shard_path(symbol: str, tf: str, session_type: str, session_date: str) -> Path:
    """Resolve shard path"""
    mapper = SessionMapper()
    return mapper.resolve_shard_path(symbol, tf, session_type, session_date)


def is_in_date_range(session_date: str, from_date: str, to_date: str) -> bool:
    """Check if date is in range"""
    mapper = SessionMapper()
    return mapper.is_in_date_range(session_date, from_date, to_date)