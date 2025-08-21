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

from ..core import (
    SESSION_TYPES, COMPILED_TF_PATTERNS, normalize_timeframe,
    SessionMappingError, create_session_mapping_error, handle_oracle_errors
)


class SessionMapper:
    """Deterministic session mapping and validation utilities"""
    
    def __init__(self, base_shard_dir: Union[str, Path] = "data/shards"):
        self.base_shard_dir = Path(base_shard_dir)
        
    @handle_oracle_errors(SessionMappingError, "TIMEFRAME_NORMALIZATION_ERROR")
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
        try:
            return normalize_timeframe(timeframe)
        except ValueError as e:
            raise create_session_mapping_error(
                "timeframe_normalization", 
                str(e),
                {'input_timeframe': timeframe}
            )
    
    @handle_oracle_errors(SessionMappingError, "SESSION_TYPE_VALIDATION_ERROR")
    def validate_session_type(self, session_type: str) -> str:
        """
        Validate and normalize session type
        
        Args:
            session_type: Session type string
            
        Returns:
            Normalized session type
            
        Raises:
            SessionMappingError: If session type is invalid
        """
        normalized = session_type.upper()
        if normalized not in SESSION_TYPES:
            raise create_session_mapping_error(
                "session_type_validation",
                f"Invalid session type: {session_type}",
                {'input_session_type': session_type, 'valid_types': list(SESSION_TYPES)}
            )
        return normalized
    
    @handle_oracle_errors(SessionMappingError, "DATE_VALIDATION_ERROR")
    def validate_date_format(self, date_str: str) -> str:
        """
        Validate date string format (YYYY-MM-DD)
        
        Args:
            date_str: Date string to validate
            
        Returns:
            Validated date string
            
        Raises:
            SessionMappingError: If date format is invalid
        """
        try:
            # Parse and reformat to ensure consistency
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError as e:
            raise create_session_mapping_error(
                "date_validation",
                f"Invalid date format: {date_str} (expected YYYY-MM-DD)",
                {'input_date': date_str}
            )
    
    @handle_oracle_errors(SessionMappingError, "SESSION_ID_PARSING_ERROR")
    def parse_session_id(self, session_id: str) -> Tuple[str, str]:
        """
        Parse session ID to extract session type and date
        
        Args:
            session_id: Session identifier (e.g., 'MIDNIGHT_2025-08-05')
            
        Returns:
            Tuple of (session_type, session_date)
            
        Raises:
            SessionMappingError: If session ID format is invalid
        """
        # Common patterns for session IDs
        patterns = [
            r'^([A-Z_]+)_(\d{4}-\d{2}-\d{2})$',  # MIDNIGHT_2025-08-05
            r'^([A-Z_]+)_(\d{4}\d{2}\d{2})$',    # MIDNIGHT_20250805
            r'^(\d{4}-\d{2}-\d{2})_([A-Z_]+)$',  # 2025-08-05_MIDNIGHT
            r'^(\d{4}\d{2}\d{2})_([A-Z_]+)$'     # 20250805_MIDNIGHT
        ]
        
        for pattern in patterns:
            match = re.match(pattern, session_id)
            if match:
                part1, part2 = match.groups()
                
                # Determine which part is session type and which is date
                if re.match(r'^\d{4}-?\d{2}-?\d{2}$', part1):
                    # part1 is date, part2 is session type
                    date_part = part1
                    session_type = part2
                else:
                    # part1 is session type, part2 is date
                    session_type = part1
                    date_part = part2
                
                # Normalize date format
                if '-' not in date_part:
                    # Convert YYYYMMDD to YYYY-MM-DD
                    date_part = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                
                # Validate components
                validated_session_type = self.validate_session_type(session_type)
                validated_date = self.validate_date_format(date_part)
                
                return validated_session_type, validated_date
        
        raise create_session_mapping_error(
            "session_id_parsing",
            f"Unable to parse session ID: {session_id}",
            {'session_id': session_id, 'supported_patterns': [
                'SESSIONTYPE_YYYY-MM-DD',
                'SESSIONTYPE_YYYYMMDD', 
                'YYYY-MM-DD_SESSIONTYPE',
                'YYYYMMDD_SESSIONTYPE'
            ]}
        )
    
    @handle_oracle_errors(SessionMappingError, "SHARD_NAME_BUILDING_ERROR")
    def build_shard_name(self, session_type: str, session_date: str) -> str:
        """
        Build standardized shard name from session type and date
        
        Args:
            session_type: Session type (e.g., 'MIDNIGHT')
            session_date: Session date in YYYY-MM-DD format
            
        Returns:
            Standardized shard name (e.g., 'shard_MIDNIGHT_2025-08-05')
        """
        validated_session_type = self.validate_session_type(session_type)
        validated_date = self.validate_date_format(session_date)
        
        return f"shard_{validated_session_type}_{validated_date}"
    
    @handle_oracle_errors(SessionMappingError, "SHARD_PATH_RESOLUTION_ERROR")
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
    
    @handle_oracle_errors(SessionMappingError, "SESSION_DISCOVERY_ERROR")
    def discover_sessions(self, symbol: str, timeframe: str, 
                         from_date: str, to_date: str) -> List[str]:
        """
        Discover all available sessions within date range
        
        Args:
            symbol: Trading symbol (e.g., 'NQ')
            timeframe: Timeframe string (e.g., 'M5')
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            List of session IDs found in the specified range
            
        Raises:
            SessionMappingError: If discovery fails
        """
        # Normalize inputs
        normalized_tf = self.normalize_timeframe(timeframe)
        validated_from = self.validate_date_format(from_date)
        validated_to = self.validate_date_format(to_date)
        
        # Build symbol-timeframe directory path
        symbol_tf_dir = self.base_shard_dir / f"{symbol}_{normalized_tf}"
        
        if not symbol_tf_dir.exists():
            return []
        
        discovered_sessions = []
        
        # Scan for shard directories
        for shard_dir in symbol_tf_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            
            # Extract session info from directory name
            if not shard_dir.name.startswith('shard_'):
                continue
            
            try:
                # Parse shard name: shard_SESSIONTYPE_YYYY-MM-DD
                parts = shard_dir.name.split('_')
                if len(parts) < 3:
                    continue
                
                session_type = '_'.join(parts[1:-1])  # Handle multi-part session types
                session_date = parts[-1]
                
                # Validate session type and date
                validated_session_type = self.validate_session_type(session_type)
                validated_session_date = self.validate_date_format(session_date)
                
                # Check if date is within range
                if validated_from <= validated_session_date <= validated_to:
                    session_id = f"{validated_session_type}_{validated_session_date}"
                    discovered_sessions.append(session_id)
                    
            except (ValueError, SessionMappingError):
                # Skip invalid shard directories
                continue
        
        return sorted(discovered_sessions)
    
    def get_session_info(self, session_id: str) -> Dict[str, str]:
        """
        Get detailed information about a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with session information
        """
        try:
            session_type, session_date = self.parse_session_id(session_id)
            
            return {
                'session_id': session_id,
                'session_type': session_type,
                'session_date': session_date,
                'date_obj': datetime.strptime(session_date, "%Y-%m-%d").date().isoformat(),
                'is_valid': True
            }
        except SessionMappingError as e:
            return {
                'session_id': session_id,
                'session_type': None,
                'session_date': None,
                'date_obj': None,
                'is_valid': False,
                'error': str(e)
            }
    
    def list_available_symbols_timeframes(self) -> List[Tuple[str, str]]:
        """
        List all available symbol-timeframe combinations
        
        Returns:
            List of (symbol, timeframe) tuples
        """
        combinations = []
        
        if not self.base_shard_dir.exists():
            return combinations
        
        for symbol_tf_dir in self.base_shard_dir.iterdir():
            if not symbol_tf_dir.is_dir():
                continue
            
            # Parse directory name: SYMBOL_TIMEFRAME
            if '_' in symbol_tf_dir.name:
                parts = symbol_tf_dir.name.rsplit('_', 1)
                if len(parts) == 2:
                    symbol, timeframe = parts
                    combinations.append((symbol, timeframe))
        
        return sorted(combinations)
