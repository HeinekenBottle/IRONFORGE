#!/usr/bin/env python3
"""
Oracle Data Module
Data processing, auditing, and session management for Oracle training system
"""

from .audit import OracleAuditor
from .session_mapping import SessionMapper

__all__ = [
    'OracleAuditor',
    'SessionMapper'
]

__version__ = "1.0.2"
__description__ = "Oracle data processing and auditing utilities"
