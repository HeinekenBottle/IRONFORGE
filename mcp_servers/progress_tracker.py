#!/usr/bin/env python3
"""
IRONFORGE Development Progress Tracker MCP Server
Tracks code development progress, new files, and script creation
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("IRONFORGE Progress Tracker")

# Database setup
DB_PATH = Path("/Users/jack/IRONFORGE/.progress_tracker.db")

def init_database():
    """Initialize SQLite database for progress tracking"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            file_path TEXT,
            description TEXT,
            metadata TEXT,
            importance INTEGER DEFAULT 1
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            created_timestamp TEXT NOT NULL,
            file_type TEXT,
            purpose TEXT,
            lines_of_code INTEGER DEFAULT 0,
            last_modified TEXT
        )
    """)
    
    conn.commit()
    conn.close()

@mcp.tool()
def track_progress_event(
    event_type: str,
    description: str,
    file_path: str = None,
    importance: int = 1,
    metadata: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Track a development progress event
    
    Args:
        event_type: Type of event (file_created, feature_implemented, bug_fixed, etc.)
        description: Human-readable description of the progress
        file_path: Optional path to related file
        importance: Importance level 1-5 (1=minor, 5=major breakthrough)
        metadata: Additional structured data about the event
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO progress_events 
        (timestamp, event_type, file_path, description, metadata, importance)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        event_type,
        file_path,
        description,
        json.dumps(metadata) if metadata else None,
        importance
    ))
    conn.commit()
    conn.close()
    
    return {
        "status": "tracked",
        "event_type": event_type,
        "description": description,
        "timestamp": datetime.now().isoformat()
    }

@mcp.tool()
def track_new_file(
    file_path: str,
    file_type: str,
    purpose: str,
    lines_of_code: int = 0
) -> Dict[str, str]:
    """
    Track creation of a new file
    
    Args:
        file_path: Full path to the new file
        file_type: Type of file (python, config, data, etc.)
        purpose: Purpose/description of the file
        lines_of_code: Initial lines of code count
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO file_tracking 
            (file_path, created_timestamp, file_type, purpose, lines_of_code, last_modified)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file_path,
            datetime.now().isoformat(),
            file_type,
            purpose,
            lines_of_code,
            datetime.now().isoformat()
        ))
        conn.commit()
    except sqlite3.IntegrityError:
        # File already exists, update it
        conn.execute("""
            UPDATE file_tracking 
            SET last_modified = ?, lines_of_code = ?, purpose = ?
            WHERE file_path = ?
        """, (
            datetime.now().isoformat(),
            lines_of_code,
            purpose,
            file_path
        ))
        conn.commit()
    
    conn.close()
    
    # Also track as progress event
    track_progress_event(
        event_type="file_created",
        description=f"Created {file_type} file: {purpose}",
        file_path=file_path,
        importance=2,
        metadata={"file_type": file_type, "lines_of_code": lines_of_code}
    )
    
    return {
        "status": "tracked",
        "file_path": file_path,
        "file_type": file_type,
        "purpose": purpose
    }

@mcp.tool()
def get_progress_summary(days: int = 7) -> Dict[str, Any]:
    """
    Get development progress summary for the last N days
    
    Args:
        days: Number of days to look back (default: 7)
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get recent progress events
    events = conn.execute("""
        SELECT timestamp, event_type, file_path, description, importance
        FROM progress_events 
        WHERE datetime(timestamp) >= datetime('now', '-{} days')
        ORDER BY timestamp DESC
    """.format(days)).fetchall()
    
    # Get new files created
    new_files = conn.execute("""
        SELECT file_path, created_timestamp, file_type, purpose, lines_of_code
        FROM file_tracking 
        WHERE datetime(created_timestamp) >= datetime('now', '-{} days')
        ORDER BY created_timestamp DESC
    """.format(days)).fetchall()
    
    # Get statistics
    stats = {
        "total_events": len(events),
        "high_importance_events": len([e for e in events if e[4] >= 4]),
        "files_created": len(new_files),
        "total_lines_added": sum(f[4] for f in new_files)
    }
    
    conn.close()
    
    return {
        "period_days": days,
        "statistics": stats,
        "recent_events": [
            {
                "timestamp": e[0],
                "event_type": e[1],
                "file_path": e[2],
                "description": e[3],
                "importance": e[4]
            } for e in events
        ],
        "new_files": [
            {
                "file_path": f[0],
                "created": f[1],
                "file_type": f[2],
                "purpose": f[3],
                "lines_of_code": f[4]
            } for f in new_files
        ]
    }

@mcp.tool()
def get_file_inventory() -> Dict[str, Any]:
    """Get complete inventory of tracked files"""
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    files = conn.execute("""
        SELECT file_path, created_timestamp, file_type, purpose, lines_of_code, last_modified
        FROM file_tracking 
        ORDER BY created_timestamp DESC
    """).fetchall()
    conn.close()
    
    return {
        "total_files": len(files),
        "files": [
            {
                "file_path": f[0],
                "created": f[1],
                "file_type": f[2],
                "purpose": f[3],
                "lines_of_code": f[4],
                "last_modified": f[5]
            } for f in files
        ]
    }

if __name__ == "__main__":
    mcp.run()