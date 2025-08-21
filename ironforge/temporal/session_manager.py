#!/usr/bin/env python3
"""
IRONFORGE Temporal Session Data Manager
Handles session loading, caching, and data preprocessing for temporal analysis
"""
import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

class SessionDataManager:
    """Manages session data loading, caching, and preprocessing for temporal analysis"""
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5", adapted_dir: str = "data/adapted"):
        self.shard_dir = shard_dir
        self.adapted_dir = adapted_dir
        self.sessions = {}
        self.graphs = {}
        self.metadata = {}
        self.session_stats = {}  # Store session high/low/open/close for each session
        
        # Session time management handled internally
        
    def load_all_sessions(self):
        """Load all available sessions into memory with price relativity calculations"""
        # Try to load adapted JSON sessions first
        adapted_files = sorted(glob.glob(f"{self.adapted_dir}/adapted_enhanced_rel_*.json"))
        
        if adapted_files:
            print(f"📊 Loading {len(adapted_files)} adapted sessions...")
            self._load_adapted_sessions(adapted_files)
        else:
            # Fallback to parquet shard format
            shard_paths = sorted(glob.glob(f"{self.shard_dir}/shard_*"))
            print(f"📊 Loading {len(shard_paths)} sessions...")
            self._load_parquet_sessions(shard_paths)
            
        print(f"✅ Loaded {len(self.sessions)} sessions with price relativity calculations")
        
    def _load_adapted_sessions(self, adapted_files: List[str]):
        """Load sessions from adapted JSON format"""
        for file_path in adapted_files:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                
                # Extract session ID from filename
                filename = Path(file_path).stem
                session_id = filename.replace('adapted_enhanced_rel_', '')
                
                # Convert to DataFrame
                nodes_data = session_data.get('nodes', [])
                if not nodes_data:
                    continue
                    
                nodes_df = pd.DataFrame(nodes_data)
                
                # Store session data
                self.sessions[session_id] = nodes_df
                self.metadata[session_id] = session_data.get('metadata', {})
                
                # Calculate session statistics
                self._calculate_session_stats(session_id, nodes_df)
                
                print(f"  ✓ Loaded {session_id}: {len(nodes_df)} events")
                
            except Exception as e:
                print(f"  ❌ Error loading {file_path}: {e}")
                continue
                
    def _load_parquet_sessions(self, shard_paths: List[str]):
        """Load sessions from parquet shard format"""
        for shard_path in shard_paths:
            try:
                # Extract session info from path
                session_id = Path(shard_path).name.replace('shard_', '')
                
                # Load nodes and edges
                nodes_path = f"{shard_path}/nodes.parquet"
                edges_path = f"{shard_path}/edges.parquet"
                
                if Path(nodes_path).exists():
                    nodes_df = pd.read_parquet(nodes_path)
                    self.sessions[session_id] = nodes_df
                    
                    # Calculate session statistics
                    self._calculate_session_stats(session_id, nodes_df)
                    
                    # Load graph if edges exist
                    if Path(edges_path).exists():
                        edges_df = pd.read_parquet(edges_path)
                        self.graphs[session_id] = edges_df
                        
                    print(f"  ✓ Loaded {session_id}: {len(nodes_df)} events")
                    
            except Exception as e:
                print(f"  ❌ Error loading {shard_path}: {e}")
                continue
                
    def _calculate_session_stats(self, session_id: str, nodes_df: pd.DataFrame):
        """Calculate session statistics (high, low, open, close)"""
        if 'price' not in nodes_df.columns:
            return
            
        prices = nodes_df['price'].dropna()
        if len(prices) == 0:
            return
            
        self.session_stats[session_id] = {
            'high': float(prices.max()),
            'low': float(prices.min()),
            'open': float(prices.iloc[0]),
            'close': float(prices.iloc[-1]),
            'range': float(prices.max() - prices.min()),
            'event_count': len(nodes_df)
        }
        
    def get_session_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """Get session data by ID"""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Session ID must be a non-empty string")
        return self.sessions.get(session_id)
        
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, float]]:
        """Get session statistics by ID"""
        return self.session_stats.get(session_id)
        
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata by ID"""
        return self.metadata.get(session_id)
        
    def list_sessions(self) -> List[str]:
        """List all available sessions with type information"""
        session_list = []
        for session_id in sorted(self.sessions.keys()):
            stats = self.session_stats.get(session_id, {})
            event_count = stats.get('event_count', 0)
            session_type = self._determine_session_type(session_id)
            
            session_list.append(f"{session_id} ({session_type}) - {event_count} events")
            
        return session_list
        
    def _determine_session_type(self, session_id: str) -> str:
        """Determine session type from session ID"""
        session_id_lower = session_id.lower()
        
        if 'ny_am' in session_id_lower or 'nyam' in session_id_lower:
            return 'NY_AM'
        elif 'ny_pm' in session_id_lower or 'nypm' in session_id_lower:
            return 'NY_PM'
        elif 'london' in session_id_lower:
            return 'LONDON'
        elif 'asia' in session_id_lower:
            return 'ASIA'
        else:
            return 'UNKNOWN'
            
    def get_enhanced_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get complete session information with price relativity analysis"""
        if not isinstance(session_id, str) or not session_id.strip():
            return {"error": "Session ID must be a non-empty string"}

        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
            
        nodes_df = self.sessions[session_id]
        stats = self.session_stats.get(session_id, {})
        metadata = self.metadata.get(session_id, {})
        session_type = self._determine_session_type(session_id)
        
        # Calculate additional statistics
        price_range = stats.get('range', 0)
        event_count = len(nodes_df)
        
        # Time analysis
        time_span = "Unknown"
        if 'timestamp' in nodes_df.columns:
            timestamps = pd.to_datetime(nodes_df['timestamp'])
            time_span = f"{timestamps.min()} to {timestamps.max()}"
            
        return {
            "session_id": session_id,
            "session_type": session_type,
            "event_count": event_count,
            "price_stats": stats,
            "time_span": time_span,
            "metadata": metadata,
            "data_quality": {
                "has_price": 'price' in nodes_df.columns,
                "has_timestamp": 'timestamp' in nodes_df.columns,
                "has_relativity": any(col.startswith('pct_from_') for col in nodes_df.columns),
                "missing_values": nodes_df.isnull().sum().to_dict()
            }
        }
        
    def validate_session_data(self, session_id: str) -> Dict[str, Any]:
        """Validate session data quality and completeness"""
        if session_id not in self.sessions:
            return {"valid": False, "error": f"Session {session_id} not found"}
            
        nodes_df = self.sessions[session_id]
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # Check required columns
        required_columns = ['price', 'timestamp']
        missing_columns = [col for col in required_columns if col not in nodes_df.columns]
        
        if missing_columns:
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            validation_results["valid"] = False
            
        # Check data quality
        if 'price' in nodes_df.columns:
            price_nulls = nodes_df['price'].isnull().sum()
            if price_nulls > 0:
                validation_results["warnings"].append(f"{price_nulls} null price values")
                
        # Check for price relativity columns
        relativity_columns = [col for col in nodes_df.columns if col.startswith('pct_from_')]
        if not relativity_columns:
            validation_results["warnings"].append("No price relativity columns found")
            
        validation_results["statistics"] = {
            "total_events": len(nodes_df),
            "columns": list(nodes_df.columns),
            "relativity_columns": relativity_columns,
            "null_counts": nodes_df.isnull().sum().to_dict()
        }
        
        return validation_results
