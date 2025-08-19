"""
Simple data loader for Oracle training demonstration

Uses existing IRONFORGE shard data to create training sessions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple data loader for Oracle training demonstrations"""
    
    def __init__(self, shard_dir: Path = Path("data")):
        """
        Initialize simple data loader
        
        Args:
            shard_dir: Directory containing session shard data
        """
        self.shard_dir = Path(shard_dir)
        logger.info(f"SimpleDataLoader initialized with shard_dir: {shard_dir}")
    
    def load_recent_sessions(
        self, 
        symbol: str = "NQ_M5", 
        limit: int = 50, 
        enhanced_format: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load recent session data for training
        
        Args:
            symbol: Trading symbol (e.g., NQ_M5)
            limit: Maximum number of sessions to load
            enhanced_format: Return enhanced session format
            
        Returns:
            List of session dictionaries
        """
        logger.info(f"Loading recent sessions for {symbol} (limit={limit})")
        
        # Look for existing shard files
        pattern = f"*{symbol}*"
        shard_files = list(self.shard_dir.rglob("*.json"))
        
        if not shard_files:
            logger.warning(f"No shard files found in {self.shard_dir}")
            # Return mock data for demonstration
            return self._create_mock_sessions(symbol, min(limit, 5))
        
        sessions = []
        for shard_file in shard_files[:limit]:
            try:
                with open(shard_file, 'r') as f:
                    session_data = json.load(f)
                
                # Ensure session has required format
                if enhanced_format:
                    session_data = self._enhance_session_format(session_data, shard_file.stem)
                
                sessions.append(session_data)
                
            except Exception as e:
                logger.warning(f"Failed to load {shard_file}: {e}")
        
        logger.info(f"Loaded {len(sessions)} sessions")
        return sessions
    
    def _enhance_session_format(self, session_data: Dict[str, Any], session_name: str) -> Dict[str, Any]:
        """Enhance session data with required fields for Oracle training"""
        
        if not isinstance(session_data, dict):
            session_data = {"raw_data": session_data}
        
        # Add required metadata
        enhanced = {
            "session_name": session_name,
            "timestamp": session_data.get("timestamp", "2025-08-19T12:00:00"),
            "events": session_data.get("events", []),
            "metadata": session_data.get("metadata", {}),
            **session_data  # Preserve original data
        }
        
        # Ensure events exist - create from any numerical data if needed
        if not enhanced["events"] and "data" in session_data:
            enhanced["events"] = self._convert_data_to_events(session_data["data"])
        
        return enhanced
    
    def _convert_data_to_events(self, data: Any) -> List[Dict[str, Any]]:
        """Convert raw data to event format"""
        events = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (int, float)):
                    events.append({
                        "index": i,
                        "price": float(item),
                        "volume": 100,
                        "timestamp": f"2025-08-19T12:{i:02d}:00"
                    })
                elif isinstance(item, dict):
                    events.append(item)
        
        return events
    
    def _create_mock_sessions(self, symbol: str, count: int) -> List[Dict[str, Any]]:
        """Create mock session data for demonstration"""
        logger.info(f"Creating {count} mock sessions for {symbol}")
        
        import random
        import numpy as np
        
        sessions = []
        base_price = 18500.0  # NQ base price
        
        for i in range(count):
            # Generate realistic price movements
            n_events = random.randint(20, 50)
            session_range = random.uniform(50, 200)
            
            prices = []
            current_price = base_price + random.uniform(-500, 500)
            
            for j in range(n_events):
                # Random walk with some trending
                change = random.gauss(0, session_range / 10)
                current_price += change
                prices.append(current_price)
            
            # Create events
            events = []
            for j, price in enumerate(prices):
                events.append({
                    "index": j,
                    "price": price,
                    "volume": random.randint(50, 200),
                    "timestamp": f"2025-08-19T{9 + j // 60}:{j % 60:02d}:00"
                })
            
            session = {
                "session_name": f"mock_{symbol}_{i:03d}",
                "timestamp": f"2025-08-19T09:00:00",
                "symbol": symbol,
                "events": events,
                "metadata": {
                    "high": max(prices),
                    "low": min(prices),
                    "range": max(prices) - min(prices),
                    "n_events": len(events),
                    "mock": True
                }
            }
            
            sessions.append(session)
        
        return sessions