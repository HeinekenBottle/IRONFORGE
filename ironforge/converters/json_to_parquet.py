"""
Enhanced JSON to Parquet Shard Converter
========================================

Converts IRONFORGE enhanced session JSON files to Parquet shards suitable for TGAT discovery.
Implements the specification for transforming event-based JSON into nodes/edges graph format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from dataclasses import dataclass

from ironforge.data_engine.parquet_writer import write_nodes, write_edges

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for JSON to Parquet conversion."""
    source_glob: str = "data/enhanced/enhanced_*_Lvl-1_*.json"
    symbol: str = "NQ"
    timeframe: str = "M5"
    source_timezone: str = "ET"
    target_timezone: str = "UTC"
    pack_mode: str = "single"  # single session per shard
    overwrite: bool = False
    dry_run: bool = False


class TimeProcessor:
    """Handles timestamp conversion from session-local to UTC milliseconds."""
    
    def __init__(self, source_tz: str = "ET", target_tz: str = "UTC"):
        # ET maps to America/New_York (handles EST/EDT automatically)
        self.source_tz = pytz.timezone("America/New_York" if source_tz == "ET" else source_tz)
        self.target_tz = pytz.timezone(target_tz)
    
    def convert_session_time(self, date_str: str, time_str: str) -> int:
        """Convert session date + time to UTC milliseconds."""
        try:
            # Parse date and time
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            
            # Combine and localize
            dt = datetime.combine(date_obj, time_obj)
            localized = self.source_tz.localize(dt)
            utc_dt = localized.astimezone(self.target_tz)
            
            # Convert to milliseconds
            return int(utc_dt.timestamp() * 1000)
        except Exception as e:
            logger.error(f"Time conversion failed for {date_str} {time_str}: {e}")
            return 0


class FeatureExtractor:
    """Extracts 45D node features and 20D edge features from session events."""
    
    def extract_node_features(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> np.ndarray:
        """Extract 45D feature vector for a node."""
        features = np.full(45, np.nan, dtype=np.float32)
        
        # Semantic features (f0-f7): 8D
        features[0] = self._get_fvg_redelivery_flag(event, session_context)
        features[1] = self._get_expansion_phase_flag(event, session_context)
        features[2] = self._get_consolidation_flag(event, session_context)
        features[3] = self._get_retracement_flag(event, session_context)
        features[4] = self._get_reversal_flag(event, session_context)
        features[5] = self._get_liq_sweep_flag(event, session_context)
        features[6] = self._get_pd_array_interaction_flag(event, session_context)
        features[7] = 0.0  # semantic_reserved
        
        # Traditional features (f8-f44): 37D
        features[8] = float(event.get("price_level", 0.0))  # raw price
        features[9] = self._get_movement_type_code(event.get("movement_type"))
        features[10] = self._get_session_relative_time(event, session_context)
        features[11] = self._get_price_relativity(event, session_context)
        
        # Fill remaining traditional features with basic computations
        for i in range(12, 45):
            features[i] = 0.0  # Placeholder for additional traditional features
            
        return features
    
    def extract_edge_features(self, src_event: Dict[str, Any], dst_event: Dict[str, Any], 
                            etype: int, session_context: Dict[str, Any]) -> np.ndarray:
        """Extract 20D feature vector for an edge."""
        features = np.full(20, np.nan, dtype=np.float32)
        
        # Semantic relationship features (e0-e2): 3D
        features[0] = 1.0 if etype in [1, 2] else 0.0  # semantic_event_link
        features[1] = self._get_event_causality(src_event, dst_event, session_context)
        features[2] = float(etype)  # relationship_type
        
        # Traditional edge features (e3-e19): 17D
        src_price = float(src_event.get("price_level", 0.0))
        dst_price = float(dst_event.get("price_level", 0.0))
        
        features[3] = abs(dst_price - src_price)  # price_delta
        features[4] = 1.0 if dst_price > src_price else -1.0  # direction
        features[5] = self._get_movement_similarity(src_event, dst_event)
        
        # Fill remaining traditional features
        for i in range(6, 20):
            features[i] = 0.0
            
        return features
    
    def _get_fvg_redelivery_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event relates to FVG redelivery."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "fpfvg" in movement_type or "rebalance" in movement_type else 0.0
    
    def _get_expansion_phase_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event occurs during expansion phase."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "expansion" in movement_type or "break" in movement_type else 0.0
    
    def _get_consolidation_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event occurs during consolidation."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "consolidation" in movement_type or "range" in movement_type else 0.0
    
    def _get_retracement_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event is a retracement."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "retracement" in movement_type or "pullback" in movement_type else 0.0
    
    def _get_reversal_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event indicates reversal."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "reversal" in movement_type or movement_type in ["session_high", "session_low"] else 0.0
    
    def _get_liq_sweep_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event involves liquidity sweep."""
        movement_type = event.get("movement_type", "")
        event_type = event.get("event_type", "")
        return 1.0 if "sweep" in movement_type or "liquidity" in event_type else 0.0
    
    def _get_pd_array_interaction_flag(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Check if event interacts with premium/discount arrays."""
        movement_type = event.get("movement_type", "")
        return 1.0 if "premium" in movement_type or "discount" in movement_type else 0.0
    
    def _get_movement_type_code(self, movement_type: Optional[str]) -> float:
        """Convert movement type to numeric code."""
        if not movement_type:
            return 0.0
        
        type_codes = {
            "open": 1.0, "close": 2.0, "session_high": 3.0, "session_low": 4.0,
            "fpfvg_formation_premium": 5.0, "fpfvg_formation_discount": 6.0,
            "rebalance": 7.0, "expansion": 8.0, "consolidation": 9.0
        }
        return type_codes.get(movement_type, 0.0)
    
    def _get_session_relative_time(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Get normalized position within session (0.0 to 1.0)."""
        # This would require session start/end times - placeholder for now
        return 0.5
    
    def _get_price_relativity(self, event: Dict[str, Any], session_context: Dict[str, Any]) -> float:
        """Get price relative to session range."""
        # This would require session high/low calculation - placeholder
        return 0.5
    
    def _get_event_causality(self, src_event: Dict[str, Any], dst_event: Dict[str, Any], 
                           session_context: Dict[str, Any]) -> float:
        """Estimate causal relationship strength between events."""
        src_type = src_event.get("movement_type", "")
        dst_type = dst_event.get("movement_type", "")
        
        # Simple heuristic: related movement types have higher causality
        if src_type == dst_type:
            return 0.8
        elif "fpfvg" in src_type and "rebalance" in dst_type:
            return 0.9
        else:
            return 0.3
    
    def _get_movement_similarity(self, src_event: Dict[str, Any], dst_event: Dict[str, Any]) -> float:
        """Calculate movement type similarity."""
        src_type = src_event.get("movement_type", "")
        dst_type = dst_event.get("movement_type", "")
        return 1.0 if src_type == dst_type else 0.0


class NodeIDManager:
    """Manages global unique node IDs across sessions."""
    
    def __init__(self, id_counter_path: str = "data/shards/_id_counter.json"):
        self.id_counter_path = Path(id_counter_path)
        self.current_id = self._load_counter()
        
    def _load_counter(self) -> int:
        """Load the current ID counter from file."""
        if self.id_counter_path.exists():
            try:
                with open(self.id_counter_path, 'r') as f:
                    data = json.load(f)
                    return data.get("next_id", 1)
            except Exception as e:
                logger.warning(f"Failed to load ID counter: {e}")
        return 1
    
    def _save_counter(self) -> None:
        """Save the current ID counter to file."""
        try:
            self.id_counter_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.id_counter_path, 'w') as f:
                json.dump({"next_id": self.current_id}, f)
        except Exception as e:
            logger.error(f"Failed to save ID counter: {e}")
    
    def get_next_ids(self, count: int) -> List[int]:
        """Get the next N unique IDs."""
        ids = list(range(self.current_id, self.current_id + count))
        self.current_id += count
        self._save_counter()
        return ids


class JSONToParquetConverter:
    """Main converter class for transforming enhanced JSON sessions to Parquet shards."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.time_processor = TimeProcessor(config.source_timezone, config.target_timezone)
        self.feature_extractor = FeatureExtractor()
        self.node_id_manager = NodeIDManager()
        
    def convert_session(self, json_path: Path) -> Optional[Path]:
        """Convert a single enhanced JSON session to Parquet shard."""
        logger.info(f"Converting session: {json_path}")
        
        try:
            # Load session data
            with open(json_path, 'r') as f:
                session_data = json.load(f)
            
            # Extract session metadata
            session_info = self._extract_session_info(json_path, session_data)
            if not session_info:
                logger.warning(f"Could not extract session info from {json_path}")
                return None
            
            # Create output directory
            shard_dir = self._get_shard_directory(session_info)
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would create {shard_dir}")
                return shard_dir
            
            if shard_dir.exists() and not self.config.overwrite:
                logger.info(f"Shard already exists (use --overwrite): {shard_dir}")
                return shard_dir
                
            shard_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert events to nodes and edges
            nodes_df, edges_df, node_map_df = self._convert_events_to_graph(
                session_data, session_info
            )
            
            # Validate data
            if not self._validate_dataframes(nodes_df, edges_df):
                logger.error(f"Validation failed for {json_path}")
                return None
            
            # Write Parquet files
            write_nodes(nodes_df, str(shard_dir / "nodes.parquet"))
            write_edges(edges_df, str(shard_dir / "edges.parquet"))
            
            # Write node mapping for traceability
            node_map_df.to_parquet(shard_dir / "node_map.parquet", compression="zstd")
            
            # Write metadata
            self._write_metadata(shard_dir, session_info, json_path, 
                               len(nodes_df), len(edges_df))
            
            logger.info(f"✅ Converted {json_path.name} -> {shard_dir.name} "
                       f"({len(nodes_df)} nodes, {len(edges_df)} edges)")
            
            return shard_dir
            
        except Exception as e:
            logger.error(f"Failed to convert {json_path}: {e}")
            return None
    
    def _extract_session_info(self, json_path: Path, session_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract session information from filename and metadata."""
        try:
            # Parse filename: enhanced_<SESSION_TYPE>_Lvl-1_YYYY_MM_DD.json
            filename = json_path.stem
            if filename.startswith("enhanced_rel_"):
                filename = filename[13:]  # Remove "enhanced_rel_" prefix
            elif filename.startswith("enhanced_"):
                filename = filename[9:]   # Remove "enhanced_" prefix
            else:
                logger.warning(f"Unexpected filename format: {json_path.name}")
                return None
            
            parts = filename.split('_')
            if len(parts) < 4:
                logger.warning(f"Could not parse filename: {json_path.name}")
                return None
            
            session_type = parts[0]
            date_parts = parts[-3:]  # Last 3 parts should be YYYY, MM, DD
            date_str = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
            
            # Validate date format
            datetime.strptime(date_str, "%Y-%m-%d")
            
            return {
                "session_type": session_type,
                "date": date_str,
                "session_id": f"{session_type}_{date_str}"
            }
            
        except Exception as e:
            logger.error(f"Failed to extract session info from {json_path}: {e}")
            return None
    
    def _get_shard_directory(self, session_info: Dict[str, str]) -> Path:
        """Generate shard directory path."""
        symbol_tf = f"{self.config.symbol}_{self.config.timeframe}"
        shard_name = f"shard_{session_info['session_type']}_{session_info['date']}"
        return Path(f"data/shards/{symbol_tf}/{shard_name}")
    
    def _convert_events_to_graph(self, session_data: Dict[str, Any], session_info: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert session events to nodes and edges DataFrames."""
        
        # Extract events from session
        price_movements = session_data.get("price_movements", [])
        liquidity_events = session_data.get("session_liquidity_events", [])
        
        all_events = []
        
        # Process price movements
        for event in price_movements:
            if "timestamp" in event and "price_level" in event:
                event_copy = event.copy()
                event_copy["source_type"] = "price_movement"
                all_events.append(event_copy)
        
        # Process liquidity events
        for event in liquidity_events:
            if "timestamp" in event:
                event_copy = event.copy()
                event_copy["source_type"] = "liquidity_event"
                # Use price_level from nearby price movement if not present
                if "price_level" not in event_copy:
                    event_copy["price_level"] = self._estimate_price_for_liquidity_event(
                        event, price_movements
                    )
                all_events.append(event_copy)
        
        if not all_events:
            logger.warning(f"No events found in session {session_info['session_id']}")
            # Return empty DataFrames with correct schemas
            return self._create_empty_dataframes()
        
        # Convert timestamps and sort
        date_str = session_info["date"]
        for event in all_events:
            event["t"] = self.time_processor.convert_session_time(
                date_str, event["timestamp"]
            )
        
        # Sort by timestamp and remove invalid times
        all_events = [e for e in all_events if e["t"] > 0]
        all_events.sort(key=lambda x: x["t"])
        
        if not all_events:
            logger.warning(f"No valid timestamps in session {session_info['session_id']}")
            return self._create_empty_dataframes()
        
        # Generate global node IDs
        node_ids = self.node_id_manager.get_next_ids(len(all_events))
        
        # Create nodes DataFrame
        nodes_data = []
        node_map_data = []
        
        for i, (event, node_id) in enumerate(zip(all_events, node_ids)):
            # Extract features
            features = self.feature_extractor.extract_node_features(event, session_data)
            
            # Determine node kind
            kind = self._get_node_kind(event)
            
            # Build node record
            node_record = {
                "node_id": node_id,
                "t": event["t"],
                "kind": kind,
                "session_id": session_info["session_id"],
                "symbol": self.config.symbol,
                "tf": self.config.timeframe,
                "price": float(event.get("price_level", 0.0))
            }
            
            # Add feature columns
            for j, feature_val in enumerate(features):
                node_record[f"f{j}"] = feature_val
            
            nodes_data.append(node_record)
            
            # Build node mapping record
            node_map_data.append({
                "session_id": session_info["session_id"],
                "local_idx": i,
                "node_id": node_id,
                "t": event["t"],
                "timestamp": event["timestamp"],
                "source_type": event["source_type"]
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        node_map_df = pd.DataFrame(node_map_data)
        
        # Create edges DataFrame
        edges_df = self._create_edges(all_events, node_ids, session_data)
        
        return nodes_df, edges_df, node_map_df
    
    def _estimate_price_for_liquidity_event(self, liq_event: Dict[str, Any], 
                                           price_movements: List[Dict[str, Any]]) -> float:
        """Estimate price for liquidity event from nearby price movements."""
        liq_time = liq_event["timestamp"]
        
        # Find closest price movement by time
        closest_price = 0.0
        min_time_diff = float('inf')
        
        for pm in price_movements:
            if "timestamp" in pm and "price_level" in pm:
                time_diff = abs(self._time_difference_seconds(liq_time, pm["timestamp"]))
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_price = float(pm["price_level"])
        
        return closest_price
    
    def _time_difference_seconds(self, time1: str, time2: str) -> float:
        """Calculate time difference in seconds between HH:MM:SS strings."""
        try:
            t1 = datetime.strptime(time1, "%H:%M:%S")
            t2 = datetime.strptime(time2, "%H:%M:%S")
            return (t2 - t1).total_seconds()
        except:
            return float('inf')
    
    def _get_node_kind(self, event: Dict[str, Any]) -> int:
        """Determine node kind enum value."""
        source_type = event.get("source_type", "")
        movement_type = event.get("movement_type", "")
        
        if source_type == "price_movement":
            return 0  # price_move
        elif source_type == "liquidity_event":
            return 1  # liq_event
        elif movement_type in ["session_high", "session_low", "open", "close"]:
            return 2  # anchor
        else:
            return 3  # other
    
    def _create_edges(self, events: List[Dict[str, Any]], node_ids: List[int], 
                     session_data: Dict[str, Any]) -> pd.DataFrame:
        """Create edges DataFrame from events."""
        edges_data = []
        
        # Create temporal_next edges (consecutive time connections)
        for i in range(len(events) - 1):
            src_id = node_ids[i]
            dst_id = node_ids[i + 1]
            dt = events[i + 1]["t"] - events[i]["t"]
            
            # Extract edge features
            features = self.feature_extractor.extract_edge_features(
                events[i], events[i + 1], 0, session_data  # etype=0 for temporal_next
            )
            
            edge_record = {
                "src": src_id,
                "dst": dst_id,
                "etype": 0,  # temporal_next
                "dt": min(dt, 3600000)  # Cap at 1 hour in milliseconds
            }
            
            # Add edge features
            for j, feature_val in enumerate(features):
                edge_record[f"e{j}"] = feature_val
            
            edges_data.append(edge_record)
        
        # Create movement_transition edges (when movement types change)
        for i in range(len(events) - 1):
            curr_type = events[i].get("movement_type", "")
            next_type = events[i + 1].get("movement_type", "")
            
            if curr_type != next_type and curr_type and next_type:
                src_id = node_ids[i]
                dst_id = node_ids[i + 1]
                dt = events[i + 1]["t"] - events[i]["t"]
                
                features = self.feature_extractor.extract_edge_features(
                    events[i], events[i + 1], 1, session_data  # etype=1 for movement_transition
                )
                
                edge_record = {
                    "src": src_id,
                    "dst": dst_id,
                    "etype": 1,  # movement_transition
                    "dt": min(dt, 3600000)
                }
                
                for j, feature_val in enumerate(features):
                    edge_record[f"e{j}"] = feature_val
                
                edges_data.append(edge_record)
        
        # Create liq_link edges (connect price movements to nearby liquidity events)
        price_indices = [i for i, e in enumerate(events) if e.get("source_type") == "price_movement"]
        liq_indices = [i for i, e in enumerate(events) if e.get("source_type") == "liquidity_event"]
        
        for liq_idx in liq_indices:
            liq_time = events[liq_idx]["t"]
            
            # Find nearest price movement within ±30 seconds
            closest_price_idx = None
            min_time_diff = 30000  # 30 seconds in milliseconds
            
            for price_idx in price_indices:
                price_time = events[price_idx]["t"]
                time_diff = abs(liq_time - price_time)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_price_idx = price_idx
            
            if closest_price_idx is not None:
                src_id = node_ids[closest_price_idx]
                dst_id = node_ids[liq_idx]
                dt = abs(events[liq_idx]["t"] - events[closest_price_idx]["t"])
                
                features = self.feature_extractor.extract_edge_features(
                    events[closest_price_idx], events[liq_idx], 2, session_data  # etype=2 for liq_link
                )
                
                edge_record = {
                    "src": src_id,
                    "dst": dst_id,
                    "etype": 2,  # liq_link
                    "dt": min(dt, 3600000)
                }
                
                for j, feature_val in enumerate(features):
                    edge_record[f"e{j}"] = feature_val
                
                edges_data.append(edge_record)
        
        if not edges_data:
            # Return empty DataFrame with correct schema
            return self._create_empty_edges_dataframe()
        
        return pd.DataFrame(edges_data)
    
    def _create_empty_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create empty DataFrames with correct schemas."""
        from ironforge.data_engine.schemas import NODE_COLS, EDGE_COLS
        
        # Empty nodes DataFrame
        nodes_df = pd.DataFrame(columns=NODE_COLS + ["session_id", "symbol", "tf", "price"])
        
        # Empty edges DataFrame  
        edges_df = pd.DataFrame(columns=EDGE_COLS)
        
        # Empty node map DataFrame
        node_map_df = pd.DataFrame(columns=["session_id", "local_idx", "node_id", "t", "timestamp", "source_type"])
        
        return nodes_df, edges_df, node_map_df
    
    def _create_empty_edges_dataframe(self) -> pd.DataFrame:
        """Create empty edges DataFrame with correct schema."""
        from ironforge.data_engine.schemas import EDGE_COLS
        return pd.DataFrame(columns=EDGE_COLS)
    
    def _validate_dataframes(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> bool:
        """Validate nodes and edges DataFrames against requirements."""
        
        # Check node count constraints
        if len(nodes_df) < 5 or len(nodes_df) > 10000:
            logger.error(f"Node count {len(nodes_df)} outside valid range [5, 10000]")
            return False
        
        # Check edge count constraints
        if len(edges_df) < 1 or len(edges_df) > 20000:
            logger.error(f"Edge count {len(edges_df)} outside valid range [1, 20000]")
            return False
        
        # Check for required columns
        from ironforge.data_engine.schemas import NODE_COLS, EDGE_COLS
        
        missing_node_cols = [col for col in NODE_COLS if col not in nodes_df.columns]
        if missing_node_cols:
            logger.error(f"Missing node columns: {missing_node_cols}")
            return False
        
        missing_edge_cols = [col for col in EDGE_COLS if col not in edges_df.columns]
        if missing_edge_cols:
            logger.error(f"Missing edge columns: {missing_edge_cols}")
            return False
        
        # Check for null node_ids
        if nodes_df["node_id"].isnull().any():
            logger.error("Found null node_id values")
            return False
        
        # Check for duplicate node_ids
        if nodes_df["node_id"].duplicated().any():
            logger.error("Found duplicate node_id values")
            return False
        
        # Check that all edge src/dst reference valid nodes
        valid_node_ids = set(nodes_df["node_id"])
        invalid_srcs = edges_df[~edges_df["src"].isin(valid_node_ids)]
        invalid_dsts = edges_df[~edges_df["dst"].isin(valid_node_ids)]
        
        if len(invalid_srcs) > 0:
            logger.error(f"Found {len(invalid_srcs)} edges with invalid src node_ids")
            return False
        
        if len(invalid_dsts) > 0:
            logger.error(f"Found {len(invalid_dsts)} edges with invalid dst node_ids")
            return False
        
        # Check time constraints
        if (nodes_df["t"] <= 0).any():
            logger.error("Found non-positive timestamps")
            return False
        
        # Check dt constraints (should be non-negative and reasonable)
        if (edges_df["dt"] < 0).any():
            logger.error("Found negative dt values")
            return False
        
        if (edges_df["dt"] > 3600000).any():  # > 1 hour
            logger.warning("Found dt values > 1 hour, this may indicate time conversion issues")
        
        return True
    
    def _write_metadata(self, shard_dir: Path, session_info: Dict[str, str], 
                       source_path: Path, node_count: int, edge_count: int) -> None:
        """Write metadata file for the shard."""
        metadata = {
            "session_id": session_info["session_id"],
            "session_type": session_info["session_type"], 
            "date": session_info["date"],
            "source_file": str(source_path),
            "node_count": node_count,
            "edge_count": edge_count,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "timezone_source": self.config.source_timezone,
            "timezone_target": self.config.target_timezone,
            "feature_set": "v1.0",
            "conversion_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(shard_dir / "meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def convert_enhanced_sessions(config: ConversionConfig) -> List[Path]:
    """Convert all enhanced sessions matching the glob pattern."""
    from glob import glob
    
    converter = JSONToParquetConverter(config)
    source_files = glob(config.source_glob)
    
    if not source_files:
        logger.warning(f"No files found matching pattern: {config.source_glob}")
        return []
    
    # Filter out batch files and duplicates (prefer rel_ versions)
    filtered_files = []
    seen_sessions = {}
    
    for file_path in source_files:
        path = Path(file_path)
        filename = path.stem
        
        # Skip batch files
        if "batch" in filename:
            continue
        
        # Extract base session identifier
        if filename.startswith("enhanced_rel_"):
            base_name = filename[13:]  # Remove "enhanced_rel_" prefix
            is_rel = True
        elif filename.startswith("enhanced_"):
            base_name = filename[9:]   # Remove "enhanced_" prefix  
            is_rel = False
        else:
            continue
        
        # Prefer rel_ versions (dedupe strategy)
        if base_name not in seen_sessions or (is_rel and not seen_sessions[base_name]["is_rel"]):
            seen_sessions[base_name] = {"path": path, "is_rel": is_rel}
    
    filtered_files = [info["path"] for info in seen_sessions.values()]
    
    logger.info(f"Found {len(filtered_files)} unique enhanced sessions to convert")
    
    # Convert sessions
    successful_conversions = []
    failed_conversions = []
    
    for file_path in sorted(filtered_files):
        result = converter.convert_session(file_path)
        if result:
            successful_conversions.append(result)
        else:
            failed_conversions.append(file_path)
    
    # Log summary
    logger.info(f"✅ Successfully converted {len(successful_conversions)} sessions")
    if failed_conversions:
        logger.warning(f"❌ Failed to convert {len(failed_conversions)} sessions:")
        for failed_path in failed_conversions:
            logger.warning(f"  - {failed_path}")
    
    return successful_conversions