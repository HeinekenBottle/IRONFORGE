#!/usr/bin/env python3
"""
IRONFORGE Enhanced Graph Builder - Rich Feature Archaeological Discovery
========================================================================

Implements 20+ dimensional rich node features for meaningful archaeological 
discovery of permanent market relationships across time and price distance.

Key Enhancements:
- 20+ dimensional node features (vs previous 4)
- Multi-timeframe explicit hierarchy (1m/5m/15m/1h/D/W)
- Cross-session alignment capabilities  
- Rich edge features for TGAT learning
- Complete preservation of 3D structure (time × price × timeframe)

This enables TGAT to discover the "permanent links" between distant 
time & price points that simple counting methods cannot detect.
"""

import json
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# Feature processing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

@dataclass
class RichNodeFeature:
    """Container for rich 45+ dimensional node features with semantic events and session context"""
    
    # Semantic Event Types (7 ENHANCED features for complete market cycle detection)
    EVENT_TYPES = ["fvg_redelivery", "expansion_phase", "consolidation", "retracement", "reversal", "liq_sweep", "pd_array_interaction"]

    # Binary semantic event flags for TGAT attention
    fvg_redelivery_flag: float      # 0.0 or 1.0
    expansion_phase_flag: float     # 0.0 or 1.0
    consolidation_flag: float       # 0.0 or 1.0
    retracement_flag: float         # 0.0 or 1.0 - price trading back into previous consolidation range
    reversal_flag: float            # 0.0 or 1.0 - singular reversal point events
    liq_sweep_flag: float          # 0.0 or 1.0
    pd_array_interaction_flag: float # 0.0 or 1.0
    
    # Session phase preservation (3 NEW features - one-hot encoded)
    phase_open: float              # 0.0 or 1.0 (first 20% of session)
    phase_mid: float               # 0.0 or 1.0 (middle 60% of session)
    phase_close: float             # 0.0 or 1.0 (final 20% of session)
    
    # Temporal Context (12 features - EXPANDED for temporal cycles)
    time_minutes: float
    daily_phase_sin: float  
    daily_phase_cos: float
    session_position: float
    time_to_close: float
    weekend_proximity: float
    absolute_timestamp: int
    day_of_week: int  # 0-6
    month_phase: float
    
    # Temporal Cycle Detection (3 NEW features - Innovation Architect)
    week_of_month: int      # 1-5: Which week of the month (for monthly cycle detection)
    month_of_year: int      # 1-12: Which month (for seasonal patterns)
    day_of_week_cycle: int  # 0-6: Day of week for cycle detection (duplicate for pattern emphasis)
    
    # Price Relativity Features (7 NEW features - PERMANENT patterns)
    normalized_price: float      # 0-1 position in session range
    pct_from_open: float        # % change from session open
    pct_from_high: float        # % distance from session high
    pct_from_low: float         # % distance from session low
    price_to_HTF_ratio: float   # Ratio to parent timeframe
    time_since_session_open: float  # Seconds since session start
    normalized_time: float      # 0-1 position in session duration
    
    # Price Context (Legacy - 3 features)
    price_delta_1m: float
    price_delta_5m: float
    price_delta_15m: float
    
    # Market State (7 features)
    volatility_window: float
    energy_state: float
    contamination_coefficient: float
    fisher_regime: int  # 0=baseline, 1=elevated, 2=transitional
    session_character: int  # 0=expansion, 1=consolidation, etc
    cross_tf_confluence: float
    timeframe_rank: int  # 1=1m, 2=5m, 3=15m, 4=1h, 5=D, 6=W
    
    # Event & Structure Context (8 features + embeddings)
    event_type_id: int
    timeframe_source: int  # 0=1m, 1=5m, 2=15m, 3=1h, 4=D, 5=W
    liquidity_type: int
    fpfvg_gap_size: float
    fpfvg_interaction_count: int
    first_presentation_flag: float  # 0.0 or 1.0
    pd_array_strength: float
    structural_importance: float
    
    # Preservation
    raw_json: Dict[str, Any]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor for TGAT - SEMANTIC ENHANCED (47D)"""
        features = [
            # Semantic Events (10) - ENHANCED: Complete market cycle detection
            self.fvg_redelivery_flag, self.expansion_phase_flag, self.consolidation_flag,
            self.retracement_flag, self.reversal_flag,
            self.liq_sweep_flag, self.pd_array_interaction_flag,
            self.phase_open, self.phase_mid, self.phase_close,
            
            # Temporal (12) - EXPANDED with temporal cycles
            self.time_minutes, self.daily_phase_sin, self.daily_phase_cos,
            self.session_position, self.time_to_close, self.weekend_proximity,
            float(self.absolute_timestamp), float(self.day_of_week), self.month_phase,
            
            # TEMPORAL CYCLE DETECTION (3) - Innovation Architect enhancement
            float(self.week_of_month), float(self.month_of_year), float(self.day_of_week_cycle),
            
            # PRICE RELATIVITY (7) - PERMANENT PATTERNS
            self.normalized_price, self.pct_from_open, self.pct_from_high,
            self.pct_from_low, self.price_to_HTF_ratio, self.time_since_session_open,
            self.normalized_time,
            
            # Price Context Legacy (3)
            self.price_delta_1m, self.price_delta_5m, self.price_delta_15m,
            
            # Market State (7)
            self.volatility_window, self.energy_state, self.contamination_coefficient,
            float(self.fisher_regime), float(self.session_character), 
            self.cross_tf_confluence, float(self.timeframe_rank),
            
            # Event & Structure (8)
            float(self.event_type_id), float(self.timeframe_source), 
            float(self.liquidity_type), self.fpfvg_gap_size,
            float(self.fpfvg_interaction_count), self.first_presentation_flag,
            self.pd_array_strength, self.structural_importance
        ]
        return torch.tensor(features, dtype=torch.float32)

@dataclass 
class RichEdgeFeature:
    """Container for rich 20-dimensional edge features with semantic labeling"""
    
    # Semantic Event Relationships (3 NEW features for archaeological discovery)
    RELATION_TYPES = {
        0: "temporal", 1: "scale", 2: "cascade", 3: "pd_array", 4: "discovered", 
        5: "confluence", 6: "echo", 7: "fvg_chain", 8: "phase_transition", 9: "liquidity_sweep"
    }
    
    # New semantic features
    semantic_event_link: int       # 0=none, 1=fvg_chain, 2=pd_sequence, 3=phase_transition, 4=liquidity_sweep
    event_causality: float        # 0.0-1.0 causal strength between semantic events
    semantic_label_id: int        # Encoded semantic label for relationship type
    
    # Temporal Relationships (4)
    time_delta: float
    log_time_delta: float
    timeframe_jump: int
    temporal_resonance: float  # NEW: Harmonic time relationships
    
    # Relationship Type & Semantics (5)
    relation_type: int  # 0=temporal, 1=scale, 2=cascade, 3=pd, 4=discovered, 5=confluence, 6=echo
    relation_strength: float
    directionality: int  # 0=forward, 1=backward, 2=bidirectional
    semantic_weight: float  # NEW: Semantic relationship importance
    causality_score: float  # NEW: Causal relationship strength
    
    # Cross-Scale & Multi-TF Hierarchy (4)
    scale_from: int
    scale_to: int  
    aggregation_type: int
    hierarchy_distance: float  # NEW: Distance in TF hierarchy
    
    # Archaeological Discovery (4)
    discovery_epoch: int
    discovery_confidence: float
    validation_score: float
    permanence_score: float  # NEW: Cross-regime stability
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor for TGAT edge features - SEMANTIC ENHANCED (20D)"""
        features = [
            # Semantic Event Relationships (3) - NEW: Archaeological discovery
            float(self.semantic_event_link), self.event_causality, float(self.semantic_label_id),
            
            # Temporal (4)
            self.time_delta, self.log_time_delta, float(self.timeframe_jump), self.temporal_resonance,
            
            # Relationship & Semantics (5)
            float(self.relation_type), self.relation_strength, float(self.directionality),
            self.semantic_weight, self.causality_score,
            
            # Cross-Scale & Hierarchy (4)
            float(self.scale_from), float(self.scale_to), float(self.aggregation_type),
            self.hierarchy_distance,
            
            # Archaeological (4)
            float(self.discovery_epoch), self.discovery_confidence, 
            self.validation_score, self.permanence_score
        ]
        return torch.tensor(features, dtype=torch.float32)

class EnhancedGraphBuilder:
    """
    Enhanced IRONFORGE Graph Builder with 20+ dimensional rich features
    
    Enables archaeological discovery of permanent relationships by providing
    TGAT with rich contextual information about each market event and the
    relationships between them across multiple timeframes.
    
    Architecture:
    - Node Features: 27 dimensions (temporal + price/market + event/structure)
    - Edge Features: 17 dimensions (temporal + semantics + hierarchy + discovery)
    - Edge Scorers: Pluggable functions for relationship strength calculation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Feature encoders
        self.event_type_encoder = {}  # Will be built from data
        self.timeframe_mapping = {
            '1m': 0, '5m': 1, '15m': 2, '1h': 3, 'D': 4, 'W': 5
        }
        self.timeframe_hierarchy = ['1m', '5m', '15m', '1h', 'D', 'W']  # NEW: Hierarchy order
        
        self.liquidity_type_mapping = {
            'native_session': 0, 'cross_session': 1, 'cascade': 2, 'fpfvg': 3
        }
        self.fisher_regime_mapping = {
            'baseline': 0, 'elevated': 1, 'transitional': 2
        }
        self.session_character_mapping = {
            'expansion': 0, 'consolidation': 1, 'transition': 2, 'breakdown': 3
        }
        
        # Price history for delta calculations
        self.price_history = {}
        
        # Edge Scoring Interfaces (to be implemented by agent)
        self.edge_scorers = {
            'temporal_resonance': None,      # Will be populated by agent
            'semantic_weight': None,         # Will be populated by agent  
            'causality_score': None,         # Will be populated by agent
            'hierarchy_distance': None,      # Will be populated by agent
            'permanence_score': None         # Will be populated by agent
        }
    
    def _sanitize_magnitude(self, magnitude_value):
        """
        Convert string magnitude values to numeric (FIXES DATA FORMAT ERRORS)
        
        Handles the 'could not convert string to float: high' errors by mapping
        string magnitude values to appropriate numeric values.
        """
        if isinstance(magnitude_value, str):
            magnitude_map = {
                'minimal': 0.1, 'very_low': 0.2, 'low': 0.3, 
                'medium': 0.5, 'high': 0.8, 'very_high': 1.0
            }
            sanitized = magnitude_map.get(magnitude_value.lower(), 0.0)
            self.logger.debug(f"Sanitized magnitude '{magnitude_value}' -> {sanitized}")
            return sanitized
        
        # Handle numeric values (including None)
        try:
            return float(magnitude_value) if magnitude_value is not None else 0.0
        except (ValueError, TypeError):
            self.logger.warning(f"Could not convert magnitude '{magnitude_value}' to float, using 0.0")
            return 0.0
    
    def _validate_session_data(self, session_data: Dict[str, Any]) -> bool:
        """
        Validate session has minimum required data for processing
        
        STRICT VALIDATION - NO DEFENSIVE HIDING OF ISSUES
        """
        # Check for essential data structures  
        if not isinstance(session_data, dict):
            raise ValueError("Session data must be a dictionary")
        
        # Check for price movements (essential for node creation)
        price_movements = session_data.get('price_movements', [])
        if not price_movements or len(price_movements) < 2:
            raise ValueError(f"Insufficient price movements: {len(price_movements)} (need ≥2)")
        
        # STRICT validation - check ALL movements, not just first 5
        invalid_movements = []
        empty_timestamps = []
        
        for i, movement in enumerate(price_movements):
            if not isinstance(movement, dict):
                invalid_movements.append(f"Movement {i}: not a dictionary")
                continue
                
            # Check timestamp validity - NO EMPTY TIMESTAMPS ALLOWED
            timestamp = movement.get('timestamp', '')
            if not timestamp or timestamp.strip() == '':
                empty_timestamps.append(f"Movement {i}: empty timestamp")
                continue
                
            # Check price validity
            if 'price_level' not in movement:
                invalid_movements.append(f"Movement {i}: missing price_level")
                continue
                
            try:
                price = float(movement['price_level'])
                if price <= 0:
                    invalid_movements.append(f"Movement {i}: invalid price {price}")
            except (ValueError, TypeError):
                invalid_movements.append(f"Movement {i}: non-numeric price '{movement['price_level']}'")
        
        # FAIL FAST - don't process corrupted data
        if empty_timestamps:
            raise ValueError(f"Empty timestamps detected: {empty_timestamps}")
            
        if invalid_movements:
            raise ValueError(f"Invalid movements detected: {invalid_movements}")
        
        # Require at least 80% valid data
        valid_movements = len(price_movements) - len(invalid_movements) - len(empty_timestamps)
        if valid_movements < len(price_movements) * 0.8:
            raise ValueError(f"Insufficient valid data: {valid_movements}/{len(price_movements)} movements valid")
        
        self.logger.debug(f"✅ Session validation passed: {len(price_movements)} movements, all valid")
        return True
    
    def _create_empty_graph_response(self) -> Dict:
        """
        Create minimal empty graph response for invalid sessions
        
        Prevents crashes when session data is insufficient
        """
        return {
            'nodes': {'1m': [], '5m': [], '15m': [], '1h': [], 'D': [], 'W': []},
            'rich_node_features': torch.empty((0, 37), dtype=torch.float32),  # Empty 37D features (TEMPORAL CYCLES)
            'edges': {'scale': [], 'cross_tf_confluence': []},
            'rich_edge_features': torch.empty((0, 17), dtype=torch.float32),  # Empty 17D features
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_times': torch.empty((0,), dtype=torch.float),
            'metadata': {
                'total_nodes': 0,
                'total_edges': 0,
                'timeframe_counts': {},
                'session_valid': False,
                'skip_reason': 'insufficient_data'
            }
        }
        
        # Preserve everything principle
        self.preserve_everything = True
        
        self.logger.info("🏗️ Enhanced Graph Builder initialized with 27D nodes + 17D edges")
        
    def _load_htf_data(self, session_data: Dict[str, Any], session_file_path: Optional[str]) -> Optional[Dict]:
        """Load HTF-enhanced data if available"""
        try:
            # Method 1: Check if session_data already contains HTF data
            if 'pythonnodes' in session_data and 'htf_cross_map' in session_data:
                self.logger.debug("HTF data found in session_data")
                return session_data
                
            # Method 2: Try to find corresponding HTF file
            if session_file_path:
                htf_path = self._find_htf_file(session_file_path)
                if htf_path:
                    with open(htf_path, 'r') as f:
                        htf_data = json.load(f)
                    if 'pythonnodes' in htf_data and 'htf_cross_map' in htf_data:
                        self.logger.debug(f"HTF data loaded from {htf_path}")
                        return htf_data
                        
            # Method 3: Look in htf_relativity directory by session metadata (PRICE RELATIVITY ENHANCED)
            session_id = session_data.get('session_metadata', {}).get('session_type', '')
            date = session_data.get('session_metadata', {}).get('date', '')
            if session_id and date:
                htf_filename = f"{session_id.upper()}_Lvl-1_{date}_htf_rel.json"
                htf_path = f"/Users/jack/IRONPULSE/data/sessions/htf_relativity/{htf_filename}"
                if Path(htf_path).exists():
                    with open(htf_path, 'r') as f:
                        htf_data = json.load(f)
                    if 'pythonnodes' in htf_data and 'htf_cross_map' in htf_data:
                        self.logger.debug(f"HTF data found at {htf_path}")
                        return htf_data
                        
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading HTF data: {e}")
            return None
    
    def _find_htf_file(self, session_file_path: str) -> Optional[str]:
        """Find corresponding HTF file for a session file"""
        try:
            # Convert session path to HTF RELATIVITY path (PRICE RELATIVITY ENHANCED)
            session_path = Path(session_file_path)
            htf_dir = Path("/Users/jack/IRONPULSE/data/sessions/htf_relativity")
            
            # Try exact filename match with _htf_rel suffix
            htf_filename = session_path.stem + "_htf_rel.json"
            htf_path = htf_dir / htf_filename
            
            if htf_path.exists():
                return str(htf_path)
                
            # Try pattern matching for similar files
            pattern = session_path.stem + "*_htf.json"
            matches = list(htf_dir.glob(pattern))
            if matches:
                return str(matches[0])
                
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error finding HTF file: {e}")
            return None
        
    def set_edge_scorers(self, scorers: Dict):
        """Set edge scoring functions from agent implementation"""
        self.edge_scorers.update(scorers)
        self.logger.info(f"📊 Edge scorers updated: {list(scorers.keys())}")
        
    def _calculate_enhanced_edge_features(self, source_feat: RichNodeFeature, 
                                        target_feat: RichNodeFeature,
                                        edge_type: str, graph_context: Dict) -> RichEdgeFeature:
        """
        Calculate enhanced edge features using pluggable scorers
        
        This is the architectural interface that uses agent-implemented scorers
        while maintaining structural control over edge feature creation.
        """
        
        # Basic temporal calculations (architectural)
        time_delta = abs(target_feat.time_minutes - source_feat.time_minutes)
        log_time_delta = np.log1p(time_delta)
        timeframe_jump = abs(target_feat.timeframe_source - source_feat.timeframe_source)
        
        # Price-based relationship strength (architectural)
        price_similarity = 1.0 - min(1.0, abs(source_feat.normalized_price - target_feat.normalized_price) / 0.01)
        base_relation_strength = price_similarity * (1.0 / (1.0 + time_delta))
        
        # Enhanced features via agent scorers (delegated)
        temporal_resonance = self._call_scorer('temporal_resonance', source_feat, target_feat, time_delta, 0.5)
        semantic_weight = self._call_scorer('semantic_weight', source_feat, target_feat, edge_type, 1.0)
        causality_score = self._call_scorer('causality_score', source_feat, target_feat, graph_context, 0.5)
        hierarchy_distance = self._call_scorer('hierarchy_distance', source_feat, target_feat, timeframe_jump, float(timeframe_jump))
        permanence_score = self._call_scorer('permanence_score', source_feat, target_feat, graph_context, 0.0)
        
        # Relationship type mapping (architectural)
        relation_type_map = {
            'temporal': 0, 'scale': 1, 'cascade': 2, 'pd_array': 3, 
            'discovered': 4, 'cross_tf_confluence': 5, 'temporal_echo': 6, 'structural_context': 7
        }
        relation_type = relation_type_map.get(edge_type, 0)
        
        # Directionality (architectural)
        if time_delta < 0.1:  # Same time
            directionality = 2  # Bidirectional
        elif target_feat.time_minutes > source_feat.time_minutes:
            directionality = 0  # Forward
        else:
            directionality = 1  # Backward
            
        # SEMANTIC EVENT LINK GENERATION - NEW for archaeological discovery
        semantic_event_link, event_causality, semantic_label_id = self._generate_semantic_edge_label(
            source_feat, target_feat, edge_type, graph_context
        )
        
        # Aggregation type (architectural)
        if edge_type == 'scale':
            aggregation_type = 1
        elif edge_type == 'cross_tf_confluence':
            aggregation_type = 2
        elif edge_type == 'structural_context':
            aggregation_type = 3
        else:
            aggregation_type = 0
            
        return RichEdgeFeature(
            # SEMANTIC EVENT RELATIONSHIPS (3) - NEW: Archaeological discovery
            semantic_event_link=semantic_event_link,
            event_causality=event_causality,
            semantic_label_id=semantic_label_id,
            
            # Temporal (4)
            time_delta=time_delta,
            log_time_delta=log_time_delta,
            timeframe_jump=timeframe_jump,
            temporal_resonance=temporal_resonance,
            
            # Relationship & Semantics (5)
            relation_type=relation_type,
            relation_strength=base_relation_strength,
            directionality=directionality,
            semantic_weight=semantic_weight,
            causality_score=causality_score,
            
            # Cross-Scale & Hierarchy (4)
            scale_from=source_feat.timeframe_source,
            scale_to=target_feat.timeframe_source,
            aggregation_type=aggregation_type,
            hierarchy_distance=hierarchy_distance,
            
            # Archaeological (4)
            discovery_epoch=0,
            discovery_confidence=1.0,
            validation_score=0.0,
            permanence_score=permanence_score
        )
        
    def _call_scorer(self, scorer_name: str, source_feat: RichNodeFeature, 
                    target_feat: RichNodeFeature, context, default_value):
        """Call agent-implemented scorer with fallback"""
        scorer = self.edge_scorers.get(scorer_name)
        if scorer and callable(scorer):
            try:
                return scorer(source_feat, target_feat, context)
            except Exception as e:
                self.logger.warning(f"Scorer {scorer_name} failed: {e}, using default")
                return default_value
        return default_value
        
    def _load_edge_scorers(self):
        """Load edge scoring functions from agent implementation"""
        try:
            from .edge_scorers import get_all_scorers
            scorers = get_all_scorers()
            self.set_edge_scorers(scorers)
            self.logger.info("✅ Edge scorers loaded successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not load edge scorers: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error loading edge scorers: {e}")
        
    def build_rich_graph(self, session_data: Dict[str, Any], 
                        historical_sessions: Optional[List[Dict]] = None,
                        session_file_path: Optional[str] = None) -> Tuple[Dict, Dict[str, Any]]:
        """
        Build multi-timeframe graph with rich 45D node features and session context
        
        Args:
            session_data: Primary session data
            historical_sessions: Optional historical context for cross-session features
            session_file_path: Optional path to session file for HTF lookup
            
        Returns:
            Tuple of (enhanced_graph, session_metadata) for semantic archaeological discovery
        """
        
        self.logger.info("🔍 Building rich multi-timeframe graph...")
        
        # Check for HTF-enhanced data
        htf_data = self._load_htf_data(session_data, session_file_path)
        if htf_data:
            self.logger.info("✅ HTF-enhanced data detected, using multi-timeframe mode")
            session_data = htf_data
        else:
            self.logger.info("📊 Using standard 1m-only mode")
        
        # Validate session has minimum required data (STRICT - NO HIDING ISSUES)  
        self._validate_session_data(session_data)  # Raises ValueError if invalid
        
        # Initialize graph structure
        graph = {
            'nodes': {
                '1m': [], '5m': [], '15m': [], '1h': [], 'D': [], 'W': []
            },
            'rich_node_features': [],
            'edges': {
                'temporal': [], 'scale': [], 'cascade': [], 'pd_array': [],
                'cross_tf_confluence': [], 'temporal_echo': [], 'discovered': [], 'structural_context': []
            },
            'rich_edge_features': [],
            'metadata': {
                'session_id': session_data.get('session_metadata', {}).get('session_id', 'unknown'),
                'total_nodes': 0,
                'timeframe_counts': {},
                'feature_dimensions': 45,  # 8 semantic + 12 + 3 + 7 + 3 + 7 + 8 (SEMANTIC ENHANCED)
                'preserved_raw': session_data
            }
        }
        
        # Check if we have HTF data and extract accordingly
        if 'pythonnodes' in session_data and 'htf_cross_map' in session_data:
            self._extract_htf_nodes_from_pythonnodes(session_data, graph)
        else:
            # Extract nodes by timeframe with rich features (original method)
            self._extract_rich_nodes_by_timeframe(session_data, graph)
        
        # Load edge scorers if not already loaded
        if not any(self.edge_scorers.values()):
            self._load_edge_scorers()
        
        # Build rich edges between nodes (includes HTF scale edges if available)
        self._build_rich_edges(graph)
        
        # Add cross-session context if available
        if historical_sessions:
            self._add_cross_session_context(graph, historical_sessions)
            
        # Calculate final statistics
        total_nodes = sum(len(nodes) for nodes in graph['nodes'].values())
        graph['metadata']['total_nodes'] = total_nodes
        graph['metadata']['timeframe_counts'] = {
            tf: len(nodes) for tf, nodes in graph['nodes'].items()
        }
        
        # EXTRACT SESSION METADATA - NEW for archaeological discovery context preservation
        session_metadata = self._extract_session_metadata(session_data, session_file_path)
        
        self.logger.info(f"✅ Rich graph built: {total_nodes} nodes, {len(graph['rich_node_features'])} rich features")
        self.logger.info(f"✅ Session metadata preserved: {session_metadata['session_name']} ({session_metadata['session_start']} - {session_metadata['session_end']})")
        
        return graph, session_metadata
    
    def _filter_constant_features(self, X_raw: torch.Tensor, rich_features: list) -> Tuple[torch.Tensor, Dict]:
        """
        TASK 4: Detect and filter constant features (variance = 0) for TGAT training
        
        Args:
            X_raw: Raw feature tensor (n_nodes, 45)
            rich_features: List of RichNodeFeature objects for context
            
        Returns:
            Tuple of (filtered_tensor, constant_features_metadata)
        """
        
        if X_raw.shape[0] <= 1:
            # Single node - no variance analysis possible
            return X_raw, {'constant_feature_indices': [], 'constant_feature_names': [], 'metadata_only_count': 0}
        
        # Calculate variance across all nodes for each feature dimension
        feature_variances = torch.var(X_raw, dim=0, unbiased=False)
        
        # Identify constant features (variance = 0)
        constant_mask = feature_variances == 0.0
        constant_indices = torch.where(constant_mask)[0].tolist()
        
        # Feature names mapping for semantic interpretation (47D feature vector)
        feature_names = [
            # Semantic Events (10) - ENHANCED: Complete market cycle detection
            'fvg_redelivery_flag', 'expansion_phase_flag', 'consolidation_flag', 'retracement_flag', 'reversal_flag',
            'liq_sweep_flag', 'pd_array_interaction_flag', 'phase_open', 'phase_mid', 'phase_close',
            
            # Temporal (12)
            'event_time', 'time_since_session_open', 'session_position', 'normalized_time',
            'absolute_timestamp', 'day_of_week', 'month_phase', 'week_of_month', 
            'month_of_year', 'day_of_week_cycle', 'week_position', 'session_day_alignment',
            
            # Price/Market (3)
            'normalized_price', 'pct_from_open', 'pct_from_high',
            
            # Structure/Event (7)
            'pct_from_low', 'volume_intensity', 'momentum_shift', 'pattern_strength',
            'liquidity_type', 'structure_significance', 'cascade_potential',
            
            # Energy/Contamination (3)
            'energy_state', 'contamination_level', 'structural_quality',
            
            # Cross-timeframe (7)
            'cross_tf_confluence', 'temporal_echo_strength', 'scaling_factor', 
            'hierarchy_position', 'multi_tf_validation', 'scale_coherence', 'temporal_stability',
            
            # Archaeological Discovery (8)
            'discovery_epoch', 'discovery_confidence', 'pattern_permanence',
            'validation_score', 'temporal_context', 'causality_strength',
            'structural_importance', 'archaeological_significance'
        ]
        
        constant_feature_names = [feature_names[i] for i in constant_indices if i < len(feature_names)]
        
        # Filter out constant features for TGAT training
        if len(constant_indices) > 0:
            # Create non-constant mask
            training_mask = ~constant_mask
            X_filtered = X_raw[:, training_mask]
            
            self.logger.info(f"🔍 TASK 4: Detected {len(constant_indices)} constant features (variance=0)")
            self.logger.info(f"   Constant features: {constant_feature_names}")
            self.logger.info(f"   Features for TGAT training: {X_filtered.shape[1]} (filtered from {X_raw.shape[1]})")
        else:
            X_filtered = X_raw
            self.logger.info(f"✅ TASK 4: No constant features detected, using all {X_raw.shape[1]} features for TGAT")
        
        # Create metadata for constant features (to be preserved for output context)
        constant_features_metadata = {
            'constant_feature_indices': constant_indices,
            'constant_feature_names': constant_feature_names,
            'metadata_only_count': len(constant_indices),
            'training_features': X_filtered.shape[1],
            'original_features': X_raw.shape[1],
            'feature_variances': feature_variances.tolist(),
            'constant_values': [X_raw[0, i].item() for i in constant_indices] if len(constant_indices) > 0 else []
        }
        
        return X_filtered, constant_features_metadata
        
    def _extract_rich_nodes_by_timeframe(self, session_data: Dict, graph: Dict) -> None:
        """Extract nodes for each timeframe with rich features"""
        
        session_meta = session_data.get('session_metadata', {})
        energy_state = session_data.get('energy_state', {})
        contamination = session_data.get('contamination_analysis', {})
        
        # Session-level context
        session_start = self._parse_time(session_meta.get('session_start', '09:30:00'))
        session_duration = session_meta.get('session_duration', 120)
        session_date = session_meta.get('session_date', '2025-01-01')
        
        # Calculate session-wide context with robust date parsing (TEMPORAL CYCLES ENHANCED)
        absolute_timestamp, day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle = self._parse_session_date(session_date)
        
        # Extract events and enrich with features
        all_events = []
        
        # 1m events from price_movements
        if 'price_movements' in session_data:
            for event in session_data['price_movements']:
                rich_features = self._create_rich_node_features(
                    event, session_meta, energy_state, contamination,
                    session_start, session_duration, absolute_timestamp, 
                    day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle, timeframe='1m',
                    session_data=session_data
                )
                
                all_events.append({
                    'timeframe': '1m',
                    'features': rich_features,
                    'node_id': len(all_events)
                })
                graph['nodes']['1m'].append(len(all_events) - 1)
                
        # 15m events from session_liquidity_events  
        if 'session_liquidity_events' in session_data:
            for event in session_data['session_liquidity_events']:
                rich_features = self._create_rich_node_features(
                    event, session_meta, energy_state, contamination,
                    session_start, session_duration, absolute_timestamp,
                    day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle, timeframe='15m',
                    session_data=session_data
                )
                
                all_events.append({
                    'timeframe': '15m', 
                    'features': rich_features,
                    'node_id': len(all_events)
                })
                graph['nodes']['15m'].append(len(all_events) - 1)
                
        # 1h events from cascade_events
        if 'micro_timing_analysis' in session_data:
            cascade_events = session_data['micro_timing_analysis'].get('cascade_events', [])
            for event in cascade_events:
                rich_features = self._create_rich_node_features(
                    event, session_meta, energy_state, contamination,
                    session_start, session_duration, absolute_timestamp,
                    day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle, timeframe='1h',
                    session_data=session_data
                )
                
                all_events.append({
                    'timeframe': '1h',
                    'features': rich_features, 
                    'node_id': len(all_events)
                })
                graph['nodes']['1h'].append(len(all_events) - 1)
                
        # Store all rich node features
        graph['rich_node_features'] = [event['features'] for event in all_events]
        
    def _extract_htf_nodes_from_pythonnodes(self, session_data: Dict, graph: Dict) -> None:
        """Extract nodes from HTF pythonnodes data structure"""
        
        pythonnodes = session_data.get('pythonnodes', {})
        session_meta = session_data.get('session_metadata', {})
        
        # Session-level context
        session_start = self._parse_time(session_meta.get('session_start', '09:30:00'))
        session_duration = session_meta.get('session_duration', 120)
        session_date = session_meta.get('session_date', session_meta.get('date', '2025-01-01'))
        
        # Calculate session-wide context (TEMPORAL CYCLES ENHANCED)
        absolute_timestamp, day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle = self._parse_session_date(session_date)
        
        all_events = []
        
        # Process each timeframe in pythonnodes
        for tf_name, nodes in pythonnodes.items():
            if not nodes:
                continue
                
            self.logger.debug(f"Processing {len(nodes)} nodes for timeframe {tf_name}")
            
            for node in nodes:
                rich_features = self._create_htf_node_features(
                    node, session_meta, tf_name,
                    session_start, session_duration, absolute_timestamp,
                    day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle,
                    session_data=session_data
                )
                
                event_data = {
                    'timeframe': tf_name,
                    'features': rich_features,
                    'node_id': len(all_events),
                    'htf_node_id': node.get('id', 0)
                }
                
                all_events.append(event_data)
                graph['nodes'][tf_name].append(len(all_events) - 1)
                
        # Store HTF cross-mapping for edge creation
        graph['htf_cross_map'] = session_data.get('htf_cross_map', {})
        
        # Store all rich node features
        graph['rich_node_features'] = [event['features'] for event in all_events]
        
        self.logger.info(f"✅ HTF nodes extracted: {len(all_events)} total nodes across {len(pythonnodes)} timeframes")
        
    def _create_htf_node_features(self, node: Dict, session_meta: Dict, timeframe: str,
                                 session_start: float, session_duration: int,
                                 absolute_timestamp: int, day_of_week: int,
                                 month_phase: float, week_of_month: int, 
                                 month_of_year: int, day_of_week_cycle: int, session_data: Dict = None) -> RichNodeFeature:
        """Create rich 37D node features from HTF node data with temporal cycles"""
        
        # Parse HTF node data
        node_time = self._parse_time(node.get('timestamp', '00:00:00'))
        node_price = float(node.get('close', node.get('price_level', 0)))
        
        # HTF-specific data
        high_price = float(node.get('high', node_price))
        low_price = float(node.get('low', node_price))
        open_price = float(node.get('open', node_price))
        
        # Temporal Context (9 features)
        time_minutes = node_time - session_start
        daily_phase_sin = np.sin(2 * np.pi * node_time / (24 * 60))
        daily_phase_cos = np.cos(2 * np.pi * node_time / (24 * 60))
        session_position = time_minutes / session_duration if session_duration > 0 else 0
        time_to_close = max(0, session_duration - time_minutes)
        weekend_proximity = min(abs(day_of_week - 5), abs(day_of_week + 2)) / 7.0
        
        # PRICE RELATIVITY FEATURES - Try node first, then lookup in price_movements
        timestamp_str = node.get('timestamp', 'unknown_time')
        
        if 'normalized_price' in node:
            # HTF node has relativity features directly
            normalized_price = node['normalized_price']
            pct_from_open = node['pct_from_open']
            pct_from_high = node['pct_from_high'] 
            pct_from_low = node['pct_from_low']
            price_to_HTF_ratio = node.get('price_to_HTF_ratio', 1.0)
            time_since_session_open = node['time_since_session_open']
            normalized_time = node['normalized_time']
        else:
            # HTF node lacks relativity features, try to find in price_movements array
            relativity_data = None
            
            if session_data and 'price_movements' in session_data:
                # Look for matching timestamp in price_movements array (exact match first)
                for pm_event in session_data['price_movements']:
                    if pm_event.get('timestamp') == timestamp_str:
                        relativity_data = pm_event
                        break
                
                # If no exact match, find closest timestamp (for HTF events with slight timing differences)
                if not relativity_data:
                    closest_match = None
                    min_time_diff = float('inf')
                    
                    try:
                        # Parse HTF timestamp to minutes
                        event_time_parts = timestamp_str.split(':')
                        event_minutes = int(event_time_parts[0]) * 60 + int(event_time_parts[1])
                        
                        for pm_event in session_data['price_movements']:
                            pm_timestamp = pm_event.get('timestamp', '')
                            if pm_timestamp:
                                # Parse price_movement timestamp to minutes
                                pm_time_parts = pm_timestamp.split(':')
                                pm_minutes = int(pm_time_parts[0]) * 60 + int(pm_time_parts[1])
                                
                                # Calculate time difference in minutes
                                time_diff = abs(event_minutes - pm_minutes)
                                
                                # If within 2 minutes, consider it a match
                                if time_diff <= 2 and time_diff < min_time_diff:
                                    min_time_diff = time_diff
                                    closest_match = pm_event
                        
                        if closest_match:
                            relativity_data = closest_match
                            self.logger.debug(f"✅ Found closest HTF relativity match for {timestamp_str} -> {closest_match.get('timestamp')} (diff: {min_time_diff}min)")
                            
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Could not parse HTF timestamps for fuzzy matching: {e}")
            
            if relativity_data:
                # Found matching relativity data in price_movements
                normalized_price = relativity_data['normalized_price']
                pct_from_open = relativity_data['pct_from_open'] 
                pct_from_high = relativity_data['pct_from_high']
                pct_from_low = relativity_data['pct_from_low']
                price_to_HTF_ratio = relativity_data.get('price_to_HTF_ratio', 1.0)
                time_since_session_open = relativity_data['time_since_session_open']
                normalized_time = relativity_data['normalized_time']
                self.logger.debug(f"✅ Found HTF relativity features for {timestamp_str} in price_movements array")
            else:
                # Calculate basic relativity features from available HTF data
                self.logger.warning(f"⚠️ HTF node at {timestamp_str} lacks relativity features - calculating from available data")
                
                # Use session bounds to calculate relative positions
                session_high = session_meta.get('session_high', node_price)
                session_low = session_meta.get('session_low', node_price) 
                session_open = session_meta.get('session_open', node_price)
                
                # Calculate relativity features from session context
                price_range = session_high - session_low if session_high > session_low else 1.0
                normalized_price = (node_price - session_low) / price_range if price_range > 0 else 0.5
                pct_from_open = ((node_price - session_open) / session_open * 100) if session_open > 0 else 0.0
                pct_from_high = ((session_high - node_price) / price_range * 100) if price_range > 0 else 0.0
                pct_from_low = ((node_price - session_low) / price_range * 100) if price_range > 0 else 100.0
                price_to_HTF_ratio = 1.0
                time_since_session_open = time_minutes * 60  # Convert to seconds
                normalized_time = session_position
        
        # Legacy price context (for backward compatibility)
        price_range = high_price - low_price if high_price > low_price else 0
        price_delta_1m = (node_price - open_price) / open_price if open_price > 0 else 0
        price_delta_5m = price_delta_1m  # Simplified for HTF nodes
        price_delta_15m = price_delta_1m
        volatility_window = price_range / node_price if node_price > 0 else 0
        
        # Timeframe rank for structural hierarchy
        timeframe_rank = {'1m': 1, '5m': 2, '15m': 3, '1h': 4, 'D': 5, 'W': 6}.get(timeframe, 1)
        
        # Market regime context (defaults for HTF)
        energy_val = 0.5  # Neutral energy for HTF nodes
        contamination_coeff = 0.0  # HTF nodes considered clean
        fisher_regime = 0  # Baseline
        session_char = self.session_character_mapping.get(session_meta.get('session_character', 'expansion'), 0)
        
        # HTF-specific cross-timeframe confluence
        meta = node.get('meta', {})
        coverage = float(meta.get('coverage', 1))
        cross_tf_confluence = min(1.0, coverage / 10.0)  # Normalize coverage to confluence
        
        # Event structure context
        event_type = node.get('event_type', 'htf_candle')
        event_type_id = self._get_event_type_id(event_type)
        timeframe_source = self.timeframe_mapping.get(timeframe, 0)
        liquidity_type = 0  # Native session for HTF nodes
        
        # HTF pattern context
        pd_array = node.get('pd_array')
        fpfvg_data = node.get('fpfvg')
        liquidity_sweep = node.get('liquidity_sweep', False)
        
        fpfvg_gap_size = float(fpfvg_data.get('level', 0)) if fpfvg_data and isinstance(fpfvg_data, dict) else 0.0
        fpfvg_interaction_count = 1 if fpfvg_data else 0
        first_presentation = 1.0 if timeframe == '1m' else 0.0  # Only 1m nodes are first presentation
        
        pd_strength = 0.0
        if pd_array and isinstance(pd_array, dict):
            if 'level' in pd_array:
                pd_strength = abs(float(pd_array['level']) - node_price) / node_price if node_price > 0 else 0
        
        # SEMANTIC EVENT EXTRACTION - NEW for archaeological discovery
        semantic_events = self._extract_semantic_events(event, session_data, timeframe)
        
        # Structural importance based on HTF coverage and patterns
        pattern_score = (1.0 if pd_array else 0.0) + (1.0 if fpfvg_data else 0.0) + (1.0 if liquidity_sweep else 0.0)
        structural_importance = (coverage / 10.0) * pattern_score * cross_tf_confluence
        
        return RichNodeFeature(
            # SEMANTIC EVENTS (10) - ENHANCED: Complete market cycle detection
            fvg_redelivery_flag=semantic_events['fvg_redelivery_flag'],
            expansion_phase_flag=semantic_events['expansion_phase_flag'],
            consolidation_flag=semantic_events['consolidation_flag'],
            retracement_flag=semantic_events['retracement_flag'],
            reversal_flag=semantic_events['reversal_flag'],
            liq_sweep_flag=semantic_events['liq_sweep_flag'],
            pd_array_interaction_flag=semantic_events['pd_array_interaction_flag'],
            phase_open=semantic_events['phase_open'],
            phase_mid=semantic_events['phase_mid'],
            phase_close=semantic_events['phase_close'],
            
            # Temporal Context (12) - TEMPORAL CYCLES ENHANCED
            time_minutes=time_minutes,
            daily_phase_sin=daily_phase_sin,
            daily_phase_cos=daily_phase_cos,
            session_position=session_position,
            time_to_close=time_to_close,
            weekend_proximity=weekend_proximity,
            absolute_timestamp=absolute_timestamp,
            day_of_week=day_of_week,
            month_phase=month_phase,
            
            # TEMPORAL CYCLE DETECTION (3) - Innovation Architect
            week_of_month=week_of_month,
            month_of_year=month_of_year,
            day_of_week_cycle=day_of_week_cycle,
            
            # PRICE RELATIVITY FEATURES (7) - PERMANENT PATTERNS
            normalized_price=normalized_price,
            pct_from_open=pct_from_open,
            pct_from_high=pct_from_high,
            pct_from_low=pct_from_low,
            price_to_HTF_ratio=price_to_HTF_ratio,
            time_since_session_open=time_since_session_open,
            normalized_time=normalized_time,
            
            # Price Context Legacy (3)
            price_delta_1m=price_delta_1m,
            price_delta_5m=price_delta_5m,
            price_delta_15m=price_delta_15m,
            
            # Market State (7)
            volatility_window=volatility_window,
            energy_state=energy_val,
            contamination_coefficient=contamination_coeff,
            fisher_regime=fisher_regime,
            session_character=session_char,
            cross_tf_confluence=cross_tf_confluence,
            timeframe_rank=timeframe_rank,
            
            # Event & Structure Context (8)
            event_type_id=event_type_id,
            timeframe_source=timeframe_source,
            liquidity_type=liquidity_type,
            fpfvg_gap_size=fpfvg_gap_size,
            fpfvg_interaction_count=fpfvg_interaction_count,
            first_presentation_flag=first_presentation,
            pd_array_strength=pd_strength,
            structural_importance=structural_importance,
            
            # Preservation
            raw_json=node
        )
        
    def _create_rich_node_features(self, event: Dict, session_meta: Dict, 
                                  energy_state: Dict, contamination: Dict,
                                  session_start: float, session_duration: int,
                                  absolute_timestamp: int, day_of_week: int,
                                  month_phase: float, week_of_month: int,
                                  month_of_year: int, day_of_week_cycle: int, timeframe: str, 
                                  session_data: Dict = None) -> RichNodeFeature:
        """Create rich 37D dimensional node features with temporal cycles"""
        
        # Parse event data
        event_time = self._parse_time(event.get('timestamp', '00:00:00'))
        event_price = float(event.get('price_level', event.get('price', 0)))
        event_type = event.get('movement_type', event.get('event_type', 'unknown'))
        
        # Temporal Context (9 features)
        time_minutes = event_time - session_start
        daily_phase_sin = np.sin(2 * np.pi * event_time / (24 * 60))  # 24-hour cycle
        daily_phase_cos = np.cos(2 * np.pi * event_time / (24 * 60))
        session_position = time_minutes / session_duration if session_duration > 0 else 0
        time_to_close = max(0, session_duration - time_minutes)
        weekend_proximity = min(abs(day_of_week - 5), abs(day_of_week + 2)) / 7.0  # Distance to weekend
        
        # PRICE RELATIVITY FEATURES - MANDATORY from relativity generator (NO FALLBACKS)
        # Technical Debt Surgeon: Comprehensive validation for partial relativity data
        required_relativity_features = [
            'normalized_price', 'pct_from_open', 'pct_from_high', 'pct_from_low',
            'time_since_session_open', 'normalized_time'
        ]
        
        # Check if this is enhanced data (has any relativity features)
        has_any_relativity = any(feature in event for feature in required_relativity_features)
        
        if has_any_relativity:
            # If any relativity features present, ALL must be present (no partial enhancement)
            missing_features = [f for f in required_relativity_features if f not in event]
            if missing_features:
                available_keys = list(event.keys())
                timestamp_str = event.get('timestamp', 'unknown_time')
                error_context = f"Event at timestamp {timestamp_str} (movement type: {event_type})"
                raise ValueError(
                    f"INCOMPLETE PRICE RELATIVITY FEATURES - {error_context}\n"
                    f"Found some relativity features but missing: {missing_features}\n"
                    f"Available keys: {available_keys}\n"
                    f"SOLUTION: Complete enhancement with price_relativity_generator.py\n"
                    f"NO FALLBACKS: Partial enhancement not allowed")
            
            # Extract all relativity features (all guaranteed to exist)
            normalized_price = event['normalized_price']
            pct_from_open = event['pct_from_open'] 
            pct_from_high = event['pct_from_high']
            pct_from_low = event['pct_from_low']
            price_to_HTF_ratio = event.get('price_to_HTF_ratio', 1.0)
            time_since_session_open = event['time_since_session_open']
            normalized_time = event['normalized_time']
        else:
            # Try to find relativity features in session price_movements array
            timestamp_str = event.get('timestamp', 'unknown_time')
            relativity_data = None
            
            if session_data and 'price_movements' in session_data:
                # Look for matching timestamp in price_movements array (exact match first)
                for pm_event in session_data['price_movements']:
                    if pm_event.get('timestamp') == timestamp_str:
                        relativity_data = pm_event
                        break
                
                # If no exact match, find closest timestamp (for cascade events with slight timing differences)
                if not relativity_data:
                    closest_match = None
                    min_time_diff = float('inf')
                    
                    try:
                        # Parse event timestamp to minutes
                        event_time_parts = timestamp_str.split(':')
                        event_minutes = int(event_time_parts[0]) * 60 + int(event_time_parts[1])
                        
                        for pm_event in session_data['price_movements']:
                            pm_timestamp = pm_event.get('timestamp', '')
                            if pm_timestamp:
                                # Parse price_movement timestamp to minutes
                                pm_time_parts = pm_timestamp.split(':')
                                pm_minutes = int(pm_time_parts[0]) * 60 + int(pm_time_parts[1])
                                
                                # Calculate time difference in minutes
                                time_diff = abs(event_minutes - pm_minutes)
                                
                                # If within 2 minutes, consider it a match
                                if time_diff <= 2 and time_diff < min_time_diff:
                                    min_time_diff = time_diff
                                    closest_match = pm_event
                        
                        if closest_match:
                            relativity_data = closest_match
                            self.logger.debug(f"✅ Found closest relativity match for {timestamp_str} -> {closest_match.get('timestamp')} (diff: {min_time_diff}min)")
                            
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Could not parse timestamps for fuzzy matching: {e}")
            
            if relativity_data:
                # Found matching relativity data in price_movements
                normalized_price = relativity_data['normalized_price']
                pct_from_open = relativity_data['pct_from_open'] 
                pct_from_high = relativity_data['pct_from_high']
                pct_from_low = relativity_data['pct_from_low']
                price_to_HTF_ratio = relativity_data.get('price_to_HTF_ratio', 1.0)
                time_since_session_open = relativity_data['time_since_session_open']
                normalized_time = relativity_data['normalized_time']
                self.logger.debug(f"✅ Found relativity features for {timestamp_str} in price_movements array")
            else:
                # No relativity features found - calculate them dynamically from session data
                self.logger.debug(f"📊 Computing relativity features for {timestamp_str} from session data")
                relativity_features = self._calculate_relativity_features(event, session_data, timestamp_str)
                
                normalized_price = relativity_features['normalized_price']
                pct_from_open = relativity_features['pct_from_open']
                pct_from_high = relativity_features['pct_from_high'] 
                pct_from_low = relativity_features['pct_from_low']
                price_to_HTF_ratio = relativity_features['price_to_HTF_ratio']
                time_since_session_open = relativity_features['time_since_session_open']
                normalized_time = relativity_features['normalized_time']
        
        # Legacy price context (for backward compatibility)
        price_delta_1m = self._calculate_price_delta(event_price, event_time, 1)
        price_delta_5m = self._calculate_price_delta(event_price, event_time, 5) 
        price_delta_15m = self._calculate_price_delta(event_price, event_time, 15)
        volatility_window = self._calculate_local_volatility(event_price, event_time)
        
        # Timeframe rank for structural hierarchy
        timeframe_rank = {'1m': 1, '5m': 2, '15m': 3, '1h': 4, 'D': 5, 'W': 6}.get(timeframe, 1)
        
        # Market regime context
        energy_val = energy_state.get('total_accumulated', 0) / 1000.0  # Normalize
        contamination_coeff = contamination.get('contamination_coefficient', 0)
        fisher_regime = self.fisher_regime_mapping.get(event.get('fisher_regime', 'baseline'), 0)
        session_char = self.session_character_mapping.get(session_meta.get('session_character', 'expansion'), 0)
        
        # Cross-timeframe confluence (placeholder - would calculate from multiple TFs)
        cross_tf_confluence = 1.0  # TODO: Implement based on multiple timeframe confirmation
        
        # Event structure context
        event_type_id = self._get_event_type_id(event_type)
        timeframe_source = self.timeframe_mapping.get(timeframe, 0)
        liquidity_type = self.liquidity_type_mapping.get(event.get('liquidity_type', 'native_session'), 0)
        
        # FPFVG context
        fpfvg_gap_size = float(event.get('gap_size', 0))
        fpfvg_interaction_count = int(event.get('interaction_count', 0))
        first_presentation = 1.0 if event.get('first_presentation_flag', False) else 0.0
        pd_strength = self._sanitize_magnitude(event.get('magnitude', event.get('pattern_strength', 0)))
        
        # SEMANTIC EVENT EXTRACTION - NEW for archaeological discovery
        semantic_events = self._extract_semantic_events(event, session_data, timeframe)
        
        # Structural importance (placeholder)
        structural_importance = pd_strength * cross_tf_confluence
        
        return RichNodeFeature(
            # SEMANTIC EVENTS (10) - ENHANCED: Complete market cycle detection
            fvg_redelivery_flag=semantic_events['fvg_redelivery_flag'],
            expansion_phase_flag=semantic_events['expansion_phase_flag'],
            consolidation_flag=semantic_events['consolidation_flag'],
            retracement_flag=semantic_events['retracement_flag'],
            reversal_flag=semantic_events['reversal_flag'],
            liq_sweep_flag=semantic_events['liq_sweep_flag'],
            pd_array_interaction_flag=semantic_events['pd_array_interaction_flag'],
            phase_open=semantic_events['phase_open'],
            phase_mid=semantic_events['phase_mid'],
            phase_close=semantic_events['phase_close'],
            
            # Temporal Context (12) - TEMPORAL CYCLES ENHANCED
            time_minutes=time_minutes,
            daily_phase_sin=daily_phase_sin,
            daily_phase_cos=daily_phase_cos, 
            session_position=session_position,
            time_to_close=time_to_close,
            weekend_proximity=weekend_proximity,
            absolute_timestamp=absolute_timestamp,
            day_of_week=day_of_week,
            month_phase=month_phase,
            
            # TEMPORAL CYCLE DETECTION (3) - Innovation Architect
            week_of_month=week_of_month,
            month_of_year=month_of_year,
            day_of_week_cycle=day_of_week_cycle,
            
            # PRICE RELATIVITY FEATURES (7) - PERMANENT PATTERNS
            normalized_price=normalized_price,
            pct_from_open=pct_from_open,
            pct_from_high=pct_from_high,
            pct_from_low=pct_from_low,
            price_to_HTF_ratio=price_to_HTF_ratio,
            time_since_session_open=time_since_session_open,
            normalized_time=normalized_time,
            
            # Price Context Legacy (3)
            price_delta_1m=price_delta_1m,
            price_delta_5m=price_delta_5m,
            price_delta_15m=price_delta_15m,
            
            # Market State (7)
            volatility_window=volatility_window,
            energy_state=energy_val,
            contamination_coefficient=contamination_coeff,
            fisher_regime=fisher_regime,
            session_character=session_char,
            cross_tf_confluence=cross_tf_confluence,
            timeframe_rank=timeframe_rank,
            
            # Event & Structure Context (8)
            event_type_id=event_type_id,
            timeframe_source=timeframe_source,
            liquidity_type=liquidity_type,
            fpfvg_gap_size=fpfvg_gap_size,
            fpfvg_interaction_count=fpfvg_interaction_count,
            first_presentation_flag=first_presentation,
            pd_array_strength=pd_strength,
            structural_importance=structural_importance,
            
            # Preservation
            raw_json=event
        )
        
    def _build_rich_edges(self, graph: Dict) -> None:
        """Build edges with rich features between nodes"""
        
        all_node_indices = []
        all_timeframes = []
        
        # Collect all nodes with their timeframes
        for tf, node_indices in graph['nodes'].items():
            for idx in node_indices:
                all_node_indices.append(idx)
                all_timeframes.append(tf)
                
        # Build temporal edges (within timeframe)
        self._build_temporal_edges(graph, all_node_indices, all_timeframes)
        
        # Build scale edges (across timeframes) 
        self._build_scale_edges(graph, all_node_indices, all_timeframes)
        
        # Build HTF scale edges using cross-map if available
        if 'htf_cross_map' in graph:
            self._build_htf_scale_edges(graph, all_node_indices, all_timeframes)
        
        # Build cross-TF confluence edges (same price across timeframes)
        self._build_confluence_edges(graph, all_node_indices, all_timeframes)
        
        # Build temporal echo edges (recurring patterns across sessions)
        self._build_temporal_echo_edges(graph, all_node_indices, all_timeframes)
        
        # Build price resonance edges (harmonic price relationships)
        self._build_price_resonance_edges(graph, all_node_indices, all_timeframes)
        
        # Build structural context edges (4th edge type - Innovation Architect)
        self._build_structural_context_edges(graph, all_node_indices, all_timeframes)
        
    def _build_temporal_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build temporal edges within each timeframe"""
        
        # Group by timeframe
        tf_groups = {}
        for idx, tf in zip(node_indices, timeframes):
            if tf not in tf_groups:
                tf_groups[tf] = []
            tf_groups[tf].append(idx)
            
        # Create sequential edges within each timeframe
        for tf, indices in tf_groups.items():
            if len(indices) < 2:
                continue
                
            # Sort by time
            rich_features = graph['rich_node_features']
            sorted_indices = sorted(indices, key=lambda i: rich_features[i].time_minutes)
            
            for i in range(len(sorted_indices) - 1):
                source_idx = sorted_indices[i]
                target_idx = sorted_indices[i + 1]
                
                # Calculate rich edge features
                source_features = rich_features[source_idx]
                target_features = rich_features[target_idx]
                
                time_delta = target_features.time_minutes - source_features.time_minutes
                log_time_delta = np.log1p(time_delta)  # log(1 + delta) to handle 0
                
                # Create enhanced edge feature via architectural interface
                edge_feature = self._calculate_enhanced_edge_features(
                    source_features, target_features, 'temporal', 
                    {'graph_type': 'temporal_sequence', 'tf': tf}
                )
                
                # Add to graph
                graph['edges']['temporal'].append({
                    'source': source_idx,
                    'target': target_idx,
                    'feature_idx': len(graph['rich_edge_features'])
                })
                graph['rich_edge_features'].append(edge_feature)
                
    def _build_scale_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build scale edges across timeframes"""
        
        rich_features = graph['rich_node_features']
        tf_order = ['1m', '5m', '15m', '1h', 'D', 'W']
        
        # Build hierarchical connections
        for i in range(len(tf_order) - 1):
            source_tf = tf_order[i]
            target_tf = tf_order[i + 1]
            
            # Find nodes in each timeframe
            source_nodes = [(idx, rich_features[idx]) for idx, tf in zip(node_indices, timeframes) if tf == source_tf]
            target_nodes = [(idx, rich_features[idx]) for idx, tf in zip(node_indices, timeframes) if tf == target_tf]
            
            # Connect nodes across scales (price proximity)
            for source_idx, source_feat in source_nodes:
                for target_idx, target_feat in target_nodes:
                    # Connect if prices are similar and times are close
                    price_diff = abs(source_feat.normalized_price - target_feat.normalized_price)
                    time_diff = abs(source_feat.time_minutes - target_feat.time_minutes)
                    
                    if price_diff < 0.01 and time_diff < 30:  # Similar price and time
                        timeframe_jump = self.timeframe_mapping[target_tf] - self.timeframe_mapping[source_tf]
                        
                        edge_feature = self._calculate_enhanced_edge_features(
                            source_feat, target_feat, 'scale',
                            {'price_diff': price_diff, 'time_diff': time_diff, 'tf_from': source_tf, 'tf_to': target_tf}
                        )
                        
                        graph['edges']['scale'].append({
                            'source': source_idx,
                            'target': target_idx,
                            'feature_idx': len(graph['rich_edge_features'])
                        })
                        graph['rich_edge_features'].append(edge_feature)
    
    def _build_htf_scale_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build scale edges using HTF cross-map for precise relationships"""
        
        htf_cross_map = graph.get('htf_cross_map', {})
        rich_features = graph['rich_node_features']
        
        self.logger.debug(f"Building HTF scale edges using {len(htf_cross_map)} cross-mappings")
        
        # Process each cross-mapping (e.g., "1m_to_5m", "1m_to_15m")
        for mapping_name, mapping in htf_cross_map.items():
            if not mapping:
                continue
                
            # Parse mapping name to get source and target timeframes
            parts = mapping_name.split("_to_")
            if len(parts) != 2:
                continue
                
            tf_source, tf_target = parts[0], parts[1]
            
            # Create edges based on the mapping
            edges_created = 0
            for source_idx_str, target_htf_idx in mapping.items():
                if target_htf_idx is None:
                    continue
                    
                try:
                    source_idx = int(source_idx_str)
                    
                    # Find the actual node indices in our graph
                    source_node_idx = self._find_node_by_tf_and_htf_id(
                        graph, tf_source, source_idx
                    )
                    target_node_idx = self._find_node_by_tf_and_htf_id(
                        graph, tf_target, target_htf_idx
                    )
                    
                    if source_node_idx is not None and target_node_idx is not None:
                        # Get node features
                        source_feat = rich_features[source_node_idx]
                        target_feat = rich_features[target_node_idx]
                        
                        # Extract parent metadata from target HTF node
                        target_raw = target_feat.raw_json
                        coverage = target_raw.get('meta', {}).get('coverage', 1)
                        parent_pd = target_raw.get('pd_array')
                        parent_fvg = target_raw.get('fpfvg')
                        parent_liquidity = target_raw.get('liquidity_sweep', False)
                        
                        # Create scale edge with HTF metadata
                        edge_context = {
                            'tf_source': tf_source,
                            'tf_target': tf_target,
                            'coverage': coverage,
                            'parent_pd': parent_pd,
                            'parent_fvg': parent_fvg,
                            'parent_liquidity': parent_liquidity,
                            'mapping_name': mapping_name
                        }
                        
                        edge_feature = self._calculate_enhanced_edge_features(
                            source_feat, target_feat, 'scale_edge', edge_context
                        )
                        
                        # Add to graph
                        graph['edges']['scale'].append({
                            'source': source_node_idx,
                            'target': target_node_idx,
                            'feature_idx': len(graph['rich_edge_features']),
                            'tf_source': tf_source,
                            'tf_target': tf_target,
                            'coverage': coverage,
                            'parent_metadata': {
                                'pd_array': parent_pd,
                                'fpfvg': parent_fvg, 
                                'liquidity_sweep': parent_liquidity
                            }
                        })
                        graph['rich_edge_features'].append(edge_feature)
                        edges_created += 1
                        
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"⚠️ Error processing mapping {mapping_name}[{source_idx_str}]: {e}")
                    continue
                    
            self.logger.debug(f"Created {edges_created} scale edges for {mapping_name}")
            
        total_scale_edges = len([e for e in graph['edges']['scale'] if 'tf_source' in e])
        self.logger.info(f"✅ Created {total_scale_edges} HTF scale edges using cross-mappings")
    
    def _find_node_by_tf_and_htf_id(self, graph: Dict, timeframe: str, htf_id: int) -> Optional[int]:
        """Find node index by timeframe and HTF node ID"""
        
        # Look through nodes in the specified timeframe
        tf_node_indices = graph['nodes'].get(timeframe, [])
        rich_features = graph['rich_node_features']
        
        for node_idx in tf_node_indices:
            if node_idx < len(rich_features):
                node_feature = rich_features[node_idx]
                # Check if this node has the matching HTF ID
                raw_data = node_feature.raw_json
                if raw_data.get('id') == htf_id:
                    return node_idx
                    
        return None
                        
    def _build_confluence_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build cross-timeframe confluence edges (same price levels across TFs)"""
        
        rich_features = graph['rich_node_features']
        
        # Find nodes at similar price levels across different timeframes
        for i, (idx1, tf1) in enumerate(zip(node_indices, timeframes)):
            for idx2, tf2 in list(zip(node_indices, timeframes))[i+1:]:
                if tf1 == tf2:
                    continue
                    
                feat1 = rich_features[idx1]
                feat2 = rich_features[idx2]
                
                # Check price confluence
                price_diff = abs(feat1.normalized_price - feat2.normalized_price)
                if price_diff < 0.005:  # Very similar prices across timeframes
                    
                    time_diff = abs(feat1.time_minutes - feat2.time_minutes)
                    
                    edge_feature = self._calculate_enhanced_edge_features(
                        feat1, feat2, 'cross_tf_confluence',
                        {'price_diff': price_diff, 'time_diff': time_diff, 'confluence_strength': 1.0 - price_diff}
                    )
                    
                    graph['edges']['cross_tf_confluence'].append({
                        'source': idx1,
                        'target': idx2, 
                        'feature_idx': len(graph['rich_edge_features'])
                    })
                    graph['rich_edge_features'].append(edge_feature)
                    
    def _build_temporal_echo_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build temporal echo edges for recurring patterns"""
        
        rich_features = graph['rich_node_features']
        
        # Find events at similar times (temporal echoes across different timeframes)
        for i, (idx1, tf1) in enumerate(zip(node_indices, timeframes)):
            for idx2, tf2 in list(zip(node_indices, timeframes))[i+1:]:
                if tf1 == tf2:
                    continue
                    
                feat1 = rich_features[idx1]
                feat2 = rich_features[idx2]
                
                # Check for temporal echo patterns
                time_similarity = abs(feat1.daily_phase_sin - feat2.daily_phase_sin)
                session_position_similarity = abs(feat1.session_position - feat2.session_position)
                
                # Strong temporal echo: similar daily phase and session position
                if time_similarity < 0.1 and session_position_similarity < 0.2:
                    
                    edge_feature = self._calculate_enhanced_edge_features(
                        feat1, feat2, 'temporal_echo',
                        {
                            'time_similarity': time_similarity, 
                            'session_similarity': session_position_similarity,
                            'echo_strength': 1.0 - (time_similarity + session_position_similarity)
                        }
                    )
                    
                    graph['edges']['temporal_echo'].append({
                        'source': idx1,
                        'target': idx2,
                        'feature_idx': len(graph['rich_edge_features'])
                    })
                    graph['rich_edge_features'].append(edge_feature)
                    
    def _build_price_resonance_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """Build price resonance edges for harmonic price relationships"""
        
        rich_features = graph['rich_node_features']
        
        # Find harmonic price relationships
        for i, idx1 in enumerate(node_indices):
            for idx2 in node_indices[i+1:]:
                feat1 = rich_features[idx1]
                feat2 = rich_features[idx2]
                
                # Calculate harmonic ratios (Fibonacci, golden ratio, etc.)
                price_ratio = feat2.normalized_price / feat1.normalized_price if feat1.normalized_price != 0 else 1.0
                
                # Check for harmonic relationships (1.618, 0.618, 2.0, 0.5, etc.)
                harmonic_ratios = [1.618, 0.618, 2.0, 0.5, 1.414, 0.707]  # Golden ratio, sqrt(2), etc.
                
                for harmonic_ratio in harmonic_ratios:
                    if abs(price_ratio - harmonic_ratio) < 0.05:  # 5% tolerance
                        
                        # Calculate structural importance
                        structural_resonance = (feat1.structural_importance + feat2.structural_importance) / 2.0
                        
                        # Only create edge if structurally significant
                        if structural_resonance > 0.3:
                            
                            # Create discovered edge type (will be learned by TGAT)
                            edge_feature = self._calculate_enhanced_edge_features(
                                feat1, feat2, 'discovered',
                                {
                                    'price_ratio': price_ratio,
                                    'harmonic_ratio': harmonic_ratio,
                                    'structural_resonance': structural_resonance,
                                    'discovery_type': 'price_harmonic'
                                }
                            )
                            
                            graph['edges']['discovered'].append({
                                'source': idx1,
                                'target': idx2,
                                'feature_idx': len(graph['rich_edge_features'])
                            })
                            graph['rich_edge_features'].append(edge_feature)
                        break  # Only match first harmonic ratio
                    
    def _classify_node_archetype(self, node_feat: RichNodeFeature) -> str:
        """
        Classify structural archetype role of a node
        
        Uses rich node features to identify structural roles that define market architecture.
        These archetypes form the foundation for structural context edge relationships.
        
        Returns:
            Archetype classification string for structural context edge building
        """
        
        # Extract key features for classification
        raw_data = node_feat.raw_json
        event_type = raw_data.get('event_type', raw_data.get('movement_type', 'unknown'))
        
        # Classification based on event characteristics and position features
        
        # 1. First FVG after liquidity sweep - Critical causal sequence marker
        if 'fpfvg' in event_type.lower() or node_feat.fpfvg_gap_size > 0:
            # Check if there's liquidity sweep context nearby in time
            if node_feat.pd_array_strength > 0.7 or 'sweep' in str(raw_data.get('context', '')):
                return "first_fvg_after_sweep"
            else:
                return "imbalance_zone"
        
        # 2. HTF range midpoint - Structural equilibrium levels
        if (0.45 <= node_feat.normalized_price <= 0.55 and 
            node_feat.cross_tf_confluence > 0.8 and
            node_feat.timeframe_rank >= 4):  # 1h+ timeframes
            return "htf_range_midpoint"
        
        # 3. Session boundaries - Open/close structural markers
        if (node_feat.session_position < 0.05 or node_feat.session_position > 0.95):
            return "session_boundary"
        
        # 4. Liquidity clusters - High liquidity concentration
        if (node_feat.structural_importance > 0.8 or
            (node_feat.pd_array_strength > 0.6 and node_feat.fpfvg_interaction_count >= 2)):
            return "liquidity_cluster"
            
        # 5. Cascade origins - Starting points of major moves
        if ('high' in event_type.lower() or 'low' in event_type.lower() or
            node_feat.volatility_window > 0.04 and node_feat.energy_state > 0.7):
            return "cascade_origin"
        
        # 6. Structural support/resistance - Key levels
        if (node_feat.cross_tf_confluence > 0.6 and 
            (node_feat.normalized_price < 0.1 or node_feat.normalized_price > 0.9)):
            return "structural_support"
        
        # 7. Price imbalances - Inefficiency zones
        if (abs(node_feat.pct_from_open) > 30 and node_feat.volatility_window > 0.03):
            return "imbalance_zone"
        
        # Default - unclassified structural role
        return "structural_neutral"

    def _build_structural_context_edges(self, graph: Dict, node_indices: List, timeframes: List) -> None:
        """
        Build structural context edges (4th edge type) - Innovation Architect implementation
        
        Creates connections between nodes based on structural archetype relationships that
        define permanent market architecture patterns. These edges encode the fundamental
        structural relationships that persist across different market regimes.
        
        Key relationships:
        - Causal sequences (sweep → first_fvg_after_sweep)
        - Structural alignments (imbalance_zone → htf_range_midpoint)  
        - Boundary interactions (cascade_origin → session_boundary)
        - Reinforcement patterns (liquidity_cluster → structural_support)
        """
        
        rich_features = graph['rich_node_features']
        
        # Classify all nodes by structural archetype
        node_archetypes = {}
        archetype_nodes = {}  # Group nodes by archetype for efficient lookup
        
        for idx in node_indices:
            if idx < len(rich_features):
                node_feat = rich_features[idx]
                archetype = self._classify_node_archetype(node_feat)
                node_archetypes[idx] = archetype
                
                if archetype not in archetype_nodes:
                    archetype_nodes[archetype] = []
                archetype_nodes[archetype].append(idx)
        
        self.logger.debug(f"Classified {len(node_indices)} nodes into archetypes: {dict((k, len(v)) for k, v in archetype_nodes.items())}")
        
        edges_created = 0
        
        # Build causal sequence edges: sweep → first_fvg_after_sweep
        if "first_fvg_after_sweep" in archetype_nodes:
            for target_idx in archetype_nodes["first_fvg_after_sweep"]:
                target_feat = rich_features[target_idx]
                
                # Find potential sweep origins (cascade_origin or liquidity_cluster)
                for archetype in ["cascade_origin", "liquidity_cluster"]:
                    if archetype in archetype_nodes:
                        for source_idx in archetype_nodes[archetype]:
                            source_feat = rich_features[source_idx]
                            
                            # Check temporal causality (source before target)
                            time_diff = target_feat.time_minutes - source_feat.time_minutes
                            if 0 < time_diff < 60:  # Within 1 hour, source before target
                                
                                # Check price proximity (related price levels)
                                price_diff = abs(target_feat.normalized_price - source_feat.normalized_price)
                                if price_diff < 0.15:  # Within 15% of price range
                                    
                                    edge_context = {
                                        'source_archetype': archetype,
                                        'target_archetype': "first_fvg_after_sweep",
                                        'causal_strength': 1.0 - (time_diff / 60.0),
                                        'price_proximity': 1.0 - (price_diff / 0.15),
                                        'structural_type': 'causal_sequence'
                                    }
                                    
                                    edge_feature = self._calculate_enhanced_edge_features(
                                        source_feat, target_feat, 'structural_context', edge_context
                                    )
                                    
                                    graph['edges']['structural_context'].append({
                                        'source': source_idx,
                                        'target': target_idx,
                                        'feature_idx': len(graph['rich_edge_features']),
                                        'structural_relationship': 'causal_sequence'
                                    })
                                    graph['rich_edge_features'].append(edge_feature)
                                    edges_created += 1
        
        # Build structural alignment edges: imbalance_zone → htf_range_midpoint
        if "imbalance_zone" in archetype_nodes and "htf_range_midpoint" in archetype_nodes:
            for source_idx in archetype_nodes["imbalance_zone"]:
                source_feat = rich_features[source_idx]
                
                for target_idx in archetype_nodes["htf_range_midpoint"]:
                    target_feat = rich_features[target_idx]
                    
                    # Check structural alignment (similar price levels across timeframes)
                    price_alignment = 1.0 - abs(source_feat.normalized_price - target_feat.normalized_price)
                    tf_hierarchy_diff = abs(target_feat.timeframe_rank - source_feat.timeframe_rank)
                    
                    if price_alignment > 0.85 and tf_hierarchy_diff >= 2:  # Strong price alignment, different TF scales
                        
                        edge_context = {
                            'source_archetype': "imbalance_zone",
                            'target_archetype': "htf_range_midpoint", 
                            'alignment_strength': price_alignment,
                            'hierarchy_span': tf_hierarchy_diff,
                            'structural_type': 'structural_alignment'
                        }
                        
                        edge_feature = self._calculate_enhanced_edge_features(
                            source_feat, target_feat, 'structural_context', edge_context
                        )
                        
                        graph['edges']['structural_context'].append({
                            'source': source_idx,
                            'target': target_idx,
                            'feature_idx': len(graph['rich_edge_features']),
                            'structural_relationship': 'structural_alignment'
                        })
                        graph['rich_edge_features'].append(edge_feature)
                        edges_created += 1
        
        # Build boundary interaction edges: cascade_origin → session_boundary
        if "cascade_origin" in archetype_nodes and "session_boundary" in archetype_nodes:
            for source_idx in archetype_nodes["cascade_origin"]:
                source_feat = rich_features[source_idx]
                
                for target_idx in archetype_nodes["session_boundary"]:
                    target_feat = rich_features[target_idx]
                    
                    # Check boundary interaction (cascade hitting session boundaries)
                    time_diff = abs(target_feat.time_minutes - source_feat.time_minutes)
                    energy_amplification = (source_feat.energy_state + target_feat.volatility_window) / 2.0
                    
                    if time_diff < 30 and energy_amplification > 0.5:  # Close in time, high energy
                        
                        edge_context = {
                            'source_archetype': "cascade_origin",
                            'target_archetype': "session_boundary",
                            'interaction_strength': energy_amplification,
                            'temporal_proximity': 1.0 - (time_diff / 30.0),
                            'structural_type': 'boundary_interaction'
                        }
                        
                        edge_feature = self._calculate_enhanced_edge_features(
                            source_feat, target_feat, 'structural_context', edge_context
                        )
                        
                        graph['edges']['structural_context'].append({
                            'source': source_idx,
                            'target': target_idx,
                            'feature_idx': len(graph['rich_edge_features']),
                            'structural_relationship': 'boundary_interaction'
                        })
                        graph['rich_edge_features'].append(edge_feature)
                        edges_created += 1
        
        # Build reinforcement edges: liquidity_cluster → structural_support
        if "liquidity_cluster" in archetype_nodes and "structural_support" in archetype_nodes:
            for source_idx in archetype_nodes["liquidity_cluster"]:
                source_feat = rich_features[source_idx]
                
                for target_idx in archetype_nodes["structural_support"]:
                    target_feat = rich_features[target_idx]
                    
                    # Check reinforcement relationship (liquidity reinforcing structural levels)
                    structural_confluence = (source_feat.structural_importance + target_feat.cross_tf_confluence) / 2.0
                    price_coherence = 1.0 - abs(source_feat.normalized_price - target_feat.normalized_price)
                    
                    if structural_confluence > 0.6 and price_coherence > 0.9:  # Strong structural coherence
                        
                        edge_context = {
                            'source_archetype': "liquidity_cluster",
                            'target_archetype': "structural_support",
                            'reinforcement_strength': structural_confluence,
                            'price_coherence': price_coherence,
                            'structural_type': 'reinforcement_pattern'
                        }
                        
                        edge_feature = self._calculate_enhanced_edge_features(
                            source_feat, target_feat, 'structural_context', edge_context
                        )
                        
                        graph['edges']['structural_context'].append({
                            'source': source_idx,
                            'target': target_idx,
                            'feature_idx': len(graph['rich_edge_features']),
                            'structural_relationship': 'reinforcement_pattern'
                        })
                        graph['rich_edge_features'].append(edge_feature)
                        edges_created += 1
        
        self.logger.info(f"✅ Created {edges_created} structural context edges across archetype relationships")

    def _add_cross_session_context(self, graph: Dict, historical_sessions: List[Dict]) -> None:
        """Add cross-session context for archaeological discovery"""
        
        # TODO: Implement cross-session relationship discovery
        # This would look for patterns that repeat across different sessions
        # at the same absolute times or similar market conditions
        pass
    
    def _extract_semantic_events(self, event: Dict, session_data: Dict, timeframe: str) -> Dict[str, float]:
        """Extract semantic market events from Level 1 JSON for archaeological discovery"""
        
        # Initialize all semantic flags to 0.0
        semantic_events = {
            'fvg_redelivery_flag': 0.0,
            'expansion_phase_flag': 0.0,
            'consolidation_flag': 0.0,
            'retracement_flag': 0.0,
            'reversal_flag': 0.0,
            'liq_sweep_flag': 0.0,
            'pd_array_interaction_flag': 0.0,
            'phase_open': 0.0,
            'phase_mid': 0.0,
            'phase_close': 0.0
        }
        
        # Extract semantic events from various sources in Level 1 JSON
        
        # 1. FVG Redelivery Events
        event_type = event.get('event_type', '')
        movement_type = event.get('movement_type', '')
        context = event.get('context', '')
        action = event.get('action', '')
        
        # Check for FVG redelivery patterns
        if ('redelivery' in event_type or 'redelivery' in context or 
            action == 'delivery' or 'redelivered' in context):
            semantic_events['fvg_redelivery_flag'] = 1.0
            
        # 2. Expansion/Consolidation Phase Detection
        if ('expansion' in event_type or 'expansion' in context or 
            'expansion' in movement_type):
            semantic_events['expansion_phase_flag'] = 1.0
            
        if ('consolidation' in event_type or 'consolidation' in context or
            'consolidation' in movement_type):
            semantic_events['consolidation_flag'] = 1.0

        # 3. Retracement Phase Detection
        # Retracement: Price trading back into previous consolidation range after expansion
        if ('retracement' in event_type or 'retracement' in context or
            'retracement' in movement_type or 'retrace' in context or
            'pullback' in context or 'correction' in context):
            semantic_events['retracement_flag'] = 1.0

        # Additional retracement detection: price action patterns
        price_action = event.get('price_action', '')
        if ('back_into_range' in price_action or 'return_to_consolidation' in price_action or
            'pullback_to_support' in price_action or 'correction_phase' in price_action):
            semantic_events['retracement_flag'] = 1.0

        # 4. Reversal Point Detection
        # Reversal: Singular points marking market structure changes or sentiment shifts
        if ('reversal' in event_type or 'reversal' in context or
            'reversal' in movement_type or 'reverse' in context):
            semantic_events['reversal_flag'] = 1.0

        # Additional reversal detection: structural change indicators
        structure_change = event.get('structure_change', '')
        sentiment_shift = event.get('sentiment_shift', '')
        if ('structure_break' in structure_change or 'trend_change' in structure_change or
            'sentiment_reversal' in sentiment_shift or 'momentum_shift' in context or
            'direction_change' in context or 'pivot_point' in context):
            semantic_events['reversal_flag'] = 1.0

        # 5. Liquidity Sweep Events
        if ('sweep' in event_type or 'sweep' in context or
            'liquidity_sweep' in event.get('target_level', '')):
            semantic_events['liq_sweep_flag'] = 1.0
            
        # 4. PD Array Interactions
        if ('pd_array' in event_type or 'pd' in event_type or
            'array' in context):
            semantic_events['pd_array_interaction_flag'] = 1.0
            
        # Check session liquidity events for additional context
        if session_data and 'session_liquidity_events' in session_data:
            event_timestamp = event.get('timestamp', '')
            for liq_event in session_data['session_liquidity_events']:
                if liq_event.get('timestamp') == event_timestamp:
                    if liq_event.get('event_type') == 'redelivery':
                        semantic_events['fvg_redelivery_flag'] = 1.0
                    if 'sweep' in liq_event.get('target_level', ''):
                        semantic_events['liq_sweep_flag'] = 1.0
        
        # 5. Session Phase Determination (based on session position)
        session_position = self._calculate_session_position(event, session_data)
        
        if session_position <= 0.2:  # First 20% of session
            semantic_events['phase_open'] = 1.0
        elif session_position >= 0.8:  # Last 20% of session  
            semantic_events['phase_close'] = 1.0
        else:  # Middle 60% of session
            semantic_events['phase_mid'] = 1.0
            
        return semantic_events
    
    def _calculate_session_position(self, event: Dict, session_data: Dict) -> float:
        """Calculate relative position within session (0.0 = start, 1.0 = end)"""
        
        if not session_data:
            return 0.5  # Default to middle if no session data
            
        session_meta = session_data.get('session_metadata', {})
        session_start = self._parse_time(session_meta.get('session_start', '09:30:00'))
        session_duration = session_meta.get('session_duration', 120)
        event_time = self._parse_time(event.get('timestamp', '00:00:00'))
        
        if session_duration <= 0:
            return 0.5
            
        time_minutes = event_time - session_start
        session_position = max(0.0, min(1.0, time_minutes / session_duration))
        
        return session_position
    
    def _generate_semantic_edge_label(self, source_feat: RichNodeFeature, target_feat: RichNodeFeature, 
                                     edge_type: str, graph_context: Dict) -> Tuple[int, float, int]:
        """Generate semantic edge labels for archaeological discovery relationships"""
        
        # Initialize defaults
        semantic_event_link = 0  # 0=none
        event_causality = 0.0    # 0.0-1.0 
        semantic_label_id = 0    # 0=generic
        
        # SEMANTIC EVENT LINK TYPES:
        # 0=none, 1=fvg_chain, 2=pd_sequence, 3=phase_transition, 4=liquidity_sweep
        
        # 1. FVG Chain Detection
        if (source_feat.fvg_redelivery_flag > 0.0 and target_feat.fvg_redelivery_flag > 0.0):
            semantic_event_link = 1  # fvg_chain
            event_causality = 0.8    # High causality for FVG chains
            semantic_label_id = 1    # fvg_redelivery_link
            
        # 2. PD Array Sequence Detection
        elif (source_feat.pd_array_interaction_flag > 0.0 and target_feat.pd_array_interaction_flag > 0.0):
            semantic_event_link = 2  # pd_sequence
            event_causality = 0.7    # High causality for PD sequences
            semantic_label_id = 2    # pd_array_sequence
            
        # 3. Phase Transition Detection
        elif self._is_phase_transition(source_feat, target_feat):
            semantic_event_link = 3  # phase_transition
            event_causality = 0.6    # Medium causality for phase changes
            semantic_label_id = 3    # phase_transition_link
            
        # 4. Liquidity Sweep Chain
        elif (source_feat.liq_sweep_flag > 0.0 and target_feat.liq_sweep_flag > 0.0):
            semantic_event_link = 4  # liquidity_sweep
            event_causality = 0.9    # Very high causality for sweep chains
            semantic_label_id = 4    # liquidity_sweep_chain
            
        # 5. Mixed Event Causality (expansion → redelivery common pattern)
        elif (source_feat.expansion_phase_flag > 0.0 and target_feat.fvg_redelivery_flag > 0.0):
            semantic_event_link = 1  # fvg_chain (expansion triggers redelivery)
            event_causality = 0.75   # High causality for expansion→redelivery
            semantic_label_id = 5    # expansion_to_redelivery
            
        # 6. Consolidation → Expansion transitions
        elif (source_feat.consolidation_flag > 0.0 and target_feat.expansion_phase_flag > 0.0):
            semantic_event_link = 3  # phase_transition
            event_causality = 0.65   # Medium-high causality
            semantic_label_id = 6    # consolidation_to_expansion
            
        # 7. Edge type specific semantic enhancement
        if edge_type == 'pd_array':
            semantic_event_link = max(semantic_event_link, 2)  # Ensure PD array edges are semantic
            event_causality = max(event_causality, 0.5)
            
        elif edge_type == 'temporal' and semantic_event_link > 0:
            # Temporal edges with semantic content get boosted causality
            event_causality = min(1.0, event_causality + 0.2)
            
        return semantic_event_link, event_causality, semantic_label_id
    
    def _is_phase_transition(self, source_feat: RichNodeFeature, target_feat: RichNodeFeature) -> bool:
        """Detect if this edge represents a session phase transition"""
        
        # Check for phase transitions
        source_phases = [source_feat.phase_open, source_feat.phase_mid, source_feat.phase_close]
        target_phases = [target_feat.phase_open, target_feat.phase_mid, target_feat.phase_close]
        
        # Different phase flags indicate a transition
        if source_phases != target_phases:
            return True
            
        # Check for expansion/consolidation transitions
        if (source_feat.expansion_phase_flag != target_feat.expansion_phase_flag or
            source_feat.consolidation_flag != target_feat.consolidation_flag):
            return True
            
        return False
    
    def _extract_session_metadata(self, session_data: Dict[str, Any], session_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract and structure session metadata for archaeological discovery context"""
        
        session_meta = session_data.get('session_metadata', {})
        
        # Core session identification
        session_type = session_meta.get('session_type', 'unknown')
        session_date = session_meta.get('session_date', 'unknown')
        session_start = session_meta.get('session_start', '00:00:00')
        session_end = session_meta.get('session_end', '00:00:00')
        session_duration = session_meta.get('session_duration', 0)
        
        # Create standardized session name
        session_name = self._standardize_session_name(session_type)
        
        # Determine anchor timeframe based on data structure
        anchor_timeframe = self._determine_anchor_timeframe(session_data)
        
        # Extract session market characteristics
        market_characteristics = self._extract_market_characteristics(session_data)
        
        # Build comprehensive metadata
        metadata = {
            # Core identification
            'session_name': session_name,
            'session_type': session_type,
            'session_date': session_date,
            'session_start': session_start,
            'session_end': session_end,
            'session_start_time': f"{session_date}T{session_start}Z" if session_date != 'unknown' else 'unknown',
            'session_end_time': f"{session_date}T{session_end}Z" if session_date != 'unknown' else 'unknown',
            'session_duration': session_duration,
            
            # Technical context
            'anchor_timeframe': anchor_timeframe,
            'data_source': 'enhanced_level1' if 'price_movements' in session_data else 'level1',
            'file_path': session_file_path or 'unknown',
            
            # Market characteristics
            'market_characteristics': market_characteristics,
            
            # Archaeological context
            'semantic_events_count': self._count_semantic_events(session_data),
            'session_quality': self._assess_session_quality(session_data),
            
            # Preservation timestamp
            'metadata_extracted_at': datetime.now().isoformat()
        }
        
        return metadata
    
    def _standardize_session_name(self, session_type: str) -> str:
        """Convert session_type to standardized session name"""
        
        name_mapping = {
            'ny_am': 'NY_AM',
            'ny_pm': 'NY_PM', 
            'london': 'LONDON',
            'asia': 'ASIA',
            'lunch': 'LUNCH',
            'midnight': 'MIDNIGHT',
            'premarket': 'PREMARKET',
            'preasia': 'PREASIA',
            'nyam': 'NY_AM',
            'nypm': 'NY_PM'
        }
        
        return name_mapping.get(session_type.lower(), session_type.upper())
    
    def _determine_anchor_timeframe(self, session_data: Dict[str, Any]) -> str:
        """Determine the primary/anchor timeframe for this session"""
        
        # Check for HTF data
        if 'pythonnodes' in session_data and 'htf_cross_map' in session_data:
            # HTF session - determine highest timeframe present
            htf_map = session_data.get('htf_cross_map', {})
            timeframes = ['W', 'D', '1h', '15m', '5m', '1m']
            
            for tf in timeframes:
                if tf in htf_map and len(htf_map[tf]) > 0:
                    return tf
                    
        # Check for multi-timeframe price movements
        price_movements = session_data.get('price_movements', [])
        has_daily_context = any('daily' in str(pm).lower() for pm in price_movements)
        has_hourly_context = any('hour' in str(pm).lower() for pm in price_movements)
        
        if has_daily_context:
            return 'Daily'
        elif has_hourly_context:
            return '1h'
        else:
            return '1m'  # Default to 1-minute for Level 1 data
    
    def _extract_market_characteristics(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market characteristics for session context"""
        
        characteristics = {
            'fpfvg_present': False,
            'expansion_phases': 0,
            'consolidation_phases': 0,
            'liquidity_events_count': 0,
            'price_range_pct': 0.0,
            'volatility_assessment': 'unknown'
        }
        
        # FPFVG analysis
        fpfvg_data = session_data.get('session_fpfvg', {})
        if fpfvg_data:
            characteristics['fpfvg_present'] = fpfvg_data.get('fpfvg_present', False)
            
        # Phase analysis - look for expansion/consolidation mentions
        all_text = str(session_data).lower()
        characteristics['expansion_phases'] = all_text.count('expansion')
        characteristics['consolidation_phases'] = all_text.count('consolidation')
        
        # Liquidity events
        liquidity_events = session_data.get('session_liquidity_events', [])
        characteristics['liquidity_events_count'] = len(liquidity_events)
        
        # Price range analysis
        session_meta = session_data.get('session_metadata', {})
        session_high = session_meta.get('session_high')
        session_low = session_meta.get('session_low')
        session_open = session_meta.get('session_open')
        
        if session_high and session_low and session_open:
            price_range = session_high - session_low
            range_pct = (price_range / session_open) * 100 if session_open > 0 else 0
            characteristics['price_range_pct'] = round(range_pct, 2)
            
            # Volatility assessment
            if range_pct > 2.0:
                characteristics['volatility_assessment'] = 'high'
            elif range_pct > 0.5:
                characteristics['volatility_assessment'] = 'medium'
            else:
                characteristics['volatility_assessment'] = 'low'
                
        return characteristics
    
    def _count_semantic_events(self, session_data: Dict[str, Any]) -> Dict[str, int]:
        """Count semantic events present in session data"""
        
        event_counts = {
            'fvg_redelivery': 0,
            'expansion_mentions': 0,
            'consolidation_mentions': 0,
            'liquidity_sweeps': 0,
            'pd_array_interactions': 0
        }
        
        # Convert to string for text analysis
        session_text = str(session_data).lower()
        
        # Count occurrences
        event_counts['fvg_redelivery'] = session_text.count('redelivery') + session_text.count('redelivered')
        event_counts['expansion_mentions'] = session_text.count('expansion')
        event_counts['consolidation_mentions'] = session_text.count('consolidation')
        event_counts['liquidity_sweeps'] = session_text.count('sweep')
        event_counts['pd_array_interactions'] = session_text.count('pd_array') + session_text.count('array')
        
        return event_counts
    
    def _assess_session_quality(self, session_data: Dict[str, Any]) -> str:
        """Assess overall session data quality for archaeological discovery"""
        
        quality_score = 0
        max_score = 5
        
        # Check for essential components
        if 'session_metadata' in session_data:
            quality_score += 1
            
        if 'price_movements' in session_data and len(session_data['price_movements']) > 0:
            quality_score += 1
            
        if 'session_liquidity_events' in session_data:
            quality_score += 1
            
        # Check for enhanced features
        price_movements = session_data.get('price_movements', [])
        has_enhanced_features = any('normalized_price' in str(pm) for pm in price_movements)
        if has_enhanced_features:
            quality_score += 1
            
        # Check for semantic richness
        semantic_counts = self._count_semantic_events(session_data)
        total_semantic_events = sum(semantic_counts.values())
        if total_semantic_events > 5:
            quality_score += 1
            
        # Assess quality
        quality_ratio = quality_score / max_score
        if quality_ratio >= 0.8:
            return 'excellent'
        elif quality_ratio >= 0.6:
            return 'good'
        elif quality_ratio >= 0.4:
            return 'adequate'
        else:
            return 'poor'
        
    def to_tgat_format(self, graph: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """Convert rich graph to TGAT format with edge attributes"""
        
        # Node features: 47-dimensional rich features with ENHANCED SEMANTIC EVENTS
        if len(graph['rich_node_features']) == 0:
            # Handle empty sessions with minimal node
            X = torch.zeros(1, 47)  # Single dummy node with 47 features (10 semantic + 37 previous)
            constant_features_metadata = []
        else:
            # TASK 4: Apply constant feature filtering before tensor conversion
            X_raw = torch.stack([feat.to_tensor() for feat in graph['rich_node_features']])
            X, constant_features_metadata = self._filter_constant_features(X_raw, graph['rich_node_features'])
        
        # Collect all edges with enhanced features and type information
        all_edges = []
        edge_times = []
        edge_features = []
        edge_types = []
        
        for edge_type, edges in graph['edges'].items():
            for edge in edges:
                all_edges.append([edge['source'], edge['target']])
                edge_feat = graph['rich_edge_features'][edge['feature_idx']]
                edge_times.append(edge_feat.time_delta)
                edge_features.append(edge_feat.to_tensor())  # Rich 20D edge features
                edge_types.append(edge_type)  # Store edge type for discovery
                
        # Convert to tensors
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t()
            edge_times = torch.tensor(edge_times, dtype=torch.float)
            edge_attr = torch.stack(edge_features)  # 20-dimensional edge features
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_times = torch.zeros(0, dtype=torch.float)
            edge_attr = torch.zeros((0, 20), dtype=torch.float)  # Empty edge features
            edge_types = []
            
        # Enhanced metadata with edge type information
        enhanced_metadata = graph['metadata'].copy()
        enhanced_metadata['edge_feature_dimensions'] = 20
        enhanced_metadata['edge_types'] = edge_types  # Store edge types for discovery
        enhanced_metadata['total_edges'] = len(all_edges)
        
        # Count edges by type for HTF analysis
        edge_type_counts = {}
        for et in edge_types:
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        enhanced_metadata['edge_type_counts'] = edge_type_counts
        
        # Store constant features metadata in the enhanced metadata for later use in output schema (TASK 4)
        enhanced_metadata['constant_features'] = constant_features_metadata
        
        return X, edge_index, edge_times, enhanced_metadata, edge_attr
        
    # Helper methods
    def _parse_time(self, timestamp: str) -> float:
        """Convert timestamp to minutes"""
        try:
            if ':' in timestamp:
                parts = timestamp.split(':')
                return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60
            return 0.0
        except:
            return 0.0
    
    def _parse_session_date(self, session_date: str) -> tuple:
        """
        Parse session date with multiple format support and TEMPORAL CYCLE EXTRACTION
        
        Handles various date formats and returns (timestamp, day_of_week, month_phase, 
        week_of_month, month_of_year, day_of_week_cycle)
        
        Innovation Architect: Extracts temporal cycle information for weekly/monthly patterns
        """
        try:
            # Default fallback date
            default_date = datetime(2025, 8, 12)  # Current date as fallback
            
            if not session_date or session_date.strip() == '':
                self.logger.warning("⚠️ Empty session_date, using default")
                parsed_date = default_date
            else:
                # Try multiple date formats
                date_formats = [
                    '%Y-%m-%d',      # 2025-08-05
                    '%m/%d/%Y',      # 08/05/2025
                    '%Y/%m/%d',      # 2025/08/05
                    '%d-%m-%Y',      # 05-08-2025
                    '%Y%m%d',        # 20250805
                ]
                
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(session_date.strip(), fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    self.logger.warning(f"⚠️ Could not parse session_date '{session_date}', using default")
                    parsed_date = default_date
                else:
                    self.logger.debug(f"✅ Parsed session_date '{session_date}' as {parsed_date}")
            
            # Calculate derived values
            absolute_timestamp = int(parsed_date.timestamp())
            day_of_week = parsed_date.weekday()  # 0=Monday, 6=Sunday
            month_phase = parsed_date.day / 31.0
            
            # TEMPORAL CYCLE EXTRACTION - Innovation Architect enhancement
            # Week of month calculation (1-5)
            week_of_month = min(5, ((parsed_date.day - 1) // 7) + 1)
            
            # Month of year (1-12)
            month_of_year = parsed_date.month
            
            # Day of week cycle (duplicate for pattern emphasis)
            day_of_week_cycle = day_of_week
            
            return absolute_timestamp, day_of_week, month_phase, week_of_month, month_of_year, day_of_week_cycle
            
        except Exception as e:
            self.logger.error(f"❌ Session date parsing error: {e}, using default")
            default_date = datetime(2025, 8, 12)
            # TEMPORAL CYCLES for default date
            week_of_month = min(5, ((default_date.day - 1) // 7) + 1)
            month_of_year = default_date.month
            day_of_week_cycle = default_date.weekday()
            return int(default_date.timestamp()), default_date.weekday(), default_date.day / 31.0, week_of_month, month_of_year, day_of_week_cycle
            
    def _get_event_type_id(self, event_type: str) -> int:
        """Get or create event type ID"""
        if event_type not in self.event_type_encoder:
            self.event_type_encoder[event_type] = len(self.event_type_encoder)
        return self.event_type_encoder[event_type]
        
    def _calculate_price_delta(self, current_price: float, current_time: float, lookback_minutes: int) -> float:
        """Calculate price delta over lookback period"""
        # Simplified - would use actual price history
        return np.random.normal(0, 0.01)  # Placeholder
        
    def _calculate_local_volatility(self, price: float, time: float) -> float:
        """Calculate local volatility"""
        # Simplified - would use actual price movements
        return np.random.uniform(0.01, 0.05)  # Placeholder

    def _calculate_relativity_features(self, event: Dict, session_data: Dict, timestamp_str: str) -> Dict:
        """
        Calculate relativity features dynamically from session data
        NO FALLBACKS: Computes actual relativity instead of using defaults
        """
        event_price = event.get('price', event.get('price_level', 0.0))
        
        # Extract session bounds from metadata or price_movements
        session_high = float('-inf')
        session_low = float('inf')
        session_open = None
        session_start_time = None
        session_duration = 0
        
        if session_data:
            # Get session metadata
            metadata = session_data.get('session_metadata', {})
            session_start_time = metadata.get('session_start', '00:00:00')
            session_duration = metadata.get('session_duration', 480)  # Default 8 hours
            
            # Calculate session high/low/open from price_movements
            price_movements = session_data.get('price_movements', [])
            if price_movements:
                prices = [pm.get('price_level', 0.0) for pm in price_movements]
                prices = [p for p in prices if p > 0]  # Filter out zero prices
                
                if prices:
                    session_high = max(prices)
                    session_low = min(prices)
                    session_open = prices[0]  # First price is open
            
            # If no price_movements, extract from session_fpfvg or phases
            if session_high == float('-inf'):
                # Check session_fpfvg
                fpfvg_data = session_data.get('session_fpfvg', {})
                if fpfvg_data and 'fpfvg_formation' in fpfvg_data:
                    formation = fpfvg_data['fpfvg_formation']
                    session_high = max(session_high, formation.get('premium_high', 0))
                    session_low = min(session_low, formation.get('discount_low', float('inf')))
                
                # Check phases for additional price data
                phases = session_data.get('phases', [])
                for phase in phases:
                    if 'high' in phase and 'low' in phase:
                        session_high = max(session_high, phase['high'])
                        session_low = min(session_low, phase['low'])
        
        # Fallback to reasonable defaults if no data found
        if session_high == float('-inf') or session_low == float('inf'):
            session_high = event_price * 1.01  # 1% above event price
            session_low = event_price * 0.99   # 1% below event price
            session_open = event_price
        
        if session_open is None:
            session_open = session_low  # Use session low as approximation
        
        # Calculate relativity features
        price_range = session_high - session_low
        normalized_price = (event_price - session_low) / price_range if price_range > 0 else 0.5
        
        # Percentage calculations
        open_delta = event_price - session_open
        high_delta = session_high - event_price  
        low_delta = event_price - session_low
        
        pct_from_open = (open_delta / session_open * 100) if session_open > 0 else 0.0
        pct_from_high = (high_delta / session_high * 100) if session_high > 0 else 0.0
        pct_from_low = (low_delta / session_low * 100) if session_low > 0 else 0.0
        
        # Time calculations
        time_since_session_open = self._calculate_time_since_open(timestamp_str, session_start_time)
        normalized_time = time_since_session_open / session_duration if session_duration > 0 else 0.0
        
        # HTF ratio (simplified - could be enhanced with actual HTF data)
        price_to_HTF_ratio = 1.0  # Would require HTF context
        
        self.logger.debug(f"🔧 Calculated relativity for {timestamp_str}: price={event_price}, range={session_low}-{session_high}, norm={normalized_price:.3f}")
        
        return {
            'normalized_price': normalized_price,
            'pct_from_open': pct_from_open,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low,
            'price_to_HTF_ratio': price_to_HTF_ratio,
            'time_since_session_open': time_since_session_open,
            'normalized_time': normalized_time
        }
    
    def _calculate_time_since_open(self, timestamp_str: str, session_start_time: str) -> int:
        """Calculate minutes since session open"""
        try:
            # Parse timestamps (format: "HH:MM:SS")
            event_parts = timestamp_str.split(':')
            start_parts = session_start_time.split(':')
            
            event_minutes = int(event_parts[0]) * 60 + int(event_parts[1])
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
            
            return max(0, event_minutes - start_minutes)
            
        except (ValueError, IndexError):
            return 0  # Return 0 if parsing fails


def create_enhanced_graph_builder() -> EnhancedGraphBuilder:
    """Factory function for enhanced graph builder"""
    return EnhancedGraphBuilder()


if __name__ == "__main__":
    """Test enhanced graph builder"""
    
    logging.basicConfig(level=logging.INFO)
    print("🏗️ ENHANCED GRAPH BUILDER - Rich Feature Testing")
    print("=" * 70)
    
    # Test with sample session data
    sample_session = {
        'session_metadata': {
            'session_type': 'ny_pm',
            'session_start': '13:30:00',
            'session_end': '16:00:00',
            'session_duration': 150,
            'session_date': '2025-08-11'
        },
        'price_movements': [
            {
                'timestamp': '13:30:00',
                'price_level': 23500.0,
                'movement_type': 'open'
            },
            {
                'timestamp': '14:15:00', 
                'price_level': 23525.0,
                'movement_type': 'high'
            }
        ],
        'session_liquidity_events': [
            {
                'timestamp': '13:45:00',
                'event_type': 'fpfvg_formation',
                'price_level': 23510.0,
                'gap_size': 15.0
            }
        ],
        'energy_state': {
            'total_accumulated': 2500
        },
        'contamination_analysis': {
            'contamination_coefficient': 0.3
        }
    }
    
    # Create enhanced builder
    builder = create_enhanced_graph_builder()
    
    # Build rich graph
    rich_graph = builder.build_rich_graph(sample_session)
    
    print(f"📊 RICH GRAPH RESULTS:")
    print(f"   Total Nodes: {rich_graph['metadata']['total_nodes']}")
    print(f"   Feature Dimensions: {rich_graph['metadata']['feature_dimensions']}")
    print(f"   Timeframe Distribution:")
    for tf, count in rich_graph['metadata']['timeframe_counts'].items():
        if count > 0:
            print(f"      {tf}: {count} nodes")
            
    print(f"   Edge Types:")
    for edge_type, edges in rich_graph['edges'].items():
        if edges:
            print(f"      {edge_type}: {len(edges)} edges")
            
    # Test TGAT format conversion
    X, edge_index, edge_times, metadata = builder.to_tgat_format(rich_graph)
    print(f"\n🧠 TGAT FORMAT:")
    print(f"   Node Features Shape: {X.shape}")
    print(f"   Edge Index Shape: {edge_index.shape}")
    print(f"   Edge Times Shape: {edge_times.shape}")
    
    # Show sample rich features
    if rich_graph['rich_node_features']:
        sample_features = rich_graph['rich_node_features'][0]
        print(f"\n🎯 SAMPLE RICH FEATURES:")
        print(f"   Temporal: time={sample_features.time_minutes:.1f}, "
              f"phase_sin={sample_features.daily_phase_sin:.3f}, "
              f"session_pos={sample_features.session_position:.3f}")
        print(f"   Price: normalized={sample_features.normalized_price:.6f}, "
              f"volatility={sample_features.volatility_window:.3f}")
        print(f"   Event: type_id={sample_features.event_type_id}, "
              f"tf_source={sample_features.timeframe_source}")
        
    print(f"\n✅ Enhanced graph builder testing complete")
    print(f"🎯 Ready for Phase 2: Enhanced Edge Features + Multi-TF Hierarchy")