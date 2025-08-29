"""
Archaeological Zone Detection Agent for IRONFORGE
==================================================

IRONFORGE Production Archaeological Zone Intelligence Agent
Integrates with Discovery Stage pipeline for temporal pattern enhancement

Core Capabilities:
- 40% dimensional anchor point detection with 7.55-point precision
- Temporal non-locality validation using Theory B forward positioning  
- Session boundary isolation and within-session learning preservation
- Enhanced Graph Builder integration for 45D/51D feature enhancement
- Sub-3s session processing performance optimization

Archaeological Principles:
- Dimensional anchors: previous_day_range * 0.40 = zone threshold
- Theory B: Events position relative to FINAL session range, not intermediate states
- Session isolation: Absolute boundary respect, no cross-session contamination
- HTF compliance: Last-closed only (f45-f50), never intra-candle data
- Forward temporal echoes: Information propagates through eventual completion patterns

Integration: Enhances TGAT discovery capabilities with archaeological intelligence
Performance: <1s zone detection, >95% anchor accuracy, >87% authenticity threshold
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from ironforge.constants import (
    EVENT_TYPES,
    EDGE_INTENTS, 
    NODE_FEATURE_DIM_STANDARD,
    NODE_FEATURE_DIM_HTF,
    EDGE_FEATURE_DIM,
)
from ironforge.integration.ironforge_container import get_ironforge_container
from ironforge.contracts.validators import ContractViolationError

# Import agent-specific modules
from .ironforge_config import ArchaeologicalConfig
from .tools import (
    ZoneAnalyzer,
    TemporalNonLocalityValidator, 
    DimensionalAnchorCalculator,
    TheoryBValidator
)
from .contracts import ArchaeologicalContractValidator
from .performance import ArchaeologicalPerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ArchaeologicalZone:
    """40% Dimensional anchor zone with temporal non-locality properties"""
    anchor_point: float
    zone_range: Tuple[float, float]
    confidence: float
    temporal_offset: float
    precision_score: float  # Target: 7.55
    session_id: str
    discovery_timestamp: float
    theory_b_alignment: bool
    forward_positioning: Dict[str, Any]
    authenticity_score: float  # Must be >87


@dataclass
class ArchaeologicalAnalysis:
    """Complete archaeological analysis for a session"""
    session_id: str
    archaeological_zones: List[ArchaeologicalZone]
    dimensional_analysis: Dict[str, Any]
    temporal_echoes: Dict[str, Any]
    performance_metrics: Dict[str, float]
    contract_validation: Dict[str, bool]
    enhanced_features: Optional[torch.Tensor] = None


class ArchaeologicalZoneDetector:
    """
    IRONFORGE Archaeological Zone Detection Agent
    
    Production-grade archaeological intelligence for temporal pattern discovery.
    Integrates seamlessly with Enhanced Graph Builder and TGAT discovery pipeline.
    
    Maintains strict IRONFORGE architectural compliance:
    - Golden invariants: 6 events, 4 edge intents, 51D/20D features
    - Session isolation: Absolute boundary respect
    - Performance requirements: <3s processing, >87% authenticity
    - HTF compliance: Last-closed only, no intra-candle contamination
    """
    
    def __init__(
        self,
        config: Optional[ArchaeologicalConfig] = None,
        container: Optional[Any] = None
    ):
        """Initialize archaeological zone detector with IRONFORGE integration"""
        self.config = config or ArchaeologicalConfig()
        self.container = container or get_ironforge_container()
        
        # Initialize core analysis components
        self.zone_analyzer = ZoneAnalyzer(self.config)
        self.temporal_validator = TemporalNonLocalityValidator(self.config)
        self.anchor_calculator = DimensionalAnchorCalculator(self.config)
        self.theory_b_validator = TheoryBValidator(self.config)
        
        # Contract and performance monitoring
        self.contract_validator = ArchaeologicalContractValidator()
        self.performance_monitor = ArchaeologicalPerformanceMonitor()
        
        # IRONFORGE component integration
        self.enhanced_graph_builder = None
        self.discovery_engine = None
        
        logger.info("Archaeological Zone Detector initialized for IRONFORGE integration")
    
    def initialize_ironforge_components(self) -> None:
        """Lazy initialization of IRONFORGE components for session independence"""
        try:
            self.enhanced_graph_builder = self.container.get_enhanced_graph_builder()
            self.discovery_engine = self.container.get_tgat_discovery()
            logger.debug("IRONFORGE components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IRONFORGE components: {e}")
            raise
    
    def detect_archaeological_zones(
        self, 
        session_data: pd.DataFrame,
        previous_session_data: Optional[pd.DataFrame] = None,
        enhanced_graph: Optional[nx.Graph] = None
    ) -> ArchaeologicalAnalysis:
        """
        Primary archaeological zone detection interface
        
        Detects 40% dimensional anchor points with temporal non-locality analysis.
        Integrates with IRONFORGE pipeline while maintaining session isolation.
        
        Args:
            session_data: Current session event data (nodes/edges)
            previous_session_data: Previous session for range calculation (isolated)
            enhanced_graph: Optional pre-built enhanced graph from IRONFORGE pipeline
            
        Returns:
            Complete archaeological analysis with zones, metrics, and validation
        """
        start_time = time.time()
        
        # Performance monitoring initialization
        self.performance_monitor.start_session_analysis()
        
        try:
            # Session identifier for tracking
            session_id = self._extract_session_id(session_data)
            logger.info(f"Starting archaeological analysis for session: {session_id}")
            
            # Contract validation - enforce golden invariants
            self._validate_session_contracts(session_data, session_id)
            
            # Initialize IRONFORGE components if needed
            if self.enhanced_graph_builder is None:
                self.initialize_ironforge_components()
            
            # Step 1: Calculate dimensional anchors (40% of previous range)
            anchor_zones = self._calculate_dimensional_anchors(
                session_data, 
                previous_session_data, 
                session_id
            )
            
            # Step 2: Apply temporal non-locality analysis
            temporal_analysis = self._analyze_temporal_nonlocality(
                session_data, 
                anchor_zones,
                session_id
            )
            
            # Step 3: Theory B validation - forward positioning
            theory_b_results = self._validate_theory_b_positioning(
                session_data,
                anchor_zones,
                temporal_analysis
            )
            
            # Step 4: Enhanced graph integration (if provided)
            enhanced_features = None
            if enhanced_graph is not None:
                enhanced_features = self._integrate_enhanced_graph_features(
                    enhanced_graph,
                    anchor_zones
                )
            
            # Step 5: Zone authenticity scoring
            authenticated_zones = self._authenticate_archaeological_zones(
                anchor_zones,
                temporal_analysis,
                theory_b_results
            )
            
            # Performance metrics calculation
            processing_time = time.time() - start_time
            performance_metrics = self.performance_monitor.get_session_metrics(
                processing_time,
                len(authenticated_zones),
                session_id
            )
            
            # Contract validation results
            contract_results = self.contract_validator.validate_archaeological_output(
                authenticated_zones,
                session_id
            )
            
            # Compile final analysis
            analysis = ArchaeologicalAnalysis(
                session_id=session_id,
                archaeological_zones=authenticated_zones,
                dimensional_analysis=self._compile_dimensional_analysis(
                    anchor_zones, 
                    previous_session_data
                ),
                temporal_echoes=temporal_analysis,
                performance_metrics=performance_metrics,
                contract_validation=contract_results,
                enhanced_features=enhanced_features
            )
            
            # Final validation and logging
            self._validate_final_analysis(analysis)
            
            logger.info(
                f"Archaeological analysis complete for {session_id}: "
                f"{len(authenticated_zones)} zones detected in {processing_time:.3f}s"
            )
            
            return analysis
            
        except Exception as e:
            self.performance_monitor.record_error(str(e))
            logger.error(f"Archaeological zone detection failed: {e}")
            raise
        finally:
            self.performance_monitor.end_session_analysis()
    
    def enhance_tgat_discovery(
        self,
        graph: nx.Graph,
        archaeological_analysis: ArchaeologicalAnalysis
    ) -> Dict[str, Any]:
        """
        Enhance TGAT discovery with archaeological intelligence
        
        Integrates archaeological zone information into TGAT graph features
        for improved temporal pattern discovery.
        """
        try:
            # Add archaeological zone features to graph nodes
            enhanced_graph = self._add_archaeological_features_to_graph(
                graph,
                archaeological_analysis.archaeological_zones
            )
            
            # Calculate temporal echo propagation features
            temporal_features = self._calculate_temporal_echo_features(
                enhanced_graph,
                archaeological_analysis.temporal_echoes
            )
            
            # Theory B forward positioning features
            positioning_features = self._calculate_positioning_features(
                enhanced_graph,
                archaeological_analysis.archaeological_zones
            )
            
            enhancement_results = {
                "enhanced_graph": enhanced_graph,
                "temporal_features": temporal_features,
                "positioning_features": positioning_features,
                "zone_count": len(archaeological_analysis.archaeological_zones),
                "authenticity_boost": self._calculate_authenticity_boost(
                    archaeological_analysis
                )
            }
            
            logger.debug(
                f"TGAT discovery enhanced with {len(archaeological_analysis.archaeological_zones)} "
                f"archaeological zones"
            )
            
            return enhancement_results
            
        except Exception as e:
            logger.error(f"TGAT discovery enhancement failed: {e}")
            raise
    
    def _extract_session_id(self, session_data: pd.DataFrame) -> str:
        """Extract session identifier from data"""
        # Implementation depends on session data structure
        if 'session_id' in session_data.columns:
            return session_data['session_id'].iloc[0]
        elif 'session_date' in session_data.columns:
            return f"session_{session_data['session_date'].iloc[0]}"
        else:
            return f"session_{hash(str(session_data.iloc[0].values))}"[:16]
    
    def _validate_session_contracts(self, session_data: pd.DataFrame, session_id: str) -> None:
        """Validate session data against IRONFORGE golden invariants"""
        try:
            # Validate event types if present
            if 'event_type' in session_data.columns:
                event_types = session_data['event_type'].unique().tolist()
                self.contract_validator.validate_event_taxonomy(event_types)
            
            # Validate feature dimensions
            feature_columns = [col for col in session_data.columns if col.startswith('f')]
            if feature_columns:
                expected_dims = NODE_FEATURE_DIM_HTF if len(feature_columns) > 45 else NODE_FEATURE_DIM_STANDARD
                self.contract_validator.validate_feature_dimensions(
                    len(feature_columns), 
                    expected_dims
                )
            
            # Validate session isolation
            self.contract_validator.validate_session_isolation(session_data, session_id)
            
        except ContractViolationError as e:
            logger.error(f"Contract validation failed for session {session_id}: {e}")
            raise
    
    def _calculate_dimensional_anchors(
        self,
        session_data: pd.DataFrame,
        previous_session_data: Optional[pd.DataFrame],
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Calculate 40% dimensional anchor points from previous session range"""
        if previous_session_data is None:
            logger.warning(f"No previous session data for {session_id}, using current session")
            range_data = session_data
        else:
            range_data = previous_session_data
        
        # Calculate previous session range
        previous_range = self.anchor_calculator.calculate_session_range(range_data)
        
        # Apply 40% dimensional anchoring
        anchor_zones = self.anchor_calculator.calculate_dimensional_anchors(
            previous_range,
            session_data,
            anchor_percentage=0.40  # Archaeological constant
        )
        
        logger.debug(f"Calculated {len(anchor_zones)} dimensional anchors for {session_id}")
        return anchor_zones
    
    def _analyze_temporal_nonlocality(
        self,
        session_data: pd.DataFrame,
        anchor_zones: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """Analyze temporal non-locality patterns using forward positioning"""
        temporal_analysis = self.temporal_validator.analyze_nonlocality(
            session_data,
            anchor_zones
        )
        
        # Add session-specific temporal echo detection
        temporal_echoes = self.temporal_validator.detect_temporal_echoes(
            session_data,
            anchor_zones
        )
        
        combined_analysis = {
            "nonlocality_patterns": temporal_analysis,
            "temporal_echoes": temporal_echoes,
            "session_id": session_id,
            "echo_count": len(temporal_echoes),
            "forward_coherence": self._calculate_forward_coherence(temporal_analysis)
        }
        
        logger.debug(f"Temporal non-locality analysis complete for {session_id}")
        return combined_analysis
    
    def _validate_theory_b_positioning(
        self,
        session_data: pd.DataFrame,
        anchor_zones: List[Dict[str, Any]],
        temporal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate Theory B forward positioning principles"""
        theory_b_results = self.theory_b_validator.validate_forward_positioning(
            session_data,
            anchor_zones,
            temporal_analysis
        )
        
        # Add precision scoring (target: 7.55 points)
        precision_scores = self.theory_b_validator.calculate_precision_scores(
            anchor_zones,
            session_data
        )
        
        results = {
            **theory_b_results,
            "precision_scores": precision_scores,
            "average_precision": np.mean([score["precision"] for score in precision_scores]),
            "theory_b_compliance": all(zone["theory_b_valid"] for zone in anchor_zones)
        }
        
        logger.debug("Theory B positioning validation complete")
        return results
    
    def _integrate_enhanced_graph_features(
        self,
        enhanced_graph: nx.Graph,
        anchor_zones: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Integrate archaeological features with enhanced graph"""
        if self.enhanced_graph_builder is None:
            logger.warning("Enhanced Graph Builder not available")
            return torch.zeros(1, NODE_FEATURE_DIM_STANDARD)
        
        # Add archaeological zone features to graph nodes
        for node_id, node_data in enhanced_graph.nodes(data=True):
            archaeological_features = self._calculate_node_archaeological_features(
                node_id,
                node_data,
                anchor_zones
            )
            
            # Extend node features with archaeological intelligence
            if 'features' in node_data:
                current_features = node_data['features']
                enhanced_features = torch.cat([current_features, archaeological_features])
                node_data['features'] = enhanced_features
        
        # Return aggregated archaeological features
        all_features = []
        for _, node_data in enhanced_graph.nodes(data=True):
            if 'features' in node_data:
                all_features.append(node_data['features'])
        
        if all_features:
            return torch.stack(all_features)
        else:
            return torch.zeros(1, NODE_FEATURE_DIM_STANDARD)
    
    def _authenticate_archaeological_zones(
        self,
        anchor_zones: List[Dict[str, Any]],
        temporal_analysis: Dict[str, Any],
        theory_b_results: Dict[str, Any]
    ) -> List[ArchaeologicalZone]:
        """Authenticate archaeological zones with >87% authenticity threshold"""
        authenticated_zones = []
        
        for zone_data in anchor_zones:
            # Calculate authenticity score
            authenticity_score = self._calculate_authenticity_score(
                zone_data,
                temporal_analysis,
                theory_b_results
            )
            
            # Apply 87% authenticity threshold
            if authenticity_score >= 87.0:
                archaeological_zone = ArchaeologicalZone(
                    anchor_point=zone_data["anchor_point"],
                    zone_range=zone_data["zone_range"],
                    confidence=zone_data["confidence"],
                    temporal_offset=zone_data.get("temporal_offset", 0.0),
                    precision_score=zone_data.get("precision_score", 0.0),
                    session_id=zone_data["session_id"],
                    discovery_timestamp=time.time(),
                    theory_b_alignment=zone_data.get("theory_b_valid", False),
                    forward_positioning=zone_data.get("forward_positioning", {}),
                    authenticity_score=authenticity_score
                )
                authenticated_zones.append(archaeological_zone)
            else:
                logger.debug(
                    f"Zone at {zone_data['anchor_point']} failed authenticity "
                    f"threshold: {authenticity_score:.1f}% < 87%"
                )
        
        logger.info(f"Authenticated {len(authenticated_zones)}/{len(anchor_zones)} zones")
        return authenticated_zones
    
    def _calculate_authenticity_score(
        self,
        zone_data: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
        theory_b_results: Dict[str, Any]
    ) -> float:
        """Calculate authenticity score for archaeological zone"""
        # Base score from zone confidence
        base_score = zone_data.get("confidence", 0.5) * 40
        
        # Temporal non-locality bonus
        temporal_bonus = 0
        if "nonlocality_patterns" in temporal_analysis:
            temporal_coherence = temporal_analysis.get("forward_coherence", 0.0)
            temporal_bonus = temporal_coherence * 25
        
        # Theory B alignment bonus
        theory_b_bonus = 0
        if zone_data.get("theory_b_valid", False):
            theory_b_bonus = 15
        
        # Precision bonus (target: 7.55 points)
        precision_bonus = 0
        precision_score = zone_data.get("precision_score", 0.0)
        if precision_score >= 7.0:
            precision_bonus = min(10, (precision_score / 7.55) * 10)
        
        total_score = base_score + temporal_bonus + theory_b_bonus + precision_bonus
        return min(100.0, max(0.0, total_score))
    
    def _compile_dimensional_analysis(
        self,
        anchor_zones: List[Dict[str, Any]],
        previous_session_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compile dimensional analysis results"""
        previous_range = 0.0
        if previous_session_data is not None:
            previous_range = self.anchor_calculator.calculate_session_range(
                previous_session_data
            )
        
        calculated_zone = previous_range * 0.40
        
        return {
            "previous_range": previous_range,
            "calculated_zone": calculated_zone,
            "zone_count": len(anchor_zones),
            "average_confidence": np.mean([
                zone.get("confidence", 0.0) for zone in anchor_zones
            ]) if anchor_zones else 0.0,
            "theory_b_alignment": all(
                zone.get("theory_b_valid", False) for zone in anchor_zones
            ),
            "forward_positioning": {
                "enabled": True,
                "zones_with_positioning": len([
                    zone for zone in anchor_zones 
                    if zone.get("forward_positioning")
                ])
            }
        }
    
    def _validate_final_analysis(self, analysis: ArchaeologicalAnalysis) -> None:
        """Final validation of archaeological analysis"""
        # Performance validation
        processing_time = analysis.performance_metrics.get("detection_time", float('inf'))
        if processing_time >= 3.0:
            logger.warning(
                f"Session processing exceeded 3s limit: {processing_time:.3f}s"
            )
        
        # Authenticity validation
        failed_zones = [
            zone for zone in analysis.archaeological_zones 
            if zone.authenticity_score < 87.0
        ]
        if failed_zones:
            logger.warning(f"{len(failed_zones)} zones below 87% authenticity threshold")
        
        # Contract validation
        contract_failures = [
            key for key, passed in analysis.contract_validation.items()
            if not passed
        ]
        if contract_failures:
            logger.error(f"Contract validation failures: {contract_failures}")
            raise ContractViolationError(f"Failed contracts: {contract_failures}")
    
    # Additional helper methods for enhanced integration
    def _add_archaeological_features_to_graph(
        self, 
        graph: nx.Graph, 
        zones: List[ArchaeologicalZone]
    ) -> nx.Graph:
        """Add archaeological features to graph nodes"""
        enhanced_graph = graph.copy()
        
        for node_id, node_data in enhanced_graph.nodes(data=True):
            archaeological_features = torch.zeros(5)  # 5D archaeological extension
            
            # Feature 0: Zone proximity
            min_distance = float('inf')
            for zone in zones:
                if hasattr(node_data, 'price') or 'price' in node_data:
                    node_price = node_data.get('price', 0.0)
                    distance = abs(node_price - zone.anchor_point)
                    min_distance = min(min_distance, distance)
            
            archaeological_features[0] = 1.0 / (1.0 + min_distance) if min_distance != float('inf') else 0.0
            
            # Feature 1: Temporal echo strength
            archaeological_features[1] = len(zones) / 10.0  # Normalized zone count
            
            # Feature 2: Theory B alignment
            theory_b_alignment = sum(zone.theory_b_alignment for zone in zones) / len(zones) if zones else 0.0
            archaeological_features[2] = theory_b_alignment
            
            # Feature 3: Average authenticity
            avg_authenticity = sum(zone.authenticity_score for zone in zones) / len(zones) if zones else 0.0
            archaeological_features[3] = avg_authenticity / 100.0
            
            # Feature 4: Precision indicator
            avg_precision = sum(zone.precision_score for zone in zones) / len(zones) if zones else 0.0
            archaeological_features[4] = min(1.0, avg_precision / 7.55)
            
            # Extend existing features
            if 'features' in node_data:
                node_data['archaeological_features'] = archaeological_features
        
        return enhanced_graph
    
    def _calculate_temporal_echo_features(
        self, 
        graph: nx.Graph, 
        temporal_echoes: Dict[str, Any]
    ) -> torch.Tensor:
        """Calculate temporal echo propagation features"""
        num_nodes = len(graph.nodes)
        echo_features = torch.zeros(num_nodes, 3)
        
        # Implementation depends on temporal echo structure
        echo_count = temporal_echoes.get("echo_count", 0)
        forward_coherence = temporal_echoes.get("forward_coherence", 0.0)
        
        # Broadcast echo information to all nodes
        echo_features[:, 0] = echo_count / 10.0  # Normalized echo count
        echo_features[:, 1] = forward_coherence
        echo_features[:, 2] = 1.0 if echo_count > 0 else 0.0  # Echo presence flag
        
        return echo_features
    
    def _calculate_positioning_features(
        self, 
        graph: nx.Graph, 
        zones: List[ArchaeologicalZone]
    ) -> torch.Tensor:
        """Calculate Theory B forward positioning features"""
        num_nodes = len(graph.nodes)
        positioning_features = torch.zeros(num_nodes, 2)
        
        if zones:
            # Average forward positioning strength
            avg_positioning = sum(
                len(zone.forward_positioning) for zone in zones
            ) / len(zones)
            
            # Theory B compliance ratio
            compliance_ratio = sum(zone.theory_b_alignment for zone in zones) / len(zones)
            
            positioning_features[:, 0] = avg_positioning / 10.0  # Normalized
            positioning_features[:, 1] = compliance_ratio
        
        return positioning_features
    
    def _calculate_authenticity_boost(self, analysis: ArchaeologicalAnalysis) -> float:
        """Calculate authenticity boost for TGAT discovery"""
        if not analysis.archaeological_zones:
            return 0.0
        
        avg_authenticity = sum(
            zone.authenticity_score for zone in analysis.archaeological_zones
        ) / len(analysis.archaeological_zones)
        
        # Convert to boost factor (87% threshold = 1.0x boost)
        boost = max(0.0, (avg_authenticity - 87.0) / 13.0)  # Scale to [0, 1]
        return boost
    
    def _calculate_forward_coherence(self, temporal_analysis: Dict[str, Any]) -> float:
        """Calculate forward coherence score for temporal analysis"""
        # Implementation depends on temporal analysis structure
        patterns = temporal_analysis.get("nonlocality_patterns", [])
        if not patterns:
            return 0.0
        
        # Simple coherence calculation based on pattern consistency
        coherence_scores = []
        for pattern in patterns:
            if isinstance(pattern, dict) and "coherence" in pattern:
                coherence_scores.append(pattern["coherence"])
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_node_archaeological_features(
        self,
        node_id: Any,
        node_data: Dict[str, Any],
        anchor_zones: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Calculate archaeological features for a specific node"""
        features = torch.zeros(3)  # 3D archaeological node features
        
        # Feature 0: Distance to nearest anchor
        node_price = node_data.get('price', 0.0)
        min_distance = float('inf')
        for zone in anchor_zones:
            distance = abs(node_price - zone.get('anchor_point', 0.0))
            min_distance = min(min_distance, distance)
        
        features[0] = 1.0 / (1.0 + min_distance) if min_distance != float('inf') else 0.0
        
        # Feature 1: Zone influence
        influence = 0.0
        for zone in anchor_zones:
            zone_range = zone.get('zone_range', (0.0, 0.0))
            if zone_range[0] <= node_price <= zone_range[1]:
                influence += zone.get('confidence', 0.0)
        features[1] = min(1.0, influence)
        
        # Feature 2: Temporal positioning
        features[2] = len(anchor_zones) / 10.0  # Normalized zone count
        
        return features


# Factory function for IRONFORGE integration
def create_archaeological_zone_detector(
    config: Optional[ArchaeologicalConfig] = None
) -> ArchaeologicalZoneDetector:
    """
    Factory function to create Archaeological Zone Detector
    
    Recommended entry point for IRONFORGE pipeline integration.
    Ensures proper container initialization and configuration.
    """
    container = get_ironforge_container()
    detector = ArchaeologicalZoneDetector(config=config, container=container)
    
    logger.info("Archaeological Zone Detector created for IRONFORGE integration")
    return detector


# Integration hooks for IRONFORGE pipeline
def enhance_discovery_with_archaeological_intelligence(
    discovery_results: Dict[str, Any],
    session_data: pd.DataFrame,
    config: Optional[ArchaeologicalConfig] = None
) -> Dict[str, Any]:
    """
    Enhance TGAT discovery results with archaeological intelligence
    
    Direct integration hook for IRONFORGE discovery pipeline.
    """
    detector = create_archaeological_zone_detector(config)
    
    # Run archaeological analysis
    archaeological_analysis = detector.detect_archaeological_zones(session_data)
    
    # Enhance discovery results
    if "enhanced_graph" in discovery_results:
        enhancement = detector.enhance_tgat_discovery(
            discovery_results["enhanced_graph"],
            archaeological_analysis
        )
        
        # Merge results
        enhanced_results = {
            **discovery_results,
            "archaeological_analysis": archaeological_analysis,
            "archaeological_enhancement": enhancement,
            "authenticity_boost": enhancement["authenticity_boost"]
        }
        
        return enhanced_results
    
    # Fallback: add archaeological analysis to results
    return {
        **discovery_results,
        "archaeological_analysis": archaeological_analysis
    }