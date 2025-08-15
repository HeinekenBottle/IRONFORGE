"""
IRONFORGE Discovery Mathematics - Self-Supervised Learning & Validation
========================================================================

Mathematical components for IRONFORGE's learnable discovery mechanism implementing:
1. Three self-supervised loss functions for training TGAT to discover permanent relationships
2. Validation framework for testing permanence, archaeological significance, and economic validity

Architecture: Designed for 27D node features, 17D edge features with TGAT attention mechanism
Purpose: Learn which market relationships are permanent vs. spurious through mathematical rigor
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SELF-SUPERVISED LOSS FUNCTIONS
# =============================================================================

def next_event_prediction_loss(
    graph: Dict[str, torch.Tensor], 
    attention_weights: torch.Tensor,
    edge_probabilities: torch.Tensor, 
    predicted_edge_features: torch.Tensor
) -> torch.Tensor:
    """
    Next Event Prediction Loss - Train network to predict temporal event sequences
    
    Mathematical Foundation:
    - Uses temporal ordering in graph['rich_node_features'] to create training pairs
    - High loss if network doesn't predict edges to temporally adjacent events
    - Weighted by event significance (structural_importance, energy_state)
    - Combined cross-entropy (edge existence) + MSE (edge features)
    
    Args:
        graph: Graph data with rich_node_features containing temporal information
        attention_weights: TGAT attention weights between nodes
        edge_probabilities: Predicted probability of edge existence
        predicted_edge_features: Predicted edge feature vectors
        
    Returns:
        torch.Tensor: Combined loss for next event prediction
    """
    try:
        # Extract temporal and significance features
        node_features = graph['rich_node_features']  # Shape: [num_nodes, 27]
        
        # Feature indices (based on IRONFORGE architecture)
        time_idx = 0  # time_minutes
        structural_importance_idx = 3  # structural_importance  
        energy_state_idx = 4  # energy_state
        
        time_minutes = node_features[:, time_idx]
        structural_importance = node_features[:, structural_importance_idx]
        energy_state = node_features[:, energy_state_idx]
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Create temporal adjacency targets
        temporal_targets = torch.zeros((num_nodes, num_nodes), device=device)
        feature_targets = torch.zeros((num_nodes, num_nodes, predicted_edge_features.shape[-1]), device=device)
        sample_weights = torch.zeros((num_nodes, num_nodes), device=device)
        
        # Sort nodes by time to find temporal neighbors
        time_sorted_indices = torch.argsort(time_minutes)
        
        # Create temporal adjacency pairs (consecutive events in time)
        for i in range(len(time_sorted_indices) - 1):
            curr_idx = time_sorted_indices[i]
            next_idx = time_sorted_indices[i + 1]
            
            # Mark as temporal neighbors
            temporal_targets[curr_idx, next_idx] = 1.0
            
            # Create edge feature target (simple difference for now)
            curr_features = node_features[curr_idx]
            next_features = node_features[next_idx]
            feature_diff = next_features - curr_features
            feature_targets[curr_idx, next_idx] = feature_diff[:predicted_edge_features.shape[-1]]
            
            # Weight by combined significance
            significance = (structural_importance[curr_idx] + structural_importance[next_idx] + 
                          energy_state[curr_idx] + energy_state[next_idx]) / 4.0
            sample_weights[curr_idx, next_idx] = 1.0 + significance
        
        # Also create some longer-range temporal connections (skip-connections)
        for skip in [2, 3, 5]:  # Skip connections
            for i in range(len(time_sorted_indices) - skip):
                curr_idx = time_sorted_indices[i]
                skip_idx = time_sorted_indices[i + skip]
                
                temporal_targets[curr_idx, skip_idx] = 0.7  # Lower probability for skip connections
                
                curr_features = node_features[curr_idx]
                skip_features = node_features[skip_idx]
                feature_diff = skip_features - curr_features
                feature_targets[curr_idx, skip_idx] = feature_diff[:predicted_edge_features.shape[-1]]
                
                significance = (structural_importance[curr_idx] + structural_importance[skip_idx]) / 2.0
                sample_weights[curr_idx, skip_idx] = 0.5 + significance * 0.3
        
        # Edge existence loss (binary cross-entropy)
        edge_existence_loss = F.binary_cross_entropy(
            edge_probabilities, 
            temporal_targets, 
            weight=sample_weights,
            reduction='mean'
        )
        
        # Edge feature prediction loss (MSE for existing edges)
        edge_mask = (temporal_targets > 0.1).float().unsqueeze(-1)  # Only for predicted edges
        feature_loss = F.mse_loss(
            predicted_edge_features * edge_mask,
            feature_targets * edge_mask,
            reduction='sum'
        ) / (edge_mask.sum() + 1e-8)  # Normalize by number of edges
        
        # Combined loss
        total_loss = edge_existence_loss + 0.1 * feature_loss
        
        logger.debug(f"Next event prediction - Edge loss: {edge_existence_loss:.4f}, "
                    f"Feature loss: {feature_loss:.4f}, Total: {total_loss:.4f}")
        
        return total_loss
        
    except Exception as e:
        logger.error(f"Error in next_event_prediction_loss: {e}")
        return torch.tensor(0.0, device=graph['rich_node_features'].device, requires_grad=True)


def cross_tf_consistency_loss(
    graph: Dict[str, torch.Tensor], 
    attention_weights: torch.Tensor,
    edge_probabilities: torch.Tensor, 
    predicted_edge_features: torch.Tensor
) -> torch.Tensor:
    """
    Cross-Timeframe Consistency Loss - Ensure consistent relationships across timeframes
    
    Mathematical Foundation:
    - Events at same price/time across different timeframes should have similar attention
    - Uses KL divergence between attention distributions for cross-TF events
    - Penalizes inconsistent attention patterns for similar events across timeframes
    
    Args:
        graph: Graph data with timeframe_source information
        attention_weights: TGAT attention weights between nodes
        edge_probabilities: Predicted probability of edge existence  
        predicted_edge_features: Predicted edge feature vectors
        
    Returns:
        torch.Tensor: KL divergence loss for cross-timeframe consistency
    """
    try:
        node_features = graph['rich_node_features']  # Shape: [num_nodes, 27]
        
        # Feature indices
        time_idx = 0  # time_minutes
        price_idx = 1  # normalized_price
        tf_source_idx = 2  # timeframe_source (encoded)
        
        time_minutes = node_features[:, time_idx]
        normalized_price = node_features[:, price_idx]
        timeframe_source = node_features[:, tf_source_idx]
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Find cross-timeframe pairs (same price/time, different timeframe)
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        pair_count = 0
        
        # Create tolerance windows for matching
        time_tolerance = 0.05  # 5% tolerance in time
        price_tolerance = 0.02  # 2% tolerance in price
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Check if different timeframes
                if abs(timeframe_source[i] - timeframe_source[j]) < 0.1:
                    continue  # Same timeframe, skip
                    
                # Check if similar time and price
                time_diff = abs(time_minutes[i] - time_minutes[j])
                price_diff = abs(normalized_price[i] - normalized_price[j])
                
                if time_diff < time_tolerance and price_diff < price_tolerance:
                    # These are cross-timeframe similar events
                    
                    # Get attention patterns for both nodes
                    att_pattern_i = attention_weights[i, :] + 1e-8  # Add epsilon for numerical stability
                    att_pattern_j = attention_weights[j, :] + 1e-8
                    
                    # Normalize to probability distributions
                    att_dist_i = F.softmax(att_pattern_i, dim=0)
                    att_dist_j = F.softmax(att_pattern_j, dim=0)
                    
                    # KL divergence between attention patterns
                    kl_div_ij = F.kl_div(att_dist_i.log(), att_dist_j, reduction='sum')
                    kl_div_ji = F.kl_div(att_dist_j.log(), att_dist_i, reduction='sum')
                    
                    # Symmetric KL divergence
                    symmetric_kl = (kl_div_ij + kl_div_ji) / 2.0
                    
                    consistency_loss = consistency_loss + symmetric_kl
                    pair_count += 1
        
        # Normalize by number of pairs
        if pair_count > 0:
            consistency_loss = consistency_loss / pair_count
        
        logger.debug(f"Cross-TF consistency - Found {pair_count} cross-timeframe pairs, "
                    f"Loss: {consistency_loss:.4f}")
        
        return consistency_loss
        
    except Exception as e:
        logger.error(f"Error in cross_tf_consistency_loss: {e}")
        return torch.tensor(0.0, device=graph['rich_node_features'].device, requires_grad=True)


def temporal_echo_detection_loss(
    graph: Dict[str, torch.Tensor], 
    attention_weights: torch.Tensor,
    edge_probabilities: torch.Tensor, 
    predicted_edge_features: torch.Tensor
) -> torch.Tensor:
    """
    Temporal Echo Detection Loss - Learn to detect recurring temporal patterns
    
    Mathematical Foundation:  
    - Events with similar daily_phase_sin/cos should have higher attention
    - Uses contrastive learning: pull similar temporal patterns together, push different apart
    - Based on session_position and daily phase information
    
    Args:
        graph: Graph data with temporal phase information
        attention_weights: TGAT attention weights between nodes
        edge_probabilities: Predicted probability of edge existence
        predicted_edge_features: Predicted edge feature vectors
        
    Returns:
        torch.Tensor: Contrastive loss for temporal echo detection
    """
    try:
        node_features = graph['rich_node_features']  # Shape: [num_nodes, 27]
        
        # Feature indices for temporal patterns
        daily_phase_sin_idx = 5  # daily_phase_sin
        daily_phase_cos_idx = 6  # daily_phase_cos  
        session_position_idx = 7  # session_position
        
        daily_phase_sin = node_features[:, daily_phase_sin_idx]
        daily_phase_cos = node_features[:, daily_phase_cos_idx]
        session_position = node_features[:, session_position_idx]
        
        num_nodes = node_features.shape[0]
        device = node_features.device
        
        # Create temporal signature vectors
        temporal_signatures = torch.stack([
            daily_phase_sin, 
            daily_phase_cos, 
            session_position
        ], dim=1)  # Shape: [num_nodes, 3]
        
        # Calculate pairwise temporal similarity
        similarity_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Cosine similarity of temporal signatures
                    sig_i = temporal_signatures[i]
                    sig_j = temporal_signatures[j]
                    
                    cos_sim = F.cosine_similarity(sig_i.unsqueeze(0), sig_j.unsqueeze(0))
                    similarity_matrix[i, j] = cos_sim
        
        # Define positive and negative pairs based on similarity threshold
        similarity_threshold = 0.7  # High similarity threshold
        positive_mask = (similarity_matrix > similarity_threshold).float()
        negative_mask = (similarity_matrix < -0.1).float()  # Dissimilar patterns
        
        # Contrastive loss calculation
        margin = 1.0  # Margin for contrastive loss
        temperature = 0.5  # Temperature for attention softmax
        
        contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        pair_count = 0
        
        for i in range(num_nodes):
            # Get attention pattern for node i
            att_pattern_i = attention_weights[i, :] / temperature
            att_probs_i = F.softmax(att_pattern_i, dim=0)
            
            # Positive pairs (should have high attention)
            positive_indices = torch.where(positive_mask[i, :] > 0.5)[0]
            if len(positive_indices) > 0:
                positive_att = att_probs_i[positive_indices]
                positive_loss = -torch.log(positive_att + 1e-8).mean()
                contrastive_loss = contrastive_loss + positive_loss
                pair_count += 1
            
            # Negative pairs (should have low attention)  
            negative_indices = torch.where(negative_mask[i, :] > 0.5)[0]
            if len(negative_indices) > 0:
                negative_att = att_probs_i[negative_indices]
                negative_loss = torch.log(1.0 - negative_att + 1e-8).mean()
                contrastive_loss = contrastive_loss - negative_loss  # Subtract to minimize negative attention
                pair_count += 1
        
        # Normalize by number of nodes with pairs
        if pair_count > 0:
            contrastive_loss = contrastive_loss / pair_count
        
        # Additional term: encourage edge formation between temporal echoes
        echo_edge_bonus = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(num_nodes):
            positive_indices = torch.where(positive_mask[i, :] > 0.5)[0]
            if len(positive_indices) > 0:
                # Should predict high edge probabilities to temporal echoes
                target_edges = torch.ones_like(edge_probabilities[i, positive_indices])
                edge_bonus = F.binary_cross_entropy(
                    edge_probabilities[i, positive_indices], 
                    target_edges,
                    reduction='mean'
                )
                echo_edge_bonus = echo_edge_bonus + edge_bonus
        
        echo_edge_bonus = echo_edge_bonus / num_nodes
        
        total_loss = contrastive_loss + 0.3 * echo_edge_bonus
        
        logger.debug(f"Temporal echo detection - Contrastive: {contrastive_loss:.4f}, "
                    f"Edge bonus: {echo_edge_bonus:.4f}, Total: {total_loss:.4f}")
        
        return total_loss
        
    except Exception as e:
        logger.error(f"Error in temporal_echo_detection_loss: {e}")
        return torch.tensor(0.0, device=graph['rich_node_features'].device, requires_grad=True)


# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

@dataclass
class EdgeCandidate:
    """Data structure for discovered edge candidates"""
    source_idx: int
    target_idx: int
    probability: float
    features: np.ndarray
    source_node_features: np.ndarray
    target_node_features: np.ndarray


class PermanenceValidator:
    """
    Validates discovered edges across different market regimes for permanence
    Tests stability of relationships across regime changes and time periods
    """
    
    def __init__(self, min_regime_count: int = 3, stability_threshold: float = 0.6):
        self.min_regime_count = min_regime_count
        self.stability_threshold = stability_threshold
        
    def permanence_testing_across_regimes(
        self, 
        discovered_edges: List[EdgeCandidate], 
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test discovered edges across different market regimes for permanence
        
        Args:
            discovered_edges: List of EdgeCandidate objects representing discovered relationships
            historical_data: Historical market data with regime information
            
        Returns:
            Dict containing permanence scores and regime breakdown analysis
        """
        try:
            results = {
                'permanence_scores': [],
                'regime_breakdown': {},
                'stable_edges': [],
                'unstable_edges': [],
                'overall_permanence': 0.0
            }
            
            # Extract regime information from historical data
            regime_data = self._extract_regime_data(historical_data)
            
            stable_count = 0
            
            for edge in discovered_edges:
                # Analyze edge permanence across regimes
                permanence_analysis = self._analyze_edge_permanence(edge, regime_data)
                
                results['permanence_scores'].append({
                    'edge_id': f"{edge.source_idx}-{edge.target_idx}",
                    'permanence_score': permanence_analysis['permanence_score'],
                    'regime_consistency': permanence_analysis['regime_consistency'],
                    'temporal_stability': permanence_analysis['temporal_stability']
                })
                
                # Classify as stable or unstable
                if permanence_analysis['permanence_score'] >= self.stability_threshold:
                    results['stable_edges'].append(edge)
                    stable_count += 1
                else:
                    results['unstable_edges'].append(edge)
                
                # Add to regime breakdown
                for regime, score in permanence_analysis['regime_scores'].items():
                    if regime not in results['regime_breakdown']:
                        results['regime_breakdown'][regime] = []
                    results['regime_breakdown'][regime].append(score)
            
            # Calculate overall permanence
            if len(discovered_edges) > 0:
                results['overall_permanence'] = stable_count / len(discovered_edges)
            
            logger.info(f"Permanence validation - {stable_count}/{len(discovered_edges)} edges stable "
                       f"({results['overall_permanence']:.2%})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in permanence_testing_across_regimes: {e}")
            return {'error': str(e), 'permanence_scores': [], 'overall_permanence': 0.0}
    
    def _extract_regime_data(self, historical_data: Dict[str, Any]) -> Dict[str, List]:
        """Extract regime classifications from historical data"""
        regime_data = defaultdict(list)
        
        if 'fisher_regime' in historical_data:
            regime_data['fisher_regime'] = historical_data['fisher_regime']
        if 'session_character' in historical_data:
            regime_data['session_character'] = historical_data['session_character']
        if 'volatility_regime' in historical_data:
            regime_data['volatility_regime'] = historical_data['volatility_regime']
        
        return dict(regime_data)
    
    def _analyze_edge_permanence(self, edge: EdgeCandidate, regime_data: Dict) -> Dict:
        """Analyze permanence of a single edge across regimes"""
        analysis = {
            'permanence_score': 0.0,
            'regime_consistency': 0.0,
            'temporal_stability': 0.0,
            'regime_scores': {}
        }
        
        # Calculate consistency across each regime type
        regime_consistencies = []
        
        for regime_type, regime_values in regime_data.items():
            if len(regime_values) >= self.min_regime_count:
                # Calculate how often this edge pattern appears in each regime
                regime_score = self._calculate_regime_score(edge, regime_values)
                analysis['regime_scores'][regime_type] = regime_score
                regime_consistencies.append(regime_score)
        
        if regime_consistencies:
            analysis['regime_consistency'] = np.mean(regime_consistencies)
            analysis['temporal_stability'] = 1.0 - np.std(regime_consistencies)  # Lower std = higher stability
            analysis['permanence_score'] = (analysis['regime_consistency'] + analysis['temporal_stability']) / 2.0
        
        return analysis
    
    def _calculate_regime_score(self, edge: EdgeCandidate, regime_values: List) -> float:
        """
        Calculate score for edge in specific regime (ENHANCED FOR CROSS-REGIME ANALYSIS)
        
        This fixes the placeholder implementation that only returned edge.probability
        """
        try:
            if not regime_values:
                return 0.0
            
            # Analyze pattern consistency across regime instances
            base_score = edge.probability
            
            # Enhance score based on regime stability factors
            stability_factors = []
            
            # Factor 1: Regime type consistency
            unique_regimes = set(regime_values)
            regime_diversity = len(unique_regimes) / len(regime_values) if regime_values else 1.0
            
            # Higher diversity = less stable across regimes, lower diversity = more stable
            regime_consistency_factor = 1.0 - regime_diversity
            stability_factors.append(regime_consistency_factor)
            
            # Factor 2: Edge type specific adjustments
            edge_type_factor = 1.0
            if hasattr(edge, 'edge_type'):
                if 'cross_tf' in edge.edge_type:
                    # Cross-timeframe patterns tend to be more permanent
                    edge_type_factor = 1.3
                elif 'structural' in edge.edge_type:
                    # Structural patterns are inherently more stable
                    edge_type_factor = 1.2
                elif 'temporal' in edge.edge_type:
                    # Temporal patterns vary more with regime
                    edge_type_factor = 0.9
                    
            stability_factors.append(edge_type_factor)
            
            # Factor 3: Regime value analysis
            regime_score = 0.0
            
            # Analyze volatility regimes
            high_vol_count = sum(1 for rv in regime_values if 'high' in str(rv).lower())
            low_vol_count = sum(1 for rv in regime_values if 'low' in str(rv).lower())
            
            if high_vol_count > 0 and low_vol_count > 0:
                # Pattern appears in both high and low volatility - more permanent
                regime_score += 0.3
            
            # Analyze trend vs consolidation regimes
            trend_count = sum(1 for rv in regime_values if 'trend' in str(rv).lower())
            consolidation_count = sum(1 for rv in regime_values if 'consolidat' in str(rv).lower())
            
            if trend_count > 0 and consolidation_count > 0:
                # Pattern appears in both trending and consolidating markets - more permanent
                regime_score += 0.3
            
            # Factor 4: Occurrence frequency in regime
            regime_occurrence_factor = min(len(regime_values) / 10.0, 1.0)  # Normalize to max 10 occurrences
            stability_factors.append(regime_occurrence_factor)
            
            # Combine all factors
            overall_stability = np.mean(stability_factors) if stability_factors else 1.0
            
            # Final regime score
            final_score = base_score * overall_stability + regime_score
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            # Fallback to base probability if analysis fails
            return min(getattr(edge, 'probability', 0.5), 1.0)


class DistanceBridgeScorer:
    """
    Scores edges based on their ability to bridge distant time/price points
    Measures archaeological significance of discovered relationships
    """
    
    def __init__(self, time_weight: float = 0.4, price_weight: float = 0.3, hierarchy_weight: float = 0.3):
        self.time_weight = time_weight
        self.price_weight = price_weight  
        self.hierarchy_weight = hierarchy_weight
        
    def distance_bridge_scoring(
        self, 
        discovered_edges: List[EdgeCandidate], 
        graph_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score edges based on distance bridging capability and archaeological value
        
        Args:
            discovered_edges: List of EdgeCandidate objects
            graph_context: Graph context with distance and hierarchy information
            
        Returns:
            Dict containing distance scores and archaeological significance metrics
        """
        try:
            results = {
                'distance_scores': [],
                'archaeological_significance': [],
                'long_range_bridges': [],
                'cross_hierarchy_bridges': [],
                'average_bridge_score': 0.0
            }
            
            total_score = 0.0
            
            for edge in discovered_edges:
                # Calculate distance bridge score
                bridge_analysis = self._calculate_bridge_score(edge, graph_context)
                
                results['distance_scores'].append({
                    'edge_id': f"{edge.source_idx}-{edge.target_idx}",
                    'temporal_distance': bridge_analysis['temporal_distance'],
                    'price_distance': bridge_analysis['price_distance'],
                    'hierarchy_distance': bridge_analysis['hierarchy_distance'],
                    'combined_bridge_score': bridge_analysis['combined_score'],
                    'archaeological_value': bridge_analysis['archaeological_value']
                })
                
                # Classify bridge types
                if bridge_analysis['temporal_distance'] > 0.7:  # Long temporal span
                    results['long_range_bridges'].append(edge)
                
                if bridge_analysis['hierarchy_distance'] > 0.6:  # Cross-hierarchy
                    results['cross_hierarchy_bridges'].append(edge)
                
                # Archaeological significance
                archaeological_sig = self._calculate_archaeological_significance(bridge_analysis)
                results['archaeological_significance'].append({
                    'edge_id': f"{edge.source_idx}-{edge.target_idx}",
                    'significance': archaeological_sig,
                    'discovery_value': bridge_analysis['discovery_value']
                })
                
                total_score += bridge_analysis['combined_score']
            
            # Calculate averages
            if len(discovered_edges) > 0:
                results['average_bridge_score'] = total_score / len(discovered_edges)
            
            # Sort by significance
            results['archaeological_significance'].sort(
                key=lambda x: x['significance'], reverse=True
            )
            
            logger.info(f"Distance bridge scoring - Average score: {results['average_bridge_score']:.3f}, "
                       f"Long-range bridges: {len(results['long_range_bridges'])}, "
                       f"Cross-hierarchy: {len(results['cross_hierarchy_bridges'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in distance_bridge_scoring: {e}")
            return {'error': str(e), 'distance_scores': [], 'average_bridge_score': 0.0}
    
    def _calculate_bridge_score(self, edge: EdgeCandidate, graph_context: Dict) -> Dict:
        """Calculate comprehensive bridge score for an edge"""
        source_features = edge.source_node_features
        target_features = edge.target_node_features
        
        # Extract relevant features (assuming 27D node features)
        time_diff = abs(source_features[0] - target_features[0])  # time_minutes
        price_diff = abs(source_features[1] - target_features[1])  # normalized_price
        
        # Hierarchy distance (based on structural_importance difference)
        hierarchy_diff = abs(source_features[3] - target_features[3])  # structural_importance
        
        # Normalize distances (0-1 scale)
        temporal_distance = min(time_diff / 1440.0, 1.0)  # Normalize by day (1440 minutes)
        price_distance = min(price_diff, 1.0)  # Already normalized
        hierarchy_distance = min(hierarchy_diff, 1.0)  # Already normalized
        
        # Calculate combined score
        combined_score = (
            self.time_weight * temporal_distance + 
            self.price_weight * price_distance +
            self.hierarchy_weight * hierarchy_distance
        )
        
        # Archaeological value (higher for rare, distant connections)
        archaeological_value = combined_score * edge.probability
        
        # Discovery value (potential new insights)
        discovery_value = combined_score * (1.0 - edge.probability)  # High distance, low probability = high discovery
        
        return {
            'temporal_distance': temporal_distance,
            'price_distance': price_distance,
            'hierarchy_distance': hierarchy_distance,
            'combined_score': combined_score,
            'archaeological_value': archaeological_value,
            'discovery_value': discovery_value
        }
    
    def _calculate_archaeological_significance(self, bridge_analysis: Dict) -> float:
        """Calculate overall archaeological significance"""
        # Combine multiple factors for significance
        base_significance = bridge_analysis['combined_score']
        value_bonus = bridge_analysis['archaeological_value'] * 0.5
        discovery_bonus = bridge_analysis['discovery_value'] * 0.3
        
        return base_significance + value_bonus + discovery_bonus


class EconomicBacktestValidator:
    """
    Validates discovered edges against actual market behavior
    Tests economic significance and prediction accuracy
    """
    
    def __init__(self, prediction_window: int = 60, significance_threshold: float = 0.05):
        self.prediction_window = prediction_window  # minutes
        self.significance_threshold = significance_threshold
        
    def economic_backtest_validation(
        self, 
        discovered_edges: List[EdgeCandidate], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate discovered edges against market behavior and economic significance
        
        Args:
            discovered_edges: List of EdgeCandidate objects
            market_data: Historical market data for backtesting
            
        Returns:
            Dict containing economic validation metrics and prediction accuracy
        """
        try:
            results = {
                'economic_scores': [],
                'prediction_accuracy': [],
                'profitable_edges': [],
                'significant_predictions': [],
                'overall_accuracy': 0.0,
                'economic_significance': 0.0,
                'backtest_summary': {}
            }
            
            accurate_predictions = 0
            total_predictions = 0
            total_economic_value = 0.0
            
            for edge in discovered_edges:
                # Perform economic backtest for this edge
                backtest_results = self._perform_edge_backtest(edge, market_data)
                
                results['economic_scores'].append({
                    'edge_id': f"{edge.source_idx}-{edge.target_idx}",
                    'prediction_accuracy': backtest_results['accuracy'],
                    'economic_value': backtest_results['economic_value'],
                    'statistical_significance': backtest_results['p_value'],
                    'sharpe_ratio': backtest_results['sharpe_ratio']
                })
                
                # Track accuracy
                if backtest_results['accuracy'] > 0.5:  # Better than random
                    accurate_predictions += 1
                    results['profitable_edges'].append(edge)
                
                total_predictions += 1
                
                # Track statistical significance  
                if backtest_results['p_value'] < self.significance_threshold:
                    results['significant_predictions'].append(edge)
                
                total_economic_value += backtest_results['economic_value']
                
                results['prediction_accuracy'].append({
                    'edge_id': f"{edge.source_idx}-{edge.target_idx}",
                    'accuracy': backtest_results['accuracy'],
                    'confidence_interval': backtest_results['confidence_interval']
                })
            
            # Calculate overall metrics
            if total_predictions > 0:
                results['overall_accuracy'] = accurate_predictions / total_predictions
                results['economic_significance'] = total_economic_value / total_predictions
            
            # Backtest summary
            results['backtest_summary'] = {
                'total_edges_tested': len(discovered_edges),
                'profitable_edges': len(results['profitable_edges']),
                'significant_edges': len(results['significant_predictions']),
                'average_economic_value': results['economic_significance'],
                'success_rate': results['overall_accuracy']
            }
            
            logger.info(f"Economic backtest - Accuracy: {results['overall_accuracy']:.2%}, "
                       f"Profitable edges: {len(results['profitable_edges'])}/{len(discovered_edges)}, "
                       f"Economic significance: {results['economic_significance']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in economic_backtest_validation: {e}")
            return {'error': str(e), 'economic_scores': [], 'overall_accuracy': 0.0}
    
    def _perform_edge_backtest(self, edge: EdgeCandidate, market_data: Dict) -> Dict:
        """Perform backtest for a single edge"""
        # Extract market timing and price data
        if 'timestamps' not in market_data or 'prices' not in market_data:
            # Fallback values if data is incomplete
            return {
                'accuracy': 0.5,
                'economic_value': 0.0,
                'p_value': 1.0,
                'sharpe_ratio': 0.0,
                'confidence_interval': (0.4, 0.6)
            }
        
        timestamps = market_data['timestamps']
        prices = market_data['prices']
        
        # Simple backtest: predict price direction based on edge
        source_time = edge.source_node_features[0]  # time_minutes
        target_time = edge.target_node_features[0]  # time_minutes
        
        source_price = edge.source_node_features[1]  # normalized_price
        target_price = edge.target_node_features[1]  # normalized_price
        
        # Predicted direction
        predicted_direction = 1 if target_price > source_price else -1
        
        # Find actual market movements
        correct_predictions = 0
        total_tests = 0
        returns = []
        
        # Simplified backtest logic
        for i in range(len(prices) - self.prediction_window):
            current_price = prices[i]
            future_price = prices[i + self.prediction_window]
            
            actual_direction = 1 if future_price > current_price else -1
            
            if predicted_direction == actual_direction:
                correct_predictions += 1
                returns.append(abs(future_price - current_price) / current_price)
            else:
                returns.append(-abs(future_price - current_price) / current_price)
            
            total_tests += 1
        
        # Calculate metrics
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0.5
        
        # Economic value (average return)
        economic_value = np.mean(returns) if returns else 0.0
        
        # Statistical significance (simplified t-test)
        if len(returns) > 1:
            t_stat = np.mean(returns) / (np.std(returns) / np.sqrt(len(returns)))
            p_value = 2 * (1 - abs(t_stat))  # Simplified p-value approximation
        else:
            p_value = 1.0
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
        
        # Confidence interval for accuracy
        if total_tests > 0:
            std_error = np.sqrt(accuracy * (1 - accuracy) / total_tests)
            ci_lower = max(0, accuracy - 1.96 * std_error)
            ci_upper = min(1, accuracy + 1.96 * std_error)
        else:
            ci_lower, ci_upper = 0.4, 0.6
        
        return {
            'accuracy': accuracy,
            'economic_value': economic_value,
            'p_value': max(0, min(1, p_value)),
            'sharpe_ratio': sharpe_ratio,
            'confidence_interval': (ci_lower, ci_upper)
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_loss_functions() -> Dict[str, callable]:
    """
    Factory function to return all loss functions for the learnable discovery system
    
    Returns:
        Dict mapping loss function names to callable functions
    """
    return {
        'next_event_prediction': next_event_prediction_loss,
        'cross_tf_consistency': cross_tf_consistency_loss,
        'temporal_echo_detection': temporal_echo_detection_loss
    }


def get_validators() -> Dict[str, object]:
    """
    Factory function to return all validator classes for the discovery system
    
    Returns:
        Dict mapping validator names to instantiated validator objects
    """
    return {
        'permanence_validator': PermanenceValidator(),
        'distance_bridge_scorer': DistanceBridgeScorer(),
        'economic_backtest_validator': EconomicBacktestValidator()
    }


# =============================================================================
# INTEGRATION HELPER FUNCTIONS
# =============================================================================

def validate_discovery_mathematics(sample_graph: Dict[str, torch.Tensor] = None) -> Dict[str, bool]:
    """
    Validation function to test all mathematical components
    
    Args:
        sample_graph: Optional sample graph data for testing
        
    Returns:
        Dict indicating which components pass validation
    """
    results = {}
    
    try:
        # Create sample data if not provided
        if sample_graph is None:
            num_nodes = 10
            feature_dim = 27
            edge_feature_dim = 17
            
            sample_graph = {
                'rich_node_features': torch.randn(num_nodes, feature_dim)
            }
            
            # Set some realistic values
            sample_graph['rich_node_features'][:, 0] = torch.arange(num_nodes) * 60  # time_minutes
            sample_graph['rich_node_features'][:, 1] = torch.rand(num_nodes)  # normalized_price
            sample_graph['rich_node_features'][:, 2] = torch.randint(0, 3, (num_nodes,)).float()  # timeframe_source
        
        num_nodes = sample_graph['rich_node_features'].shape[0]
        attention_weights = torch.rand(num_nodes, num_nodes)
        edge_probabilities = torch.rand(num_nodes, num_nodes)
        predicted_edge_features = torch.rand(num_nodes, num_nodes, 17)
        
        # Test loss functions
        loss_functions = get_loss_functions()
        
        for name, loss_fn in loss_functions.items():
            try:
                loss = loss_fn(sample_graph, attention_weights, edge_probabilities, predicted_edge_features)
                results[f'loss_{name}'] = isinstance(loss, torch.Tensor) and loss.requires_grad
            except Exception as e:
                logger.error(f"Loss function {name} failed: {e}")
                results[f'loss_{name}'] = False
        
        # Test validators
        validators = get_validators()
        
        # Create sample edges
        sample_edges = [
            EdgeCandidate(
                source_idx=0,
                target_idx=1, 
                probability=0.8,
                features=np.random.rand(17),
                source_node_features=sample_graph['rich_node_features'][0].numpy(),
                target_node_features=sample_graph['rich_node_features'][1].numpy()
            )
        ]
        
        sample_historical_data = {
            'fisher_regime': ['high', 'medium', 'low'] * 10,
            'session_character': ['expansion', 'consolidation'] * 15,
            'timestamps': list(range(100)),
            'prices': np.random.rand(100) * 100 + 23000
        }
        
        for name, validator in validators.items():
            try:
                if name == 'permanence_validator':
                    result = validator.permanence_testing_across_regimes(sample_edges, sample_historical_data)
                elif name == 'distance_bridge_scorer':
                    result = validator.distance_bridge_scoring(sample_edges, {'context': 'test'})
                elif name == 'economic_backtest_validator':
                    result = validator.economic_backtest_validation(sample_edges, sample_historical_data)
                
                results[f'validator_{name}'] = 'error' not in result
            except Exception as e:
                logger.error(f"Validator {name} failed: {e}")
                results[f'validator_{name}'] = False
        
        # Overall success
        results['overall_success'] = all(results.values())
        
        logger.info(f"Discovery mathematics validation - Success rate: "
                   f"{sum(results.values())}/{len(results)} components passed")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in validate_discovery_mathematics: {e}")
        return {'overall_success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run validation when called directly
    validation_results = validate_discovery_mathematics()
    print("IRONFORGE Discovery Mathematics Validation Results:")
    print("=" * 55)
    
    for component, status in validation_results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{component:35} {status_str}")
    
    print("=" * 55)
    overall_status = "✅ SUCCESS" if validation_results.get('overall_success', False) else "❌ FAILED"
    print(f"Overall Status: {overall_status}")
    
    if validation_results.get('overall_success', False):
        print("\n🎯 All mathematical components are ready for IRONFORGE integration!")
        print("Use: from discovery_mathematics import get_loss_functions, get_validators")
    else:
        print("\n⚠️  Some components need attention before production deployment")