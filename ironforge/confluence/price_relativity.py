"""
IRONFORGE Price Relativity Module
================================

Price-relative hierarchical clustering and archaeological zone analysis.
Ensures pattern detection is meaningful across different price levels and market conditions.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PriceRelativeConfig:
    """Configuration for price-relative hierarchical analysis"""
    
    # Price normalization settings
    use_percentage_moves: bool = True
    baseline_price: Optional[float] = None  # For relative calculations
    price_band_width: float = 0.02  # 2% price bands for pattern grouping
    
    # Scale factor adjustments
    relative_scale_factors: Dict[str, float] = None
    absolute_scale_factors: Dict[str, float] = None
    
    # Archaeological zone settings
    zone_precision_mode: str = 'relative'  # 'relative' or 'absolute'
    min_relative_precision: float = 0.01   # 1% minimum relative precision
    max_relative_precision: float = 0.10   # 10% maximum relative precision
    
    def __post_init__(self):
        if self.relative_scale_factors is None:
            self.relative_scale_factors = {
                'sub_zone': 0.618,
                'base_zone': 1.0,
                'super_zone': 1.618,
                'macro_zone': 2.618
            }
        
        if self.absolute_scale_factors is None:
            self.absolute_scale_factors = {
                'sub_zone': 0.75,
                'base_zone': 1.0, 
                'super_zone': 1.25,
                'macro_zone': 1.5
            }


class PriceRelativeHierarchicalAnalyzer:
    """
    Price-relative hierarchical clustering and pattern analysis.
    
    Addresses the fundamental challenge that patterns must be meaningful
    across different price levels and market conditions.
    """
    
    def __init__(self, config: Optional[PriceRelativeConfig] = None):
        self.config = config or PriceRelativeConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"🔄 Price-relative hierarchical analysis initialized")
        self.logger.info(f"   Mode: {self.config.zone_precision_mode}")
        self.logger.info(f"   Price bands: {self.config.price_band_width*100:.1f}%")
    
    def calculate_price_relative_precision(self, 
                                         zone_percentage: float,
                                         current_price: float,
                                         target_precision_points: float = 7.55,
                                         market_volatility: float = 0.02) -> Dict[str, float]:
        """
        Calculate price-relative precision targets that scale appropriately with price levels.
        
        Args:
            zone_percentage: Archaeological zone percentage (0.40 for 40% zone)
            current_price: Current market price level
            target_precision_points: Target precision in absolute points (legacy)
            market_volatility: Current market volatility (as decimal percentage)
            
        Returns:
            Dictionary with both absolute and relative precision metrics
        """
        try:
            # Convert absolute target to relative percentage
            relative_target = (target_precision_points / current_price) * 100.0
            
            # Adjust for market volatility (higher volatility = looser precision targets)
            volatility_adjustment = 1.0 + (market_volatility * 2.0)  # 2% vol = 1.04x adjustment
            
            # Zone-specific precision factors (relative to price)
            zone_precision_factors = {
                0.236: 0.028,  # Fibonacci levels tend to be more precise
                0.382: 0.025,
                0.40: 0.035,   # 40% zone baseline (key archaeological level)
                0.50: 0.040,   # Round number, slightly less precise
                0.618: 0.030,  # Golden ratio precision
                0.786: 0.038,  # Higher Fibonacci levels
            }
            
            # Find closest zone or interpolate
            closest_zone = min(zone_precision_factors.keys(), 
                             key=lambda x: abs(x - zone_percentage))
            base_relative_precision = zone_precision_factors.get(closest_zone, 0.035)
            
            # Apply volatility adjustment
            adjusted_relative_precision = base_relative_precision * volatility_adjustment
            
            # Convert back to absolute points
            adjusted_absolute_precision = (adjusted_relative_precision / 100.0) * current_price
            
            # Price band classification for pattern grouping
            price_band = self._classify_price_band(current_price)
            
            return {
                'relative_precision_pct': adjusted_relative_precision,
                'absolute_precision_points': adjusted_absolute_precision,
                'original_target_points': target_precision_points,
                'price_level': current_price,
                'volatility_adjustment': volatility_adjustment,
                'zone_percentage': zone_percentage,
                'price_band': price_band,
                'precision_mode': 'price_relative'
            }
            
        except Exception as e:
            self.logger.warning(f"Price-relative precision calculation failed: {e}")
            return {
                'relative_precision_pct': 3.5,  # Default 3.5% relative
                'absolute_precision_points': target_precision_points,
                'precision_mode': 'fallback_absolute'
            }
    
    def calculate_hierarchical_scale_factors(self, 
                                           current_price: float,
                                           price_range: Dict[str, float],
                                           base_zone_percentage: float) -> Dict[str, Dict[str, float]]:
        """
        Calculate price-relative hierarchical scale factors for multi-scale analysis.
        
        Args:
            current_price: Current market price
            price_range: Session price range {'high': float, 'low': float}
            base_zone_percentage: Base archaeological zone (e.g., 0.40)
            
        Returns:
            Scale factors adjusted for price relativity
        """
        try:
            range_size = price_range['high'] - price_range['low']
            range_percentage = (range_size / current_price) * 100.0
            
            # Adjust scale factors based on price range relative to current price
            if range_percentage < 1.0:  # Tight range session
                # Use smaller relative scale factors for tight ranges
                scale_adjustment = 0.8
                scale_factors = self.config.relative_scale_factors
            elif range_percentage > 5.0:  # Wide range session
                # Use larger absolute scale factors for wide ranges
                scale_adjustment = 1.2  
                scale_factors = self.config.absolute_scale_factors
            else:  # Normal range session
                scale_adjustment = 1.0
                scale_factors = self.config.relative_scale_factors
            
            # Calculate price-adjusted zone levels for each scale
            hierarchical_zones = {}
            
            for scale_name, scale_factor in scale_factors.items():
                # Apply price-relative scaling
                adjusted_scale_factor = scale_factor * scale_adjustment
                
                # Calculate scaled zone percentage (bounded)
                scaled_percentage = min(0.95, max(0.05, base_zone_percentage * adjusted_scale_factor))
                
                # Calculate absolute zone level
                scaled_zone_level = price_range['low'] + (range_size * scaled_percentage)
                
                # Calculate relative distance from current price
                relative_distance = abs(scaled_zone_level - current_price) / current_price * 100.0
                
                hierarchical_zones[scale_name] = {
                    'scale_factor': adjusted_scale_factor,
                    'zone_percentage': scaled_percentage,
                    'zone_level': scaled_zone_level,
                    'relative_distance_pct': relative_distance,
                    'price_significance': self._calculate_price_significance(
                        scaled_zone_level, current_price, range_percentage
                    )
                }
                
            return {
                'hierarchical_zones': hierarchical_zones,
                'range_percentage': range_percentage,
                'scale_adjustment_factor': scale_adjustment,
                'price_context': {
                    'current_price': current_price,
                    'range_high': price_range['high'],
                    'range_low': price_range['low'],
                    'range_size': range_size
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Hierarchical scale factor calculation failed: {e}")
            return {'hierarchical_zones': {}, 'error': str(e)}
    
    def normalize_attention_weights_by_price(self, 
                                           attention_matrix: np.ndarray,
                                           price_levels: List[float],
                                           current_price: float) -> np.ndarray:
        """
        Normalize TGAT attention weights by price relativity for meaningful cross-price comparisons.
        
        Args:
            attention_matrix: Original TGAT attention weights [N, N]
            price_levels: Price level for each node/event
            current_price: Current market price for normalization
            
        Returns:
            Price-normalized attention matrix
        """
        try:
            if len(price_levels) != attention_matrix.shape[0]:
                self.logger.warning("Price levels length mismatch with attention matrix")
                return attention_matrix
                
            normalized_matrix = attention_matrix.copy()
            
            # Calculate price distance normalization factors
            for i in range(len(price_levels)):
                for j in range(len(price_levels)):
                    if i != j:
                        # Price distance between events
                        price_distance = abs(price_levels[i] - price_levels[j])
                        relative_price_distance = price_distance / current_price
                        
                        # Normalize attention by price distance
                        # Closer prices get higher relative attention weights
                        if relative_price_distance > 0:
                            price_normalization_factor = 1.0 / (1.0 + relative_price_distance * 10.0)
                            normalized_matrix[i, j] *= price_normalization_factor
                        
            return normalized_matrix
            
        except Exception as e:
            self.logger.warning(f"Price normalization failed: {e}")
            return attention_matrix
    
    def calculate_cross_price_coherence(self, 
                                      patterns_by_price_band: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate coherence of hierarchical patterns across different price bands.
        
        Args:
            patterns_by_price_band: Patterns grouped by price bands
            
        Returns:
            Cross-price coherence metrics
        """
        try:
            coherence_metrics = {}
            
            # Calculate intra-band coherence (patterns within same price band)
            for band_name, patterns in patterns_by_price_band.items():
                if len(patterns) > 1:
                    intra_coherence = self._calculate_intra_band_coherence(patterns)
                    coherence_metrics[f'intra_coherence_{band_name}'] = intra_coherence
                    
            # Calculate inter-band coherence (patterns across different price bands)
            band_names = list(patterns_by_price_band.keys())
            if len(band_names) > 1:
                inter_coherence_scores = []
                
                for i, band1 in enumerate(band_names):
                    for j, band2 in enumerate(band_names[i+1:], i+1):
                        inter_coherence = self._calculate_inter_band_coherence(
                            patterns_by_price_band[band1],
                            patterns_by_price_band[band2]
                        )
                        inter_coherence_scores.append(inter_coherence)
                        coherence_metrics[f'inter_coherence_{band1}_{band2}'] = inter_coherence
                
                # Overall inter-band coherence
                coherence_metrics['overall_inter_coherence'] = np.mean(inter_coherence_scores)
            
            # Price stability metric
            price_stability = self._calculate_price_stability(patterns_by_price_band)
            coherence_metrics['price_stability'] = price_stability
            
            return coherence_metrics
            
        except Exception as e:
            self.logger.warning(f"Cross-price coherence calculation failed: {e}")
            return {'error': str(e)}
    
    def _classify_price_band(self, price: float) -> str:
        """Classify price into bands for pattern grouping"""
        # Example for NQ futures (adjust for other instruments)
        if price < 16000:
            return 'low_band'
        elif price < 18000:
            return 'mid_low_band'
        elif price < 20000:
            return 'mid_high_band'
        else:
            return 'high_band'
    
    def _calculate_price_significance(self, zone_level: float, current_price: float, range_pct: float) -> float:
        """Calculate price significance of a zone level"""
        relative_distance = abs(zone_level - current_price) / current_price
        
        # Closer zones are more significant, adjusted for range
        base_significance = 1.0 / (1.0 + relative_distance * 5.0)
        
        # Adjust for range context
        range_adjustment = min(2.0, 1.0 + range_pct / 10.0)
        
        return min(1.0, base_significance * range_adjustment)
    
    def _calculate_intra_band_coherence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate coherence within a price band"""
        if len(patterns) < 2:
            return 0.5
            
        # Extract pattern features for coherence calculation
        coherence_scores = []
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Simple coherence based on pattern similarity
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_inter_band_coherence(self, patterns1: List[Dict[str, Any]], patterns2: List[Dict[str, Any]]) -> float:
        """Calculate coherence between different price bands"""
        if not patterns1 or not patterns2:
            return 0.5
            
        # Cross-band pattern comparison
        cross_similarities = []
        for pattern1 in patterns1[:3]:  # Limit for performance
            for pattern2 in patterns2[:3]:
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                cross_similarities.append(similarity)
        
        return np.mean(cross_similarities) if cross_similarities else 0.5
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns"""
        # Simplified pattern similarity based on available metrics
        try:
            # Compare significance scores if available
            if 'significance_scores' in pattern1 and 'significance_scores' in pattern2:
                sig1 = pattern1['significance_scores']
                sig2 = pattern2['significance_scores']
                if isinstance(sig1, (list, np.ndarray)) and isinstance(sig2, (list, np.ndarray)):
                    sig1_mean = np.mean(sig1)
                    sig2_mean = np.mean(sig2)
                    significance_similarity = 1.0 - abs(sig1_mean - sig2_mean)
                    return max(0.0, min(1.0, significance_similarity))
            
            return 0.5  # Default neutral similarity
            
        except Exception as e:
            self.logger.warning(f"Pattern similarity calculation failed: {e}")
            return 0.5
    
    def _calculate_price_stability(self, patterns_by_price_band: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall price stability across bands"""
        try:
            band_sizes = [len(patterns) for patterns in patterns_by_price_band.values()]
            if not band_sizes:
                return 0.5
                
            # Price stability = inverse of price band distribution variance
            # More evenly distributed patterns across bands = higher stability
            variance = np.var(band_sizes)
            max_variance = np.var([len(patterns_by_price_band), 0] * (len(patterns_by_price_band) // 2))
            
            if max_variance > 0:
                stability = 1.0 - (variance / max_variance)
                return max(0.0, min(1.0, stability))
            
            return 0.5
            
        except Exception as e:
            return 0.5


def create_price_relative_archaeological_analysis(session_data: Dict[str, Any], 
                                                current_price: float,
                                                config: Optional[PriceRelativeConfig] = None) -> Dict[str, Any]:
    """
    Create comprehensive price-relative archaeological analysis for a session.
    
    Args:
        session_data: Session data with events and price information
        current_price: Current market price level
        config: Optional configuration for price-relative analysis
        
    Returns:
        Complete price-relative archaeological analysis results
    """
    analyzer = PriceRelativeHierarchicalAnalyzer(config)
    
    # Extract price range from session data
    events = session_data.get('session_liquidity_events', [])
    if events:
        price_levels = [event.get('price_level', current_price) for event in events]
        price_range = {
            'high': max(price_levels),
            'low': min(price_levels)
        }
    else:
        # Fallback: assume 2% range around current price
        range_size = current_price * 0.02
        price_range = {
            'high': current_price + range_size,
            'low': current_price - range_size
        }
    
    # Calculate price-relative precision for key archaeological zones
    key_zones = [0.236, 0.382, 0.40, 0.50, 0.618, 0.786]
    precision_analysis = {}
    
    for zone_pct in key_zones:
        precision_metrics = analyzer.calculate_price_relative_precision(
            zone_pct, current_price, target_precision_points=7.55
        )
        precision_analysis[f'zone_{zone_pct:.3f}'] = precision_metrics
    
    # Calculate hierarchical scale factors for 40% zone (primary archaeological level)
    hierarchical_analysis = analyzer.calculate_hierarchical_scale_factors(
        current_price, price_range, base_zone_percentage=0.40
    )
    
    return {
        'price_relative_config': config.__dict__ if config else PriceRelativeConfig().__dict__,
        'price_context': {
            'current_price': current_price,
            'price_range': price_range,
            'range_percentage': ((price_range['high'] - price_range['low']) / current_price) * 100.0
        },
        'precision_analysis': precision_analysis,
        'hierarchical_analysis': hierarchical_analysis,
        'analysis_timestamp': np.datetime64('now').isoformat()
    }