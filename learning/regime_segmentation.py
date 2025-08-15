#!/usr/bin/env python3
"""
IRONFORGE Regime Segmentation - Innovation Architect Implementation
=================================================================

Auto-clusters discovered patterns into market regimes using unsupervised learning.
Builds on 37D temporal cycle features and structural context for regime identification.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

@dataclass
class RegimeCharacteristics:
    """Container for regime cluster characteristics"""
    regime_id: int
    regime_label: str
    pattern_count: int
    centroid_features: np.ndarray
    stability_score: float
    temporal_dominance: str  # 'weekly', 'monthly', 'mixed'
    structural_dominance: str  # 'breakout', 'consolidation', 'reversal'
    price_range_preference: str  # 'high', 'mid', 'low'

class RegimeSegmentation:
    """
    Innovation Architect implementation for market regime segmentation
    Auto-clusters TGAT-discovered patterns into meaningful market regimes
    """
    
    def __init__(self, clustering_method='DBSCAN', 
                 min_patterns_per_regime=3, stability_threshold=0.5):
        self.logger = logging.getLogger(__name__)
        self.clustering_method = clustering_method
        self.min_patterns_per_regime = min_patterns_per_regime
        self.stability_threshold = stability_threshold
        
        # Initialize clusterer based on method
        if clustering_method == 'DBSCAN':
            self.clusterer = DBSCAN(eps=0.3, min_samples=min_patterns_per_regime)
        elif clustering_method == 'HDBSCAN':
            self.clusterer = HDBSCAN(min_cluster_size=min_patterns_per_regime)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        self.scaler = StandardScaler()
        self.regime_characteristics = {}
        
    def segment_patterns(self, tgat_patterns: List[Dict]) -> Dict[str, Any]:
        """
        Auto-cluster discovered patterns into market regimes
        
        Args:
            tgat_patterns: List of patterns from TGAT discovery
            
        Returns:
            Dictionary containing regime segmentation results
        """
        
        if len(tgat_patterns) < self.min_patterns_per_regime:
            self.logger.warning(f"Insufficient patterns for clustering: {len(tgat_patterns)} < {self.min_patterns_per_regime}")
            return self._create_single_regime_result(tgat_patterns)
        
        # Extract feature vectors for clustering
        feature_matrix = self._extract_regime_features(tgat_patterns)
        
        # Normalize features for clustering
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(scaled_features)
        
        # Analyze clustering quality
        quality_metrics = self._analyze_clustering_quality(scaled_features, cluster_labels)
        
        # Create regime characteristics
        regime_chars = self._characterize_regimes(
            tgat_patterns, feature_matrix, cluster_labels
        )
        
        # Generate regime labels mapping
        regime_labels = self._generate_regime_labels(tgat_patterns, cluster_labels)
        
        return {
            'regime_labels': regime_labels,
            'regime_characteristics': regime_chars,
            'quality_metrics': quality_metrics,
            'total_regimes': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'noise_patterns': sum(1 for label in cluster_labels if label == -1),
            'clustering_method': self.clustering_method,
            'feature_dimensions': feature_matrix.shape[1]
        }
    
    def _extract_regime_features(self, patterns: List[Dict]) -> np.ndarray:
        """
        Extract 12D feature vectors from TGAT patterns for regime clustering
        
        Features:
        - Temporal characteristics (4D): duration, frequency, periodicity, temporal_spread
        - Structural characteristics (4D): price_range, volatility, node_count, edge_density  
        - Relativity characteristics (4D): avg_normalized_price, relativity_consistency, 
                                          htf_alignment, cycle_alignment
        """
        
        feature_vectors = []
        
        for pattern in patterns:
            features = []
            
            # === Temporal Characteristics (4D) ===
            
            # Duration: Pattern timespan
            duration = self._calculate_pattern_duration(pattern)
            features.append(duration)
            
            # Frequency: How often this pattern type occurs
            frequency = pattern.get('frequency_score', 0.0)
            features.append(frequency)
            
            # Periodicity: Temporal regularity
            periodicity = self._calculate_periodicity(pattern)
            features.append(periodicity)
            
            # Temporal spread: How distributed across time
            temporal_spread = self._calculate_temporal_spread(pattern)
            features.append(temporal_spread)
            
            # === Structural Characteristics (4D) ===
            
            # Price range: Normalized price movement span
            price_range = self._calculate_price_range(pattern)
            features.append(price_range)
            
            # Volatility: Price movement intensity
            volatility = pattern.get('volatility_score', 0.0)
            features.append(volatility)
            
            # Node count: Pattern complexity
            node_count = len(pattern.get('nodes', []))
            features.append(node_count)
            
            # Edge density: Connection richness
            edge_density = self._calculate_edge_density(pattern)
            features.append(edge_density)
            
            # === Relativity Characteristics (4D) ===
            
            # Average normalized price: Price level preference
            avg_normalized_price = self._calculate_avg_normalized_price(pattern)
            features.append(avg_normalized_price)
            
            # Relativity consistency: How stable price relationships are
            relativity_consistency = self._calculate_relativity_consistency(pattern)
            features.append(relativity_consistency)
            
            # HTF alignment: Higher timeframe correlation
            htf_alignment = pattern.get('htf_correlation', 0.0)
            features.append(htf_alignment)
            
            # Cycle alignment: Temporal cycle correlation
            cycle_alignment = self._calculate_cycle_alignment(pattern)
            features.append(cycle_alignment)
            
            feature_vectors.append(features)
        
        return np.array(feature_vectors)
    
    def _calculate_pattern_duration(self, pattern: Dict) -> float:
        """Calculate pattern duration in normalized time units"""
        nodes = pattern.get('nodes', [])
        if len(nodes) < 2:
            return 0.0
        
        # Extract timestamps from nodes
        timestamps = []
        for node in nodes:
            timestamp = node.get('timestamp', 0)
            if isinstance(timestamp, str):
                # Convert timestamp string to minutes
                try:
                    parts = timestamp.split(':')
                    timestamp = int(parts[0]) * 60 + int(parts[1])
                except:
                    timestamp = 0
            timestamps.append(timestamp)
        
        if not timestamps:
            return 0.0
        
        # Duration in minutes, normalized to [0,1] (assuming max session = 480 minutes)
        duration = (max(timestamps) - min(timestamps)) / 480.0
        return min(1.0, duration)
    
    def _calculate_periodicity(self, pattern: Dict) -> float:
        """Calculate temporal periodicity score"""
        # Use temporal cycle information if available
        pattern_type = pattern.get('type', '')
        
        if 'weekly' in pattern_type or 'day_of_week' in pattern_type:
            return 0.8  # High periodicity for weekly patterns
        elif 'monthly' in pattern_type or 'month' in pattern_type:
            return 0.6  # Medium periodicity for monthly patterns  
        elif 'temporal' in pattern_type:
            return 0.4  # Some periodicity for temporal patterns
        else:
            return 0.2  # Low periodicity for other patterns
    
    def _calculate_temporal_spread(self, pattern: Dict) -> float:
        """Calculate how spread out pattern is across time"""
        nodes = pattern.get('nodes', [])
        if len(nodes) < 3:
            return 0.0
        
        # Calculate time gaps between consecutive nodes
        timestamps = []
        for node in nodes:
            timestamp = node.get('timestamp', 0)
            if isinstance(timestamp, str):
                try:
                    parts = timestamp.split(':')
                    timestamp = int(parts[0]) * 60 + int(parts[1])
                except:
                    timestamp = 0
            timestamps.append(timestamp)
        
        timestamps.sort()
        
        if len(timestamps) < 2:
            return 0.0
        
        # Calculate variance in time gaps (normalized)
        gaps = np.diff(timestamps)
        if len(gaps) == 0:
            return 0.0
        
        gap_variance = np.var(gaps)
        # Normalize by typical session variance (assuming ~30min typical gap)
        normalized_spread = min(1.0, gap_variance / (30.0 ** 2))
        
        return normalized_spread
    
    def _calculate_price_range(self, pattern: Dict) -> float:
        """Calculate normalized price range for pattern"""
        nodes = pattern.get('nodes', [])
        
        normalized_prices = []
        for node in nodes:
            normalized_price = node.get('normalized_price', 0.5)
            normalized_prices.append(normalized_price)
        
        if not normalized_prices:
            return 0.0
        
        price_range = max(normalized_prices) - min(normalized_prices)
        return price_range
    
    def _calculate_edge_density(self, pattern: Dict) -> float:
        """Calculate edge density (connections per node)"""
        nodes = pattern.get('nodes', [])
        edges = pattern.get('edges', [])
        
        if len(nodes) == 0:
            return 0.0
        
        # Edge density = edges / max_possible_edges
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        if max_edges == 0:
            return 0.0
        
        density = len(edges) / max_edges
        return min(1.0, density)
    
    def _calculate_avg_normalized_price(self, pattern: Dict) -> float:
        """Calculate average normalized price level for pattern"""
        nodes = pattern.get('nodes', [])
        
        normalized_prices = []
        for node in nodes:
            normalized_price = node.get('normalized_price', 0.5)
            normalized_prices.append(normalized_price)
        
        if not normalized_prices:
            return 0.5  # Neutral if no data
        
        return np.mean(normalized_prices)
    
    def _calculate_relativity_consistency(self, pattern: Dict) -> float:
        """Calculate how consistent price relativity features are"""
        nodes = pattern.get('nodes', [])
        
        relativity_features = ['pct_from_open', 'pct_from_high', 'pct_from_low', 'price_to_HTF_ratio']
        feature_variances = []
        
        for feature_name in relativity_features:
            values = []
            for node in nodes:
                value = node.get(feature_name, 0.0)
                values.append(value)
            
            if values:
                # High consistency = low variance
                variance = np.var(values)
                consistency = 1.0 / (1.0 + variance)  # Transform variance to consistency
                feature_variances.append(consistency)
        
        if not feature_variances:
            return 0.5
        
        return np.mean(feature_variances)
    
    def _calculate_cycle_alignment(self, pattern: Dict) -> float:
        """Calculate temporal cycle alignment score"""
        pattern_type = pattern.get('type', '')
        description = pattern.get('description', '')
        
        # Score based on temporal cycle features in pattern
        cycle_score = 0.0
        
        if 'weekly_cycle' in pattern_type:
            cycle_score += 0.4
        if 'monthly_cycle' in pattern_type:
            cycle_score += 0.3
        if 'cross_cycle_confluence' in pattern_type:
            cycle_score += 0.5
        if 'week' in description.lower():
            cycle_score += 0.2
        if 'month' in description.lower():
            cycle_score += 0.2
        
        return min(1.0, cycle_score)
    
    def _analyze_clustering_quality(self, features: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, float]:
        """Analyze the quality of clustering results"""
        
        quality_metrics = {}
        
        # Remove noise points (-1 labels) for quality analysis
        valid_mask = labels >= 0
        if np.sum(valid_mask) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'noise_ratio': 1.0,
                'cluster_count': 0
            }
        
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Silhouette score (higher is better, [-1, 1])
        try:
            if len(set(valid_labels)) > 1:
                sil_score = silhouette_score(valid_features, valid_labels)
            else:
                sil_score = 0.0
        except:
            sil_score = 0.0
        
        # Calinski-Harabasz score (higher is better)
        try:
            if len(set(valid_labels)) > 1:
                ch_score = calinski_harabasz_score(valid_features, valid_labels)
            else:
                ch_score = 0.0
        except:
            ch_score = 0.0
        
        # Noise ratio (lower is better)
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Cluster count
        cluster_count = len(set(valid_labels))
        
        quality_metrics = {
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'noise_ratio': noise_ratio,
            'cluster_count': cluster_count
        }
        
        return quality_metrics
    
    def _characterize_regimes(self, patterns: List[Dict], features: np.ndarray, 
                            labels: np.ndarray) -> Dict[int, RegimeCharacteristics]:
        """Create detailed characteristics for each regime cluster"""
        
        regime_chars = {}
        unique_labels = set(labels)
        
        for regime_id in unique_labels:
            if regime_id == -1:  # Skip noise cluster
                continue
            
            # Get patterns and features for this regime
            regime_mask = labels == regime_id
            regime_patterns = [patterns[i] for i in np.where(regime_mask)[0]]
            regime_features = features[regime_mask]
            
            if len(regime_patterns) == 0:
                continue
            
            # Calculate centroid
            centroid = np.mean(regime_features, axis=0)
            
            # Calculate stability (inverse of within-cluster variance)
            if len(regime_features) > 1:
                stability = 1.0 / (1.0 + np.mean(np.var(regime_features, axis=0)))
            else:
                stability = 1.0
            
            # Analyze temporal dominance
            temporal_dominance = self._analyze_temporal_dominance(regime_patterns)
            
            # Analyze structural dominance  
            structural_dominance = self._analyze_structural_dominance(regime_patterns)
            
            # Analyze price range preference
            price_range_pref = self._analyze_price_range_preference(regime_features, centroid)
            
            # Generate regime label
            regime_label = self._generate_regime_label(
                temporal_dominance, structural_dominance, price_range_pref
            )
            
            regime_chars[regime_id] = RegimeCharacteristics(
                regime_id=regime_id,
                regime_label=regime_label,
                pattern_count=len(regime_patterns),
                centroid_features=centroid,
                stability_score=stability,
                temporal_dominance=temporal_dominance,
                structural_dominance=structural_dominance,
                price_range_preference=price_range_pref
            )
        
        return regime_chars
    
    def _analyze_temporal_dominance(self, patterns: List[Dict]) -> str:
        """Determine temporal dominance for regime"""
        weekly_count = sum(1 for p in patterns if 'weekly' in p.get('type', ''))
        monthly_count = sum(1 for p in patterns if 'monthly' in p.get('type', ''))
        
        if weekly_count > monthly_count * 1.5:
            return 'weekly'
        elif monthly_count > weekly_count * 1.5:
            return 'monthly'
        else:
            return 'mixed'
    
    def _analyze_structural_dominance(self, patterns: List[Dict]) -> str:
        """Determine structural dominance for regime"""
        breakout_keywords = ['breakout', 'sweep', 'cascade', 'momentum']
        consolidation_keywords = ['consolidation', 'range', 'equilibrium', 'balance']
        reversal_keywords = ['reversal', 'rejection', 'bounce', 'retracement']
        
        breakout_count = 0
        consolidation_count = 0
        reversal_count = 0
        
        for pattern in patterns:
            desc = pattern.get('description', '').lower()
            pattern_type = pattern.get('type', '').lower()
            text = f"{desc} {pattern_type}"
            
            for keyword in breakout_keywords:
                if keyword in text:
                    breakout_count += 1
                    break
            
            for keyword in consolidation_keywords:
                if keyword in text:
                    consolidation_count += 1
                    break
            
            for keyword in reversal_keywords:
                if keyword in text:
                    reversal_count += 1
                    break
        
        # Determine dominance
        counts = [breakout_count, consolidation_count, reversal_count]
        max_count = max(counts)
        
        if max_count == 0:
            return 'mixed'
        elif breakout_count == max_count:
            return 'breakout'
        elif consolidation_count == max_count:
            return 'consolidation'
        else:
            return 'reversal'
    
    def _analyze_price_range_preference(self, features: np.ndarray, 
                                      centroid: np.ndarray) -> str:
        """Determine price range preference for regime"""
        # Use avg_normalized_price feature (index 8 in 12D features)
        avg_price_level = centroid[8]
        
        if avg_price_level < 0.33:
            return 'low'
        elif avg_price_level > 0.67:
            return 'high'
        else:
            return 'mid'
    
    def _generate_regime_label(self, temporal: str, structural: str, 
                             price_pref: str) -> str:
        """Generate human-readable regime label"""
        return f"{temporal}_{structural}_{price_pref}"
    
    def _generate_regime_labels(self, patterns: List[Dict], 
                              cluster_labels: np.ndarray) -> Dict[str, int]:
        """Generate pattern_id -> regime_label mapping"""
        regime_labels = {}
        
        for i, pattern in enumerate(patterns):
            pattern_id = pattern.get('pattern_id', f"pattern_{i}")
            regime_id = int(cluster_labels[i])
            regime_labels[pattern_id] = regime_id
        
        return regime_labels
    
    def _create_single_regime_result(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Create single regime result for insufficient patterns"""
        
        regime_labels = {}
        for i, pattern in enumerate(patterns):
            pattern_id = pattern.get('pattern_id', f"pattern_{i}")
            regime_labels[pattern_id] = 0  # Single regime
        
        return {
            'regime_labels': regime_labels,
            'regime_characteristics': {
                0: RegimeCharacteristics(
                    regime_id=0,
                    regime_label='insufficient_data_single_regime',
                    pattern_count=len(patterns),
                    centroid_features=np.zeros(12),
                    stability_score=1.0,
                    temporal_dominance='mixed',
                    structural_dominance='mixed',
                    price_range_preference='mid'
                )
            },
            'quality_metrics': {
                'silhouette_score': 1.0,
                'calinski_harabasz_score': 0.0,
                'noise_ratio': 0.0,
                'cluster_count': 1
            },
            'total_regimes': 1,
            'noise_patterns': 0,
            'clustering_method': self.clustering_method,
            'feature_dimensions': 12
        }

def main():
    """Command-line interface for regime segmentation testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IRONFORGE regime segmentation")
    parser.add_argument('patterns_file', help="JSON file containing TGAT patterns")
    parser.add_argument('--method', choices=['DBSCAN', 'HDBSCAN'], 
                       default='DBSCAN', help="Clustering method")
    parser.add_argument('--min-patterns', type=int, default=3, 
                       help="Minimum patterns per regime")
    parser.add_argument('--output', '-o', help="Output file for regime results")
    
    args = parser.parse_args()
    
    # Load patterns
    try:
        with open(args.patterns_file, 'r') as f:
            patterns_data = json.load(f)
        
        patterns = patterns_data.get('patterns', [])
        if not patterns:
            print(f"❌ No patterns found in {args.patterns_file}")
            return 1
            
    except Exception as e:
        print(f"❌ Error loading patterns: {e}")
        return 1
    
    # Run regime segmentation
    try:
        segmenter = RegimeSegmentation(
            clustering_method=args.method,
            min_patterns_per_regime=args.min_patterns
        )
        
        results = segmenter.segment_patterns(patterns)
        
        print(f"\n🎯 Regime Segmentation Results:")
        print(f"   Total regimes: {results['total_regimes']}")
        print(f"   Noise patterns: {results['noise_patterns']}")
        print(f"   Silhouette score: {results['quality_metrics']['silhouette_score']:.3f}")
        print(f"   Clustering method: {results['clustering_method']}")
        
        # Show regime characteristics
        regime_chars = results['regime_characteristics']
        for regime_id, char in regime_chars.items():
            if isinstance(char, RegimeCharacteristics):
                print(f"\n📊 Regime {regime_id}: {char.regime_label}")
                print(f"     Patterns: {char.pattern_count}")
                print(f"     Stability: {char.stability_score:.3f}")
                print(f"     Temporal: {char.temporal_dominance}")
                print(f"     Structural: {char.structural_dominance}")
        
        # Save results if requested
        if args.output:
            # Convert RegimeCharacteristics to dict for JSON serialization
            serializable_results = results.copy()
            serializable_chars = {}
            
            for regime_id, char in regime_chars.items():
                if isinstance(char, RegimeCharacteristics):
                    serializable_chars[regime_id] = {
                        'regime_id': char.regime_id,
                        'regime_label': char.regime_label,
                        'pattern_count': char.pattern_count,
                        'stability_score': char.stability_score,
                        'temporal_dominance': char.temporal_dominance,
                        'structural_dominance': char.structural_dominance,
                        'price_range_preference': char.price_range_preference,
                        'centroid_features': char.centroid_features.tolist()
                    }
            
            serializable_results['regime_characteristics'] = serializable_chars
            
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Regime segmentation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())