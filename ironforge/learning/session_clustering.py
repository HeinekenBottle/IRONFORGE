"""
Session Clustering - Offline Library Builder

Builds persistent session fingerprint library using k-means clustering:
- Enumerates and processes all discoverable historical sessions
- Clusters with k-means (default k=5, fixed random_state)
- Persists centroids, scaler, and comprehensive metadata
- Provides reproducible clustering with coverage tracking
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter, defaultdict

from .session_fingerprinting import (
    SessionFingerprintExtractor, 
    SessionFingerprintConfig,
    SessionFingerprint
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for session clustering"""
    k_clusters: int = 5
    random_state: int = 42
    distance_metric: str = "euclidean"  # "euclidean" or "cosine"
    max_iter: int = 300
    n_init: int = 10
    min_sessions_threshold: int = 10
    
    @classmethod
    def default(cls) -> 'ClusteringConfig':
        """Default clustering configuration"""
        return cls()


@dataclass
class ClusterStats:
    """Statistics for a single cluster"""
    cluster_id: int
    n_sessions: int
    session_ids: List[str]
    centroid: np.ndarray
    
    # Range/volatility characteristics
    normalized_range_p50: float
    volatility_proxy_mean: float
    volatility_class: str  # "low", "medium", "high"
    
    # Phase characteristics
    top_semantic_phases: List[Tuple[str, float]]  # (phase_name, rate)
    htf_regime_dominant: int  # 0, 1, or 2
    
    # Temporal characteristics
    avg_session_duration: float
    avg_events_per_session: float
    temporal_consistency_score: float
    
    # Session type distribution
    session_type_distribution: Dict[str, int]
    date_range: Tuple[str, str]  # (earliest, latest)


@dataclass
class ClusteringMetadata:
    """Comprehensive metadata for clustering run"""
    # Processing summary
    total_sessions_discovered: int
    sessions_processed: int
    sessions_skipped: int
    skipped_reasons: Dict[str, int]
    coverage_percent: float
    
    # Data characteristics
    timeframe: str  # e.g., "M5"
    timezone: str
    date_range: Tuple[str, str]
    session_types: List[str]
    
    # Feature schema
    feature_schema_version: str
    feature_names: List[str]
    feature_dimensions: int
    scaler_type: str
    
    # Clustering parameters
    k_clusters: int
    distance_metric: str
    random_state: int
    
    # Model performance
    inertia: float
    silhouette_score: float
    cluster_separation_score: float
    
    # Timestamp
    created_at: str
    version: str = "1.0.2"


class SessionClusteringLibrary:
    """Offline library builder for session fingerprint clustering"""
    
    def __init__(self, 
                 clustering_config: Optional[ClusteringConfig] = None,
                 fingerprint_config: Optional[SessionFingerprintConfig] = None):
        self.clustering_config = clustering_config or ClusteringConfig.default()
        self.fingerprint_config = fingerprint_config or SessionFingerprintConfig.default()
        
        # Components
        self.extractor = SessionFingerprintExtractor(self.fingerprint_config)
        self.kmeans = None
        self.cluster_stats = {}
        self.metadata = None
        
        self.logger = logging.getLogger(__name__)
    
    def enumerate_historical_sessions(self, data_dirs: Optional[List[Path]] = None) -> List[Path]:
        """Enumerate all discoverable historical session files"""
        if data_dirs is None:
            data_dirs = [Path("data/adapted"), Path("data/enhanced")]
        
        session_files = []
        discovery_log = defaultdict(int)
        
        for data_dir in data_dirs:
            if not data_dir.exists():
                self.logger.warning(f"Data directory not found: {data_dir}")
                continue
                
            # Adapted enhanced sessions (primary)
            if data_dir.name == "adapted":
                adapted_files = list(data_dir.glob("adapted_enhanced_rel_*.json"))
                session_files.extend(adapted_files)
                discovery_log["adapted_enhanced"] = len(adapted_files)
                
            # Enhanced sessions (secondary)
            elif data_dir.name == "enhanced":
                enhanced_files = list(data_dir.glob("enhanced_*.json"))
                # Filter out adapted versions to avoid duplicates
                enhanced_files = [f for f in enhanced_files if not f.name.startswith("enhanced_rel_")]
                session_files.extend(enhanced_files)
                discovery_log["enhanced"] = len(enhanced_files)
        
        self.logger.info(f"Discovered {len(session_files)} historical sessions")
        for source, count in discovery_log.items():
            self.logger.info(f"  {source}: {count} sessions")
            
        return sorted(session_files)  # Deterministic ordering
    
    def extract_all_fingerprints(self, session_files: List[Path]) -> Tuple[List[SessionFingerprint], Dict[str, int]]:
        """Extract fingerprints from all session files with detailed tracking"""
        fingerprints = []
        skipped_reasons = defaultdict(int)
        
        self.logger.info(f"Processing {len(session_files)} session files...")
        
        for i, session_file in enumerate(session_files):
            if i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(session_files)} sessions processed")
                
            try:
                fingerprint = self.extractor.extract_session_fingerprint(session_file)
                
                if fingerprint is None:
                    skipped_reasons["extraction_failed"] += 1
                    continue
                    
                if fingerprint.n_events < self.clustering_config.min_sessions_threshold:
                    skipped_reasons["insufficient_events"] += 1
                    continue
                    
                fingerprints.append(fingerprint)
                
            except Exception as e:
                self.logger.warning(f"Error processing {session_file.name}: {e}")
                skipped_reasons["processing_error"] += 1
                continue
        
        self.logger.info(f"Successfully extracted {len(fingerprints)} fingerprints")
        self.logger.info(f"Skipped {sum(skipped_reasons.values())} sessions: {dict(skipped_reasons)}")
        
        return fingerprints, dict(skipped_reasons)
    
    def fit_clustering(self, fingerprints: List[SessionFingerprint]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit k-means clustering on fingerprint vectors"""
        if len(fingerprints) < self.clustering_config.k_clusters:
            raise ValueError(f"Too few fingerprints ({len(fingerprints)}) for k={self.clustering_config.k_clusters}")
        
        # Extract feature matrix
        feature_matrix = np.vstack([fp.feature_vector for fp in fingerprints])
        
        # NON-DETERMINISTIC CLUSTERING PROTECTION: Enforce fixed random_state
        if self.clustering_config.random_state is None:
            raise ValueError(
                "Non-deterministic clustering detected: random_state must be fixed for reproducible results. "
                "This is critical for production consistency and validation. "
                "Please set clustering_config.random_state to a fixed integer (e.g., 42)."
            )
        
        # Fit k-means with enforced deterministic settings
        self.kmeans = KMeans(
            n_clusters=self.clustering_config.k_clusters,
            random_state=self.clustering_config.random_state,  # REQUIRED: Fixed for determinism
            max_iter=self.clustering_config.max_iter,
            n_init=self.clustering_config.n_init
        )
        
        # Log deterministic settings for validation
        self.logger.info(f"Deterministic clustering initialized: random_state={self.clustering_config.random_state}")
        self.logger.info(f"Clustering parameters: k={self.clustering_config.k_clusters}, max_iter={self.clustering_config.max_iter}, n_init={self.clustering_config.n_init}")
        
        cluster_labels = self.kmeans.fit_predict(feature_matrix)
        
        self.logger.info(f"K-means clustering completed (k={self.clustering_config.k_clusters})")
        self.logger.info(f"Inertia: {self.kmeans.inertia_:.3f}")
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
        self.logger.info(f"Silhouette score: {silhouette_avg:.3f}")
        
        return feature_matrix, cluster_labels
    
    def calculate_cluster_stats(self, 
                              fingerprints: List[SessionFingerprint], 
                              cluster_labels: np.ndarray,
                              feature_matrix: np.ndarray) -> Dict[int, ClusterStats]:
        """Calculate comprehensive statistics for each cluster"""
        cluster_stats = {}
        feature_names = fingerprints[0].feature_names
        
        for cluster_id in range(self.clustering_config.k_clusters):
            # Get sessions in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_fingerprints = [fp for i, fp in enumerate(fingerprints) if cluster_mask[i]]
            cluster_features = feature_matrix[cluster_mask]
            
            if len(cluster_fingerprints) == 0:
                continue
            
            # Extract feature indices
            range_idx = None
            volatility_idx = None
            for i, name in enumerate(feature_names):
                if "normalized_range" in name:
                    range_idx = i
                elif "volatility_proxy" in name:
                    volatility_idx = i
            
            # Basic statistics
            centroid = self.kmeans.cluster_centers_[cluster_id]
            n_sessions = len(cluster_fingerprints)
            session_ids = [fp.session_id for fp in cluster_fingerprints]
            
            # Range characteristics
            if range_idx is not None:
                range_values = cluster_features[:, range_idx]
                normalized_range_p50 = np.percentile(range_values, 50)
            else:
                normalized_range_p50 = 0.0
            
            # Volatility characteristics
            if volatility_idx is not None:
                volatility_values = cluster_features[:, volatility_idx]
                volatility_proxy_mean = np.mean(volatility_values)
                
                # Classify volatility
                if volatility_proxy_mean < -0.5:
                    volatility_class = "low"
                elif volatility_proxy_mean > 0.5:
                    volatility_class = "high"
                else:
                    volatility_class = "medium"
            else:
                volatility_proxy_mean = 0.0
                volatility_class = "medium"
            
            # Semantic phase analysis
            semantic_phase_rates = []
            for i, name in enumerate(feature_names):
                if name.startswith("rate_"):
                    phase_name = name.replace("rate_", "")
                    mean_rate = np.mean(cluster_features[:, i])
                    semantic_phase_rates.append((phase_name, mean_rate))
            
            top_semantic_phases = sorted(semantic_phase_rates, key=lambda x: x[1], reverse=True)[:3]
            
            # HTF regime analysis
            htf_regime_scores = []
            for regime in [0, 1, 2]:
                regime_col = f"htf_regime_{regime}"
                if regime_col in feature_names:
                    idx = feature_names.index(regime_col)
                    mean_score = np.mean(cluster_features[:, idx])
                    htf_regime_scores.append((regime, mean_score))
            
            htf_regime_dominant = max(htf_regime_scores, key=lambda x: x[1])[0] if htf_regime_scores else 1
            
            # Temporal characteristics
            avg_events_per_session = np.mean([fp.n_events for fp in cluster_fingerprints])
            
            # Session duration (placeholder - would need actual duration data)
            avg_session_duration = 6.5  # Default session hours
            
            # Temporal consistency (using event spacing regularity if available)
            temporal_consistency_score = 0.8  # Placeholder
            for i, name in enumerate(feature_names):
                if "event_spacing_regularity" in name:
                    temporal_consistency_score = np.mean(cluster_features[:, i])
                    break
            
            # Session type distribution
            session_types = [fp.session_type for fp in cluster_fingerprints]
            session_type_distribution = dict(Counter(session_types))
            
            # Date range
            session_dates = [fp.session_date for fp in cluster_fingerprints]
            date_range = (min(session_dates), max(session_dates))
            
            cluster_stats[cluster_id] = ClusterStats(
                cluster_id=cluster_id,
                n_sessions=n_sessions,
                session_ids=session_ids,
                centroid=centroid,
                normalized_range_p50=normalized_range_p50,
                volatility_proxy_mean=volatility_proxy_mean,
                volatility_class=volatility_class,
                top_semantic_phases=top_semantic_phases,
                htf_regime_dominant=htf_regime_dominant,
                avg_session_duration=avg_session_duration,
                avg_events_per_session=avg_events_per_session,
                temporal_consistency_score=temporal_consistency_score,
                session_type_distribution=session_type_distribution,
                date_range=date_range
            )
        
        self.logger.info(f"Calculated statistics for {len(cluster_stats)} clusters")
        return cluster_stats
    
    def calculate_cluster_separation(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate cluster separation score"""
        centroids = self.kmeans.cluster_centers_
        
        # Inter-cluster distances
        inter_distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                inter_distances.append(dist)
        
        # Intra-cluster distances (average distance to centroid)
        intra_distances = []
        for cluster_id in range(self.clustering_config.k_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_points = feature_matrix[cluster_mask]
                centroid = centroids[cluster_id]
                distances = [np.linalg.norm(point - centroid) for point in cluster_points]
                intra_distances.extend(distances)
        
        # Separation score: ratio of inter-cluster to intra-cluster distances
        if len(intra_distances) > 0 and len(inter_distances) > 0:
            separation_score = np.mean(inter_distances) / np.mean(intra_distances)
        else:
            separation_score = 0.0
            
        return separation_score
    
    def build_metadata(self, 
                      session_files: List[Path],
                      fingerprints: List[SessionFingerprint],
                      skipped_reasons: Dict[str, int],
                      feature_matrix: np.ndarray,
                      cluster_labels: np.ndarray) -> ClusteringMetadata:
        """Build comprehensive clustering metadata"""
        
        # Processing summary
        total_discovered = len(session_files)
        sessions_processed = len(fingerprints)
        sessions_skipped = sum(skipped_reasons.values())
        coverage_percent = (sessions_processed / total_discovered) * 100 if total_discovered > 0 else 0
        
        # Data characteristics
        session_dates = [fp.session_date for fp in fingerprints]
        session_types = list(set(fp.session_type for fp in fingerprints))
        date_range = (min(session_dates), max(session_dates)) if session_dates else ("", "")
        
        # Feature schema
        feature_names = fingerprints[0].feature_names if fingerprints else []
        feature_dimensions = len(feature_names)
        
        # Performance metrics
        silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
        separation_score = self.calculate_cluster_separation(feature_matrix, cluster_labels)
        
        metadata = ClusteringMetadata(
            total_sessions_discovered=total_discovered,
            sessions_processed=sessions_processed,
            sessions_skipped=sessions_skipped,
            skipped_reasons=skipped_reasons,
            coverage_percent=coverage_percent,
            timeframe="M5",  # Default timeframe
            timezone="US/Eastern",  # Default timezone
            date_range=date_range,
            session_types=session_types,
            feature_schema_version="1.0.2",
            feature_names=feature_names,
            feature_dimensions=feature_dimensions,
            scaler_type=self.fingerprint_config.scaler_type,
            k_clusters=self.clustering_config.k_clusters,
            distance_metric=self.clustering_config.distance_metric,
            random_state=self.clustering_config.random_state,
            inertia=self.kmeans.inertia_,
            silhouette_score=silhouette_avg,
            cluster_separation_score=separation_score,
            created_at=datetime.now().isoformat(),
            version="1.0.2"
        )
        
        return metadata
    
    def save_library(self, output_dir: Path, 
                    fingerprints: List[SessionFingerprint],
                    cluster_stats: Dict[int, ClusterStats],
                    metadata: ClusteringMetadata):
        """Save complete clustering library to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save k-means model (centroids + parameters)
        kmeans_path = output_dir / "kmeans_model.pkl"
        with open(kmeans_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        self.logger.info(f"Saved k-means model: {kmeans_path}")
        
        # Save fitted scaler with hash calculation
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.extractor.scaler, f)
        
        # Calculate scaler hash for integrity validation
        with open(scaler_path, 'rb') as f:
            scaler_data = f.read()
            scaler_hash = hashlib.sha256(scaler_data).hexdigest()
        
        self.logger.info(f"Saved scaler: {scaler_path} (hash: {scaler_hash[:16]}...)")
        
        # Save cluster statistics
        cluster_stats_path = output_dir / "cluster_stats.json"
        # Convert ClusterStats to serializable format
        serializable_stats = {}
        for cluster_id, stats in cluster_stats.items():
            stats_dict = asdict(stats)
            # Convert numpy arrays and float32 to JSON serializable types
            stats_dict['centroid'] = stats.centroid.tolist()
            
            # Convert numpy float32 values to regular floats
            for key, value in stats_dict.items():
                if isinstance(value, np.floating):
                    stats_dict[key] = float(value)
                elif isinstance(value, np.integer):
                    stats_dict[key] = int(value)
                elif isinstance(value, list) and len(value) > 0:
                    # Handle lists that might contain numpy types
                    if isinstance(value[0], tuple):
                        # Handle top_semantic_phases list of tuples
                        stats_dict[key] = [(str(item[0]), float(item[1])) for item in value]
                    else:
                        stats_dict[key] = [float(item) if isinstance(item, np.floating) else item for item in value]
            
            serializable_stats[str(cluster_id)] = stats_dict
            
        with open(cluster_stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        self.logger.info(f"Saved cluster statistics: {cluster_stats_path}")
        
        # Save metadata with scaler hash for integrity validation
        metadata_path = output_dir / "metadata.json"
        metadata_dict = asdict(metadata)
        
        # Add scaler hash for integrity validation
        metadata_dict["scaler_hash"] = scaler_hash
        
        # Convert numpy types to JSON serializable types
        for key, value in metadata_dict.items():
            if isinstance(value, np.floating):
                metadata_dict[key] = float(value)
            elif isinstance(value, np.integer):
                metadata_dict[key] = int(value)
                
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        self.logger.info(f"Saved metadata: {metadata_path}")
        
        # Save fingerprints DataFrame for analysis
        fingerprints_df = self.extractor.fingerprints_to_dataframe(fingerprints)
        fingerprints_path = output_dir / "session_fingerprints.parquet"
        fingerprints_df.to_parquet(fingerprints_path, index=False)
        self.logger.info(f"Saved fingerprints: {fingerprints_path}")
        
        self.logger.info(f"Clustering library saved to: {output_dir}")
    
    def build_offline_library(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Build complete offline session clustering library"""
        if output_dir is None:
            output_dir = Path("models/session_fingerprints/v1.0.2")
        
        self.logger.info("Starting offline library build...")
        
        # 1. Enumerate historical sessions
        session_files = self.enumerate_historical_sessions()
        
        # 2. Extract fingerprints
        fingerprints, skipped_reasons = self.extract_all_fingerprints(session_files)
        
        if len(fingerprints) < self.clustering_config.k_clusters:
            raise ValueError(f"Insufficient fingerprints ({len(fingerprints)}) for clustering")
        
        # 3. Fit scaler on all fingerprints
        self.extractor.fit_scaler(fingerprints)
        scaled_fingerprints = self.extractor.transform_fingerprints(fingerprints)
        
        # 4. Fit clustering
        feature_matrix, cluster_labels = self.fit_clustering(scaled_fingerprints)
        
        # 5. Calculate cluster statistics
        self.cluster_stats = self.calculate_cluster_stats(scaled_fingerprints, cluster_labels, feature_matrix)
        
        # 6. Build metadata
        self.metadata = self.build_metadata(session_files, fingerprints, skipped_reasons, 
                                          feature_matrix, cluster_labels)
        
        # 7. Save library
        self.save_library(output_dir, scaled_fingerprints, self.cluster_stats, self.metadata)
        
        # 8. Return summary
        summary = {
            "sessions_discovered": len(session_files),
            "sessions_processed": len(fingerprints),
            "coverage_percent": self.metadata.coverage_percent,
            "k_clusters": self.clustering_config.k_clusters,
            "silhouette_score": self.metadata.silhouette_score,
            "cluster_separation": self.metadata.cluster_separation_score,
            "output_directory": str(output_dir)
        }
        
        self.logger.info("Offline library build complete!")
        return summary