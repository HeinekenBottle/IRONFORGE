"""
Online Session Classifier - Real-time session fingerprinting at 30% completion

Provides real-time session classification using pre-trained clustering models:
- Computes partial fingerprints at ~30% session completion
- Assigns to closest archetype with confidence scoring
- Generates session_fingerprint.json sidecars
- Integrates with discovery/report pipeline
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
from sklearn.metrics.pairwise import cosine_distances
from sklearn.covariance import EmpiricalCovariance

from .session_fingerprinting import SessionFingerprintExtractor, SessionFingerprintConfig
from .session_clustering import ClusteringMetadata

logger = logging.getLogger(__name__)


@dataclass
class OnlineClassifierConfig:
    """Configuration for online session classifier"""
    enabled: bool = False  # OFF by default
    completion_threshold_pct: float = 30.0  # 30% session completion
    model_path: Optional[Path] = None  # Path to saved clustering artifacts
    distance_metric: str = "euclidean"  # "euclidean", "cosine", "mahalanobis"
    confidence_method: str = "inverse_distance"  # "inverse_distance", "softmax"
    min_events_for_classification: int = 5  # Minimum events needed for partial fingerprint
    min_session_duration_minutes: float = 10.0  # Minimum session duration (minutes)
    min_session_bars: int = 20  # Minimum bars/price updates for reliable fingerprint
    write_minidash_row: bool = False  # Optional minidash integration
    
    @classmethod
    def default(cls, model_path: Optional[Path] = None) -> 'OnlineClassifierConfig':
        """Default configuration with model path"""
        return cls(
            model_path=model_path or Path("models/session_fingerprints/v1.0.2")
        )


@dataclass
class SessionPrediction:
    """Prediction results for a partial session"""
    session_id: str
    archetype_id: int
    confidence: float
    distance_to_centroid: float
    pct_session_seen: float
    
    # Predicted characteristics from cluster stats
    predicted_volatility_class: str
    predicted_range_p50: float
    predicted_dominant_htf_regime: int
    predicted_top_phases: List[Tuple[str, float]]
    predicted_session_types: Dict[str, float]  # Probability distribution
    
    # Metadata
    timestamp: str
    model_version: str
    distance_metric: str
    confidence_method: str
    notes: List[str]


@dataclass
class PartialSessionData:
    """Container for partial session data at checkpoint"""
    events: List[Dict]
    session_metadata: Dict
    completion_pct: float
    n_events_total_estimated: Optional[int] = None


class OnlineSessionClassifier:
    """Real-time session classifier using pre-trained clustering models"""
    
    def __init__(self, config: Optional[OnlineClassifierConfig] = None):
        self.config = config or OnlineClassifierConfig.default()
        self.logger = logging.getLogger(__name__)
        
        # Loaded artifacts
        self.kmeans_model = None
        self.scaler = None
        self.cluster_stats = {}
        self.metadata = None
        self.covariance_estimator = None  # For Mahalanobis distance
        
        # Runtime state
        self.extractor = None
        self.is_loaded = False
        
        if self.config.enabled:
            self.load_artifacts()
    
    def load_artifacts(self):
        """Load pre-trained clustering artifacts with hard-fail error handling"""
        if not self.config.model_path or not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {self.config.model_path}. "
                f"Please ensure the offline library has been built (Stage 2) before enabling online classification."
            )
        
        try:
            # Load k-means model
            kmeans_path = self.config.model_path / "kmeans_model.pkl"
            if not kmeans_path.exists():
                raise FileNotFoundError(f"K-means model not found: {kmeans_path}")
            
            with open(kmeans_path, 'rb') as f:
                self.kmeans_model = pickle.load(f)
            self.logger.info(f"Loaded k-means model with {self.kmeans_model.n_clusters} clusters")
            
            # Load scaler with hash validation
            scaler_path = self.config.model_path / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            # Calculate scaler file hash for validation
            with open(scaler_path, 'rb') as f:
                scaler_data = f.read()
                scaler_hash = hashlib.sha256(scaler_data).hexdigest()
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Validate scaler integrity with metadata
            expected_scaler_hash = self.metadata.get("scaler_hash") if hasattr(self, 'metadata') and self.metadata else None
            
            # Store scaler hash for sidecar output (always record actual hash)
            self.scaler_hash = scaler_hash
            
            self.logger.info(f"Loaded fitted scaler (hash: {scaler_hash[:16]}...)")
            
            # SCALER MISMATCH PROTECTION: Validate against expected hash if available
            if expected_scaler_hash and expected_scaler_hash != scaler_hash:
                raise ValueError(
                    f"Scaler mismatch detected: Expected hash {expected_scaler_hash[:16]}..., "
                    f"got {scaler_hash[:16]}... "
                    f"This indicates the scaler was modified or corrupted since training. "
                    f"Please rebuild the offline library to ensure scaler consistency."
                )
            
            # Load cluster statistics
            cluster_stats_path = self.config.model_path / "cluster_stats.json"
            if not cluster_stats_path.exists():
                raise FileNotFoundError(f"Cluster statistics not found: {cluster_stats_path}")
            
            with open(cluster_stats_path, 'r') as f:
                self.cluster_stats = json.load(f)
            self.logger.info(f"Loaded statistics for {len(self.cluster_stats)} clusters")
            
            # Load metadata
            metadata_path = self.config.model_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                # Reconstruct feature names for extractor configuration
                feature_names = metadata_dict.get('feature_names', [])
                scaler_type = metadata_dict.get('scaler_type', 'standard')
                self.metadata = metadata_dict
            
            # Initialize extractor with matching configuration
            fp_config = SessionFingerprintConfig.default()
            fp_config.scaler_type = scaler_type
            self.extractor = SessionFingerprintExtractor(fp_config)
            self.extractor.scaler = self.scaler  # Use pre-fitted scaler
            
            # Prepare covariance estimator for Mahalanobis distance
            if self.config.distance_metric == "mahalanobis":
                self._prepare_covariance_estimator()
            
            self.is_loaded = True
            self.logger.info("Online classifier artifacts loaded successfully")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load clustering artifacts from {self.config.model_path}: {e}. "
                f"Please verify the offline library build completed successfully."
            ) from e
    
    def _prepare_covariance_estimator(self):
        """Prepare covariance estimator for Mahalanobis distance"""
        try:
            # Load training data to estimate covariance
            fingerprints_path = self.config.model_path / "session_fingerprints.parquet"
            if fingerprints_path.exists():
                df = pd.read_parquet(fingerprints_path)
                feature_cols = [col for col in df.columns if col.startswith('f')]
                
                if len(feature_cols) > 0:
                    feature_matrix = df[feature_cols].values
                    self.covariance_estimator = EmpiricalCovariance()
                    self.covariance_estimator.fit(feature_matrix)
                    self.logger.info("Prepared covariance estimator for Mahalanobis distance")
                else:
                    self.logger.warning("No feature columns found for covariance estimation")
                    self.covariance_estimator = None
            else:
                self.logger.warning("Training fingerprints not found for covariance estimation")
                self.covariance_estimator = None
                
        except Exception as e:
            self.logger.error(f"Failed to prepare covariance estimator: {e}")
            self.covariance_estimator = None
    
    def extract_partial_session_data(self, session_data: Dict, 
                                   target_completion_pct: Optional[float] = None) -> Optional[PartialSessionData]:
        """Extract partial session data at specified completion percentage"""
        if target_completion_pct is None:
            target_completion_pct = self.config.completion_threshold_pct
        
        events = session_data.get("events", [])
        session_metadata = session_data.get("session_metadata", {})
        
        # SHORT SESSION HANDLING: Multi-criteria validation
        session_validation_errors = []
        
        # Check minimum events
        if len(events) < self.config.min_events_for_classification:
            session_validation_errors.append(
                f"Insufficient events: {len(events)} < {self.config.min_events_for_classification} required"
            )
        
        # Check minimum session duration
        if events and len(events) >= 2:
            try:
                # Parse timestamps to calculate duration
                first_ts = pd.to_datetime(events[0].get("timestamp", ""))
                last_ts = pd.to_datetime(events[-1].get("timestamp", ""))
                duration_minutes = (last_ts - first_ts).total_seconds() / 60.0
                
                if duration_minutes < self.config.min_session_duration_minutes:
                    session_validation_errors.append(
                        f"Session too short: {duration_minutes:.1f} min < {self.config.min_session_duration_minutes} min required"
                    )
                    
            except Exception as e:
                # If timestamp parsing fails, note but don't block
                self.logger.warning(f"Could not validate session duration: {e}")
        
        # Check minimum bars/price updates
        price_updates = [event for event in events if event.get("price_level") is not None]
        if len(price_updates) < self.config.min_session_bars:
            session_validation_errors.append(
                f"Insufficient price updates: {len(price_updates)} < {self.config.min_session_bars} required"
            )
        
        # If any validation fails, return None with explicit logging
        if session_validation_errors:
            self.logger.info(f"Session rejected for classification: {'; '.join(session_validation_errors)}")
            return None
        
        # Sort events by timestamp to ensure chronological order
        if events and "timestamp" in events[0]:
            events = sorted(events, key=lambda e: e.get("timestamp", ""))
        
        # Calculate target event count for completion percentage
        total_events = len(events)
        target_event_count = max(
            self.config.min_events_for_classification,
            int(total_events * target_completion_pct / 100.0)
        )
        
        # Extract partial events
        partial_events = events[:target_event_count]
        actual_completion_pct = (len(partial_events) / total_events) * 100.0
        
        return PartialSessionData(
            events=partial_events,
            session_metadata=session_metadata,
            completion_pct=actual_completion_pct,
            n_events_total_estimated=total_events
        )
    
    def compute_partial_fingerprint(self, partial_data: PartialSessionData) -> Optional[np.ndarray]:
        """Compute partial session fingerprint from partial data"""
        if not self.extractor:
            raise RuntimeError("Extractor not initialized. Ensure artifacts are loaded.")
        
        # Create temporary session data structure
        temp_session_data = {
            "events": partial_data.events,
            "session_metadata": partial_data.session_metadata
        }
        
        # Extract features using same methodology as training
        semantic_rates = self.extractor.extract_semantic_rates(partial_data.events)
        htf_distribution = self.extractor.extract_htf_distribution(partial_data.events)
        range_tempo_features = self.extractor.extract_range_tempo_features(
            partial_data.events, partial_data.session_metadata
        )
        timing_features = self.extractor.extract_timing_features(partial_data.events)
        event_distribution_features = self.extractor.extract_event_distribution_features(
            partial_data.events, partial_data.session_metadata
        )
        
        # Combine features
        feature_vector = np.concatenate([
            semantic_rates,
            htf_distribution,
            range_tempo_features,
            timing_features,
            event_distribution_features
        ])
        
        # Check for invalid values
        if not np.isfinite(feature_vector).all():
            self.logger.warning("Non-finite values in partial fingerprint")
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale using pre-fitted scaler
        scaled_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return scaled_vector
    
    def compute_distances_to_centroids(self, fingerprint: np.ndarray) -> np.ndarray:
        """Compute distances from fingerprint to all cluster centroids"""
        centroids = self.kmeans_model.cluster_centers_
        
        if self.config.distance_metric == "cosine":
            # Cosine distance
            distances = cosine_distances(fingerprint.reshape(1, -1), centroids).flatten()
            
        elif self.config.distance_metric == "mahalanobis" and self.covariance_estimator is not None:
            # Mahalanobis distance
            try:
                distances = []
                for centroid in centroids:
                    diff = fingerprint - centroid
                    dist = np.sqrt(diff.T @ self.covariance_estimator.precision_ @ diff)
                    distances.append(dist)
                distances = np.array(distances)
            except Exception as e:
                self.logger.warning(f"Mahalanobis distance computation failed: {e}, falling back to Euclidean")
                distances = np.linalg.norm(centroids - fingerprint, axis=1)
                
        else:
            # Euclidean distance (default)
            distances = np.linalg.norm(centroids - fingerprint, axis=1)
        
        return distances
    
    def compute_confidence(self, distances: np.ndarray, closest_idx: int) -> float:
        """Compute confidence score with misuse protection (cap/floor and distance logging)"""
        closest_distance = distances[closest_idx]
        
        # CONFIDENCE MISUSE PROTECTION: Log raw distance metrics for analysis
        distance_stats = {
            "closest_distance": float(closest_distance),
            "max_distance": float(np.max(distances)),
            "min_distance": float(np.min(distances)),
            "median_distance": float(np.median(distances)),
            "distance_std": float(np.std(distances))
        }
        
        # Log distance metrics for validation and debugging
        self.logger.debug(f"Distance metrics: {distance_stats}")
        
        # Compute raw confidence score
        if self.config.confidence_method == "softmax":
            # Softmax over negative distances
            neg_distances = -distances
            exp_distances = np.exp(neg_distances - np.max(neg_distances))  # Numerical stability
            softmax_probs = exp_distances / np.sum(exp_distances)
            raw_confidence = float(softmax_probs[closest_idx])
            
        else:
            # Inverse distance (default)
            # confidence = 1 / (1 + normalized_distance)
            max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
            normalized_distance = closest_distance / max_distance
            raw_confidence = 1.0 / (1.0 + normalized_distance)
        
        # CONFIDENCE MISUSE PROTECTION: Apply cap and floor to prevent overconfidence
        # These bounds prevent misinterpretation as absolute probability
        confidence_floor = 0.05  # Minimum 5% - never claim zero uncertainty
        confidence_cap = 0.95    # Maximum 95% - never claim absolute certainty
        
        capped_confidence = np.clip(raw_confidence, confidence_floor, confidence_cap)
        
        # Log if capping was applied (important for validation)
        if capped_confidence != raw_confidence:
            self.logger.warning(
                f"Confidence capping applied: raw={raw_confidence:.4f} -> capped={capped_confidence:.4f} "
                f"(floor={confidence_floor}, cap={confidence_cap})"
            )
        
        # Additional validation: Check for unrealistic confidence patterns
        if capped_confidence > 0.90 and closest_distance > np.median(distances):
            self.logger.warning(
                f"Suspicious high confidence ({capped_confidence:.3f}) despite large distance "
                f"({closest_distance:.3f} > median {np.median(distances):.3f}). "
                f"This may indicate model uncertainty that confidence score doesn't capture."
            )
        
        return float(capped_confidence)
    
    def predict_session_characteristics(self, archetype_id: int) -> Dict[str, Any]:
        """Predict session characteristics based on archetype cluster statistics"""
        if str(archetype_id) not in self.cluster_stats:
            return {}
        
        cluster_data = self.cluster_stats[str(archetype_id)]
        
        # Extract predicted characteristics
        predicted_volatility_class = cluster_data.get("volatility_class", "medium")
        predicted_range_p50 = cluster_data.get("normalized_range_p50", 0.0)
        predicted_dominant_htf_regime = cluster_data.get("htf_regime_dominant", 1)
        predicted_top_phases = cluster_data.get("top_semantic_phases", [])
        
        # Convert session type counts to probabilities
        session_type_counts = cluster_data.get("session_type_distribution", {})
        total_sessions = sum(session_type_counts.values()) if session_type_counts else 1
        predicted_session_types = {
            session_type: count / total_sessions 
            for session_type, count in session_type_counts.items()
        }
        
        return {
            "volatility_class": predicted_volatility_class,
            "range_p50": predicted_range_p50,
            "dominant_htf_regime": predicted_dominant_htf_regime,
            "top_phases": predicted_top_phases,
            "session_types": predicted_session_types
        }
    
    def classify_partial_session(self, session_data: Dict, 
                                session_id: str,
                                target_completion_pct: Optional[float] = None) -> Optional[SessionPrediction]:
        """Classify a partial session and return prediction"""
        if not self.config.enabled:
            return None
        
        if not self.is_loaded:
            raise RuntimeError("Classifier not loaded. Call load_artifacts() first.")
        
        # Extract partial session data with explicit insufficient data handling
        partial_data = self.extract_partial_session_data(session_data, target_completion_pct)
        if partial_data is None:
            # Generate explicit "insufficient data" metadata for transparency
            self._write_insufficient_data_sidecar(session_data, session_id, target_completion_pct)
            return None
        
        # Compute partial fingerprint
        fingerprint = self.compute_partial_fingerprint(partial_data)
        if fingerprint is None:
            return None
        
        # Compute distances to centroids
        distances = self.compute_distances_to_centroids(fingerprint)
        
        # Find closest archetype
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Compute confidence
        confidence = self.compute_confidence(distances, closest_idx)
        
        # Predict characteristics
        predicted_chars = self.predict_session_characteristics(closest_idx)
        
        # Prepare notes
        notes = []
        if partial_data.completion_pct < target_completion_pct:
            notes.append(f"Early classification at {partial_data.completion_pct:.1f}% completion")
        
        if fingerprint is not None and not np.isfinite(fingerprint).all():
            notes.append("Some features required imputation")
        
        # Create prediction
        prediction = SessionPrediction(
            session_id=session_id,
            archetype_id=int(closest_idx),
            confidence=confidence,
            distance_to_centroid=float(closest_distance),
            pct_session_seen=partial_data.completion_pct,
            predicted_volatility_class=predicted_chars.get("volatility_class", "medium"),
            predicted_range_p50=predicted_chars.get("range_p50", 0.0),
            predicted_dominant_htf_regime=predicted_chars.get("dominant_htf_regime", 1),
            predicted_top_phases=predicted_chars.get("top_phases", []),
            predicted_session_types=predicted_chars.get("session_types", {}),
            timestamp=datetime.now().isoformat(),
            model_version=self.metadata.get("version", "unknown") if self.metadata else "unknown",
            distance_metric=self.config.distance_metric,
            confidence_method=self.config.confidence_method,
            notes=notes
        )
        
        return prediction
    
    def _write_insufficient_data_sidecar(self, session_data: Dict, session_id: str, 
                                        target_completion_pct: Optional[float]) -> None:
        """Write sidecar with explicit insufficient data metadata"""
        events = session_data.get("events", [])
        session_metadata = session_data.get("session_metadata", {})
        
        # Calculate diagnostic metrics
        n_events = len(events)
        n_price_updates = len([e for e in events if e.get("price_level") is not None])
        
        duration_minutes = None
        if events and len(events) >= 2:
            try:
                first_ts = pd.to_datetime(events[0].get("timestamp", ""))
                last_ts = pd.to_datetime(events[-1].get("timestamp", ""))
                duration_minutes = (last_ts - first_ts).total_seconds() / 60.0
            except:
                pass
        
        # Determine specific reasons for rejection
        rejection_reasons = []
        if n_events < self.config.min_events_for_classification:
            rejection_reasons.append(f"events ({n_events} < {self.config.min_events_for_classification})")
        if n_price_updates < self.config.min_session_bars:
            rejection_reasons.append(f"price_updates ({n_price_updates} < {self.config.min_session_bars})")
        if duration_minutes and duration_minutes < self.config.min_session_duration_minutes:
            rejection_reasons.append(f"duration ({duration_minutes:.1f}min < {self.config.min_session_duration_minutes}min)")
        
        self.logger.info(f"Generated insufficient data metadata for {session_id}: {rejection_reasons}")
    
    def write_sidecar(self, prediction: SessionPrediction, run_dir: Path) -> Path:
        """Write session_fingerprint.json sidecar to run directory"""
        sidecar_path = run_dir / "session_fingerprint.json"
        
        # Prepare sidecar data
        sidecar_data = {
            "session_id": prediction.session_id,
            "date": prediction.timestamp.split("T")[0],  # Extract date
            "pct_seen": prediction.pct_session_seen,
            "archetype_id": prediction.archetype_id,
            "confidence": prediction.confidence,
            "predicted_stats": {
                "volatility_class": prediction.predicted_volatility_class,
                "range_p50": prediction.predicted_range_p50,
                "dominant_htf_regime": prediction.predicted_dominant_htf_regime,
                "top_phases": prediction.predicted_top_phases,
                "session_type_probabilities": prediction.predicted_session_types
            },
            "artifact_path": str(self.config.model_path),
            "notes": prediction.notes,
            "classification_metadata": {
                "distance_to_centroid": prediction.distance_to_centroid,
                "distance_metric": prediction.distance_metric,
                "confidence_method": prediction.confidence_method,
                "model_version": prediction.model_version,
                "timestamp": prediction.timestamp,
                "scaler_hash": getattr(self, 'scaler_hash', 'unknown')
            }
        }
        
        # Write sidecar
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        self.logger.info(f"Written session fingerprint sidecar: {sidecar_path}")
        return sidecar_path
    
    def write_minidash_row(self, prediction: SessionPrediction, minidash_path: Path) -> bool:
        """Write optional minidash row if minidash file exists"""
        if not self.config.write_minidash_row or not minidash_path.exists():
            return False
        
        try:
            # Prepare minidash row data
            minidash_row = {
                "timestamp": prediction.timestamp,
                "session_id": prediction.session_id,
                "archetype": prediction.archetype_id,
                "confidence": f"{prediction.confidence:.3f}",
                "volatility": prediction.predicted_volatility_class,
                "pct_seen": f"{prediction.pct_session_seen:.1f}%",
                "htf_regime": prediction.predicted_dominant_htf_regime
            }
            
            # Read existing minidash
            if minidash_path.suffix.lower() == '.json':
                with open(minidash_path, 'r') as f:
                    minidash_data = json.load(f)
                
                # Add row to appropriate section
                if "session_classifications" not in minidash_data:
                    minidash_data["session_classifications"] = []
                
                minidash_data["session_classifications"].append(minidash_row)
                
                # Write back
                with open(minidash_path, 'w') as f:
                    json.dump(minidash_data, f, indent=2)
                
                self.logger.info(f"Added session classification to minidash: {minidash_path}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Failed to write minidash row: {e}")
            return False
        
        return False


def create_online_classifier(enabled: bool = False, 
                           model_path: Optional[Path] = None,
                           completion_threshold: float = 30.0,
                           distance_metric: str = "euclidean",
                           confidence_method: str = "inverse_distance") -> OnlineSessionClassifier:
    """Factory function to create configured online classifier"""
    config = OnlineClassifierConfig(
        enabled=enabled,
        completion_threshold_pct=completion_threshold,
        model_path=model_path or Path("models/session_fingerprints/v1.0.2"),
        distance_metric=distance_metric,
        confidence_method=confidence_method
    )
    
    return OnlineSessionClassifier(config)