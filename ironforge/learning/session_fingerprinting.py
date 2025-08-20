"""
Session Fingerprinting - Per-session feature vector extraction

Extracts session-level fingerprints from enhanced session data:
- Semantic phase rates per 100 events (6 components)
- HTF regime distribution over {0,1,2} (3 components) 
- Range/tempo features (2-4 components)
- Timing summaries (2-4 components)

Target: 20-32 dimensional vectors with deterministic, repeatable values.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SessionFingerprintConfig:
    """Configuration for session fingerprinting"""
    semantic_event_types: List[str]
    min_events_threshold: int = 10
    scaler_type: str = "standard"  # "standard" or "robust"
    tempo_method: str = "std_diff"  # "std_diff" or "mad_based"
    
    @classmethod
    def default(cls) -> 'SessionFingerprintConfig':
        """Default configuration based on enhanced session adapter event types"""
        return cls(
            semantic_event_types=[
                "fvg_interaction",
                "expansion_phase", 
                "consolidation_phase",
                "retracement_event",
                "reversal_signal",
                "liquidity_sweep"
            ]
        )


@dataclass 
class SessionFingerprint:
    """Per-session fingerprint vector with metadata"""
    session_id: str
    feature_vector: np.ndarray
    feature_names: List[str]
    n_events: int
    session_date: str
    session_type: str
    data_quality: str
    htf_mode: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation"""
        result = {
            "session_id": self.session_id,
            "n_events": self.n_events,
            "session_date": self.session_date,
            "session_type": self.session_type,
            "data_quality": self.data_quality,
            "htf_mode": self.htf_mode
        }
        
        # Add feature values with names
        for i, (name, value) in enumerate(zip(self.feature_names, self.feature_vector)):
            result[f"f{i:02d}_{name}"] = value
            
        return result


class SessionFingerprintExtractor:
    """Extract session-level fingerprint vectors from enhanced session data"""
    
    def __init__(self, config: Optional[SessionFingerprintConfig] = None):
        self.config = config or SessionFingerprintConfig.default()
        self.scaler = None
        self.feature_names = self._define_feature_names()
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler
        if self.config.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
    
    def _define_feature_names(self) -> List[str]:
        """Define stable feature ordering and names"""
        names = []
        
        # Semantic phase rates per 100 events (6 features)
        for event_type in self.config.semantic_event_types:
            names.append(f"rate_{event_type}")
        
        # HTF regime distribution {0,1,2} (3 features) 
        names.extend(["htf_regime_0", "htf_regime_1", "htf_regime_2"])
        
        # Range/tempo features (8 features)
        names.extend([
            "normalized_range_vs_median",
            "volatility_proxy", 
            "price_momentum_std",
            "energy_density_mean",
            "price_amplitude_ratio",
            "session_range_efficiency", 
            "price_velocity_mean",
            "momentum_acceleration_std"
        ])
        
        # Timing summaries (8 features)
        names.extend([
            "barpos_m15_mean",
            "barpos_h1_mean", 
            "normalized_time_std",
            "event_spacing_regularity",
            "session_duration_ratio",
            "time_cluster_density",
            "event_burst_intensity",
            "temporal_momentum_consistency"
        ])
        
        # Event distribution features (5 features)
        names.extend([
            "event_density_per_hour",
            "session_completion_ratio",
            "price_action_complexity",
            "structural_coherence_score",
            "archaeological_significance_mean"
        ])
        
        return names
    
    def extract_semantic_rates(self, events: List[Dict]) -> np.ndarray:
        """Extract semantic phase rates per 100 events"""
        if not events:
            return np.zeros(len(self.config.semantic_event_types))
            
        # Count semantic events by type
        semantic_counts = Counter()
        
        for event in events:
            event_family = event.get("event_family", "")
            event_type = event.get("type", "")
            
            # Map event types to semantic categories
            if "fvg" in event_type.lower() or "fvg" in event_family.lower():
                semantic_counts["fvg_interaction"] += 1
            elif "expansion" in event_type.lower() or event.get("structural_role") == "expansion":
                semantic_counts["expansion_phase"] += 1
            elif "consolidation" in event_type.lower() or event.get("structural_role") == "consolidation":
                semantic_counts["consolidation_phase"] += 1
            elif "retracement" in event_type.lower():
                semantic_counts["retracement_event"] += 1
            elif "reversal" in event_type.lower():
                semantic_counts["reversal_signal"] += 1
            elif "liquidity" in event_type.lower() or "sweep" in event_type.lower():
                semantic_counts["liquidity_sweep"] += 1
        
        # Convert to rates per 100 events
        rates = []
        for event_type in self.config.semantic_event_types:
            rate = (semantic_counts[event_type] / len(events)) * 100
            rates.append(rate)
            
        return np.array(rates, dtype=np.float32)
    
    def extract_htf_distribution(self, events: List[Dict]) -> np.ndarray:
        """Extract HTF regime distribution over {0,1,2} with leakage protection"""
        if not events:
            return np.array([0.33, 0.33, 0.34])  # Default uniform distribution
            
        # HTF LEAKAGE PROTECTION: Only use allowed context
        # Ensure f50 distribution uses historical-only context, not future-looking signals
        allowed_htf_fields = ["htf_confluence", "htf_regime", "historical_htf_state"]
        forbidden_htf_fields = ["future_htf", "htf_forecast", "f50_forward", "htf_prediction"]
        
        htf_values = []
        leakage_detected = False
        
        for event in events:
            # Check for forbidden future-looking HTF fields (CRITICAL SAFETY)
            for forbidden_field in forbidden_htf_fields:
                if forbidden_field in event:
                    leakage_detected = True
                    self.logger.warning(f"HTF leakage detected: forbidden field '{forbidden_field}' in event data")
            
            # Use only allowed historical HTF context
            # Priority: htf_regime (if available) > htf_confluence > default
            if "htf_regime" in event:
                # Direct regime value (0, 1, 2) - most reliable
                regime_val = event["htf_regime"]
                if regime_val in [0, 1, 2]:
                    htf_values.append(regime_val / 2.0)  # Normalize to [0, 1] for consistency
                else:
                    htf_values.append(0.5)  # Default if invalid regime
            elif "htf_confluence" in event:
                # Historical confluence (baseline pathway)
                htf_confluence = event["htf_confluence"]
                # Validate confluence is in reasonable range
                if 0.0 <= htf_confluence <= 1.0:
                    htf_values.append(htf_confluence)
                else:
                    htf_values.append(0.5)  # Default if out of range
            else:
                # No HTF context available - use neutral default
                htf_values.append(0.5)
        
        if leakage_detected:
            raise ValueError(
                "HTF leakage protection triggered: Future-looking HTF signals detected. "
                "Session fingerprinting must use only historical HTF context to prevent lookahead bias."
            )
        
        if not htf_values:
            return np.array([0.33, 0.33, 0.34])
            
        # Convert continuous HTF values to discrete regimes {0,1,2}
        # Use conservative thresholds to separate baseline vs HTF pathways
        regimes = []
        for val in htf_values:
            if val < 0.33:
                regimes.append(0)  # Low HTF regime
            elif val < 0.67:
                regimes.append(1)  # Medium HTF regime  
            else:
                regimes.append(2)  # High HTF regime
        
        # Calculate distribution
        regime_counts = Counter(regimes)
        total = len(regimes)
        distribution = np.array([
            regime_counts[0] / total,
            regime_counts[1] / total, 
            regime_counts[2] / total
        ], dtype=np.float32)
        
        return distribution
    
    def extract_range_tempo_features(self, events: List[Dict], session_metadata: Dict) -> np.ndarray:
        """Extract range/tempo features"""
        if not events:
            return np.zeros(8)
            
        # Extract price levels
        prices = [event.get("price_level", 0) for event in events if event.get("price_level")]
        if not prices:
            return np.zeros(8)
            
        prices = np.array(prices)
        
        # 1. Normalized range vs 30-day median (placeholder - using session range)
        session_range = prices.max() - prices.min()
        # For now, normalize by the session range itself (TODO: implement 30-day median)
        normalized_range = session_range / np.median(prices) if np.median(prices) > 0 else 0
        
        # 2. Volatility proxy - std of 1st differences / median
        if self.config.tempo_method == "mad_based":
            # MAD-based proxy
            price_diffs = np.diff(prices) if len(prices) > 1 else np.array([0])
            mad = np.median(np.abs(price_diffs - np.median(price_diffs)))
            volatility_proxy = mad / np.median(prices) if np.median(prices) > 0 else 0
        else:
            # Standard deviation of 1st differences
            price_diffs = np.diff(prices) if len(prices) > 1 else np.array([0])
            volatility_proxy = np.std(price_diffs) / np.median(prices) if np.median(prices) > 0 else 0
        
        # 3. Price momentum standard deviation
        momentum_values = [event.get("price_momentum", 0) for event in events]
        momentum_std = np.std(momentum_values) if momentum_values else 0
        
        # 4. Energy density mean
        energy_values = [event.get("energy_density", 0) for event in events]
        energy_mean = np.mean(energy_values) if energy_values else 0
        
        # 5. Price amplitude ratio (high-low / median)
        price_amplitude_ratio = session_range / np.median(prices) if np.median(prices) > 0 else 0
        
        # 6. Session range efficiency (range / total price movement)
        total_movement = np.sum(np.abs(np.diff(prices))) if len(prices) > 1 else session_range
        range_efficiency = session_range / total_movement if total_movement > 0 else 1.0
        
        # 7. Price velocity mean (average price change per event)
        price_velocity_mean = np.mean(np.abs(np.diff(prices))) if len(prices) > 1 else 0
        
        # 8. Momentum acceleration std (2nd derivative of momentum)
        if len(momentum_values) > 2:
            momentum_diffs = np.diff(momentum_values)
            momentum_acceleration_std = np.std(np.diff(momentum_diffs)) if len(momentum_diffs) > 1 else 0
        else:
            momentum_acceleration_std = 0
        
        return np.array([
            normalized_range, volatility_proxy, momentum_std, energy_mean,
            price_amplitude_ratio, range_efficiency, price_velocity_mean, momentum_acceleration_std
        ], dtype=np.float32)
    
    def extract_timing_features(self, events: List[Dict]) -> np.ndarray:
        """Extract timing summary features"""
        if not events:
            return np.zeros(8)
            
        # Extract timing-related values
        normalized_times = [event.get("normalized_time", 0) for event in events]
        durations = [event.get("duration", 0) for event in events if event.get("duration")]
        
        # Placeholder for barpos features (not in current data structure)
        # In a real implementation, these would come from f47/f48 HTF features
        barpos_m15_values = [event.get("range_position", 0.5) for event in events]  # Proxy
        barpos_h1_values = [event.get("range_position", 0.5) for event in events]   # Proxy
        
        # 1-2. Barpos means for f47/f48 windows
        barpos_m15_mean = np.mean(barpos_m15_values) if barpos_m15_values else 0.5
        barpos_h1_mean = np.mean(barpos_h1_values) if barpos_h1_values else 0.5
        
        # 3. Normalized time standard deviation  
        normalized_time_std = np.std(normalized_times) if normalized_times else 0
        
        # 4. Event spacing regularity
        if len(normalized_times) > 1:
            time_diffs = np.diff(sorted(normalized_times))
            event_spacing_regularity = 1.0 - np.std(time_diffs) if len(time_diffs) > 0 else 1.0
        else:
            event_spacing_regularity = 1.0
        
        # 5. Session duration ratio (actual vs standard session)
        if durations:
            total_duration = sum(durations)
            # Assume standard session is 6.5 hours (23400 seconds)
            session_duration_ratio = total_duration / 23400.0
        else:
            session_duration_ratio = 1.0
        
        # 6. Time cluster density (events per time window)
        if len(normalized_times) > 1:
            time_windows = np.linspace(0, 1, 11)  # 10 windows
            time_counts = np.histogram(normalized_times, bins=time_windows)[0]
            time_cluster_density = np.std(time_counts) / np.mean(time_counts) if np.mean(time_counts) > 0 else 0
        else:
            time_cluster_density = 0
            
        # 7. Event burst intensity (max events in any 10% window)
        if len(normalized_times) > 1:
            max_burst = 0
            for start in np.arange(0, 0.91, 0.1):
                window_events = sum(1 for t in normalized_times if start <= t < start + 0.1)
                max_burst = max(max_burst, window_events)
            event_burst_intensity = max_burst / len(normalized_times)
        else:
            event_burst_intensity = 1.0
            
        # 8. Temporal momentum consistency 
        time_since_open = [event.get("time_since_session_open", 0) for event in events]
        if len(time_since_open) > 2:
            time_momentum = np.diff(time_since_open)
            temporal_momentum_consistency = 1.0 - (np.std(time_momentum) / np.mean(time_momentum)) if np.mean(time_momentum) > 0 else 1.0
        else:
            temporal_momentum_consistency = 1.0
            
        return np.array([
            barpos_m15_mean, barpos_h1_mean, normalized_time_std, event_spacing_regularity,
            session_duration_ratio, time_cluster_density, event_burst_intensity, temporal_momentum_consistency
        ], dtype=np.float32)
    
    def extract_event_distribution_features(self, events: List[Dict], session_metadata: Dict) -> np.ndarray:
        """Extract event distribution features"""
        if not events:
            return np.zeros(5)
            
        n_events = len(events)
        
        # 1. Event density per hour (events / session duration in hours)
        # Assume standard 6.5 hour session if no duration available
        session_duration_hours = 6.5  # Default
        if session_metadata.get("session_duration"):
            session_duration_hours = session_metadata["session_duration"] / 3600.0
        event_density_per_hour = n_events / session_duration_hours
        
        # 2. Session completion ratio (proxy using normalized time coverage)
        normalized_times = [event.get("normalized_time", 0) for event in events]
        if normalized_times:
            time_coverage = max(normalized_times) - min(normalized_times)
            session_completion_ratio = min(time_coverage, 1.0)
        else:
            session_completion_ratio = 1.0
            
        # 3. Price action complexity (variety of event types / magnitudes)
        magnitudes = [abs(event.get("magnitude", 0)) for event in events]
        if magnitudes:
            # Coefficient of variation as complexity measure
            price_action_complexity = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0
        else:
            price_action_complexity = 0
            
        # 4. Structural coherence score (consistency of structural roles)
        structural_roles = [event.get("structural_role", "unknown") for event in events]
        role_counts = Counter(structural_roles)
        if len(role_counts) > 1:
            # Entropy-based measure (lower entropy = more coherent)
            total = len(structural_roles)
            entropy = -sum((count/total) * np.log2(count/total) for count in role_counts.values())
            max_entropy = np.log2(len(role_counts))
            structural_coherence_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        else:
            structural_coherence_score = 1.0
            
        # 5. Archaeological significance mean
        archaeological_significance = [event.get("archaeological_significance", 0) for event in events]
        archaeological_significance_mean = np.mean(archaeological_significance) if archaeological_significance else 0
        
        return np.array([
            event_density_per_hour, session_completion_ratio, price_action_complexity,
            structural_coherence_score, archaeological_significance_mean
        ], dtype=np.float32)
    
    def extract_session_fingerprint(self, session_file: Path) -> Optional[SessionFingerprint]:
        """Extract fingerprint from single session file"""
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            events = session_data.get("events", [])
            session_metadata = session_data.get("session_metadata", {})
            
            # Check minimum events threshold
            if len(events) < self.config.min_events_threshold:
                self.logger.warning(f"Session {session_file.stem} has only {len(events)} events (min: {self.config.min_events_threshold})")
                return None
            
            # Extract feature components
            semantic_rates = self.extract_semantic_rates(events)
            htf_distribution = self.extract_htf_distribution(events)
            range_tempo_features = self.extract_range_tempo_features(events, session_metadata)
            timing_features = self.extract_timing_features(events)
            event_distribution_features = self.extract_event_distribution_features(events, session_metadata)
            
            # Combine into single vector
            feature_vector = np.concatenate([
                semantic_rates,
                htf_distribution,
                range_tempo_features,
                timing_features,
                event_distribution_features
            ])
            
            # Check for NaN/Inf values
            if not np.isfinite(feature_vector).all():
                self.logger.warning(f"Non-finite values in {session_file.stem}: {feature_vector}")
                # Replace with zeros
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract session metadata
            session_id = session_file.stem.replace("adapted_enhanced_rel_", "")
            session_date = session_metadata.get("session_date", "unknown")
            session_type = session_metadata.get("session_type", "unknown")
            
            return SessionFingerprint(
                session_id=session_id,
                feature_vector=feature_vector,
                feature_names=self.feature_names,
                n_events=len(events),
                session_date=session_date,
                session_type=session_type,
                data_quality="good" if len(events) >= self.config.min_events_threshold else "low",
                htf_mode=session_metadata.get("htf_mode")
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting fingerprint from {session_file}: {e}")
            return None
    
    def extract_batch_fingerprints(self, session_files: List[Path]) -> List[SessionFingerprint]:
        """Extract fingerprints from multiple session files"""
        fingerprints = []
        
        for session_file in session_files:
            fingerprint = self.extract_session_fingerprint(session_file)
            if fingerprint is not None:
                fingerprints.append(fingerprint)
        
        self.logger.info(f"Extracted {len(fingerprints)} valid fingerprints from {len(session_files)} session files")
        return fingerprints
    
    def fit_scaler(self, fingerprints: List[SessionFingerprint]):
        """Fit scaler on training fingerprints (offline)"""
        if not fingerprints:
            raise ValueError("No fingerprints provided for scaler fitting")
            
        # Stack feature vectors
        feature_matrix = np.vstack([fp.feature_vector for fp in fingerprints])
        
        # Fit scaler
        self.scaler.fit(feature_matrix)
        self.logger.info(f"Fitted {self.config.scaler_type} scaler on {len(fingerprints)} sessions")
    
    def transform_fingerprints(self, fingerprints: List[SessionFingerprint]) -> List[SessionFingerprint]:
        """Apply fitted scaler to fingerprints (online)"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
            
        transformed_fingerprints = []
        for fp in fingerprints:
            # Scale the feature vector
            scaled_vector = self.scaler.transform(fp.feature_vector.reshape(1, -1)).flatten()
            
            # Create new fingerprint with scaled features
            transformed_fp = SessionFingerprint(
                session_id=fp.session_id,
                feature_vector=scaled_vector,
                feature_names=fp.feature_names,
                n_events=fp.n_events,
                session_date=fp.session_date,
                session_type=fp.session_type,
                data_quality=fp.data_quality,
                htf_mode=fp.htf_mode
            )
            transformed_fingerprints.append(transformed_fp)
            
        return transformed_fingerprints
    
    def fingerprints_to_dataframe(self, fingerprints: List[SessionFingerprint]) -> pd.DataFrame:
        """Convert fingerprints to pandas DataFrame"""
        if not fingerprints:
            return pd.DataFrame()
            
        # Convert to list of dictionaries
        data = [fp.to_dict() for fp in fingerprints]
        df = pd.DataFrame(data)
        
        # Sort by session_date for deterministic ordering
        if 'session_date' in df.columns:
            df = df.sort_values('session_date').reset_index(drop=True)
            
        return df
    
    def validate_fingerprints(self, fingerprints: List[SessionFingerprint]) -> Dict[str, Any]:
        """Validate fingerprint vectors meet success criteria"""
        if not fingerprints:
            return {"valid": False, "error": "No fingerprints to validate"}
            
        # Stack all feature vectors
        feature_matrix = np.vstack([fp.feature_vector for fp in fingerprints])
        
        # Check dimensions
        expected_dim = len(self.feature_names)
        actual_dim = feature_matrix.shape[1]
        dim_valid = 20 <= actual_dim <= 32
        
        # Check for NaN/Inf
        finite_check = np.isfinite(feature_matrix).all()
        
        # Check repeatability (feature vectors should be deterministic)
        # For now, just check that we have reasonable variance
        feature_stds = np.std(feature_matrix, axis=0)
        has_variance = (feature_stds > 1e-10).any()  # At least some features should vary
        
        validation_results = {
            "valid": dim_valid and finite_check,
            "n_fingerprints": len(fingerprints),
            "vector_dimension": actual_dim,
            "expected_dimension": expected_dim,
            "dimension_in_range": dim_valid,
            "no_nan_inf": finite_check,
            "has_variance": has_variance,
            "feature_std_min": feature_stds.min(),
            "feature_std_max": feature_stds.max(),
            "feature_mean_range": (feature_matrix.mean(axis=0).min(), feature_matrix.mean(axis=0).max())
        }
        
        return validation_results