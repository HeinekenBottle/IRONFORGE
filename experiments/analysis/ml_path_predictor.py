#!/usr/bin/env python3
"""
Machine Learning Path Predictor for Experiment Set E
Implements one-vs-rest classifiers with isotonic calibration and hazard curve modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False
    print("⚠️  Warning: lifelines not available. Hazard curves will use simplified implementation.")


class MLPathPredictor:
    """Machine Learning Path Predictor with Statistical Rigor"""
    
    def __init__(self):
        self.classifiers = {
            'CONT': LogisticRegression(random_state=42, max_iter=1000),
            'MR': LogisticRegression(random_state=42, max_iter=1000), 
            'ACCEL': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.isotonic_regressors = {
            'CONT': IsotonicRegression(out_of_bounds='clip'),
            'MR': IsotonicRegression(out_of_bounds='clip'),
            'ACCEL': IsotonicRegression(out_of_bounds='clip')
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def extract_features(self, session_data: pd.DataFrame, event_idx: int) -> np.ndarray:
        """Extract features for ML prediction from session data"""
        if event_idx >= len(session_data):
            return np.array([])
        
        event_row = session_data.iloc[event_idx]
        features = []
        feature_names = []
        
        # Basic event features
        features.extend([
            event_row.get('magnitude', 0),
            event_row.get('energy_density', 0.5),
            event_row.get('archaeological_significance', 0),
            event_row.get('htf_confluence', 0.5),
            event_row.get('range_position', 0.5)
        ])
        feature_names.extend([
            'magnitude', 'energy_density', 'archaeological_significance',
            'htf_confluence', 'range_position'
        ])
        
        # Derived f8 features (from experiment_e_analyzer)
        if 'f8' in session_data.columns:
            f8_values = session_data['f8'].dropna()
            if len(f8_values) > 0:
                f8_q = event_row.get('f8', 0) / f8_values.quantile(0.95) if f8_values.quantile(0.95) > 0 else 0
                features.append(f8_q)
                feature_names.append('f8_q')
            else:
                features.append(0.5)
                feature_names.append('f8_q')
        else:
            features.append(0.5)
            feature_names.append('f8_q')
        
        # f8 slope sign (3-bar rolling slope)
        if event_idx >= 2 and 'f8' in session_data.columns:
            f8_window = session_data['f8'].iloc[max(0, event_idx-2):event_idx+1]
            if len(f8_window) >= 2:
                slope = np.polyfit(range(len(f8_window)), f8_window, 1)[0]
                f8_slope_sign = 1 if slope > 0 else -1 if slope < 0 else 0
            else:
                f8_slope_sign = 0
        else:
            f8_slope_sign = 0
        
        features.append(f8_slope_sign)
        feature_names.append('f8_slope_sign')
        
        # HTF features (f47-f50)
        htf_features = ['f47_barpos_m15', 'f48_barpos_h1', 'f49_dist_daily_mid', 'f50_htf_regime']
        for htf_feat in htf_features:
            features.append(event_row.get(htf_feat, 1 if htf_feat == 'f50_htf_regime' else 0))
            feature_names.append(htf_feat)
        
        # Temporal features
        features.extend([
            event_row.get('time_since_session_open', 0) / 3600,  # Normalize to hours
            event_row.get('normalized_time', 0.5),
            event_row.get('price_momentum', 0)
        ])
        feature_names.extend(['hours_since_open', 'normalized_time', 'price_momentum'])
        
        # Gap age proxy (derived from price momentum and time)
        gap_age = 0 if abs(event_row.get('price_momentum', 0)) > 0.02 else 1
        features.append(gap_age)
        feature_names.append('gap_age_binary')
        
        # Session context features
        session_stats = self._calculate_session_context(session_data, event_idx)
        features.extend([
            session_stats.get('position_in_range', 0.4),  # Should be ~40% for RD@40% events
            session_stats.get('distance_to_session_high', 0.5),
            session_stats.get('distance_to_session_low', 0.5),
            session_stats.get('session_volatility', 0.1)
        ])
        feature_names.extend([
            'position_in_range', 'dist_to_high', 'dist_to_low', 'session_volatility'
        ])
        
        # Store feature names on first extraction
        if not self.feature_names:
            self.feature_names = feature_names
        
        # Handle NaN values by replacing with defaults
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.5, posinf=1.0, neginf=0.0)
        
        return features
    
    def _calculate_session_context(self, session_data: pd.DataFrame, event_idx: int) -> Dict[str, float]:
        """Calculate session-level context features"""
        price_col = 'price_level' if 'price_level' in session_data.columns else 'absolute_price'
        if price_col not in session_data.columns:
            return {'position_in_range': 0.4, 'distance_to_session_high': 0.5, 
                   'distance_to_session_low': 0.5, 'session_volatility': 0.1}
        
        prices = session_data[price_col].dropna()
        if len(prices) == 0:
            return {'position_in_range': 0.4, 'distance_to_session_high': 0.5,
                   'distance_to_session_low': 0.5, 'session_volatility': 0.1}
        
        session_high = prices.max()
        session_low = prices.min()
        session_range = session_high - session_low
        current_price = session_data.iloc[event_idx].get(price_col, session_low + session_range * 0.4)
        
        if session_range > 0:
            position_in_range = (current_price - session_low) / session_range
            dist_to_high = (session_high - current_price) / session_range
            dist_to_low = (current_price - session_low) / session_range
        else:
            position_in_range = 0.4
            dist_to_high = 0.5
            dist_to_low = 0.5
        
        # Session volatility (coefficient of variation)
        session_volatility = prices.std() / prices.mean() if prices.mean() > 0 else 0.1
        
        return {
            'position_in_range': position_in_range,
            'distance_to_session_high': dist_to_high,
            'distance_to_session_low': dist_to_low,
            'session_volatility': min(session_volatility, 1.0)  # Cap at 1.0
        }
    
    def prepare_training_data(self, rd40_events: List[Dict], sessions: Dict[str, pd.DataFrame],
                            path_classifications: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix X and target labels y for training"""
        X_list = []
        y_list = []
        
        for event in rd40_events:
            session_id = event.get('session_id')
            event_idx = event.get('event_index', 0)
            
            if session_id not in sessions:
                continue
            
            # Extract features
            features = self.extract_features(sessions[session_id], event_idx)
            if len(features) == 0:
                continue
            
            # Get path label
            event_key = f"{session_id}_{event_idx}"
            path_data = path_classifications.get(event_key, {})
            path_label = path_data.get('path', 'UNKNOWN')
            
            # Skip unknown paths for training
            if path_label == 'UNKNOWN' or path_label.startswith('NOT_'):
                continue
            
            # Map E1/E2/E3 labels to base path types
            if path_label.startswith('E1_'):
                path_label = 'CONT'
            elif path_label.startswith('E2_'):
                path_label = 'MR'
            elif path_label.startswith('E3_'):
                path_label = 'ACCEL'
            
            X_list.append(features)
            y_list.append(path_label)
        
        if len(X_list) == 0:
            return np.array([]), np.array([])
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit one-vs-rest classifiers with isotonic calibration"""
        if len(X) == 0 or len(y) == 0:
            return {"error": "No training data available"}
        
        print(f"🤖 Training ML Path Predictor with {len(X)} samples...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        training_results = {
            "total_samples": len(X),
            "class_distribution": {},
            "cross_validation_scores": {},
            "calibration_metrics": {},
            "feature_importance": {}
        }
        
        # Get class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            training_results["class_distribution"][cls] = count
        
        # Train one-vs-rest classifiers
        for path_type in ['CONT', 'MR', 'ACCEL']:
            print(f"  📊 Training {path_type} classifier...")
            
            # Create binary labels (one-vs-rest)
            y_binary = (y == path_type).astype(int)
            
            # Skip if no positive samples
            if y_binary.sum() == 0:
                print(f"    ⚠️  No {path_type} samples found, skipping...")
                continue
            
            # Cross-validation
            cv_scores = cross_val_score(self.classifiers[path_type], X_scaled, y_binary, 
                                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                      scoring='roc_auc')
            training_results["cross_validation_scores"][path_type] = {
                "mean_auc": cv_scores.mean(),
                "std_auc": cv_scores.std(),
                "scores": cv_scores.tolist()
            }
            
            # Fit classifier
            self.classifiers[path_type].fit(X_scaled, y_binary)
            
            # Get uncalibrated probabilities
            y_pred_proba = self.classifiers[path_type].predict_proba(X_scaled)[:, 1]
            
            # Fit isotonic calibration
            self.isotonic_regressors[path_type].fit(y_pred_proba, y_binary)
            
            # Calculate calibration metrics
            y_calibrated = self.isotonic_regressors[path_type].predict(y_pred_proba)
            calibration_auc = roc_auc_score(y_binary, y_calibrated) if y_binary.sum() > 0 else 0
            
            training_results["calibration_metrics"][path_type] = {
                "uncalibrated_auc": roc_auc_score(y_binary, y_pred_proba),
                "calibrated_auc": calibration_auc,
                "calibration_improvement": calibration_auc - roc_auc_score(y_binary, y_pred_proba)
            }
            
            # Feature importance (for logistic regression)
            if hasattr(self.classifiers[path_type], 'coef_'):
                feature_importance = np.abs(self.classifiers[path_type].coef_[0])
                top_features = np.argsort(feature_importance)[-5:][::-1]  # Top 5 features
                
                training_results["feature_importance"][path_type] = {
                    "top_features": [
                        {"feature": self.feature_names[idx], "importance": feature_importance[idx]}
                        for idx in top_features if idx < len(self.feature_names)
                    ]
                }
        
        self.is_fitted = True
        self.performance_metrics = training_results
        
        print(f"✅ ML Path Predictor training complete!")
        return training_results
    
    def predict_path_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict calibrated path probabilities using isotonic regression"""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        probabilities = {}
        
        for path_type in ['CONT', 'MR', 'ACCEL']:
            if path_type not in self.classifiers:
                probabilities[path_type] = np.zeros(len(X))
                continue
            
            try:
                # Get uncalibrated probabilities
                uncalibrated_proba = self.classifiers[path_type].predict_proba(X_scaled)[:, 1]
                
                # Apply isotonic calibration
                calibrated_proba = self.isotonic_regressors[path_type].predict(uncalibrated_proba)
                probabilities[path_type] = calibrated_proba
            except Exception as e:
                # If classifier not properly fitted, return zeros
                print(f"    ⚠️  {path_type} classifier not available: {e}")
                probabilities[path_type] = np.zeros(len(X))
        
        return probabilities
    
    def generate_confusion_matrix(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive confusion matrix analysis"""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        # Get predictions for each path
        path_probabilities = self.predict_path_probabilities(X)
        
        # Convert to multi-class predictions (argmax)
        prob_matrix = np.column_stack([
            path_probabilities.get('CONT', np.zeros(len(X))),
            path_probabilities.get('MR', np.zeros(len(X))),
            path_probabilities.get('ACCEL', np.zeros(len(X)))
        ])
        
        y_pred = np.array(['CONT', 'MR', 'ACCEL'])[np.argmax(prob_matrix, axis=1)]
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['CONT', 'MR', 'ACCEL'])
        
        # Classification report
        report = classification_report(y_true, y_pred, labels=['CONT', 'MR', 'ACCEL'], 
                                     output_dict=True, zero_division=0)
        
        # Per-path metrics
        path_metrics = {}
        for i, path_type in enumerate(['CONT', 'MR', 'ACCEL']):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            path_metrics[path_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": cm[i, :].sum()
            }
        
        return {
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "path_metrics": path_metrics,
            "overall_accuracy": (cm.diagonal().sum() / cm.sum()) if cm.sum() > 0 else 0
        }
    
    def analyze_hazard_curves(self, rd40_events: List[Dict], sessions: Dict[str, pd.DataFrame],
                            path_classifications: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze time-to-event (hazard) curves for path resolution"""
        print("📈 Analyzing hazard curves for path resolution timing...")
        
        hazard_data = defaultdict(list)  # {path_type: [(duration, event_occurred)]}
        
        for event in rd40_events:
            session_id = event.get('session_id')
            event_idx = event.get('event_index', 0)
            
            if session_id not in sessions:
                continue
            
            event_key = f"{session_id}_{event_idx}"
            path_data = path_classifications.get(event_key, {})
            path_label = path_data.get('path', 'UNKNOWN')
            
            # Map E1/E2/E3 to base types
            if path_label.startswith('E1_'):
                path_type = 'CONT'
            elif path_label.startswith('E2_'):
                path_type = 'MR'
            elif path_label.startswith('E3_'):
                path_type = 'ACCEL'
            else:
                continue
            
            # Calculate time to resolution from different sources
            timing_data = path_data.get('timing_analysis', {})
            kpis = path_data.get('kpis', {})
            
            time_to_resolution = None
            
            if path_type == 'CONT':
                time_to_resolution = (timing_data.get('time_to_80') or 
                                    kpis.get('expected_time_to_80') or
                                    kpis.get('time_to_80'))
            elif path_type == 'MR':
                time_to_resolution = (timing_data.get('time_to_mid') or 
                                    kpis.get('time_to_mid'))
            elif path_type == 'ACCEL':
                time_to_resolution = (timing_data.get('time_to_80') or 
                                    kpis.get('time_to_80'))
            
            # Use a default observation window if no resolution time
            if time_to_resolution is None or time_to_resolution == float('inf'):
                time_to_resolution = 90  # 90-minute observation window
                event_occurred = False
            else:
                time_to_resolution = min(time_to_resolution, 120)  # Cap at 120 minutes
                event_occurred = True
            
            hazard_data[path_type].append((time_to_resolution, event_occurred))
        
        # Calculate hazard statistics
        hazard_results = {
            "path_hazard_analysis": {},
            "comparative_survival": {},
            "median_resolution_times": {}
        }
        
        for path_type, duration_data in hazard_data.items():
            if not duration_data:
                continue
            
            durations = [d[0] for d in duration_data]
            events = [d[1] for d in duration_data]
            
            # Basic statistics
            hazard_results["path_hazard_analysis"][path_type] = {
                "total_events": len(duration_data),
                "resolved_events": sum(events),
                "resolution_rate": sum(events) / len(events) if events else 0,
                "median_time": np.median(durations),
                "mean_time": np.mean(durations),
                "std_time": np.std(durations)
            }
            
            # Kaplan-Meier survival analysis (if available)
            if SURVIVAL_AVAILABLE and len(durations) > 2:
                try:
                    kmf = KaplanMeierFitter()
                    kmf.fit(durations, events, label=f'{path_type} Path')
                    
                    # Median survival time
                    median_survival = kmf.median_survival_time_
                    hazard_results["median_resolution_times"][path_type] = median_survival
                    
                    # Survival probability at key time points
                    time_points = [15, 30, 45, 60, 90, 120]  # minutes
                    survival_probs = {}
                    for t in time_points:
                        try:
                            prob = kmf.survival_function_at_times(t).iloc[0]
                            survival_probs[f"t_{t}min"] = prob
                        except:
                            survival_probs[f"t_{t}min"] = None
                    
                    hazard_results["comparative_survival"][path_type] = survival_probs
                    
                except Exception as e:
                    print(f"    ⚠️  Survival analysis error for {path_type}: {e}")
            else:
                # Simplified hazard analysis
                hazard_results["median_resolution_times"][path_type] = np.median(durations)
        
        return hazard_results
    
    def get_feature_attributions(self, X: np.ndarray, target_path: str = 'CONT') -> Dict[str, Any]:
        """Get feature attributions that influence path selection"""
        if not self.is_fitted or target_path not in self.classifiers:
            return {"error": f"Model not fitted or {target_path} classifier not available"}
        
        X_scaled = self.scaler.transform(X)
        
        # Get feature importance from logistic regression coefficients
        if hasattr(self.classifiers[target_path], 'coef_'):
            coefficients = self.classifiers[target_path].coef_[0]
            feature_attribution = {}
            
            for i, coef in enumerate(coefficients):
                if i < len(self.feature_names):
                    feature_attribution[self.feature_names[i]] = coef
            
            # Sort by absolute importance
            sorted_features = sorted(feature_attribution.items(), key=lambda x: abs(x[1]), reverse=True)
            
            return {
                "target_path": target_path,
                "feature_attributions": dict(sorted_features),
                "top_positive_features": [f for f, coef in sorted_features if coef > 0][:5],
                "top_negative_features": [f for f, coef in sorted_features if coef < 0][:5]
            }
        
        return {"error": "Feature attribution not available for this classifier type"}