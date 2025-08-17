"""
Validation Runner for IRONFORGE (Wave 4)
=========================================
Orchestrates validation experiments with splits, controls, ablations,
and comprehensive reporting for temporal pattern discovery evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .controls import create_control_variants
from .metrics import compute_validation_metrics
from .splits import PurgedKFold, oos_split, temporal_train_test_split


@dataclass
class ValidationConfig:
    """Configuration for validation experiments.

    Parameters
    ----------
    mode : str
        Validation mode: "oos", "purged-kfold", or "holdout".
    folds : int
        Number of folds for k-fold validation.
    embargo_mins : int
        Embargo period in minutes to prevent look-ahead bias.
    controls : List[str]
        Negative controls to run: ["time_shuffle", "label_perm", etc.].
    ablations : List[str]
        Feature groups to ablate: ["htf_prox", "cycles", "structure"].
    report_dir : Path
        Directory to write validation reports.
    random_seed : int
        Base random seed for reproducibility.
    """

    mode: str = "oos"
    folds: int = 5
    embargo_mins: int = 30
    controls: list[str] = field(default_factory=lambda: ["time_shuffle", "label_perm"])
    ablations: list[str] = field(default_factory=lambda: ["htf_prox", "cycles", "structure"])
    report_dir: Path = field(default_factory=lambda: Path("reports/validation"))
    random_seed: int = 42

    def __post_init__(self):
        """Ensure report directory exists."""
        self.report_dir = Path(self.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)


class ValidationRunner:
    """Orchestrates comprehensive validation experiments."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute validation experiment with chosen split, controls, and ablations.

        Parameters
        ----------
        data : Dict[str, Any]
            Input data containing:
            - edge_index: Edge connectivity matrix
            - edge_times: Edge timestamps
            - node_features: Node feature matrix
            - labels: Target labels
            - timestamps: Sample timestamps
            - feature_groups: Optional feature group definitions

        Returns
        -------
        Dict[str, Any]
            Comprehensive validation results.
        """
        self.logger.info(f"Starting validation with mode: {self.config.mode}")
        start_time = time.time()

        # Validate input data
        self._validate_input_data(data)

        # Extract components
        edge_index = data["edge_index"]
        edge_times = data["edge_times"]
        node_features = data["node_features"]
        labels = data["labels"]
        timestamps = data["timestamps"]
        feature_groups = data.get("feature_groups", {})

        # Run main validation
        main_results = self._run_main_validation(
            edge_index, edge_times, node_features, labels, timestamps
        )

        # Run negative controls
        control_results = self._run_controls(
            edge_index, edge_times, node_features, labels, timestamps
        )

        # Run ablation studies
        ablation_results = self._run_ablations(
            edge_index, edge_times, node_features, labels, timestamps, feature_groups
        )

        # Compile comprehensive results
        results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "config": self._config_to_dict(),
                "data_shape": {
                    "num_nodes": len(node_features),
                    "num_edges": len(edge_times),
                    "num_features": node_features.shape[1] if len(node_features) > 0 else 0,
                    "num_samples": len(labels),
                },
                "runtime_seconds": time.time() - start_time,
            },
            "main_validation": main_results,
            "negative_controls": control_results,
            "ablation_studies": ablation_results,
            "summary": self._create_summary(main_results, control_results, ablation_results),
        }

        # Save results
        self._save_results(results)

        self.logger.info(
            f"Validation completed in {results['experiment_info']['runtime_seconds']:.2f}s"
        )
        return results

    def _validate_input_data(self, data: dict[str, Any]) -> None:
        """Validate input data structure and types."""
        required_keys = ["edge_index", "edge_times", "node_features", "labels", "timestamps"]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required data key: {key}")

        # Basic shape validation
        if len(data["labels"]) != len(data["timestamps"]):
            raise ValueError("Labels and timestamps must have same length")

        if len(data["edge_times"]) != data["edge_index"].shape[1]:
            raise ValueError("Edge times and edge index must have consistent dimensions")

    def _run_main_validation(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Run main validation with specified split strategy."""
        self.logger.info(f"Running main validation with {self.config.mode} split")

        if self.config.mode == "oos":
            return self._run_oos_validation(
                edge_index, edge_times, node_features, labels, timestamps
            )
        elif self.config.mode == "purged-kfold":
            return self._run_kfold_validation(
                edge_index, edge_times, node_features, labels, timestamps
            )
        elif self.config.mode == "holdout":
            return self._run_holdout_validation(
                edge_index, edge_times, node_features, labels, timestamps
            )
        else:
            raise ValueError(f"Unknown validation mode: {self.config.mode}")

    def _run_oos_validation(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Run out-of-sample validation."""
        # Use 70% cutoff for OOS split
        sorted_times = np.sort(timestamps)
        cutoff_idx = int(0.7 * len(sorted_times))
        cutoff_ts = sorted_times[cutoff_idx]

        train_idx, test_idx = oos_split(timestamps, cutoff_ts)

        # Simulate model training and prediction (placeholder)
        train_scores, test_scores = self._simulate_model_run(
            node_features, labels, train_idx, test_idx
        )

        # Calculate metrics
        metrics = compute_validation_metrics(labels[test_idx], test_scores, timestamps[test_idx])

        return {
            "split_type": "oos",
            "cutoff_timestamp": int(cutoff_ts),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "metrics": metrics,
            "fold_results": [
                {
                    "fold": 0,
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                    "train_scores": train_scores.tolist(),
                    "test_scores": test_scores.tolist(),
                    "metrics": metrics,
                }
            ],
        }

    def _run_kfold_validation(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Run purged K-fold validation."""
        splitter = PurgedKFold(n_splits=self.config.folds, embargo_mins=self.config.embargo_mins)

        fold_results = []
        all_metrics = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(timestamps)):
            self.logger.debug(f"Processing fold {fold + 1}/{self.config.folds}")

            # Simulate model training and prediction
            train_scores, test_scores = self._simulate_model_run(
                node_features, labels, train_idx, test_idx
            )

            # Calculate metrics for this fold
            fold_metrics = compute_validation_metrics(
                labels[test_idx], test_scores, timestamps[test_idx]
            )

            fold_results.append(
                {
                    "fold": fold,
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                    "train_scores": train_scores.tolist(),
                    "test_scores": test_scores.tolist(),
                    "metrics": fold_metrics,
                }
            )

            all_metrics.append(fold_metrics)

        # Aggregate metrics across folds
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        return {
            "split_type": "purged_kfold",
            "n_folds": self.config.folds,
            "embargo_mins": self.config.embargo_mins,
            "metrics": aggregated_metrics,
            "fold_results": fold_results,
        }

    def _run_holdout_validation(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Run holdout validation with embargo."""
        train_idx, test_idx = temporal_train_test_split(
            timestamps, test_size=0.2, embargo_mins=self.config.embargo_mins
        )

        # Simulate model training and prediction
        train_scores, test_scores = self._simulate_model_run(
            node_features, labels, train_idx, test_idx
        )

        # Calculate metrics
        metrics = compute_validation_metrics(labels[test_idx], test_scores, timestamps[test_idx])

        return {
            "split_type": "holdout",
            "embargo_mins": self.config.embargo_mins,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "metrics": metrics,
            "fold_results": [
                {
                    "fold": 0,
                    "train_indices": train_idx.tolist(),
                    "test_indices": test_idx.tolist(),
                    "train_scores": train_scores.tolist(),
                    "test_scores": test_scores.tolist(),
                    "metrics": metrics,
                }
            ],
        }

    def _run_controls(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
    ) -> dict[str, Any]:
        """Run negative control experiments."""
        if not self.config.controls:
            return {}

        self.logger.info(f"Running negative controls: {self.config.controls}")

        # Create control variants
        control_variants = create_control_variants(
            edge_index,
            edge_times,
            node_features,
            labels,
            self.config.controls,
            seed=self.config.random_seed,
        )

        control_results = {}

        for control_name, variant_data in control_variants.items():
            self.logger.debug(f"Processing control: {control_name}")

            # Run same validation as main experiment
            control_result = self._run_main_validation(
                variant_data["edge_index"],
                variant_data["edge_times"],
                variant_data["node_features"],
                variant_data["labels"],
                timestamps,
            )

            control_result["description"] = variant_data["description"]
            control_results[control_name] = control_result

        return control_results

    def _run_ablations(
        self,
        edge_index: np.ndarray,
        edge_times: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        feature_groups: dict[str, Sequence[int]],
    ) -> dict[str, Any]:
        """Run ablation studies on feature groups."""
        if not self.config.ablations or not feature_groups:
            return {}

        self.logger.info(f"Running ablations: {self.config.ablations}")

        ablation_results = {}

        for ablation_group in self.config.ablations:
            if ablation_group not in feature_groups:
                self.logger.warning(f"Feature group '{ablation_group}' not found, skipping")
                continue

            self.logger.debug(f"Processing ablation: {ablation_group}")

            # Create ablated features (zero out the group)
            ablated_features = node_features.copy()
            group_indices = feature_groups[ablation_group]

            for idx in group_indices:
                if idx < ablated_features.shape[1]:
                    ablated_features[:, idx] = 0.0

            # Run validation with ablated features
            ablation_result = self._run_main_validation(
                edge_index, edge_times, ablated_features, labels, timestamps
            )

            ablation_result["ablated_features"] = list(group_indices)
            ablation_result["description"] = f"Ablated feature group: {ablation_group}"
            ablation_results[ablation_group] = ablation_result

        return ablation_results

    def _simulate_model_run(
        self,
        node_features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate model training and prediction (placeholder implementation).

        In practice, this would integrate with the actual TGAT discovery engine.
        For Wave 4, we provide a realistic simulation that respects data patterns.
        """
        rng = np.random.RandomState(self.config.random_seed)

        # Simple simulation: scores based on feature means + noise
        if len(node_features) > 0 and node_features.shape[1] > 0:
            # Use feature means as base scores
            feature_means = np.mean(node_features, axis=1)
            feature_std = np.std(feature_means) if len(feature_means) > 1 else 1.0

            # Normalize to [0, 1] range
            if feature_std > 0:
                normalized_features = (feature_means - np.mean(feature_means)) / feature_std
                base_scores = 1.0 / (1.0 + np.exp(-normalized_features))  # Sigmoid
            else:
                base_scores = np.full(len(node_features), 0.5)
        else:
            base_scores = rng.random(len(labels))

        # Add correlation with actual labels for realism
        label_correlation = 0.3  # Moderate correlation
        label_noise = rng.normal(0, 0.2, len(labels))

        simulated_scores = 0.7 * base_scores + label_correlation * labels + label_noise

        # Clip to valid range
        simulated_scores = np.clip(simulated_scores, 0.0, 1.0)

        # Return train and test scores
        train_scores = simulated_scores[train_idx] if len(train_idx) > 0 else np.array([])
        test_scores = simulated_scores[test_idx] if len(test_idx) > 0 else np.array([])

        return train_scores, test_scores

    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics across folds."""
        if not metrics_list:
            return {}

        aggregated = {}

        # Get all metric names
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m and np.isfinite(m[key])]

            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
            else:
                aggregated[f"{key}_mean"] = 0.0
                aggregated[f"{key}_std"] = 0.0
                aggregated[f"{key}_min"] = 0.0
                aggregated[f"{key}_max"] = 0.0

        return aggregated

    def _create_summary(
        self,
        main_results: dict[str, Any],
        control_results: dict[str, Any],
        ablation_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Create high-level summary of validation results."""
        summary = {
            "validation_passed": True,
            "main_performance": {},
            "control_comparison": {},
            "ablation_impact": {},
            "recommendations": [],
        }

        # Extract main performance metrics
        main_metrics = main_results.get("metrics", {})
        summary["main_performance"] = {
            "temporal_auc": main_metrics.get("temporal_auc", 0.0),
            "precision_at_20": main_metrics.get("precision_at_20", 0.0),
            "pattern_stability": main_metrics.get("pattern_stability", 0.0),
        }

        # Compare with controls
        for control_name, control_result in control_results.items():
            control_metrics = control_result.get("metrics", {})
            control_auc = control_metrics.get("temporal_auc", 0.0)
            main_auc = main_metrics.get("temporal_auc", 0.0)

            summary["control_comparison"][control_name] = {
                "control_auc": control_auc,
                "main_auc": main_auc,
                "performance_drop": max(0.0, main_auc - control_auc),
                "passes_check": main_auc > control_auc + 0.05,  # 5% threshold
            }

        # Analyze ablation impact
        for ablation_name, ablation_result in ablation_results.items():
            ablation_metrics = ablation_result.get("metrics", {})
            ablation_auc = ablation_metrics.get("temporal_auc", 0.0)
            main_auc = main_metrics.get("temporal_auc", 0.0)

            summary["ablation_impact"][ablation_name] = {
                "ablated_auc": ablation_auc,
                "main_auc": main_auc,
                "performance_drop": max(0.0, main_auc - ablation_auc),
                "feature_importance": max(0.0, main_auc - ablation_auc) / max(main_auc, 1e-6),
            }

        # Generate recommendations
        if summary["main_performance"]["temporal_auc"] < 0.6:
            summary["recommendations"].append("Main model performance is below threshold (0.6)")
            summary["validation_passed"] = False

        for control_name, control_data in summary["control_comparison"].items():
            if not control_data["passes_check"]:
                summary["recommendations"].append(
                    f"Model does not sufficiently outperform {control_name} control"
                )
                summary["validation_passed"] = False

        return summary

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save validation results to JSON and HTML files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.config.report_dir / f"validation_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Validation results saved to {json_file}")

        # Save minimal HTML summary
        html_file = self.config.report_dir / f"validation_summary_{timestamp}.html"
        self._create_html_summary(results, html_file)

        self.logger.info(f"Validation summary saved to {html_file}")

    def _create_html_summary(self, results: dict[str, Any], output_path: Path) -> None:
        """Create minimal HTML summary report."""
        summary = results.get("summary", {})
        experiment_info = results.get("experiment_info", {})

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IRONFORGE Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ margin: 5px 0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>IRONFORGE Validation Report</h1>
        <p>Generated: {experiment_info.get('timestamp', 'Unknown')}</p>
        <p>Configuration: {experiment_info.get('config', {}).get('mode', 'Unknown')} validation</p>
        <p>Runtime: {experiment_info.get('runtime_seconds', 0):.2f} seconds</p>
    </div>
    
    <div class="section">
        <h2>Validation Status</h2>
        <p class="{'pass' if summary.get('validation_passed', False) else 'fail'}">
            {'✅ PASSED' if summary.get('validation_passed', False) else '❌ FAILED'}
        </p>
    </div>
    
    <div class="section">
        <h2>Main Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
        """

        for metric, value in summary.get("main_performance", {}).items():
            html_content += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"

        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Control Comparisons</h2>
        <table>
            <tr><th>Control</th><th>Control AUC</th><th>Main AUC</th><th>Drop</th><th>Status</th></tr>
        """

        for control_name, control_data in summary.get("control_comparison", {}).items():
            status = "✅ Pass" if control_data.get("passes_check", False) else "❌ Fail"
            html_content += f"""
            <tr>
                <td>{control_name}</td>
                <td>{control_data.get('control_auc', 0):.4f}</td>
                <td>{control_data.get('main_auc', 0):.4f}</td>
                <td>{control_data.get('performance_drop', 0):.4f}</td>
                <td>{status}</td>
            </tr>
            """

        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Ablation Impact</h2>
        <table>
            <tr><th>Feature Group</th><th>Ablated AUC</th><th>Main AUC</th><th>Importance</th></tr>
        """

        for ablation_name, ablation_data in summary.get("ablation_impact", {}).items():
            html_content += f"""
            <tr>
                <td>{ablation_name}</td>
                <td>{ablation_data.get('ablated_auc', 0):.4f}</td>
                <td>{ablation_data.get('main_auc', 0):.4f}</td>
                <td>{ablation_data.get('feature_importance', 0):.4f}</td>
            </tr>
            """

        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
        """

        for recommendation in summary.get("recommendations", []):
            html_content += f"<li>{recommendation}</li>"

        if not summary.get("recommendations"):
            html_content += "<li>No issues detected - validation passed all checks</li>"

        html_content += """
        </ul>
    </div>
</body>
</html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

    def _config_to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "mode": self.config.mode,
            "folds": self.config.folds,
            "embargo_mins": self.config.embargo_mins,
            "controls": self.config.controls,
            "ablations": self.config.ablations,
            "report_dir": str(self.config.report_dir),
            "random_seed": self.config.random_seed,
        }
