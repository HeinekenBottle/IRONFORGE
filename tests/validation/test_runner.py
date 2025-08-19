"""Unit tests for validation runner (Wave 4)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ironforge.validation.runner import ValidationConfig, ValidationRunner


class TestValidationConfig:
    """Test cases for ValidationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()

        assert config.mode == "oos"
        assert config.folds == 5
        assert config.embargo_mins == 30
        assert config.controls == ["time_shuffle", "label_perm"]
        assert config.ablations == ["htf_prox", "cycles", "structure"]
        assert config.random_seed == 42
        assert config.report_dir == Path("reports/validation")

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            mode="purged-kfold",
            folds=10,
            embargo_mins=60,
            controls=["time_shuffle"],
            ablations=["htf_prox"],
            random_seed=123,
            report_dir=Path("/tmp/reports"),
        )

        assert config.mode == "purged-kfold"
        assert config.folds == 10
        assert config.embargo_mins == 60
        assert config.controls == ["time_shuffle"]
        assert config.ablations == ["htf_prox"]
        assert config.random_seed == 123
        assert config.report_dir == Path("/tmp/reports")

    def test_report_dir_creation(self):
        """Test that report directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_reports"
            ValidationConfig(report_dir=report_path)

            assert report_path.exists()
            assert report_path.is_dir()


class TestValidationRunner:
    """Test cases for ValidationRunner."""

    def setup_method(self):
        """Set up test data."""
        self.config = ValidationConfig(
            mode="oos",
            folds=3,
            embargo_mins=10,
            controls=["time_shuffle"],
            ablations=["htf_prox"],
            random_seed=42,
        )

        self.test_data = {
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
            "edge_times": np.array([100, 200, 300]),
            "node_features": np.random.random((10, 20)),
            "labels": np.random.randint(0, 2, 10),
            "timestamps": np.arange(10) * 10 + 100,
            "feature_groups": {
                "htf_prox": [0, 1, 2, 3, 4],
                "cycles": [5, 6, 7, 8, 9],
                "structure": [10, 11, 12, 13, 14],
            },
        }

    def test_runner_initialization(self):
        """Test ValidationRunner initialization."""
        runner = ValidationRunner(self.config)

        assert runner.config == self.config
        assert hasattr(runner, "logger")

    def test_input_validation(self):
        """Test input data validation."""
        runner = ValidationRunner(self.config)

        # Missing required key
        invalid_data = {key: value for key, value in self.test_data.items() if key != "labels"}

        with pytest.raises(ValueError, match="Missing required data key"):
            runner.run(invalid_data)

        # Mismatched lengths
        invalid_data = self.test_data.copy()
        invalid_data["timestamps"] = np.array([100, 200])  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            runner.run(invalid_data)

    def test_oos_validation(self):
        """Test out-of-sample validation."""
        config = ValidationConfig(mode="oos", controls=[], ablations=[])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check result structure
        assert "experiment_info" in results
        assert "main_validation" in results
        assert "negative_controls" in results
        assert "ablation_studies" in results
        assert "summary" in results

        # Check main validation results
        main_results = results["main_validation"]
        assert main_results["split_type"] == "oos"
        assert "metrics" in main_results
        assert "fold_results" in main_results

    def test_kfold_validation(self):
        """Test purged k-fold validation."""
        config = ValidationConfig(mode="purged-kfold", folds=3, controls=[], ablations=[])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check main validation results
        main_results = results["main_validation"]
        assert main_results["split_type"] == "purged_kfold"
        assert main_results["n_folds"] == 3
        assert len(main_results["fold_results"]) <= 3  # May be fewer due to embargo

    def test_holdout_validation(self):
        """Test holdout validation."""
        config = ValidationConfig(mode="holdout", controls=[], ablations=[])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check main validation results
        main_results = results["main_validation"]
        assert main_results["split_type"] == "holdout"
        assert "embargo_mins" in main_results

    def test_negative_controls(self):
        """Test negative controls execution."""
        config = ValidationConfig(controls=["time_shuffle", "label_perm"], ablations=[])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check control results
        control_results = results["negative_controls"]
        assert "time_shuffle" in control_results
        assert "label_perm" in control_results

        for _control_name, control_result in control_results.items():
            assert "description" in control_result
            assert "metrics" in control_result

    def test_ablation_studies(self):
        """Test ablation studies execution."""
        config = ValidationConfig(controls=[], ablations=["htf_prox", "cycles"])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check ablation results
        ablation_results = results["ablation_studies"]
        assert "htf_prox" in ablation_results
        assert "cycles" in ablation_results

        for _ablation_name, ablation_result in ablation_results.items():
            assert "description" in ablation_result
            assert "ablated_features" in ablation_result
            assert "metrics" in ablation_result

    def test_missing_feature_group(self):
        """Test ablation with missing feature group."""
        config = ValidationConfig(controls=[], ablations=["nonexistent_group"])
        runner = ValidationRunner(config)

        # Should handle gracefully
        results = runner.run(self.test_data)

        # Should not include the missing group
        ablation_results = results["ablation_studies"]
        assert "nonexistent_group" not in ablation_results

    def test_unknown_validation_mode(self):
        """Test with unknown validation mode."""
        config = ValidationConfig(mode="unknown_mode")
        runner = ValidationRunner(config)

        with pytest.raises(ValueError, match="Unknown validation mode"):
            runner.run(self.test_data)

    def test_experiment_info(self):
        """Test experiment info generation."""
        runner = ValidationRunner(self.config)

        results = runner.run(self.test_data)

        # Check experiment info
        exp_info = results["experiment_info"]
        assert "timestamp" in exp_info
        assert "config" in exp_info
        assert "data_shape" in exp_info
        assert "runtime_seconds" in exp_info

        # Check data shape info
        data_shape = exp_info["data_shape"]
        assert data_shape["num_nodes"] == len(self.test_data["node_features"])
        assert data_shape["num_edges"] == len(self.test_data["edge_times"])
        assert data_shape["num_features"] == self.test_data["node_features"].shape[1]
        assert data_shape["num_samples"] == len(self.test_data["labels"])

    def test_summary_generation(self):
        """Test validation summary generation."""
        config = ValidationConfig(controls=["time_shuffle"], ablations=["htf_prox"])
        runner = ValidationRunner(config)

        results = runner.run(self.test_data)

        # Check summary structure
        summary = results["summary"]
        assert "validation_passed" in summary
        assert "main_performance" in summary
        assert "control_comparison" in summary
        assert "ablation_impact" in summary
        assert "recommendations" in summary

        # Check control comparison
        control_comp = summary["control_comparison"]
        assert "time_shuffle" in control_comp
        control_data = control_comp["time_shuffle"]
        assert "control_auc" in control_data
        assert "main_auc" in control_data
        assert "performance_drop" in control_data
        assert "passes_check" in control_data

        # Check ablation impact
        ablation_impact = summary["ablation_impact"]
        assert "htf_prox" in ablation_impact
        ablation_data = ablation_impact["htf_prox"]
        assert "ablated_auc" in ablation_data
        assert "main_auc" in ablation_data
        assert "feature_importance" in ablation_data

    def test_results_saving(self):
        """Test that results are saved to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ValidationConfig(report_dir=Path(temp_dir))
            runner = ValidationRunner(config)

            runner.run(self.test_data)

            # Check that files were created
            report_files = list(Path(temp_dir).glob("*"))
            assert len(report_files) >= 2  # JSON + HTML

            # Check for JSON file
            json_files = list(Path(temp_dir).glob("validation_results_*.json"))
            assert len(json_files) == 1

            # Check for HTML file
            html_files = list(Path(temp_dir).glob("validation_summary_*.html"))
            assert len(html_files) == 1

    def test_html_summary_content(self):
        """Test HTML summary content generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ValidationConfig(
                report_dir=Path(temp_dir), controls=["time_shuffle"], ablations=["htf_prox"]
            )
            runner = ValidationRunner(config)

            runner.run(self.test_data)

            # Read HTML file
            html_files = list(Path(temp_dir).glob("validation_summary_*.html"))
            html_content = html_files[0].read_text()

            # Check for expected content
            assert "IRONFORGE Validation Report" in html_content
            assert "Validation Status" in html_content
            assert "Main Performance" in html_content
            assert "Control Comparisons" in html_content
            assert "Ablation Impact" in html_content

    def test_metric_aggregation(self):
        """Test metric aggregation across folds."""
        config = ValidationConfig(mode="purged-kfold", folds=3)
        runner = ValidationRunner(config)

        # Mock metrics for multiple folds
        mock_metrics = [
            {"temporal_auc": 0.8, "precision_at_20": 0.6},
            {"temporal_auc": 0.7, "precision_at_20": 0.5},
            {"temporal_auc": 0.9, "precision_at_20": 0.7},
        ]

        aggregated = runner._aggregate_metrics(mock_metrics)

        # Check aggregated metrics
        assert "temporal_auc_mean" in aggregated
        assert "temporal_auc_std" in aggregated
        assert "temporal_auc_min" in aggregated
        assert "temporal_auc_max" in aggregated

        # Check values
        assert aggregated["temporal_auc_mean"] == 0.8
        assert aggregated["temporal_auc_min"] == 0.7
        assert aggregated["temporal_auc_max"] == 0.9

    def test_config_serialization(self):
        """Test configuration serialization."""
        runner = ValidationRunner(self.config)

        config_dict = runner._config_to_dict()

        # Check all config fields are present
        assert config_dict["mode"] == self.config.mode
        assert config_dict["folds"] == self.config.folds
        assert config_dict["embargo_mins"] == self.config.embargo_mins
        assert config_dict["controls"] == self.config.controls
        assert config_dict["ablations"] == self.config.ablations
        assert config_dict["random_seed"] == self.config.random_seed
        assert isinstance(config_dict["report_dir"], str)

    def test_model_simulation(self):
        """Test model simulation functionality."""
        runner = ValidationRunner(self.config)

        # Test with realistic data
        node_features = np.random.random((100, 20))
        labels = np.random.randint(0, 2, 100)
        train_idx = np.arange(70)
        test_idx = np.arange(70, 100)

        train_scores, test_scores = runner._simulate_model_run(
            node_features, labels, train_idx, test_idx
        )

        # Check output shapes
        assert len(train_scores) == len(train_idx)
        assert len(test_scores) == len(test_idx)

        # Check score ranges
        assert np.all(train_scores >= 0.0) and np.all(train_scores <= 1.0)
        assert np.all(test_scores >= 0.0) and np.all(test_scores <= 1.0)

    def test_edge_case_data(self):
        """Test validation with edge case data."""
        # Very small dataset
        small_data = {
            "edge_index": np.array([[0], [0]]),
            "edge_times": np.array([100]),
            "node_features": np.random.random((2, 5)),
            "labels": np.array([1, 0]),
            "timestamps": np.array([100, 200]),
            "feature_groups": {"group1": [0, 1]},
        }

        config = ValidationConfig(folds=2, controls=[], ablations=[])
        runner = ValidationRunner(config)

        # Should handle small data gracefully
        results = runner.run(small_data)

        assert "experiment_info" in results
        assert "main_validation" in results

    def test_performance_within_budget(self):
        """Test that validation completes within performance budget."""
        import time

        # Create moderately sized dataset
        large_data = {
            "edge_index": np.random.randint(0, 50, (2, 100)),
            "edge_times": np.random.randint(100, 1000, 100),
            "node_features": np.random.random((50, 30)),
            "labels": np.random.randint(0, 2, 50),
            "timestamps": np.arange(50) * 10 + 100,
            "feature_groups": {
                "group1": list(range(10)),
                "group2": list(range(10, 20)),
                "group3": list(range(20, 30)),
            },
        }

        config = ValidationConfig(
            mode="oos", controls=["time_shuffle", "label_perm"], ablations=["group1", "group2"]
        )
        runner = ValidationRunner(config)

        start_time = time.time()
        results = runner.run(large_data)
        elapsed_time = time.time() - start_time

        # Should complete within 5 seconds (Wave 4 budget)
        assert elapsed_time < 5.0

        # Check that runtime is recorded
        assert results["experiment_info"]["runtime_seconds"] < 5.0


@pytest.mark.parametrize("mode", ["oos", "purged-kfold", "holdout"])
def test_all_validation_modes(mode):
    """Test all validation modes work correctly."""
    test_data = {
        "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
        "edge_times": np.array([100, 200, 300]),
        "node_features": np.random.random((20, 10)),
        "labels": np.random.randint(0, 2, 20),
        "timestamps": np.arange(20) * 10 + 100,
        "feature_groups": {"test_group": [0, 1, 2]},
    }

    config = ValidationConfig(mode=mode, folds=3, controls=[], ablations=[])
    runner = ValidationRunner(config)

    results = runner.run(test_data)

    # Should complete successfully for all modes
    assert "main_validation" in results
    assert results["main_validation"]["split_type"] in ["oos", "purged_kfold", "holdout"]


def test_validation_consistency():
    """Test that validation produces consistent results with same seed."""
    test_data = {
        "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
        "edge_times": np.array([100, 200, 300]),
        "node_features": np.random.RandomState(42).random((10, 8)),
        "labels": np.random.RandomState(42).randint(0, 2, 10),
        "timestamps": np.arange(10) * 10 + 100,
        "feature_groups": {"test_group": [0, 1, 2]},
    }

    config1 = ValidationConfig(random_seed=123, controls=["time_shuffle"], ablations=[])
    config2 = ValidationConfig(random_seed=123, controls=["time_shuffle"], ablations=[])

    runner1 = ValidationRunner(config1)
    runner2 = ValidationRunner(config2)

    results1 = runner1.run(test_data.copy())
    results2 = runner2.run(test_data.copy())

    # Should produce similar results (within tolerance for floating point)
    main_auc1 = results1["main_validation"]["metrics"]["temporal_auc"]
    main_auc2 = results2["main_validation"]["metrics"]["temporal_auc"]

    assert abs(main_auc1 - main_auc2) < 1e-6  # Should be nearly identical
