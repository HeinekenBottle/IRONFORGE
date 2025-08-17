"""Performance tests for validation components (Wave 4)."""

import gc
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from ironforge.validation.controls import create_control_variants
from ironforge.validation.metrics import compute_validation_metrics
from ironforge.validation.runner import ValidationConfig, ValidationRunner
from ironforge.validation.splits import PurgedKFold, oos_split


class TestValidationPerformance:
    """Performance tests for validation system with Wave 4 budgets."""

    def setup_method(self):
        """Set up performance test environment."""
        # Force garbage collection before each test
        gc.collect()

        # Record baseline memory
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def check_memory_budget(self, max_memory_mb: float = 150.0):
        """Check that memory usage is within budget."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.baseline_memory

        assert (
            memory_used < max_memory_mb
        ), f"Memory usage {memory_used:.1f}MB exceeds budget {max_memory_mb}MB"

        return memory_used

    def test_purged_kfold_performance(self):
        """Test PurgedKFold performance with large dataset."""
        # Create large synthetic dataset
        n_samples = 10000
        timestamps = np.arange(n_samples) * 60 + 1000000  # 1 minute intervals

        splitter = PurgedKFold(n_splits=5, embargo_mins=30)

        start_time = time.time()

        splits = list(splitter.split(timestamps))

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(50.0)  # Lower budget for splits

        # Performance requirements
        assert elapsed_time < 1.0, f"PurgedKFold took {elapsed_time:.2f}s, should be <1s"
        assert len(splits) == 5, "Should produce correct number of splits"

        print(f"PurgedKFold: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_oos_split_performance(self):
        """Test OOS split performance with large dataset."""
        # Create large synthetic dataset
        n_samples = 50000
        timestamps = np.arange(n_samples) * 10 + 1000000
        cutoff_ts = timestamps[int(0.7 * len(timestamps))]

        start_time = time.time()

        train_idx, test_idx = oos_split(timestamps, cutoff_ts)

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(30.0)

        # Performance requirements
        assert elapsed_time < 0.5, f"OOS split took {elapsed_time:.2f}s, should be <0.5s"
        assert len(train_idx) + len(test_idx) == len(timestamps)

        print(f"OOS split: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_controls_performance(self):
        """Test negative controls performance with moderate dataset."""
        # Create synthetic graph data
        n_nodes = 1000
        n_edges = 5000

        edge_index = np.random.randint(0, n_nodes, (2, n_edges))
        edge_times = np.random.randint(100000, 200000, n_edges)
        node_features = np.random.random((n_nodes, 45))  # 45D features
        labels = np.random.randint(0, 2, n_nodes)

        controls = ["time_shuffle", "label_perm", "node_shuffle", "edge_direction"]

        start_time = time.time()

        variants = create_control_variants(
            edge_index, edge_times, node_features, labels, controls, seed=42
        )

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(100.0)

        # Performance requirements
        assert elapsed_time < 2.0, f"Controls took {elapsed_time:.2f}s, should be <2s"
        assert len(variants) == len(controls)

        print(f"Controls creation: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_metrics_computation_performance(self):
        """Test metrics computation performance with large dataset."""
        # Create large synthetic prediction data
        n_samples = 20000

        y_true = np.random.randint(0, 2, n_samples)
        y_score = np.random.random(n_samples)
        timestamps = np.arange(n_samples) * 30 + 1000000

        # Pattern metadata for archaeological metrics
        pattern_metadata = {
            "scores": np.random.random(1000),
            "types": np.random.choice(["fvg", "poi", "bos", "liq"], 1000),
            "spans": np.random.uniform(5.0, 60.0, 1000),
        }

        start_time = time.time()

        metrics = compute_validation_metrics(
            y_true,
            y_score,
            timestamps,
            pattern_metadata=pattern_metadata,
            k_values=[5, 10, 20, 50, 100],
        )

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(80.0)

        # Performance requirements
        assert elapsed_time < 1.5, f"Metrics took {elapsed_time:.2f}s, should be <1.5s"
        assert len(metrics) >= 10, "Should compute multiple metrics"

        print(f"Metrics computation: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_full_validation_performance(self):
        """Test complete validation pipeline performance."""
        # Create realistic synthetic dataset
        n_nodes = 500
        n_edges = 2000
        n_samples = n_nodes

        test_data = {
            "edge_index": np.random.randint(0, n_nodes, (2, n_edges)),
            "edge_times": np.random.randint(100000, 200000, n_edges),
            "node_features": np.random.random((n_nodes, 45)),  # 45D features
            "labels": np.random.randint(0, 2, n_samples),
            "timestamps": np.arange(n_samples) * 60 + 1000000,
            "feature_groups": {
                "htf_prox": list(range(5)),
                "cycles": list(range(5, 10)),
                "structure": list(range(10, 15)),
            },
        }

        config = ValidationConfig(
            mode="oos",
            controls=["time_shuffle", "label_perm"],
            ablations=["htf_prox", "cycles"],
            random_seed=42,
        )

        runner = ValidationRunner(config)

        start_time = time.time()

        results = runner.run(test_data)

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(150.0)  # Full budget

        # Performance requirements (Wave 4 budget: <5s, <150MB)
        assert elapsed_time < 5.0, f"Full validation took {elapsed_time:.2f}s, should be <5s"

        # Check result completeness
        assert "experiment_info" in results
        assert "main_validation" in results
        assert "negative_controls" in results
        assert "ablation_studies" in results
        assert "summary" in results

        print(f"Full validation: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_kfold_validation_performance(self):
        """Test purged k-fold validation performance."""
        # Moderate dataset for k-fold
        n_nodes = 300
        n_samples = n_nodes

        test_data = {
            "edge_index": np.random.randint(0, n_nodes, (2, 1000)),
            "edge_times": np.random.randint(100000, 200000, 1000),
            "node_features": np.random.random((n_nodes, 30)),
            "labels": np.random.randint(0, 2, n_samples),
            "timestamps": np.arange(n_samples) * 120 + 1000000,  # 2-minute intervals
            "feature_groups": {"test_group": list(range(10))},
        }

        config = ValidationConfig(
            mode="purged-kfold",
            folds=5,
            embargo_mins=60,
            controls=["time_shuffle"],
            ablations=["test_group"],
            random_seed=42,
        )

        runner = ValidationRunner(config)

        start_time = time.time()

        results = runner.run(test_data)

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(150.0)

        # K-fold should complete within budget
        assert elapsed_time < 5.0, f"K-fold validation took {elapsed_time:.2f}s, should be <5s"

        # Check k-fold specific results
        main_results = results["main_validation"]
        assert main_results["split_type"] == "purged_kfold"
        assert "fold_results" in main_results

        print(f"K-fold validation: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    def test_batch_processing_performance(self):
        """Test batch processing performance across multiple runs."""
        # Simulate processing multiple datasets
        n_datasets = 10
        dataset_size = 200

        config = ValidationConfig(
            mode="oos", controls=["time_shuffle"], ablations=[], random_seed=42
        )
        runner = ValidationRunner(config)

        start_time = time.time()

        for i in range(n_datasets):
            test_data = {
                "edge_index": np.random.randint(0, dataset_size, (2, 500)),
                "edge_times": np.random.randint(100000, 200000, 500),
                "node_features": np.random.random((dataset_size, 20)),
                "labels": np.random.randint(0, 2, dataset_size),
                "timestamps": np.arange(dataset_size) * 30 + 1000000 + i * 100000,
                "feature_groups": {},
            }

            results = runner.run(test_data)

            # Each run should be fast
            batch_time = time.time() - start_time
            per_batch_time = batch_time / (i + 1)

            # Average per batch should be under 100ms (Wave 4 budget)
            if i > 0:  # Skip first iteration (warmup)
                assert per_batch_time < 0.1, f"Avg batch time {per_batch_time:.3f}s exceeds 100ms"

        total_time = time.time() - start_time
        memory_used = self.check_memory_budget(150.0)

        print(f"Batch processing ({n_datasets} runs): {total_time:.3f}s, {memory_used:.1f}MB")
        print(f"Average per batch: {total_time/n_datasets:.3f}s")

    def test_memory_cleanup_performance(self):
        """Test that memory is properly cleaned up between runs."""
        config = ValidationConfig(
            mode="oos", controls=["time_shuffle", "label_perm"], ablations=[], random_seed=42
        )
        runner = ValidationRunner(config)

        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Run multiple validations
        for i in range(5):
            test_data = {
                "edge_index": np.random.randint(0, 300, (2, 800)),
                "edge_times": np.random.randint(100000, 200000, 800),
                "node_features": np.random.random((300, 30)),
                "labels": np.random.randint(0, 2, 300),
                "timestamps": np.arange(300) * 60 + 1000000,
                "feature_groups": {},
            }

            results = runner.run(test_data)

            # Force garbage collection
            del results
            del test_data
            gc.collect()

            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory

            # Memory growth should be bounded
            assert memory_growth < 100.0, f"Memory growth {memory_growth:.1f}MB after {i+1} runs"

        final_memory = self.process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        print(f"Memory growth after 5 runs: {total_growth:.1f}MB")

    def test_edge_case_performance(self):
        """Test performance with edge cases."""
        config = ValidationConfig(mode="oos", controls=[], ablations=[])
        runner = ValidationRunner(config)

        # Test with minimal data
        start_time = time.time()

        minimal_data = {
            "edge_index": np.array([[0], [0]]),
            "edge_times": np.array([100]),
            "node_features": np.random.random((2, 5)),
            "labels": np.array([1, 0]),
            "timestamps": np.array([100, 200]),
            "feature_groups": {},
        }

        results = runner.run(minimal_data)

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(50.0)

        # Edge cases should be very fast
        assert elapsed_time < 1.0, f"Edge case took {elapsed_time:.2f}s, should be <1s"

        print(f"Edge case performance: {elapsed_time:.3f}s, {memory_used:.1f}MB")

    @pytest.mark.slow
    def test_stress_performance(self):
        """Stress test with large dataset (marked as slow)."""
        # Large dataset for stress testing
        n_nodes = 2000
        n_edges = 10000

        test_data = {
            "edge_index": np.random.randint(0, n_nodes, (2, n_edges)),
            "edge_times": np.random.randint(100000, 300000, n_edges),
            "node_features": np.random.random((n_nodes, 45)),
            "labels": np.random.randint(0, 2, n_nodes),
            "timestamps": np.arange(n_nodes) * 30 + 1000000,
            "feature_groups": {
                "htf_prox": list(range(15)),
                "cycles": list(range(15, 30)),
                "structure": list(range(30, 45)),
            },
        }

        config = ValidationConfig(
            mode="oos",
            controls=["time_shuffle", "label_perm", "node_shuffle"],
            ablations=["htf_prox", "cycles", "structure"],
            random_seed=42,
        )

        runner = ValidationRunner(config)

        start_time = time.time()

        results = runner.run(test_data)

        elapsed_time = time.time() - start_time
        memory_used = self.check_memory_budget(150.0)

        # Even stress test should complete within budget
        assert elapsed_time < 10.0, f"Stress test took {elapsed_time:.2f}s, should be <10s"

        print(f"Stress test: {elapsed_time:.3f}s, {memory_used:.1f}MB")


@pytest.mark.parametrize(
    "n_samples,expected_time",
    [
        (100, 0.1),
        (1000, 0.5),
        (5000, 2.0),
    ],
)
def test_scalability_performance(n_samples, expected_time):
    """Test performance scalability with different dataset sizes."""
    # Create dataset of specified size
    test_data = {
        "edge_index": np.random.randint(0, n_samples // 2, (2, n_samples)),
        "edge_times": np.random.randint(100000, 200000, n_samples),
        "node_features": np.random.random((n_samples // 2, 20)),
        "labels": np.random.randint(0, 2, n_samples // 2),
        "timestamps": np.arange(n_samples // 2) * 60 + 1000000,
        "feature_groups": {"test": list(range(10))},
    }

    config = ValidationConfig(mode="oos", controls=["time_shuffle"], ablations=[])
    runner = ValidationRunner(config)

    start_time = time.time()

    results = runner.run(test_data)

    elapsed_time = time.time() - start_time

    # Should scale reasonably
    assert (
        elapsed_time < expected_time
    ), f"Size {n_samples} took {elapsed_time:.2f}s, expected <{expected_time}s"

    print(f"Size {n_samples}: {elapsed_time:.3f}s")


def test_report_generation_performance():
    """Test that report generation doesn't significantly impact performance."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        config = ValidationConfig(
            mode="oos", controls=["time_shuffle"], ablations=[], report_dir=Path(temp_dir)
        )
        runner = ValidationRunner(config)

        test_data = {
            "edge_index": np.random.randint(0, 100, (2, 300)),
            "edge_times": np.random.randint(100000, 200000, 300),
            "node_features": np.random.random((100, 20)),
            "labels": np.random.randint(0, 2, 100),
            "timestamps": np.arange(100) * 60 + 1000000,
            "feature_groups": {},
        }

        start_time = time.time()

        results = runner.run(test_data)

        elapsed_time = time.time() - start_time

        # Report generation should not add significant overhead
        assert elapsed_time < 2.0, f"With reports took {elapsed_time:.2f}s, should be <2s"

        # Check that reports were created
        report_files = list(Path(temp_dir).glob("*"))
        assert len(report_files) >= 2  # JSON + HTML

        print(f"With report generation: {elapsed_time:.3f}s")
