"""
Wave 4 Validation Examples for IRONFORGE
========================================
Comprehensive examples demonstrating validation rails usage for temporal
pattern discovery evaluation with time-series safe methods.
"""

from pathlib import Path

import numpy as np

from ironforge.validation.controls import (
    create_control_variants,
    label_permutation,
    time_shuffle_edges,
)
from ironforge.validation.metrics import (
    archaeological_significance,
    motif_half_life,
    pattern_stability_score,
    precision_at_k,
    temporal_auc,
)

# Import Wave 4 validation components
from ironforge.validation.runner import ValidationConfig, ValidationRunner
from ironforge.validation.splits import PurgedKFold, oos_split, temporal_train_test_split


def create_synthetic_validation_data(n_nodes=500, n_edges=2000, seed=42):
    """Create realistic synthetic data for validation examples."""
    np.random.seed(seed)

    # Create temporal graph data
    edge_index = np.random.randint(0, n_nodes, (2, n_edges))
    edge_times = np.sort(np.random.randint(100000, 200000, n_edges))

    # 45D node features (matching IRONFORGE schema)
    node_features = np.random.random((n_nodes, 45))

    # Add some structure to make realistic patterns
    # HTF proximity features (0-4): correlation with time
    time_factor = (edge_times - edge_times.min()) / (edge_times.max() - edge_times.min())
    for i in range(5):
        node_features[:, i] = (
            0.3 * time_factor[: len(node_features[:, i])] + 0.7 * node_features[:, i]
        )

    # Cycle features (5-9): periodic patterns
    for i in range(5, 10):
        cycle_phase = 2 * np.pi * np.arange(n_nodes) / 50  # 50-sample cycle
        node_features[:, i] = 0.5 * np.sin(cycle_phase + i) + 0.5

    # Binary labels with some correlation to features
    feature_sum = np.sum(node_features[:, :10], axis=1)
    label_probs = 1.0 / (1.0 + np.exp(-(feature_sum - np.median(feature_sum))))
    labels = (np.random.random(n_nodes) < label_probs).astype(int)

    # Sample timestamps
    timestamps = np.sort(np.random.randint(100000, 200000, n_nodes))

    # Feature groups for ablation studies
    feature_groups = {
        "htf_prox": list(range(5)),  # HTF proximity features
        "cycles": list(range(5, 10)),  # Cycle analysis features
        "structure": list(range(10, 15)),  # Market structure features
        "semantic": list(range(15, 20)),  # Semantic event features
        "technical": list(range(20, 25)),  # Technical indicators
        "volume": list(range(25, 30)),  # Volume analysis features
        "volatility": list(range(30, 35)),  # Volatility features
        "momentum": list(range(35, 40)),  # Momentum features
        "mean_reversion": list(range(40, 45)),  # Mean reversion features
    }

    return {
        "edge_index": edge_index,
        "edge_times": edge_times,
        "node_features": node_features,
        "labels": labels,
        "timestamps": timestamps,
        "feature_groups": feature_groups,
    }


def example_1_basic_validation():
    """Example 1: Basic out-of-sample validation."""
    print("=" * 60)
    print("Example 1: Basic Out-of-Sample Validation")
    print("=" * 60)

    # Create synthetic data
    data = create_synthetic_validation_data(n_nodes=300, n_edges=1000)

    # Configure basic OOS validation
    config = ValidationConfig(
        mode="oos",
        controls=["time_shuffle", "label_perm"],
        ablations=[],  # No ablations for basic example
        random_seed=42,
    )

    # Run validation
    runner = ValidationRunner(config)
    results = runner.run(data)

    # Display results
    summary = results["summary"]
    main_perf = summary["main_performance"]

    print(f"Validation Status: {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}")
    print(f"Main AUC: {main_perf['temporal_auc']:.4f}")
    print(f"Precision@20: {main_perf['precision_at_20']:.4f}")
    print(f"Pattern Stability: {main_perf['pattern_stability']:.4f}")

    # Control comparisons
    print("\nControl Comparisons:")
    for control_name, control_data in summary["control_comparison"].items():
        performance_drop = control_data["performance_drop"]
        status = "✅ Pass" if control_data["passes_check"] else "❌ Fail"
        print(f"  {control_name}: -{performance_drop:.4f} AUC ({status})")

    # Recommendations
    if summary["recommendations"]:
        print("\nRecommendations:")
        for rec in summary["recommendations"]:
            print(f"  ⚠️  {rec}")

    return results


def example_2_purged_kfold():
    """Example 2: Purged K-fold validation with embargo."""
    print("\n" + "=" * 60)
    print("Example 2: Purged K-Fold with Embargo")
    print("=" * 60)

    # Create larger dataset for k-fold
    data = create_synthetic_validation_data(n_nodes=800, n_edges=3000)

    # Configure purged k-fold validation
    config = ValidationConfig(
        mode="purged-kfold",
        folds=5,
        embargo_mins=60,  # 1-hour embargo
        controls=["time_shuffle", "label_perm", "node_shuffle"],
        ablations=["htf_prox", "cycles"],
        random_seed=42,
    )

    # Run validation
    runner = ValidationRunner(config)
    results = runner.run(data)

    # Display fold-specific results
    main_results = results["main_validation"]
    print(f"Split Type: {main_results['split_type']}")
    print(f"Number of Folds: {main_results['n_folds']}")
    print(f"Embargo Period: {main_results['embargo_mins']} minutes")

    # Aggregated metrics
    metrics = main_results["metrics"]
    print("\nAggregated Performance:")
    print(f"  Temporal AUC: {metrics['temporal_auc_mean']:.4f} ± {metrics['temporal_auc_std']:.4f}")
    print(
        f"  Precision@20: {metrics['precision_at_20_mean']:.4f} ± {metrics['precision_at_20_std']:.4f}"
    )
    print(f"  Range: [{metrics['temporal_auc_min']:.4f}, {metrics['temporal_auc_max']:.4f}]")

    # Fold details
    print("\nFold Details:")
    for fold_result in main_results["fold_results"]:
        fold_metrics = fold_result["metrics"]
        print(
            f"  Fold {fold_result['fold']}: AUC={fold_metrics['temporal_auc']:.4f}, "
            f"Train={len(fold_result['train_indices'])}, Test={len(fold_result['test_indices'])}"
        )

    # Ablation impact
    print("\nAblation Studies:")
    for ablation_name, ablation_data in results["summary"]["ablation_impact"].items():
        importance = ablation_data["feature_importance"]
        print(f"  {ablation_name}: {importance:.4f} importance")

    return results


def example_3_comprehensive_validation():
    """Example 3: Comprehensive validation with all controls and ablations."""
    print("\n" + "=" * 60)
    print("Example 3: Comprehensive Validation Suite")
    print("=" * 60)

    # Create realistic dataset
    data = create_synthetic_validation_data(n_nodes=1000, n_edges=4000)

    # Configure comprehensive validation
    config = ValidationConfig(
        mode="purged-kfold",
        folds=7,
        embargo_mins=120,  # 2-hour embargo
        controls=[
            "time_shuffle",
            "label_perm",
            "node_shuffle",
            "edge_direction",
            "temporal_blocks",
        ],
        ablations=["htf_prox", "cycles", "structure", "semantic", "technical"],
        random_seed=123,
    )

    # Run comprehensive validation
    runner = ValidationRunner(config)
    results = runner.run(data)

    # Detailed analysis
    summary = results["summary"]

    print("Comprehensive Validation Results:")
    print(f"Status: {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}")

    # Main performance breakdown
    main_perf = summary["main_performance"]
    print("\nMain Performance:")
    print(f"  Temporal AUC: {main_perf['temporal_auc']:.4f}")
    print(f"  Precision@20: {main_perf['precision_at_20']:.4f}")
    print(f"  Pattern Stability: {main_perf['pattern_stability']:.4f}")

    # Comprehensive control analysis
    print("\nNegative Control Analysis:")
    for control_name, control_data in summary["control_comparison"].items():
        main_auc = control_data["main_auc"]
        control_auc = control_data["control_auc"]
        drop = control_data["performance_drop"]
        status = "✅" if control_data["passes_check"] else "❌"

        print(f"  {control_name:15s}: {main_auc:.4f} → {control_auc:.4f} (Δ{drop:+.4f}) {status}")

    # Feature importance ranking
    print("\nFeature Group Importance Ranking:")
    ablation_impact = summary["ablation_impact"]
    sorted_ablations = sorted(
        ablation_impact.items(), key=lambda x: x[1]["feature_importance"], reverse=True
    )

    for rank, (group_name, group_data) in enumerate(sorted_ablations, 1):
        importance = group_data["feature_importance"]
        print(f"  {rank}. {group_name:15s}: {importance:.4f}")

    return results


def example_4_individual_components():
    """Example 4: Using individual validation components."""
    print("\n" + "=" * 60)
    print("Example 4: Individual Component Usage")
    print("=" * 60)

    # Create small dataset for demonstration
    n_samples = 200
    timestamps = np.arange(n_samples) * 60 + 100000  # 1-minute intervals
    y_true = np.random.randint(0, 2, n_samples)
    y_score = np.random.random(n_samples)

    print("A. Time-Series Safe Splits")
    print("-" * 30)

    # Demonstrate different splitting methods
    train_idx, test_idx = oos_split(timestamps, cutoff_ts=timestamps[140])
    print(f"OOS Split: {len(train_idx)} train, {len(test_idx)} test")

    train_idx, test_idx = temporal_train_test_split(timestamps, test_size=0.3, embargo_mins=30)
    print(f"Temporal Split: {len(train_idx)} train, {len(test_idx)} test")

    splitter = PurgedKFold(n_splits=5, embargo_mins=15)
    folds = list(splitter.split(timestamps))
    print(f"Purged K-Fold: {len(folds)} folds")
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {i}: {len(train_idx)} train, {len(test_idx)} test")

    print("\nB. Validation Metrics")
    print("-" * 30)

    # Individual metric calculations
    p_at_k = precision_at_k(y_true, y_score, k=20)
    temp_auc = temporal_auc(y_true, y_score, timestamps)
    stability = pattern_stability_score(y_score, timestamps, window_size=30)

    print(f"Precision@20: {p_at_k:.4f}")
    print(f"Temporal AUC: {temp_auc:.4f}")
    print(f"Pattern Stability: {stability:.4f}")

    # Motif half-life with hit timestamps
    hit_indices = np.where(y_true == 1)[0]
    if len(hit_indices) > 1:
        hit_timestamps = timestamps[hit_indices]
        half_life = motif_half_life(hit_timestamps)
        print(f"Motif Half-Life: {half_life:.2f}")

    print("\nC. Negative Controls")
    print("-" * 30)

    # Create simple graph data
    edge_index = np.array([[0, 1, 2, 1], [1, 2, 0, 0]])
    edge_times = np.array([100, 200, 300, 400])
    node_features = np.random.random((3, 10))
    labels = np.array([1, 0, 1])

    # Individual control operations
    _, shuffled_times = time_shuffle_edges(edge_index, edge_times, seed=42)
    permuted_labels = label_permutation(labels, seed=42)

    print(f"Original edge times: {edge_times}")
    print(f"Shuffled edge times: {shuffled_times}")
    print(f"Original labels: {labels}")
    print(f"Permuted labels: {permuted_labels}")

    # Create control variants
    controls = ["time_shuffle", "label_perm", "node_shuffle"]
    variants = create_control_variants(
        edge_index, edge_times, node_features, labels, controls, seed=42
    )

    print(f"\nControl Variants Created: {list(variants.keys())}")
    for control_name, variant in variants.items():
        print(f"  {control_name}: {variant['description']}")

    print("\nD. Archaeological Significance")
    print("-" * 30)

    # Archaeological pattern analysis
    pattern_scores = np.array([0.85, 0.92, 0.78, 0.88, 0.81])
    pattern_types = np.array(["fvg", "poi", "fvg", "bos", "liq"])
    temporal_spans = np.array([15.5, 22.3, 12.1, 18.7, 20.2])

    arch_metrics = archaeological_significance(pattern_scores, pattern_types, temporal_spans)

    print(f"Pattern Diversity Index: {arch_metrics['diversity_index']:.4f}")
    print(f"Temporal Coverage: {arch_metrics['temporal_coverage']:.4f}")
    print(f"Pattern Density: {arch_metrics['pattern_density']:.4f}")
    print(f"Significance-Weighted Score: {arch_metrics['significance_weighted_score']:.4f}")

    return arch_metrics


def example_5_cli_equivalent():
    """Example 5: Programmatic equivalent of CLI usage."""
    print("\n" + "=" * 60)
    print("Example 5: CLI Equivalent Programmatic Usage")
    print("=" * 60)

    # This example shows how CLI commands translate to programmatic usage

    print("CLI Command:")
    print("ironforge sdk validate \\")
    print("    --data-path ./data/validation \\")
    print("    --mode purged-kfold \\")
    print("    --folds 5 \\")
    print("    --embargo-mins 30 \\")
    print("    --controls time_shuffle label_perm \\")
    print("    --ablations htf_prox cycles \\")
    print("    --report-dir ./reports/validation \\")
    print("    --seed 42")

    print("\nProgrammatic Equivalent:")
    print("-" * 30)

    # Create validation data (would normally load from data_path)
    data = create_synthetic_validation_data(n_nodes=400, n_edges=1500)

    # Configure exactly as CLI would
    config = ValidationConfig(
        mode="purged-kfold",
        folds=5,
        embargo_mins=30,
        controls=["time_shuffle", "label_perm"],
        ablations=["htf_prox", "cycles"],
        report_dir=Path("./reports/validation"),
        random_seed=42,
    )

    # Run validation
    runner = ValidationRunner(config)
    results = runner.run(data)

    # Output equivalent to CLI
    summary = results["summary"]
    main_perf = summary["main_performance"]

    print("🚀 Starting IRONFORGE validation with purged-kfold mode...")
    print("📊 Configuration: 5 folds, 30min embargo")
    print("🧪 Controls: time_shuffle, label_perm")
    print("🔬 Ablations: htf_prox, cycles")
    print("📁 Reports will be saved to: ./reports/validation")
    print("")
    print("✅ Validation completed!")
    print(f"📈 Main AUC: {main_perf['temporal_auc']:.4f}")
    print(f"🎯 Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")

    return results


def example_6_batch_validation():
    """Example 6: Batch validation across multiple datasets."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Validation Workflow")
    print("=" * 60)

    # Simulate multiple trading sessions or datasets
    session_configs = [
        {"name": "ASIA_Session", "n_nodes": 300, "n_edges": 1000, "seed": 101},
        {"name": "LONDON_Session", "n_nodes": 500, "n_edges": 1800, "seed": 102},
        {"name": "NY_AM_Session", "n_nodes": 700, "n_edges": 2500, "seed": 103},
        {"name": "NY_PM_Session", "n_nodes": 600, "n_edges": 2200, "seed": 104},
    ]

    # Configure validation for batch processing
    config = ValidationConfig(
        mode="oos",  # Faster for batch processing
        controls=["time_shuffle", "label_perm"],
        ablations=["htf_prox"],  # Limited ablations for speed
        random_seed=42,
    )

    runner = ValidationRunner(config)
    batch_results = {}

    print("Running batch validation across trading sessions...")
    print("-" * 50)

    for session_config in session_configs:
        session_name = session_config["name"]
        print(f"Processing {session_name}...")

        # Generate session data
        session_data = create_synthetic_validation_data(
            n_nodes=session_config["n_nodes"],
            n_edges=session_config["n_edges"],
            seed=session_config["seed"],
        )

        # Run validation
        results = runner.run(session_data)
        batch_results[session_name] = results

        # Quick status
        summary = results["summary"]
        status = "✅ PASSED" if summary["validation_passed"] else "❌ FAILED"
        auc = summary["main_performance"]["temporal_auc"]
        print(f"  {session_name}: {status} (AUC: {auc:.4f})")

    # Aggregate batch results
    print("\nBatch Validation Summary:")
    print("-" * 30)

    passed_sessions = []
    failed_sessions = []
    auc_scores = []

    for session_name, results in batch_results.items():
        summary = results["summary"]
        auc = summary["main_performance"]["temporal_auc"]
        auc_scores.append(auc)

        if summary["validation_passed"]:
            passed_sessions.append(session_name)
        else:
            failed_sessions.append(session_name)

    print(f"Sessions Passed: {len(passed_sessions)}/{len(session_configs)}")
    print(f"Average AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"AUC Range: [{np.min(auc_scores):.4f}, {np.max(auc_scores):.4f}]")

    if failed_sessions:
        print(f"Failed Sessions: {', '.join(failed_sessions)}")

    return batch_results


def example_7_production_pipeline():
    """Example 7: Production validation pipeline integration."""
    print("\n" + "=" * 60)
    print("Example 7: Production Validation Pipeline")
    print("=" * 60)

    def production_validation_gate(model_results, validation_data):
        """Production validation gate for model deployment."""
        print("🔍 Running production validation gate...")

        # Configure strict validation for production
        config = ValidationConfig(
            mode="purged-kfold",
            folds=7,  # More folds for robust validation
            embargo_mins=60,
            controls=["time_shuffle", "label_perm", "node_shuffle"],
            ablations=["htf_prox", "cycles", "structure"],
            random_seed=42,
        )

        runner = ValidationRunner(config)
        results = runner.run(validation_data)

        # Strict quality gates for production
        summary = results["summary"]
        main_perf = summary["main_performance"]

        # Production thresholds
        MIN_AUC = 0.70
        MIN_STABILITY = 0.40
        MIN_CONTROL_GAP = 0.08

        quality_checks = {
            "auc_threshold": main_perf["temporal_auc"] >= MIN_AUC,
            "stability_threshold": main_perf["pattern_stability"] >= MIN_STABILITY,
            "control_gaps": all(
                ctrl["performance_drop"] >= MIN_CONTROL_GAP
                for ctrl in summary["control_comparison"].values()
            ),
        }

        print("Quality Gate Results:")
        for check_name, passed in quality_checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check_name}: {status}")

        deployment_approved = all(quality_checks.values())

        if deployment_approved:
            print("🚀 Model APPROVED for production deployment")
            # log_validation_success(results)
            return True, results
        else:
            print("🚫 Model REJECTED - quality gates failed")
            # log_validation_failure(results)
            return False, results

    # Simulate production scenario
    print("Simulating production model validation...")

    # Create production-like validation data
    production_data = create_synthetic_validation_data(n_nodes=1200, n_edges=5000, seed=999)

    # Mock model results (would come from actual model)
    mock_model_results = {
        "model_version": "ironforge_v2.1.0",
        "training_data": "2025_Q1_sessions",
        "architecture": "TGAT_enhanced",
    }

    # Run production validation gate
    approved, validation_results = production_validation_gate(mock_model_results, production_data)

    # Production deployment decision
    if approved:
        print("\n📦 Proceeding with model deployment...")
        print("  - Updating production model weights")
        print("  - Notifying monitoring systems")
        print("  - Scheduling validation monitoring")
    else:
        print("\n🔧 Model requires improvement before deployment:")
        for rec in validation_results["summary"]["recommendations"]:
            print(f"  - {rec}")

    return approved, validation_results


def main():
    """Run all Wave 4 validation examples."""
    print("IRONFORGE Wave 4 Validation Examples")
    print("=" * 80)
    print("Comprehensive demonstration of validation rails for temporal pattern discovery.")
    print("=" * 80)

    # Track execution time
    import time

    start_time = time.time()

    # Run all examples
    examples = [
        example_1_basic_validation,
        example_2_purged_kfold,
        example_3_comprehensive_validation,
        example_4_individual_components,
        example_5_cli_equivalent,
        example_6_batch_validation,
        example_7_production_pipeline,
    ]

    example_results = {}

    for example_func in examples:
        try:
            result = example_func()
            example_results[example_func.__name__] = result
        except Exception as e:
            print(f"❌ {example_func.__name__} failed: {e}")
            example_results[example_func.__name__] = None

    # Summary
    elapsed_time = time.time() - start_time
    successful_examples = sum(1 for result in example_results.values() if result is not None)

    print("\n" + "=" * 80)
    print("Wave 4 Validation Examples Summary")
    print("=" * 80)
    print(f"Examples completed: {successful_examples}/{len(examples)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Performance budget: {'✅ PASSED' if elapsed_time < 30.0 else '❌ EXCEEDED'} (<30s)")

    print("\nExample Status:")
    for example_name, result in example_results.items():
        status = "✅ SUCCESS" if result is not None else "❌ FAILED"
        print(f"  {example_name}: {status}")

    print("\nWave 4 validation framework ready for production use! 🚀")

    return example_results


if __name__ == "__main__":
    # Run examples when script is executed directly
    results = main()
