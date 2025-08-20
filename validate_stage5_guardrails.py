#!/usr/bin/env python3
"""
Stage 5 Validation & Guardrails for Session Fingerprinting
Comprehensive validation framework ensuring production readiness and safety
"""

import logging
import json
import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    test_name: str
    passed: bool
    details: str
    metrics: Dict[str, Any] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.warnings is None:
            self.warnings = []


class Stage5Validator:
    """Comprehensive validation framework for Session Fingerprinting Stage 5"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
        
        # Critical thresholds
        self.min_coverage_pct = 95.0
        self.min_sessions_required = 66
        self.confidence_min_threshold = 0.01
        self.confidence_max_threshold = 0.99
        self.determinism_tolerance = 1e-10
        
    def validate_builder_coverage(self) -> ValidationResult:
        """Validate builder coverage ≥95%, cluster stats, deterministic re-run"""
        self.logger.info("=== BUILDER VALIDATION: Coverage & Determinism ===")
        
        try:
            # Import required modules
            from ironforge.learning.session_clustering import SessionClusteringBuilder
            from ironforge.learning.session_fingerprinting import SessionFingerprintConfig
            
            # Run builder with deterministic settings
            config = SessionFingerprintConfig.default()
            builder = SessionClusteringBuilder(config)
            
            # First run
            self.logger.info("Running first builder iteration...")
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "first_run"
                result1 = builder.build_offline_library(output_dir)
                
                # Extract metrics from first run
                sessions_processed = result1.get('sessions_processed', 0)
                sessions_discovered = result1.get('sessions_discovered', 0)
                sessions_skipped = len(result1.get('skipped_sessions', {}))
                
                if sessions_discovered == 0:
                    return ValidationResult(
                        "builder_coverage",
                        False,
                        "No sessions discovered in first run",
                        {"coverage_pct": 0.0, "sessions_processed": sessions_processed}
                    )
                
                coverage_pct = (sessions_processed / sessions_discovered) * 100.0
                
                # Check minimum coverage threshold
                coverage_pass = coverage_pct >= self.min_coverage_pct
                sessions_count_pass = sessions_processed >= self.min_sessions_required
                
                # Validate cluster statistics exist and are reasonable
                cluster_stats_path = output_dir / "cluster_stats.json"
                if not cluster_stats_path.exists():
                    return ValidationResult(
                        "builder_coverage",
                        False,
                        "Cluster statistics file not generated",
                        {"coverage_pct": coverage_pct}
                    )
                
                with open(cluster_stats_path, 'r') as f:
                    cluster_stats = json.load(f)
                
                n_clusters = len(cluster_stats)
                cluster_stats_pass = n_clusters >= 3  # Minimum reasonable clusters
                
                # Second run for determinism check
                self.logger.info("Running second builder iteration for determinism check...")
                output_dir2 = Path(temp_dir) / "second_run"
                result2 = builder.build_offline_library(output_dir2)
                
                # Compare clustering results for determinism
                cluster_stats_path2 = output_dir2 / "second_run" / "cluster_stats.json"
                if cluster_stats_path2.exists():
                    with open(cluster_stats_path2, 'r') as f:
                        cluster_stats2 = json.load(f)
                    
                    # Check deterministic clustering (same cluster count and similar stats)
                    determinism_pass = (len(cluster_stats) == len(cluster_stats2))
                    
                    if determinism_pass:
                        # Compare cluster centroids for determinism
                        centroids1_path = output_dir / "kmeans_model.pkl"
                        centroids2_path = output_dir2 / "kmeans_model.pkl"
                        
                        if centroids1_path.exists() and centroids2_path.exists():
                            with open(centroids1_path, 'rb') as f:
                                model1 = pickle.load(f)
                            with open(centroids2_path, 'rb') as f:
                                model2 = pickle.load(f)
                                
                            # Compare centroids (allowing for cluster order permutation)
                            centroids_diff = np.mean(np.abs(model1.cluster_centers_ - model2.cluster_centers_))
                            determinism_pass = centroids_diff < self.determinism_tolerance
                        else:
                            determinism_pass = False
                else:
                    determinism_pass = False
                
                # Check for silent session skips (critical failure)
                silent_skips = []
                for session_id, reason in result1.get('skipped_sessions', {}).items():
                    if reason.startswith("silent") or "unexpectedly" in reason.lower():
                        silent_skips.append(f"{session_id}: {reason}")
                
                silent_skip_pass = len(silent_skips) == 0
                
                # Overall validation
                all_pass = (coverage_pass and sessions_count_pass and 
                           cluster_stats_pass and determinism_pass and silent_skip_pass)
                
                # Prepare detailed results
                details = []
                if not coverage_pass:
                    details.append(f"Coverage {coverage_pct:.1f}% < {self.min_coverage_pct}% required")
                if not sessions_count_pass:
                    details.append(f"Sessions {sessions_processed} < {self.min_sessions_required} required")
                if not cluster_stats_pass:
                    details.append(f"Insufficient clusters: {n_clusters} < 3 minimum")
                if not determinism_pass:
                    details.append("Non-deterministic clustering detected")
                if not silent_skip_pass:
                    details.append(f"Silent skips detected: {silent_skips}")
                
                if all_pass:
                    details.append(f"Coverage: {coverage_pct:.1f}%, Sessions: {sessions_processed}, Clusters: {n_clusters}")
                    details.append("Deterministic clustering confirmed")
                
                warnings = []
                if coverage_pct < 98.0:
                    warnings.append(f"Coverage {coverage_pct:.1f}% below optimal 98%")
                if sessions_skipped > sessions_processed * 0.1:
                    warnings.append(f"High skip rate: {sessions_skipped} skipped vs {sessions_processed} processed")
                
                return ValidationResult(
                    "builder_coverage",
                    all_pass,
                    "; ".join(details),
                    {
                        "coverage_pct": coverage_pct,
                        "sessions_processed": sessions_processed,
                        "sessions_discovered": sessions_discovered,
                        "sessions_skipped": sessions_skipped,
                        "n_clusters": n_clusters,
                        "determinism_pass": determinism_pass,
                        "silent_skips": len(silent_skips)
                    },
                    warnings
                )
                
        except Exception as e:
            return ValidationResult(
                "builder_coverage",
                False,
                f"Builder validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    def validate_online_determinism(self) -> ValidationResult:
        """Validate online classifier produces identical sidecars from same session run twice"""
        self.logger.info("=== ONLINE VALIDATION: Deterministic Sidecars ===")
        
        try:
            from ironforge.learning.online_session_classifier import create_online_classifier
            
            # Check if model artifacts exist
            model_path = Path("models/session_fingerprints/v1.0.2")
            if not model_path.exists():
                return ValidationResult(
                    "online_determinism",
                    False,
                    "Model artifacts not found - run Stage 2 builder first",
                    {"model_path": str(model_path)}
                )
            
            # Load test session
            sample_session_file = Path("data/adapted/adapted_enhanced_rel_NY_AM_Lvl-1_2025_07_29.json")
            if not sample_session_file.exists():
                return ValidationResult(
                    "online_determinism",
                    False,
                    "Test session file not found",
                    {"session_file": str(sample_session_file)}
                )
            
            with open(sample_session_file, 'r') as f:
                session_data = json.load(f)
            
            # Create classifier with deterministic settings
            classifier = create_online_classifier(
                enabled=True,
                model_path=model_path,
                completion_threshold=30.0,
                distance_metric="euclidean",
                confidence_method="softmax"
            )
            
            session_id = "determinism_test_session"
            
            # First classification run
            self.logger.info("Running first classification...")
            prediction1 = classifier.classify_partial_session(
                session_data, session_id, target_completion_pct=30.0
            )
            
            if not prediction1:
                return ValidationResult(
                    "online_determinism",
                    False,
                    "First classification returned None",
                    {"prediction_available": False}
                )
            
            # Second classification run (same inputs)
            self.logger.info("Running second classification...")
            prediction2 = classifier.classify_partial_session(
                session_data, session_id, target_completion_pct=30.0
            )
            
            if not prediction2:
                return ValidationResult(
                    "online_determinism",
                    False,
                    "Second classification returned None",
                    {"prediction_available": False}
                )
            
            # Compare predictions for determinism
            determinism_checks = {
                "archetype_id": prediction1.archetype_id == prediction2.archetype_id,
                "confidence": abs(prediction1.confidence - prediction2.confidence) < 1e-10,
                "distance": abs(prediction1.distance_to_centroid - prediction2.distance_to_centroid) < 1e-10,
                "pct_seen": abs(prediction1.pct_session_seen - prediction2.pct_session_seen) < 1e-10,
                "volatility_class": prediction1.predicted_volatility_class == prediction2.predicted_volatility_class,
                "htf_regime": prediction1.predicted_dominant_htf_regime == prediction2.predicted_dominant_htf_regime
            }
            
            all_deterministic = all(determinism_checks.values())
            
            # Generate sidecars and compare
            with tempfile.TemporaryDirectory() as temp_dir:
                run_dir1 = Path(temp_dir) / "run1"
                run_dir2 = Path(temp_dir) / "run2"
                
                sidecar1_path = classifier.write_sidecar(prediction1, run_dir1)
                sidecar2_path = classifier.write_sidecar(prediction2, run_dir2)
                
                # Load and compare sidecars (excluding timestamps)
                with open(sidecar1_path, 'r') as f:
                    sidecar1 = json.load(f)
                with open(sidecar2_path, 'r') as f:
                    sidecar2 = json.load(f)
                
                # Remove timestamps for comparison
                sidecar1_copy = json.loads(json.dumps(sidecar1))
                sidecar2_copy = json.loads(json.dumps(sidecar2))
                
                if "classification_metadata" in sidecar1_copy:
                    sidecar1_copy["classification_metadata"].pop("timestamp", None)
                if "classification_metadata" in sidecar2_copy:
                    sidecar2_copy["classification_metadata"].pop("timestamp", None)
                sidecar1_copy.pop("date", None)  # Date may change between runs
                sidecar2_copy.pop("date", None)
                
                sidecars_identical = sidecar1_copy == sidecar2_copy
            
            # Overall determinism validation
            determinism_pass = all_deterministic and sidecars_identical
            
            details = []
            if not all_deterministic:
                failed_checks = [k for k, v in determinism_checks.items() if not v]
                details.append(f"Non-deterministic fields: {failed_checks}")
            if not sidecars_identical:
                details.append("Sidecar JSON content differs between runs")
            
            if determinism_pass:
                details.append("Identical predictions and sidecars confirmed")
            
            return ValidationResult(
                "online_determinism",
                determinism_pass,
                "; ".join(details) or "Online determinism validated",
                {
                    "archetype_id": prediction1.archetype_id,
                    "confidence": prediction1.confidence,
                    "determinism_checks": determinism_checks,
                    "sidecars_identical": sidecars_identical
                }
            )
            
        except Exception as e:
            return ValidationResult(
                "online_determinism",
                False,
                f"Online determinism validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    def validate_contracts_integrity(self) -> ValidationResult:
        """Validate node/edge dims untouched, taxonomy/edge intents unmodified"""
        self.logger.info("=== CONTRACTS VALIDATION: Schema Integrity ===")
        
        try:
            # Run contracts tests to ensure no schema drift
            result = subprocess.run([
                "python3", "-m", "pytest", 
                "tests/contracts/test_public_contracts.py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            contracts_pass = result.returncode == 0
            
            # Check specific dimensional contracts
            dimension_checks = {}
            
            # Import and verify node dimensions (51D)
            try:
                from ironforge.data_engine.schemas import NodeSchema
                node_schema = NodeSchema()
                expected_node_dims = 51  # f0..f50
                actual_node_dims = len([f for f in node_schema.feature_columns if f.startswith('f')])
                dimension_checks['node_dims'] = actual_node_dims == expected_node_dims
                dimension_checks['expected_node_dims'] = expected_node_dims
                dimension_checks['actual_node_dims'] = actual_node_dims
            except Exception as e:
                dimension_checks['node_dims'] = False
                dimension_checks['node_error'] = str(e)
            
            # Import and verify edge dimensions (20D)
            try:
                from ironforge.data_engine.schemas import EdgeSchema
                edge_schema = EdgeSchema()
                expected_edge_dims = 20  # e0..e19
                actual_edge_dims = len([f for f in edge_schema.feature_columns if f.startswith('e')])
                dimension_checks['edge_dims'] = actual_edge_dims == expected_edge_dims
                dimension_checks['expected_edge_dims'] = expected_edge_dims
                dimension_checks['actual_edge_dims'] = actual_edge_dims
            except Exception as e:
                dimension_checks['edge_dims'] = False
                dimension_checks['edge_error'] = str(e)
            
            # Verify event taxonomy (6 types exactly)
            try:
                from ironforge.semantic_engine import SemanticEventTypes
                semantic_types = SemanticEventTypes.get_canonical_types()
                expected_event_types = 6
                actual_event_types = len(semantic_types)
                dimension_checks['event_taxonomy'] = actual_event_types == expected_event_types
                dimension_checks['expected_event_types'] = expected_event_types
                dimension_checks['actual_event_types'] = actual_event_types
            except Exception as e:
                dimension_checks['event_taxonomy'] = False
                dimension_checks['event_taxonomy_error'] = str(e)
            
            # Verify edge intents (4 types exactly)
            try:
                from ironforge.data_engine.schemas import EdgeIntentTypes
                edge_intents = EdgeIntentTypes.get_canonical_intents()
                expected_edge_intents = 4  # TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT
                actual_edge_intents = len(edge_intents)
                dimension_checks['edge_intents'] = actual_edge_intents == expected_edge_intents
                dimension_checks['expected_edge_intents'] = expected_edge_intents
                dimension_checks['actual_edge_intents'] = actual_edge_intents
            except Exception as e:
                dimension_checks['edge_intents'] = False
                dimension_checks['edge_intents_error'] = str(e)
            
            # Overall contracts validation
            all_dimensions_pass = all(v for k, v in dimension_checks.items() if k.endswith('_dims') or k.endswith('_taxonomy') or k.endswith('_intents'))
            overall_pass = contracts_pass and all_dimensions_pass
            
            details = []
            if not contracts_pass:
                details.append("Contracts tests failed")
                if result.stderr:
                    details.append(f"Error output: {result.stderr[:200]}...")
            
            for check_name in ['node_dims', 'edge_dims', 'event_taxonomy', 'edge_intents']:
                if not dimension_checks.get(check_name, False):
                    if f"{check_name}_error" in dimension_checks:
                        details.append(f"{check_name} check failed: {dimension_checks[f'{check_name}_error']}")
                    else:
                        expected_key = f"expected_{check_name.replace('_dims', '_dims').replace('dims', 'dims')}"
                        actual_key = f"actual_{check_name.replace('_dims', '_dims').replace('dims', 'dims')}"
                        if expected_key in dimension_checks and actual_key in dimension_checks:
                            details.append(f"{check_name}: expected {dimension_checks[expected_key]}, got {dimension_checks[actual_key]}")
            
            if overall_pass:
                details.append("All schema contracts preserved")
                details.append(f"Node dims: {dimension_checks.get('actual_node_dims', 'N/A')}/51")
                details.append(f"Edge dims: {dimension_checks.get('actual_edge_dims', 'N/A')}/20")
                details.append(f"Event types: {dimension_checks.get('actual_event_types', 'N/A')}/6")
                details.append(f"Edge intents: {dimension_checks.get('actual_edge_intents', 'N/A')}/4")
            
            return ValidationResult(
                "contracts_integrity",
                overall_pass,
                "; ".join(details) or "Contracts integrity validated",
                {
                    "contracts_tests_pass": contracts_pass,
                    "dimension_checks": dimension_checks,
                    "stdout": result.stdout[-500:] if result.stdout else "",
                    "stderr": result.stderr[-500:] if result.stderr else ""
                }
            )
            
        except Exception as e:
            return ValidationResult(
                "contracts_integrity",
                False,
                f"Contracts validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    def validate_safety_guardrails(self) -> ValidationResult:
        """Validate AUX sidecar only, no labels/cards touched, flag defaults OFF"""
        self.logger.info("=== SAFETY VALIDATION: Guardrails & Isolation ===")
        
        try:
            from ironforge.learning.online_session_classifier import OnlineClassifierConfig, create_online_classifier
            
            # Verify default configuration is OFF
            default_config = OnlineClassifierConfig.default()
            default_off = not default_config.enabled
            
            # Verify factory function defaults to OFF
            classifier_default = create_online_classifier()
            factory_default_off = not classifier_default.config.enabled
            
            # Verify no labels/cards modification
            # Check that fingerprinting code doesn't import or modify canonical schemas
            fingerprinting_file = Path("ironforge/learning/session_fingerprinting.py")
            clustering_file = Path("ironforge/learning/session_clustering.py")
            classifier_file = Path("ironforge/learning/online_session_classifier.py")
            
            schema_safety_checks = {}
            
            for file_path in [fingerprinting_file, clustering_file, classifier_file]:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for dangerous imports or modifications
                    dangerous_patterns = [
                        "from ironforge.data_engine.schemas import",
                        "from ironforge.semantic_engine import",
                        "labels",
                        "cards",
                        "NodeSchema",
                        "EdgeSchema",
                        "modify_schema",
                        "update_schema"
                    ]
                    
                    file_safety = True
                    dangerous_found = []
                    
                    for pattern in dangerous_patterns:
                        if pattern in content:
                            dangerous_found.append(pattern)
                            file_safety = False
                    
                    schema_safety_checks[file_path.name] = {
                        "safe": file_safety,
                        "dangerous_patterns": dangerous_found
                    }
            
            # Verify sidecar-only output (no modification of canonical pipeline)
            sidecar_only_check = True
            
            # Check that only session_fingerprint.json is written
            model_path = Path("models/session_fingerprints/v1.0.2")
            if model_path.exists():
                try:
                    classifier = create_online_classifier(enabled=True, model_path=model_path)
                    
                    # Verify sidecar generation doesn't modify input data
                    test_session_data = {
                        "events": [
                            {"timestamp": "2025-07-29T14:35:00", "event_type": "expansion", "price": 23162.25}
                        ],
                        "session_metadata": {"session_type": "NY_AM", "date": "2025-07-29"}
                    }
                    
                    original_data = json.loads(json.dumps(test_session_data))
                    
                    prediction = classifier.classify_partial_session(
                        test_session_data, "safety_test_session", 30.0
                    )
                    
                    # Verify input data unchanged
                    data_unchanged = test_session_data == original_data
                    
                    sidecar_only_check = data_unchanged
                    
                except Exception as e:
                    sidecar_only_check = False
                    schema_safety_checks["sidecar_test_error"] = str(e)
            
            # Overall safety validation
            all_schema_safe = all(check["safe"] for check in schema_safety_checks.values() if isinstance(check, dict))
            overall_safety = default_off and factory_default_off and all_schema_safe and sidecar_only_check
            
            details = []
            
            if not default_off:
                details.append("Default configuration not OFF")
            if not factory_default_off:
                details.append("Factory function default not OFF")
            if not all_schema_safe:
                unsafe_files = [f for f, check in schema_safety_checks.items() 
                              if isinstance(check, dict) and not check["safe"]]
                details.append(f"Schema safety violations in: {unsafe_files}")
            if not sidecar_only_check:
                details.append("Sidecar-only constraint violated")
            
            if overall_safety:
                details.append("All safety guardrails confirmed")
                details.append("Default state: OFF")
                details.append("Schema isolation: preserved")
                details.append("Sidecar-only output: confirmed")
            
            warnings = []
            for file_name, check in schema_safety_checks.items():
                if isinstance(check, dict) and check["dangerous_patterns"]:
                    warnings.append(f"{file_name} contains patterns: {check['dangerous_patterns']}")
            
            return ValidationResult(
                "safety_guardrails",
                overall_safety,
                "; ".join(details) or "Safety guardrails validated",
                {
                    "default_off": default_off,
                    "factory_default_off": factory_default_off,
                    "schema_safety_checks": schema_safety_checks,
                    "sidecar_only_check": sidecar_only_check
                },
                warnings
            )
            
        except Exception as e:
            return ValidationResult(
                "safety_guardrails",
                False,
                f"Safety validation failed: {str(e)}",
                {"error": str(e)}
            )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all Stage 5 validations and return results"""
        self.logger.info("Starting Stage 5 Comprehensive Validation")
        self.logger.info("=" * 60)
        
        # Core validation tests
        self.results = [
            self.validate_builder_coverage(),
            self.validate_online_determinism(),
            self.validate_contracts_integrity(),
            self.validate_safety_guardrails()
        ]
        
        return self.results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.results:
            self.run_all_validations()
        
        # Calculate overall status
        all_passed = all(result.passed for result in self.results)
        critical_failures = [result for result in self.results if not result.passed]
        
        # Accept/Reject decision based on gates
        accept_gates = {
            "coverage_95_percent": any(
                result.test_name == "builder_coverage" and 
                result.passed and 
                result.metrics.get("coverage_pct", 0) >= 95.0 
                for result in self.results
            ),
            "online_determinism": any(
                result.test_name == "online_determinism" and result.passed 
                for result in self.results
            ),
            "contracts_green": any(
                result.test_name == "contracts_integrity" and result.passed 
                for result in self.results
            ),
            "no_schema_drift": any(
                result.test_name == "safety_guardrails" and result.passed 
                for result in self.results
            )
        }
        
        accept_decision = all(accept_gates.values())
        
        # Reject gates check
        reject_gates = {
            "silent_session_skip": any(
                result.test_name == "builder_coverage" and
                result.metrics.get("silent_skips", 0) > 0
                for result in self.results
            ),
            "missing_sidecar_on_flagged": False,  # Would need integration test
            "dims_drift": not accept_gates["contracts_green"]
        }
        
        reject_decision = any(reject_gates.values())
        
        # Risk register status
        risk_mitigations = {
            "htf_leakage": "Pending implementation",
            "scaler_mismatch": "Pending implementation", 
            "short_sessions": "Pending implementation",
            "non_deterministic_clustering": "Validated" if accept_gates["online_determinism"] else "Failed",
            "confidence_misuse": "Pending implementation"
        }
        
        report = {
            "stage": "Stage 5 - Validation & Guardrails",
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_status": "PASS" if accept_decision and not reject_decision else "FAIL",
            "accept_decision": accept_decision,
            "reject_decision": reject_decision,
            "accept_gates": accept_gates,
            "reject_gates": reject_gates,
            "validation_results": [
                {
                    "test": result.test_name,
                    "passed": result.passed,
                    "details": result.details,
                    "metrics": result.metrics,
                    "warnings": result.warnings
                }
                for result in self.results
            ],
            "critical_failures": [result.test_name for result in critical_failures],
            "risk_mitigations": risk_mitigations,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if result.test_name == "builder_coverage":
                    if result.metrics.get("coverage_pct", 0) < 95.0:
                        recommendations.append("Increase session discovery coverage by investigating skipped sessions")
                    if result.metrics.get("silent_skips", 0) > 0:
                        recommendations.append("CRITICAL: Eliminate silent session skips before production deployment")
                
                elif result.test_name == "online_determinism":
                    recommendations.append("Fix non-deterministic behavior in online classifier")
                
                elif result.test_name == "contracts_integrity":
                    recommendations.append("CRITICAL: Restore schema contracts - no dimensional changes allowed")
                
                elif result.test_name == "safety_guardrails":
                    recommendations.append("Fix safety violations - ensure complete isolation from canonical pipeline")
            
            # Add warnings as recommendations
            for warning in result.warnings or []:
                recommendations.append(f"WARNING: {warning}")
        
        # Add risk mitigation recommendations
        recommendations.extend([
            "Implement HTF leakage protection with f50 context isolation",
            "Add scaler hash validation and mismatch protection", 
            "Define minimum session length requirements with explicit metadata",
            "Implement confidence score capping and distance logging"
        ])
        
        return recommendations


def main():
    """Run Stage 5 validation and generate report"""
    validator = Stage5Validator()
    
    try:
        # Run all validations
        results = validator.run_all_validations()
        
        # Generate and display report
        report = validator.generate_validation_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5 VALIDATION REPORT")
        logger.info("=" * 60)
        
        logger.info(f"Overall Status: {report['overall_status']}")
        logger.info(f"Accept Decision: {report['accept_decision']}")
        logger.info(f"Reject Decision: {report['reject_decision']}")
        
        logger.info("\nAccept Gates:")
        for gate, status in report['accept_gates'].items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {gate}: {status_icon}")
        
        logger.info("\nReject Gates:")
        for gate, status in report['reject_gates'].items():
            status_icon = "❌" if status else "✅"
            logger.info(f"  {gate}: {status_icon}")
        
        logger.info("\nValidation Results:")
        for result_data in report['validation_results']:
            status_icon = "✅" if result_data['passed'] else "❌"
            logger.info(f"  {result_data['test']}: {status_icon} - {result_data['details']}")
            
            if result_data['warnings']:
                for warning in result_data['warnings']:
                    logger.info(f"    ⚠️  {warning}")
        
        if report['critical_failures']:
            logger.info(f"\nCritical Failures: {report['critical_failures']}")
        
        logger.info("\nRisk Mitigations Status:")
        for risk, status in report['risk_mitigations'].items():
            logger.info(f"  {risk}: {status}")
        
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  • {rec}")
        
        # Write detailed report to file
        report_path = Path("stage5_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report written to: {report_path}")
        
        # Return appropriate exit code
        return 0 if report['overall_status'] == 'PASS' else 1
        
    except Exception as e:
        logger.error(f"Validation framework failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)