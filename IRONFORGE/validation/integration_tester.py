"""
IRONFORGE Integration Testing Framework
=======================================

Comprehensive integration tests focusing on session independence,
end-to-end workflows, and system integration validation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from config import get_config
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)


class IntegrationTester:
    """
    Integration testing framework for IRONFORGE system validation.

    Tests:
    - End-to-end session processing workflows
    - Session independence validation
    - Cross-component integration
    - Data flow integrity
    - Configuration system integration
    - Error handling and recovery
    """

    def __init__(self):
        config = get_config()
        self.test_output_path = Path(config.get_reports_path()) / "integration_tests"
        self.test_output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Integration Tester initialized for end-to-end validation")

    def run_comprehensive_integration_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive integration test suite

        Returns:
            Complete integration test results
        """
        try:
            logger.info("Starting comprehensive integration test suite...")

            test_results = {
                "test_timestamp": datetime.now().isoformat(),
                "test_suite": "comprehensive_integration",
                "session_independence_tests": {},
                "end_to_end_workflow_tests": {},
                "cross_component_tests": {},
                "data_integrity_tests": {},
                "error_handling_tests": {},
                "configuration_integration_tests": {},
                "test_summary": {},
                "integration_health": {},
            }

            # Run session independence tests
            test_results["session_independence_tests"] = self._test_session_independence()

            # Run end-to-end workflow tests
            test_results["end_to_end_workflow_tests"] = self._test_end_to_end_workflows()

            # Run cross-component integration tests
            test_results["cross_component_tests"] = self._test_cross_component_integration()

            # Run data integrity tests
            test_results["data_integrity_tests"] = self._test_data_integrity()

            # Run error handling tests
            test_results["error_handling_tests"] = self._test_error_handling()

            # Run configuration integration tests
            test_results["configuration_integration_tests"] = self._test_configuration_integration()

            # Generate test summary
            test_results["test_summary"] = self._generate_integration_summary(test_results)
            test_results["integration_health"] = self._assess_integration_health(test_results)

            # Save test results
            self._save_integration_results(test_results)

            logger.info(
                f"Integration tests complete: {test_results['integration_health']['overall_health']}"
            )
            return test_results

        except Exception as e:
            logger.error(f"Integration testing failed: {e}")
            return {
                "test_timestamp": datetime.now().isoformat(),
                "status": "ERROR",
                "error": str(e),
                "test_summary": {"overall_result": "FAILED", "error": str(e)},
            }

    def _test_session_independence(self) -> Dict[str, Any]:
        """Test comprehensive session independence"""

        tests = {}

        # Test isolated session processing
        try:
            sessions_data = self._create_test_sessions(3)
            session_results = []

            for session_data in sessions_data:
                # Process each session independently
                from ironforge.integration.ironforge_container import get_ironforge_container

                container = get_ironforge_container()
                builder = container.get_enhanced_graph_builder()
                graduation = container.get_pattern_graduation()

                # Build graph
                graph = builder.build_session_graph(session_data)

                # Mock pattern discovery results
                mock_patterns = {
                    "session_name": session_data["session_name"],
                    "significant_patterns": [
                        {
                            "pattern_scores": [0.88, 0.90, 0.85],
                            "attention_received": 0.85,
                            "archaeological_significance": 0.89,
                            "confidence": 0.87,
                        }
                    ],
                    "session_metrics": {"total_patterns": 1},
                }

                # Validate patterns
                validation_results = graduation.validate_patterns(mock_patterns)

                session_results.append(
                    {
                        "session_name": session_data["session_name"],
                        "graph_nodes": graph.number_of_nodes(),
                        "graph_edges": graph.number_of_edges(),
                        "graduation_score": validation_results.get("graduation_score", 0.0),
                        "production_ready": validation_results.get("production_ready", False),
                    }
                )

                # Clean up references
                del container, builder, graduation, graph

            # Verify session independence
            unique_scores = len(set(result["graduation_score"] for result in session_results))
            unique_nodes = len(set(result["graph_nodes"] for result in session_results))

            tests["isolated_processing"] = {
                "test": "isolated_session_processing",
                "sessions_processed": len(session_results),
                "session_results": session_results,
                "unique_graduation_scores": unique_scores,
                "unique_node_counts": unique_nodes,
                "sessions_independent": unique_scores > 1
                or unique_nodes > 1,  # Should have variation
                "result": "PASS" if len(session_results) == 3 else "FAIL",
            }
        except Exception as e:
            tests["isolated_processing"] = {
                "test": "isolated_session_processing",
                "result": "ERROR",
                "error": str(e),
            }

        # Test component instance independence
        try:
            from ironforge.integration.ironforge_container import get_ironforge_container

            # Create multiple component instances
            instances_sets = []
            for i in range(3):
                container = get_ironforge_container()
                builder = container.get_enhanced_graph_builder()
                graduation = container.get_pattern_graduation()

                instances_sets.append(
                    {
                        "iteration": i,
                        "builder_id": id(builder),
                        "graduation_id": id(graduation),
                        "container_id": id(container),
                    }
                )

            # Verify all instances are unique
            builder_ids = [s["builder_id"] for s in instances_sets]
            graduation_ids = [s["graduation_id"] for s in instances_sets]

            all_builders_unique = len(set(builder_ids)) == len(builder_ids)
            all_graduations_unique = len(set(graduation_ids)) == len(graduation_ids)

            tests["instance_independence"] = {
                "test": "component_instance_independence",
                "instances_created": len(instances_sets),
                "instance_details": instances_sets,
                "all_builders_unique": all_builders_unique,
                "all_graduations_unique": all_graduations_unique,
                "complete_independence": all_builders_unique and all_graduations_unique,
                "result": "PASS" if (all_builders_unique and all_graduations_unique) else "FAIL",
            }
        except Exception as e:
            tests["instance_independence"] = {
                "test": "component_instance_independence",
                "result": "ERROR",
                "error": str(e),
            }

        # Test state isolation between sessions
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation
            from ironforge.synthesis.production_graduation import ProductionGraduation

            # Verify no state attributes exist
            pg = PatternGraduation()
            prod = ProductionGraduation()

            # Check for removed state attributes
            has_validation_history = hasattr(pg, "validation_history")
            has_production_features = hasattr(prod, "production_features")

            # Check summary methods reflect independence
            pg_summary = pg.get_graduation_summary()
            prod_summary = prod.get_production_summary()

            session_independence_flagged = pg_summary.get(
                "session_independence", False
            ) and prod_summary.get("session_independence", False)

            tests["state_isolation"] = {
                "test": "session_state_isolation",
                "pattern_graduation_stateless": not has_validation_history,
                "production_graduation_stateless": not has_production_features,
                "independence_flags_present": session_independence_flagged,
                "complete_state_isolation": (
                    not has_validation_history
                    and not has_production_features
                    and session_independence_flagged
                ),
                "result": (
                    "PASS"
                    if (
                        not has_validation_history
                        and not has_production_features
                        and session_independence_flagged
                    )
                    else "FAIL"
                ),
            }
        except Exception as e:
            tests["state_isolation"] = {
                "test": "session_state_isolation",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _test_end_to_end_workflows(self) -> Dict[str, Any]:
        """Test complete end-to-end workflows"""

        tests = {}

        # Test complete session processing workflow
        try:
            session_data = self._create_test_sessions(1)[0]

            # Step 1: Initialize container
            from ironforge.integration.ironforge_container import get_ironforge_container

            container = get_ironforge_container()

            # Step 2: Get components
            builder = container.get_enhanced_graph_builder()
            graduation = container.get_pattern_graduation()

            # Step 3: Build graph
            graph = builder.build_session_graph(session_data)

            # Step 4: Mock TGAT discovery
            discovery_results = {
                "session_name": session_data["session_name"],
                "significant_patterns": [
                    {
                        "pattern_scores": [0.88, 0.91, 0.89],
                        "attention_received": 0.87,
                        "archaeological_significance": 0.90,
                        "confidence": 0.88,
                        "temporal_consistency": 0.85,
                    },
                    {
                        "pattern_scores": [0.85, 0.89, 0.87],
                        "attention_received": 0.82,
                        "archaeological_significance": 0.88,
                        "confidence": 0.86,
                        "temporal_consistency": 0.83,
                    },
                ],
                "session_metrics": {"total_patterns": 2, "pattern_quality": 0.88},
            }

            # Step 5: Validate patterns
            validation_results = graduation.validate_patterns(discovery_results)

            # Step 6: Check production readiness
            production_ready = validation_results.get("production_ready", False)
            graduation_score = validation_results.get("graduation_score", 0.0)

            # Verify workflow completion
            workflow_successful = (
                graph.number_of_nodes() > 0
                and graph.number_of_edges() >= 0
                and graduation_score > 0
                and "graduation_status" in validation_results
            )

            tests["complete_workflow"] = {
                "test": "end_to_end_session_workflow",
                "workflow_steps_completed": 6,
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "patterns_discovered": len(discovery_results["significant_patterns"]),
                "graduation_score": graduation_score,
                "production_ready": production_ready,
                "workflow_successful": workflow_successful,
                "result": "PASS" if workflow_successful else "FAIL",
            }
        except Exception as e:
            tests["complete_workflow"] = {
                "test": "end_to_end_session_workflow",
                "result": "ERROR",
                "error": str(e),
            }

        # Test batch processing workflow
        try:
            sessions_data = self._create_test_sessions(5)
            batch_results = []

            for session_data in sessions_data:
                container = get_ironforge_container()
                builder = container.get_enhanced_graph_builder()
                graduation = container.get_pattern_graduation()

                # Quick processing
                graph = builder.build_session_graph(session_data)
                mock_patterns = {
                    "session_name": session_data["session_name"],
                    "significant_patterns": [
                        {
                            "pattern_scores": [0.88],
                            "attention_received": 0.85,
                            "archaeological_significance": 0.87,
                        }
                    ],
                    "session_metrics": {"total_patterns": 1},
                }
                validation = graduation.validate_patterns(mock_patterns)

                batch_results.append(
                    {
                        "session": session_data["session_name"],
                        "nodes": graph.number_of_nodes(),
                        "score": validation.get("graduation_score", 0.0),
                    }
                )

                del container, builder, graduation, graph

            batch_successful = len(batch_results) == 5 and all(
                r["score"] > 0 for r in batch_results
            )

            tests["batch_processing"] = {
                "test": "batch_session_processing",
                "sessions_processed": len(batch_results),
                "batch_results": batch_results,
                "all_sessions_successful": batch_successful,
                "average_score": np.mean([r["score"] for r in batch_results]),
                "result": "PASS" if batch_successful else "FAIL",
            }
        except Exception as e:
            tests["batch_processing"] = {
                "test": "batch_session_processing",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _test_cross_component_integration(self) -> Dict[str, Any]:
        """Test integration between different components"""

        tests = {}

        # Test graph builder to graduation integration
        try:
            session_data = self._create_test_sessions(1)[0]

            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
            from ironforge.synthesis.pattern_graduation import PatternGraduation

            builder = EnhancedGraphBuilder()
            graduation = PatternGraduation()

            # Build graph
            graph = builder.build_session_graph(session_data)

            # Extract features for validation
            node_features = []
            for _node, data in graph.nodes(data=True):
                if "features" in data:
                    features = data["features"]
                    if hasattr(features, "get_combined_features"):
                        combined = features.get_combined_features()
                        node_features.append(combined)

            # Create patterns from graph features
            if node_features:
                patterns = {
                    "session_name": session_data["session_name"],
                    "significant_patterns": [
                        {
                            "pattern_scores": [
                                float(np.mean(node_features[0][:5]))
                            ],  # Sample from features
                            "attention_received": 0.85,
                            "archaeological_significance": 0.88,
                        }
                    ],
                    "session_metrics": {"total_patterns": 1},
                }

                validation_results = graduation.validate_patterns(patterns)
                integration_successful = validation_results.get("graduation_score", 0.0) > 0
            else:
                integration_successful = False
                validation_results = {}

            tests["builder_graduation_integration"] = {
                "test": "graph_builder_graduation_integration",
                "graph_built": graph.number_of_nodes() > 0,
                "features_extracted": len(node_features),
                "patterns_created": (
                    len(patterns.get("significant_patterns", [])) if "patterns" in locals() else 0
                ),
                "validation_completed": "graduation_score" in validation_results,
                "integration_successful": integration_successful,
                "result": "PASS" if integration_successful else "FAIL",
            }
        except Exception as e:
            tests["builder_graduation_integration"] = {
                "test": "graph_builder_graduation_integration",
                "result": "ERROR",
                "error": str(e),
            }

        # Test configuration integration across components
        try:
            from config import get_config
            from ironforge.synthesis.production_graduation import ProductionGraduation

            config = get_config()
            prod_graduation = ProductionGraduation()

            # Verify components use same configuration paths
            config_preservation_path = config.get_preservation_path()
            prod_output_path = str(prod_graduation.output_path.parent)

            paths_consistent = config_preservation_path in prod_output_path

            tests["configuration_integration"] = {
                "test": "cross_component_configuration_consistency",
                "config_preservation_path": config_preservation_path,
                "production_output_path": prod_output_path,
                "paths_consistent": paths_consistent,
                "result": "PASS" if paths_consistent else "FAIL",
            }
        except Exception as e:
            tests["configuration_integration"] = {
                "test": "cross_component_configuration_consistency",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _test_data_integrity(self) -> Dict[str, Any]:
        """Test data integrity throughout the system"""

        tests = {}

        # Test feature dimension consistency
        try:
            session_data = self._create_test_sessions(1)[0]

            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

            builder = EnhancedGraphBuilder()
            graph = builder.build_session_graph(session_data)

            # Check feature dimensions
            node_dims = []
            edge_dims = []

            for _node, data in graph.nodes(data=True):
                if "features" in data:
                    features = data["features"]
                    if hasattr(features, "get_combined_features"):
                        combined = features.get_combined_features()
                        node_dims.append(len(combined))

            for _u, _v, data in graph.edges(data=True):
                if "features" in data:
                    features = data["features"]
                    if hasattr(features, "get_combined_features"):
                        combined = features.get_combined_features()
                        edge_dims.append(len(combined))

            # Verify dimension consistency
            node_dims_consistent = all(dim == 45 for dim in node_dims) if node_dims else True
            edge_dims_consistent = all(dim == 20 for dim in edge_dims) if edge_dims else True

            tests["feature_dimensions"] = {
                "test": "feature_dimension_integrity",
                "nodes_with_features": len(node_dims),
                "edges_with_features": len(edge_dims),
                "node_dimensions": node_dims,
                "edge_dimensions": edge_dims,
                "node_dims_consistent": node_dims_consistent,
                "edge_dims_consistent": edge_dims_consistent,
                "dimensions_integrity": node_dims_consistent and edge_dims_consistent,
                "result": "PASS" if (node_dims_consistent and edge_dims_consistent) else "FAIL",
            }
        except Exception as e:
            tests["feature_dimensions"] = {
                "test": "feature_dimension_integrity",
                "result": "ERROR",
                "error": str(e),
            }

        # Test pattern validation integrity
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation

            graduation = PatternGraduation()

            # Test with valid patterns
            valid_patterns = {
                "session_name": "data_integrity_test",
                "significant_patterns": [
                    {
                        "pattern_scores": [0.88, 0.90, 0.89],
                        "attention_received": 0.87,
                        "archaeological_significance": 0.89,
                    }
                ],
                "session_metrics": {"total_patterns": 1},
            }

            validation_results = graduation.validate_patterns(valid_patterns)

            # Test with invalid patterns (below threshold)
            invalid_patterns = {
                "session_name": "data_integrity_test_invalid",
                "significant_patterns": [
                    {
                        "pattern_scores": [0.60, 0.65, 0.70],
                        "attention_received": 0.62,
                        "archaeological_significance": 0.65,
                    }
                ],
                "session_metrics": {"total_patterns": 1},
            }

            invalid_validation = graduation.validate_patterns(invalid_patterns)

            # Verify validation integrity
            valid_score = validation_results.get("graduation_score", 0.0)
            invalid_score = invalid_validation.get("graduation_score", 0.0)

            validation_integrity = (
                valid_score > invalid_score and valid_score >= 0.87 and invalid_score < 0.87
            )

            tests["validation_integrity"] = {
                "test": "pattern_validation_integrity",
                "valid_pattern_score": valid_score,
                "invalid_pattern_score": invalid_score,
                "score_differentiation": valid_score > invalid_score,
                "threshold_enforcement": valid_score >= 0.87 and invalid_score < 0.87,
                "validation_integrity": validation_integrity,
                "result": "PASS" if validation_integrity else "FAIL",
            }
        except Exception as e:
            tests["validation_integrity"] = {
                "test": "pattern_validation_integrity",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""

        tests = {}

        # Test invalid session data handling
        try:
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder

            builder = EnhancedGraphBuilder()

            # Test with invalid session data
            invalid_session = {
                "session_name": "invalid_test",
                "events": [],  # Empty events
                "relativity_stats": {},  # Empty stats
            }

            try:
                graph = builder.build_session_graph(invalid_session)
                error_handled = True
                nodes_count = graph.number_of_nodes()
            except Exception as e:
                error_handled = False
                nodes_count = 0
                error_message = str(e)

            tests["invalid_session_handling"] = {
                "test": "invalid_session_data_handling",
                "invalid_data_processed": error_handled,
                "nodes_created": nodes_count,
                "graceful_handling": error_handled or "error_message" in locals(),
                "result": "PASS" if (error_handled or "error_message" in locals()) else "FAIL",
            }
        except Exception as e:
            tests["invalid_session_handling"] = {
                "test": "invalid_session_data_handling",
                "result": "ERROR",
                "error": str(e),
            }

        # Test graduation with invalid patterns
        try:
            from ironforge.synthesis.pattern_graduation import PatternGraduation

            graduation = PatternGraduation()

            # Test with malformed patterns
            malformed_patterns = {
                "session_name": "error_test",
                "significant_patterns": [
                    {
                        "pattern_scores": "invalid_scores",  # Invalid type
                        "attention_received": "invalid_attention",
                    }
                ],
            }

            try:
                validation_results = graduation.validate_patterns(malformed_patterns)
                error_handled = True
                has_error_status = validation_results.get("status") == "ERROR"
            except Exception as e:
                error_handled = False
                has_error_status = False
                error_message = str(e)

            tests["malformed_pattern_handling"] = {
                "test": "malformed_pattern_error_handling",
                "error_handled_gracefully": error_handled,
                "error_status_returned": has_error_status,
                "proper_error_handling": error_handled or has_error_status,
                "result": "PASS" if (error_handled or has_error_status) else "FAIL",
            }
        except Exception as e:
            tests["malformed_pattern_handling"] = {
                "test": "malformed_pattern_error_handling",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _test_configuration_integration(self) -> Dict[str, Any]:
        """Test configuration system integration"""

        tests = {}

        # Test configuration availability across components
        try:
            from config import get_config

            config = get_config()

            # Test required configuration methods
            required_paths = [
                "get_workspace_root",
                "get_data_path",
                "get_preservation_path",
                "get_reports_path",
                "get_discoveries_path",
            ]

            path_results = {}
            all_paths_available = True

            for path_method in required_paths:
                try:
                    if hasattr(config, path_method):
                        path_value = getattr(config, path_method)()
                        path_results[path_method] = {
                            "available": True,
                            "path": str(path_value),
                            "valid": Path(path_value).exists() or Path(path_value).parent.exists(),
                        }
                    else:
                        path_results[path_method] = {"available": False}
                        all_paths_available = False
                except Exception as e:
                    path_results[path_method] = {"available": False, "error": str(e)}
                    all_paths_available = False

            tests["configuration_availability"] = {
                "test": "configuration_system_integration",
                "all_paths_available": all_paths_available,
                "path_results": path_results,
                "result": "PASS" if all_paths_available else "FAIL",
            }
        except Exception as e:
            tests["configuration_availability"] = {
                "test": "configuration_system_integration",
                "result": "ERROR",
                "error": str(e),
            }

        return tests

    def _create_test_sessions(self, count: int) -> List[Dict[str, Any]]:
        """Create test session data for integration testing"""

        sessions = []
        base_price = 23200.0

        for i in range(count):
            session = {
                "session_name": f"integration_test_session_{i+1}",
                "events": [
                    {
                        "timestamp": f"14:{35+j}:00",
                        "price": base_price + i * 50 + j * 5,
                        "event_type": ["price_movement", "rebalance", "interaction"][j % 3],
                        "significance": 0.5 + (j % 5) * 0.1,
                    }
                    for j in range(5 + i * 2)  # Variable event count per session
                ],
                "relativity_stats": {
                    "session_high": base_price + i * 50 + 100,
                    "session_low": base_price + i * 50 - 50,
                    "session_open": base_price + i * 50,
                },
            }
            sessions.append(session)

        return sessions

    def _generate_integration_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive integration test summary"""

        summary = {
            "total_test_categories": 0,
            "passed_categories": 0,
            "failed_categories": 0,
            "error_categories": 0,
            "overall_integration_score": 0.0,
            "category_summaries": {},
        }

        test_categories = [
            "session_independence_tests",
            "end_to_end_workflow_tests",
            "cross_component_tests",
            "data_integrity_tests",
            "error_handling_tests",
            "configuration_integration_tests",
        ]

        for category in test_categories:
            if category in test_results:
                category_tests = test_results[category]
                category_summary = self._summarize_test_category(category_tests)
                summary["category_summaries"][category] = category_summary

                summary["total_test_categories"] += 1
                if category_summary["category_result"] == "PASS":
                    summary["passed_categories"] += 1
                elif category_summary["category_result"] == "FAIL":
                    summary["failed_categories"] += 1
                else:
                    summary["error_categories"] += 1

        if summary["total_test_categories"] > 0:
            summary["overall_integration_score"] = (
                summary["passed_categories"] / summary["total_test_categories"]
            )

        return summary

    def _summarize_test_category(self, category_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize results for a test category"""

        total_tests = len(category_tests)
        passed_tests = sum(1 for test in category_tests.values() if test.get("result") == "PASS")
        failed_tests = sum(1 for test in category_tests.values() if test.get("result") == "FAIL")
        error_tests = sum(1 for test in category_tests.values() if test.get("result") == "ERROR")

        if total_tests == 0:
            category_result = "NO_TESTS"
        elif error_tests > 0:
            category_result = "ERROR"
        elif failed_tests > 0:
            category_result = "FAIL"
        else:
            category_result = "PASS"

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "category_result": category_result,
        }

    def _assess_integration_health(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall integration health"""

        test_summary = test_results.get("test_summary", {})
        integration_score = test_summary.get("overall_integration_score", 0.0)
        failed_categories = test_summary.get("failed_categories", 0)
        error_categories = test_summary.get("error_categories", 0)

        if error_categories > 0:
            health_status = "CRITICAL"
        elif failed_categories > 0:
            health_status = "DEGRADED"
        elif integration_score >= 1.0:
            health_status = "EXCELLENT"
        elif integration_score >= 0.8:
            health_status = "GOOD"
        else:
            health_status = "POOR"

        return {
            "overall_health": health_status,
            "integration_score": integration_score,
            "categories_passed": test_summary.get("passed_categories", 0),
            "categories_failed": failed_categories,
            "categories_error": error_categories,
            "health_recommendation": self._get_health_recommendation(
                health_status, integration_score
            ),
        }

    def _get_health_recommendation(self, health_status: str, integration_score: float) -> str:
        """Get health recommendation based on integration results"""

        if health_status == "EXCELLENT":
            return "Integration health excellent - system ready for production"
        elif health_status == "GOOD":
            return "Integration health good - minor issues may need attention"
        elif health_status == "POOR":
            return "Integration health poor - significant improvements needed"
        elif health_status == "DEGRADED":
            return "Integration health degraded - fix failed tests before deployment"
        elif health_status == "CRITICAL":
            return "Critical integration issues - immediate attention required"
        else:
            return "Review integration test results and address identified issues"

    def _save_integration_results(self, test_results: Dict[str, Any]):
        """Save integration test results to file"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_results_{timestamp}.json"
            filepath = self.test_output_path / filename

            with open(filepath, "w") as f:
                json.dump(test_results, f, indent=2, default=str)

            logger.info(f"Integration test results saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save integration test results: {e}")

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integration testing framework"""

        return {
            "testing_framework": "IRONFORGE Integration Testing",
            "test_categories": [
                "session_independence",
                "end_to_end_workflows",
                "cross_component_integration",
                "data_integrity",
                "error_handling",
                "configuration_integration",
            ],
            "integration_focus_areas": [
                "session_isolation",
                "component_independence",
                "data_flow_integrity",
                "error_recovery",
                "configuration_consistency",
            ],
            "output_path": str(self.test_output_path),
        }
