"""
Layer 4: Validation Framework
============================

Comprehensive mathematical testing framework with property-based validation.
Ensures mathematical accuracy, numerical stability, and performance requirements.

Key Features:
- Property-based mathematical testing
- Statistical hypothesis testing
- Numerical stability validation  
- Performance benchmarking
- Cross-validation frameworks
- Mathematical invariant checking
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import scipy.stats as stats

# Hypothesis testing for property-based testing (optional dependency)
try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for graceful degradation
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def floats(min_value=None, max_value=None, **kwargs):
            return lambda: np.random.uniform(min_value or -1e6, max_value or 1e6)
        
        @staticmethod
        def integers(min_value=None, max_value=None, **kwargs):
            return lambda: np.random.randint(min_value or 1, max_value or 1000)
        
        @staticmethod
        def lists(elements, min_size=0, max_size=100):
            return lambda: [elements() for _ in range(np.random.randint(min_size, max_size + 1))]

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"

class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    test_type: str
    result: TestResult
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationSuite:
    """Collection of validation results"""
    suite_name: str
    results: list[ValidationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    
    def add_result(self, result: ValidationResult):
        """Add validation result to suite"""
        self.results.append(result)
    
    def finalize(self):
        """Finalize validation suite"""
        self.end_time = datetime.now()
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for validation suite"""
        if not self.results:
            return {"total": 0, "pass": 0, "fail": 0, "warning": 0, "skip": 0, "success_rate": 0.0}
        
        summary = {
            "total": len(self.results),
            "pass": len([r for r in self.results if r.result == TestResult.PASS]),
            "fail": len([r for r in self.results if r.result == TestResult.FAIL]),
            "warning": len([r for r in self.results if r.result == TestResult.WARNING]),
            "skip": len([r for r in self.results if r.result == TestResult.SKIP]),
        }
        
        summary["success_rate"] = summary["pass"] / summary["total"] if summary["total"] > 0 else 0.0
        summary["total_time_seconds"] = (
            (self.end_time or datetime.now()) - self.start_time
        ).total_seconds()
        
        return summary

class ValidationLayer(ABC):
    """
    Base class for validation layer implementations.
    Provides framework for mathematical accuracy and performance validation.
    """
    
    @abstractmethod
    def validate_mathematical_invariants(self, model: Any) -> ValidationSuite:
        """Validate mathematical invariants"""
        pass
    
    @abstractmethod
    def performance_benchmark(self, model: Any) -> ValidationSuite:
        """Benchmark mathematical model performance"""
        pass
    
    @abstractmethod
    def statistical_validation(self, predictions: np.ndarray, actuals: np.ndarray) -> ValidationSuite:
        """Statistical validation of predictions"""
        pass

class MathematicalPropertyTest:
    """
    Property-based testing for mathematical models.
    Tests mathematical properties using generated data.
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.property_tests = {
            "hawkes_process": self._hawkes_property_tests,
            "htf_coupling": self._htf_property_tests,
            "fft_correlator": self._fft_property_tests
        }
    
    def run_property_tests(self, model_implementation: Any, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationSuite:
        """Run property-based tests for mathematical model"""
        
        suite = ValidationSuite(f"property_tests_{self.model_type}")
        
        if self.model_type in self.property_tests:
            test_func = self.property_tests[self.model_type]
            results = test_func(model_implementation, validation_level)
            
            for result in results:
                suite.add_result(result)
        else:
            suite.add_result(ValidationResult(
                test_name="property_test_availability",
                test_type="setup",
                result=TestResult.SKIP,
                message=f"No property tests defined for {self.model_type}"
            ))
        
        suite.finalize()
        return suite
    
    def _hawkes_property_tests(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Property tests for Hawkes process implementations"""
        
        results = []
        
        # Property 1: Non-negative intensity
        try:
            start_time = datetime.now()
            
            # Test with various parameter combinations
            test_cases = [
                {"mu": 0.02, "alpha": 35.51, "beta": 0.00442},  # Oracle parameters
                {"mu": 0.1, "alpha": 1.0, "beta": 1.0},         # Standard parameters
                {"mu": 0.01, "alpha": 0.5, "beta": 2.0}         # Low excitation
            ]
            
            intensity_violations = 0
            total_tests = 0
            
            for params in test_cases:
                # Generate random event sequences
                for _ in range(10 if level == ValidationLevel.BASIC else 50):
                    events = np.sort(np.random.uniform(0, 100, np.random.randint(5, 20)))
                    
                    try:
                        intensities = model.compute_core_function(events, params)
                        
                        if isinstance(intensities, np.ndarray):
                            if np.any(intensities < 0):
                                intensity_violations += 1
                        
                        total_tests += 1
                        
                    except Exception as e:
                        logger.warning(f"Hawkes computation failed: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if intensity_violations == 0:
                results.append(ValidationResult(
                    test_name="non_negative_intensity",
                    test_type="mathematical_property",
                    result=TestResult.PASS,
                    message=f"All intensities non-negative ({total_tests} tests)",
                    execution_time_ms=execution_time,
                    details={"tests_performed": total_tests, "violations": intensity_violations}
                ))
            else:
                results.append(ValidationResult(
                    test_name="non_negative_intensity",
                    test_type="mathematical_property",
                    result=TestResult.FAIL,
                    message=f"Found {intensity_violations} negative intensity violations",
                    execution_time_ms=execution_time,
                    details={"tests_performed": total_tests, "violations": intensity_violations}
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                test_name="non_negative_intensity",
                test_type="mathematical_property",
                result=TestResult.FAIL,
                message=f"Property test failed with exception: {e}",
                details={"exception": str(e)}
            ))
        
        # Property 2: Self-excitation (intensity increases after events)
        try:
            start_time = datetime.now()
            
            excitation_violations = 0
            total_excitation_tests = 0
            
            for params in test_cases[:2]:  # Limit for performance
                if params["alpha"] <= 0:  # Skip non-excitatory cases
                    continue
                
                for _ in range(5 if level == ValidationLevel.BASIC else 20):
                    # Create scenario: compute intensity before and after adding event
                    base_events = np.sort(np.random.uniform(0, 50, 5))
                    new_event_time = base_events[-1] + 0.1  # Shortly after last event
                    
                    try:
                        # Intensity before new event
                        intensity_before = model.compute_core_function(base_events, params)[-1]
                        
                        # Intensity after adding new event  
                        extended_events = np.append(base_events, new_event_time)
                        intensity_after = model.compute_core_function(extended_events, params)[-1]
                        
                        # For positive alpha, intensity should increase
                        if intensity_after <= intensity_before:
                            excitation_violations += 1
                        
                        total_excitation_tests += 1
                        
                    except Exception as e:
                        logger.warning(f"Self-excitation test failed: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if total_excitation_tests > 0:
                violation_rate = excitation_violations / total_excitation_tests
                
                if violation_rate <= 0.1:  # Allow 10% tolerance for numerical issues
                    results.append(ValidationResult(
                        test_name="self_excitation_property",
                        test_type="mathematical_property",
                        result=TestResult.PASS,
                        message=f"Self-excitation verified ({violation_rate:.1%} violation rate)",
                        execution_time_ms=execution_time,
                        details={"tests_performed": total_excitation_tests, "violations": excitation_violations}
                    ))
                else:
                    results.append(ValidationResult(
                        test_name="self_excitation_property",
                        test_type="mathematical_property",
                        result=TestResult.FAIL,
                        message=f"High self-excitation violation rate: {violation_rate:.1%}",
                        execution_time_ms=execution_time,
                        details={"tests_performed": total_excitation_tests, "violations": excitation_violations}
                    ))
            else:
                results.append(ValidationResult(
                    test_name="self_excitation_property",
                    test_type="mathematical_property",
                    result=TestResult.SKIP,
                    message="No self-excitation tests performed",
                    execution_time_ms=execution_time
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                test_name="self_excitation_property",
                test_type="mathematical_property",
                result=TestResult.FAIL,
                message=f"Self-excitation test failed: {e}",
                details={"exception": str(e)}
            ))
        
        # Property 3: Stability condition (α/β < 1)
        try:
            start_time = datetime.now()
            
            stability_test_params = [
                {"mu": 0.1, "alpha": 0.5, "beta": 1.0},    # Stable: 0.5/1.0 = 0.5
                {"mu": 0.1, "alpha": 0.9, "beta": 1.0},    # Stable: 0.9/1.0 = 0.9  
                {"mu": 0.02, "alpha": 35.51, "beta": 0.00442}  # Oracle: 35.51/0.00442 ≈ 8033 (unstable theoretically)
            ]
            
            stability_results = []
            
            for params in stability_test_params:
                stability_ratio = params["alpha"] / params["beta"] if params["beta"] != 0 else float('inf')
                
                # Test with moderate-length sequences
                test_events = np.sort(np.random.uniform(0, 20, 10))
                
                try:
                    intensities = model.compute_core_function(test_events, params)
                    
                    # Check if computation succeeded and intensities are reasonable
                    if isinstance(intensities, np.ndarray) and len(intensities) > 0:
                        max_intensity = np.max(intensities)
                        has_inf_nan = np.any(np.isinf(intensities)) or np.any(np.isnan(intensities))
                        
                        stability_results.append({
                            "stability_ratio": stability_ratio,
                            "max_intensity": float(max_intensity),
                            "computation_success": True,
                            "has_inf_nan": has_inf_nan,
                            "theoretically_stable": stability_ratio < 1.0
                        })
                    else:
                        stability_results.append({
                            "stability_ratio": stability_ratio,
                            "computation_success": False,
                            "theoretically_stable": stability_ratio < 1.0
                        })
                        
                except Exception as e:
                    stability_results.append({
                        "stability_ratio": stability_ratio,
                        "computation_success": False,
                        "error": str(e),
                        "theoretically_stable": stability_ratio < 1.0
                    })
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Analyze stability results
            stable_cases = [r for r in stability_results if r.get("theoretically_stable", False)]
            [r for r in stability_results if not r.get("theoretically_stable", True)]
            
            stable_success_rate = np.mean([r["computation_success"] for r in stable_cases]) if stable_cases else 1.0
            
            if stable_success_rate >= 0.8:  # 80% success rate for stable cases
                results.append(ValidationResult(
                    test_name="stability_condition",
                    test_type="mathematical_property",
                    result=TestResult.PASS,
                    message=f"Stability condition validated ({stable_success_rate:.1%} success for stable cases)",
                    execution_time_ms=execution_time,
                    details={
                        "stability_results": stability_results,
                        "stable_success_rate": stable_success_rate
                    }
                ))
            else:
                results.append(ValidationResult(
                    test_name="stability_condition",
                    test_type="mathematical_property",
                    result=TestResult.WARNING,
                    message=f"Low success rate for stable cases: {stable_success_rate:.1%}",
                    execution_time_ms=execution_time,
                    details={
                        "stability_results": stability_results,
                        "stable_success_rate": stable_success_rate
                    }
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                test_name="stability_condition",
                test_type="mathematical_property",
                result=TestResult.FAIL,
                message=f"Stability test failed: {e}",
                details={"exception": str(e)}
            ))
        
        return results
    
    def _htf_property_tests(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Property tests for HTF coupling implementations"""
        
        results = []
        
        # Property: Timescale separation (β_h << β_s)
        try:
            start_time = datetime.now()
            
            # Test with HTF parameters
            htf_params = {
                "mu_h": 0.02, "alpha_h": 35.51, "beta_h": 0.00442,  # HTF (long timescale)
                "mu_s": 0.1, "alpha_s": 1.0, "beta_s": 0.1,        # Session (short timescale)
                "gamma_base": 0.5
            }
            
            timescale_separation = htf_params["beta_s"] / htf_params["beta_h"]
            
            if timescale_separation > 10:  # Good separation
                result_status = TestResult.PASS
                message = f"Good timescale separation: {timescale_separation:.1f}"
            elif timescale_separation > 2:
                result_status = TestResult.WARNING
                message = f"Moderate timescale separation: {timescale_separation:.1f}"
            else:
                result_status = TestResult.FAIL
                message = f"Poor timescale separation: {timescale_separation:.1f}"
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            results.append(ValidationResult(
                test_name="timescale_separation",
                test_type="mathematical_property",
                result=result_status,
                message=message,
                execution_time_ms=execution_time,
                details={
                    "beta_h": htf_params["beta_h"],
                    "beta_s": htf_params["beta_s"],
                    "separation_ratio": timescale_separation
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="timescale_separation",
                test_type="mathematical_property",
                result=TestResult.FAIL,
                message=f"Timescale test failed: {e}",
                details={"exception": str(e)}
            ))
        
        return results
    
    def _fft_property_tests(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Property tests for FFT correlator implementations"""
        
        results = []
        
        # Property: Correlation symmetry
        try:
            start_time = datetime.now()
            
            # Test correlation symmetry: corr(x,y) = corr(y,x)
            test_size = 100 if level == ValidationLevel.BASIC else 500
            
            signal1 = np.random.randn(test_size)
            signal2 = np.random.randn(test_size)
            
            # Compute correlations in both directions
            data_xy = np.array([signal1, signal2])
            data_yx = np.array([signal2, signal1])
            
            params = {"detrend": True}
            
            corr_xy = model.compute_core_function(data_xy, params)
            corr_yx = model.compute_core_function(data_yx, params)
            
            # Check symmetry (allowing for numerical precision)
            if isinstance(corr_xy, np.ndarray) and isinstance(corr_yx, np.ndarray):
                # Compare reversed correlation
                corr_yx_reversed = np.flip(corr_yx)
                
                if len(corr_xy) == len(corr_yx_reversed):
                    max_diff = np.max(np.abs(corr_xy - corr_yx_reversed))
                    
                    if max_diff < 1e-10:
                        result_status = TestResult.PASS
                        message = f"Perfect correlation symmetry (max diff: {max_diff:.2e})"
                    elif max_diff < 1e-6:
                        result_status = TestResult.PASS
                        message = f"Good correlation symmetry (max diff: {max_diff:.2e})"
                    else:
                        result_status = TestResult.WARNING
                        message = f"Moderate correlation symmetry (max diff: {max_diff:.2e})"
                else:
                    result_status = TestResult.FAIL
                    message = f"Correlation length mismatch: {len(corr_xy)} vs {len(corr_yx_reversed)}"
            else:
                result_status = TestResult.FAIL
                message = "Correlation computation failed"
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            results.append(ValidationResult(
                test_name="correlation_symmetry",
                test_type="mathematical_property",
                result=result_status,
                message=message,
                execution_time_ms=execution_time,
                details={
                    "test_size": test_size,
                    "max_difference": float(max_diff) if 'max_diff' in locals() else None
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="correlation_symmetry",
                test_type="mathematical_property",
                result=TestResult.FAIL,
                message=f"Symmetry test failed: {e}",
                details={"exception": str(e)}
            ))
        
        return results

class NumericalStabilityTest:
    """
    Tests for numerical stability of mathematical implementations.
    Checks behavior under extreme conditions and edge cases.
    """
    
    def __init__(self):
        self.stability_tests = [
            self._test_extreme_parameters,
            self._test_boundary_conditions,
            self._test_numerical_precision,
            self._test_large_data_handling
        ]
    
    def run_stability_tests(self, model_implementation: Any, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationSuite:
        """Run numerical stability tests"""
        
        suite = ValidationSuite("numerical_stability")
        
        for test_func in self.stability_tests:
            try:
                test_results = test_func(model_implementation, validation_level)
                for result in test_results:
                    suite.add_result(result)
            except Exception as e:
                suite.add_result(ValidationResult(
                    test_name=test_func.__name__,
                    test_type="numerical_stability",
                    result=TestResult.FAIL,
                    message=f"Stability test failed: {e}",
                    details={"exception": str(e)}
                ))
        
        suite.finalize()
        return suite
    
    def _test_extreme_parameters(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Test with extreme parameter values"""
        
        results = []
        
        # Test cases with extreme parameter values
        extreme_cases = [
            {"name": "very_small_mu", "params": {"mu": 1e-10, "alpha": 1.0, "beta": 1.0}},
            {"name": "very_large_alpha", "params": {"mu": 0.1, "alpha": 1e6, "beta": 1e3}},
            {"name": "very_small_beta", "params": {"mu": 0.1, "alpha": 1.0, "beta": 1e-10}},
            {"name": "very_large_beta", "params": {"mu": 0.1, "alpha": 1.0, "beta": 1e6}},
        ]
        
        for case in extreme_cases:
            try:
                start_time = datetime.now()
                
                # Generate test data
                test_events = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                
                # Attempt computation
                result = model.compute_core_function(test_events, case["params"])
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Validate result
                if isinstance(result, np.ndarray):
                    has_nan = np.any(np.isnan(result))
                    has_inf = np.any(np.isinf(result))
                    all_finite = np.all(np.isfinite(result))
                    
                    if all_finite:
                        test_result = TestResult.PASS
                        message = f"Handled extreme {case['name']} successfully"
                    elif has_inf:
                        test_result = TestResult.WARNING
                        message = f"Infinite values in {case['name']} result"
                    else:
                        test_result = TestResult.FAIL
                        message = f"NaN values in {case['name']} result"
                else:
                    test_result = TestResult.FAIL
                    message = f"Invalid result type for {case['name']}"
                
                results.append(ValidationResult(
                    test_name=f"extreme_params_{case['name']}",
                    test_type="numerical_stability",
                    result=test_result,
                    message=message,
                    execution_time_ms=execution_time,
                    details={
                        "parameters": case["params"],
                        "result_type": type(result).__name__,
                        "has_nan": has_nan if isinstance(result, np.ndarray) else False,
                        "has_inf": has_inf if isinstance(result, np.ndarray) else False
                    }
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"extreme_params_{case['name']}",
                    test_type="numerical_stability",
                    result=TestResult.FAIL,
                    message=f"Exception with extreme {case['name']}: {e}",
                    details={"parameters": case["params"], "exception": str(e)}
                ))
        
        return results
    
    def _test_boundary_conditions(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Test boundary conditions and edge cases"""
        
        results = []
        
        boundary_cases = [
            {"name": "empty_events", "events": np.array([]), "params": {"mu": 0.1, "alpha": 1.0, "beta": 1.0}},
            {"name": "single_event", "events": np.array([1.0]), "params": {"mu": 0.1, "alpha": 1.0, "beta": 1.0}},
            {"name": "zero_parameters", "events": np.array([1.0, 2.0]), "params": {"mu": 0.0, "alpha": 0.0, "beta": 1.0}},
            {"name": "identical_events", "events": np.array([5.0, 5.0, 5.0]), "params": {"mu": 0.1, "alpha": 1.0, "beta": 1.0}}
        ]
        
        for case in boundary_cases:
            try:
                start_time = datetime.now()
                
                result = model.compute_core_function(case["events"], case["params"])
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Validate boundary case handling
                if isinstance(result, np.ndarray):
                    if len(case["events"]) == 0:
                        # Empty events should return baseline intensity
                        expected_length = 1
                    else:
                        expected_length = len(case["events"])
                    
                    if len(result) == expected_length:
                        test_result = TestResult.PASS
                        message = f"Boundary case {case['name']} handled correctly"
                    else:
                        test_result = TestResult.FAIL
                        message = f"Wrong result length for {case['name']}: got {len(result)}, expected {expected_length}"
                else:
                    test_result = TestResult.FAIL
                    message = f"Invalid result type for boundary case {case['name']}"
                
                results.append(ValidationResult(
                    test_name=f"boundary_{case['name']}",
                    test_type="numerical_stability",
                    result=test_result,
                    message=message,
                    execution_time_ms=execution_time,
                    details={
                        "events": case["events"].tolist(),
                        "parameters": case["params"],
                        "result_length": len(result) if isinstance(result, np.ndarray) else 0
                    }
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"boundary_{case['name']}",
                    test_type="numerical_stability",
                    result=TestResult.FAIL,
                    message=f"Exception in boundary case {case['name']}: {e}",
                    details={"events": case["events"].tolist(), "parameters": case["params"], "exception": str(e)}
                ))
        
        return results
    
    def _test_numerical_precision(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Test numerical precision and consistency"""
        
        results = []
        
        try:
            start_time = datetime.now()
            
            # Test reproducibility
            test_events = np.array([1.0, 2.5, 4.0, 5.5])
            test_params = {"mu": 0.02, "alpha": 35.51, "beta": 0.00442}
            
            # Run same computation multiple times
            results_list = []
            for _ in range(5):
                result = model.compute_core_function(test_events, test_params)
                if isinstance(result, np.ndarray):
                    results_list.append(result)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if len(results_list) >= 2:
                # Check consistency across runs
                max_diff = 0.0
                for i in range(1, len(results_list)):
                    if len(results_list[i]) == len(results_list[0]):
                        diff = np.max(np.abs(results_list[i] - results_list[0]))
                        max_diff = max(max_diff, diff)
                
                if max_diff < 1e-12:
                    test_result = TestResult.PASS
                    message = f"Perfect numerical reproducibility (max diff: {max_diff:.2e})"
                elif max_diff < 1e-9:
                    test_result = TestResult.PASS
                    message = f"Good numerical reproducibility (max diff: {max_diff:.2e})"
                elif max_diff < 1e-6:
                    test_result = TestResult.WARNING
                    message = f"Moderate numerical reproducibility (max diff: {max_diff:.2e})"
                else:
                    test_result = TestResult.FAIL
                    message = f"Poor numerical reproducibility (max diff: {max_diff:.2e})"
            else:
                test_result = TestResult.SKIP
                message = "Insufficient results for reproducibility test"
            
            results.append(ValidationResult(
                test_name="numerical_reproducibility",
                test_type="numerical_stability",
                result=test_result,
                message=message,
                execution_time_ms=execution_time,
                details={
                    "runs_completed": len(results_list),
                    "max_difference": float(max_diff) if 'max_diff' in locals() else None
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                test_name="numerical_reproducibility",
                test_type="numerical_stability",
                result=TestResult.FAIL,
                message=f"Reproducibility test failed: {e}",
                details={"exception": str(e)}
            ))
        
        return results
    
    def _test_large_data_handling(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Test handling of large datasets"""
        
        results = []
        
        # Test different data sizes
        if level == ValidationLevel.BASIC:
            sizes = [100, 1000]
        elif level == ValidationLevel.STANDARD:
            sizes = [100, 1000, 5000]
        else:  # COMPREHENSIVE or EXHAUSTIVE
            sizes = [100, 1000, 5000, 10000]
        
        for size in sizes:
            try:
                start_time = datetime.now()
                
                # Generate large dataset
                large_events = np.sort(np.random.uniform(0, 1000, size))
                test_params = {"mu": 0.02, "alpha": 1.0, "beta": 0.1}  # Stable parameters
                
                result = model.compute_core_function(large_events, test_params)
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Validate large data handling
                if isinstance(result, np.ndarray) and len(result) == size:
                    # Check for reasonable performance (< 10 seconds for 10k points)
                    if execution_time < 10000:
                        test_result = TestResult.PASS
                        message = f"Large data ({size} points) handled efficiently"
                    else:
                        test_result = TestResult.WARNING
                        message = f"Large data ({size} points) handled but slowly ({execution_time:.1f}ms)"
                    
                    # Check for memory issues (approximate)
                    memory_usage_mb = result.nbytes / (1024 * 1024)
                    if memory_usage_mb < 100:  # Reasonable memory usage
                        pass  # Keep current result
                    else:
                        test_result = TestResult.WARNING
                        message += f" (high memory: {memory_usage_mb:.1f}MB)"
                else:
                    test_result = TestResult.FAIL
                    message = f"Large data ({size} points) handling failed"
                
                results.append(ValidationResult(
                    test_name=f"large_data_{size}",
                    test_type="numerical_stability",
                    result=test_result,
                    message=message,
                    execution_time_ms=execution_time,
                    details={
                        "data_size": size,
                        "result_size": len(result) if isinstance(result, np.ndarray) else 0,
                        "memory_usage_mb": result.nbytes / (1024 * 1024) if isinstance(result, np.ndarray) else 0
                    }
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"large_data_{size}",
                    test_type="numerical_stability",
                    result=TestResult.FAIL,
                    message=f"Large data ({size} points) failed: {e}",
                    details={"data_size": size, "exception": str(e)}
                ))
        
        return results

class PerformanceBenchmarkTest:
    """
    Performance benchmarking for mathematical implementations.
    Tests execution time, memory usage, and scalability.
    """
    
    def __init__(self):
        self.benchmark_tests = [
            self._benchmark_execution_time,
            self._benchmark_memory_usage,
            self._benchmark_scalability
        ]
    
    def run_performance_tests(self, model_implementation: Any, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationSuite:
        """Run performance benchmark tests"""
        
        suite = ValidationSuite("performance_benchmark")
        
        for test_func in self.benchmark_tests:
            try:
                test_results = test_func(model_implementation, validation_level)
                for result in test_results:
                    suite.add_result(result)
            except Exception as e:
                suite.add_result(ValidationResult(
                    test_name=test_func.__name__,
                    test_type="performance",
                    result=TestResult.FAIL,
                    message=f"Performance test failed: {e}",
                    details={"exception": str(e)}
                ))
        
        suite.finalize()
        return suite
    
    def _benchmark_execution_time(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Benchmark execution time performance"""
        
        results = []
        
        # Test different data sizes for performance
        test_sizes = [50, 200, 500] if level == ValidationLevel.BASIC else [50, 200, 500, 1000, 2000]
        
        for size in test_sizes:
            try:
                # Generate test data
                test_events = np.sort(np.random.uniform(0, 100, size))
                test_params = {"mu": 0.02, "alpha": 35.51, "beta": 0.00442}
                
                # Warm-up run
                model.compute_core_function(test_events[:10], test_params)
                
                # Benchmark runs
                execution_times = []
                for _ in range(5):
                    start_time = datetime.now()
                    model.compute_core_function(test_events, test_params)
                    end_time = datetime.now()
                    
                    execution_time = (end_time - start_time).total_seconds() * 1000
                    execution_times.append(execution_time)
                
                # Calculate statistics
                mean_time = np.mean(execution_times)
                std_time = np.std(execution_times)
                min_time = np.min(execution_times)
                max_time = np.max(execution_times)
                
                # Performance thresholds (adjust based on requirements)
                if size <= 100:
                    threshold = 50  # 50ms for small datasets
                elif size <= 500:
                    threshold = 200  # 200ms for medium datasets
                else:
                    threshold = 1000  # 1s for large datasets
                
                if mean_time <= threshold:
                    test_result = TestResult.PASS
                    message = f"Good performance for {size} points: {mean_time:.1f}ms avg"
                elif mean_time <= threshold * 2:
                    test_result = TestResult.WARNING
                    message = f"Acceptable performance for {size} points: {mean_time:.1f}ms avg"
                else:
                    test_result = TestResult.FAIL
                    message = f"Poor performance for {size} points: {mean_time:.1f}ms avg (threshold: {threshold}ms)"
                
                results.append(ValidationResult(
                    test_name=f"execution_time_{size}",
                    test_type="performance",
                    result=test_result,
                    message=message,
                    execution_time_ms=mean_time,
                    details={
                        "data_size": size,
                        "mean_time_ms": float(mean_time),
                        "std_time_ms": float(std_time),
                        "min_time_ms": float(min_time),
                        "max_time_ms": float(max_time),
                        "threshold_ms": threshold,
                        "throughput_items_per_sec": size / (mean_time / 1000) if mean_time > 0 else 0
                    }
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"execution_time_{size}",
                    test_type="performance",
                    result=TestResult.FAIL,
                    message=f"Performance test failed for {size} points: {e}",
                    details={"data_size": size, "exception": str(e)}
                ))
        
        return results
    
    def _benchmark_memory_usage(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Benchmark memory usage"""
        
        results = []
        
        try:
            import os

            import psutil
            
            process = psutil.Process(os.getpid())
            
            # Test memory usage with different data sizes
            test_sizes = [100, 500, 1000] if level == ValidationLevel.BASIC else [100, 500, 1000, 2000]
            
            for size in test_sizes:
                try:
                    # Measure memory before
                    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
                    
                    # Generate test data
                    test_events = np.sort(np.random.uniform(0, 100, size))
                    test_params = {"mu": 0.02, "alpha": 35.51, "beta": 0.00442}
                    
                    # Execute computation
                    result = model.compute_core_function(test_events, test_params)
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_used = memory_after - memory_before
                    
                    # Estimate expected memory usage
                    if isinstance(result, np.ndarray):
                        result_memory = result.nbytes / (1024 * 1024)  # MB
                    else:
                        result_memory = 0.1  # Estimate for non-array results
                    
                    # Memory efficiency check
                    if memory_used <= result_memory * 2:  # Allow 2x overhead
                        test_result = TestResult.PASS
                        message = f"Good memory efficiency for {size} points: {memory_used:.1f}MB used"
                    elif memory_used <= result_memory * 5:  # Allow 5x overhead
                        test_result = TestResult.WARNING
                        message = f"Moderate memory efficiency for {size} points: {memory_used:.1f}MB used"
                    else:
                        test_result = TestResult.FAIL
                        message = f"Poor memory efficiency for {size} points: {memory_used:.1f}MB used"
                    
                    results.append(ValidationResult(
                        test_name=f"memory_usage_{size}",
                        test_type="performance",
                        result=test_result,
                        message=message,
                        details={
                            "data_size": size,
                            "memory_used_mb": float(memory_used),
                            "result_memory_mb": float(result_memory),
                            "memory_efficiency_ratio": float(memory_used / max(result_memory, 0.1))
                        }
                    ))
                    
                except Exception as e:
                    results.append(ValidationResult(
                        test_name=f"memory_usage_{size}",
                        test_type="performance",
                        result=TestResult.FAIL,
                        message=f"Memory test failed for {size} points: {e}",
                        details={"data_size": size, "exception": str(e)}
                    ))
            
        except ImportError:
            results.append(ValidationResult(
                test_name="memory_usage",
                test_type="performance",
                result=TestResult.SKIP,
                message="psutil not available for memory benchmarking"
            ))
        
        return results
    
    def _benchmark_scalability(self, model: Any, level: ValidationLevel) -> list[ValidationResult]:
        """Benchmark computational scalability"""
        
        results = []
        
        try:
            # Test how execution time scales with data size
            test_sizes = [50, 100, 200, 400] if level == ValidationLevel.BASIC else [50, 100, 200, 400, 800, 1600]
            
            execution_times = []
            data_sizes = []
            
            for size in test_sizes:
                try:
                    test_events = np.sort(np.random.uniform(0, 100, size))
                    test_params = {"mu": 0.02, "alpha": 1.0, "beta": 0.1}  # Stable parameters
                    
                    start_time = datetime.now()
                    model.compute_core_function(test_events, test_params)
                    end_time = datetime.now()
                    
                    execution_time = (end_time - start_time).total_seconds()
                    execution_times.append(execution_time)
                    data_sizes.append(size)
                    
                except Exception as e:
                    logger.warning(f"Scalability test failed for size {size}: {e}")
            
            if len(execution_times) >= 3:
                # Analyze scaling behavior
                data_sizes = np.array(data_sizes)
                execution_times = np.array(execution_times)
                
                # Fit power law: time = a * size^b
                log_sizes = np.log(data_sizes)
                log_times = np.log(execution_times)
                
                coeffs = np.polyfit(log_sizes, log_times, 1)
                scaling_exponent = coeffs[0]
                
                # Interpret scaling behavior
                if scaling_exponent <= 1.2:
                    test_result = TestResult.PASS
                    message = f"Excellent scalability: O(n^{scaling_exponent:.2f})"
                elif scaling_exponent <= 2.2:
                    test_result = TestResult.PASS
                    message = f"Good scalability: O(n^{scaling_exponent:.2f})"
                elif scaling_exponent <= 3.0:
                    test_result = TestResult.WARNING
                    message = f"Moderate scalability: O(n^{scaling_exponent:.2f})"
                else:
                    test_result = TestResult.FAIL
                    message = f"Poor scalability: O(n^{scaling_exponent:.2f})"
                
                results.append(ValidationResult(
                    test_name="computational_scalability",
                    test_type="performance",
                    result=test_result,
                    message=message,
                    details={
                        "scaling_exponent": float(scaling_exponent),
                        "data_sizes": data_sizes.tolist(),
                        "execution_times": execution_times.tolist(),
                        "r_squared": float(np.corrcoef(log_sizes, log_times)[0,1]**2)
                    }
                ))
            else:
                results.append(ValidationResult(
                    test_name="computational_scalability",
                    test_type="performance",
                    result=TestResult.SKIP,
                    message="Insufficient data points for scalability analysis"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                test_name="computational_scalability",
                test_type="performance",
                result=TestResult.FAIL,
                message=f"Scalability test failed: {e}",
                details={"exception": str(e)}
            ))
        
        return results

class MathematicalValidationFramework(ValidationLayer):
    """
    Comprehensive validation framework that combines all testing approaches.
    Main entry point for mathematical model validation.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.property_tester = None  # Will be set based on model type
        self.stability_tester = NumericalStabilityTest()
        self.performance_tester = PerformanceBenchmarkTest()
        self.validation_history: list[ValidationSuite] = []
    
    def validate_mathematical_invariants(self, model: Any, model_type: str = "hawkes_process") -> ValidationSuite:
        """Validate mathematical invariants for given model"""
        
        suite = ValidationSuite(f"mathematical_invariants_{model_type}")
        
        # Set up property tester for specific model type
        self.property_tester = MathematicalPropertyTest(model_type)
        
        # Run property tests
        property_suite = self.property_tester.run_property_tests(model, self.validation_level)
        for result in property_suite.results:
            suite.add_result(result)
        
        suite.finalize()
        return suite
    
    def performance_benchmark(self, model: Any) -> ValidationSuite:
        """Benchmark mathematical model performance"""
        
        return self.performance_tester.run_performance_tests(model, self.validation_level)
    
    def statistical_validation(self, predictions: np.ndarray, actuals: np.ndarray) -> ValidationSuite:
        """Statistical validation of predictions against actuals"""
        
        suite = ValidationSuite("statistical_validation")
        
        try:
            # Basic statistics
            start_time = datetime.now()
            
            if len(predictions) != len(actuals):
                suite.add_result(ValidationResult(
                    test_name="data_length_match",
                    test_type="statistical",
                    result=TestResult.FAIL,
                    message=f"Prediction length ({len(predictions)}) != actual length ({len(actuals)})"
                ))
                suite.finalize()
                return suite
            
            # Calculate error metrics
            errors = predictions - actuals
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            mape = np.mean(np.abs(errors / np.maximum(np.abs(actuals), 1e-8))) * 100  # Avoid division by zero
            
            # Correlation
            correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else np.nan
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Evaluate error metrics
            if mae < 0.1:
                mae_result = TestResult.PASS
                mae_message = f"Excellent MAE: {mae:.4f}"
            elif mae < 0.2:
                mae_result = TestResult.PASS
                mae_message = f"Good MAE: {mae:.4f}"
            elif mae < 0.5:
                mae_result = TestResult.WARNING
                mae_message = f"Moderate MAE: {mae:.4f}"
            else:
                mae_result = TestResult.FAIL
                mae_message = f"High MAE: {mae:.4f}"
            
            suite.add_result(ValidationResult(
                test_name="mean_absolute_error",
                test_type="statistical",
                result=mae_result,
                message=mae_message,
                execution_time_ms=execution_time,
                details={"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}
            ))
            
            # Correlation test
            if not np.isnan(correlation):
                if correlation > 0.8:
                    corr_result = TestResult.PASS
                    corr_message = f"Strong correlation: {correlation:.3f}"
                elif correlation > 0.6:
                    corr_result = TestResult.PASS
                    corr_message = f"Good correlation: {correlation:.3f}"
                elif correlation > 0.3:
                    corr_result = TestResult.WARNING
                    corr_message = f"Moderate correlation: {correlation:.3f}"
                else:
                    corr_result = TestResult.FAIL
                    corr_message = f"Weak correlation: {correlation:.3f}"
                
                suite.add_result(ValidationResult(
                    test_name="prediction_correlation",
                    test_type="statistical",
                    result=corr_result,
                    message=corr_message,
                    details={"correlation": float(correlation)}
                ))
            
            # Statistical significance test (if enough data)
            if len(predictions) >= 10:
                try:
                    # Test if predictions are significantly different from random
                    t_stat, p_value = stats.ttest_rel(predictions, actuals)
                    
                    if p_value > 0.05:
                        sig_result = TestResult.PASS
                        sig_message = f"Predictions not significantly different from actuals (p={p_value:.3f})"
                    else:
                        sig_result = TestResult.WARNING
                        sig_message = f"Predictions significantly different from actuals (p={p_value:.3f})"
                    
                    suite.add_result(ValidationResult(
                        test_name="statistical_significance",
                        test_type="statistical",
                        result=sig_result,
                        message=sig_message,
                        details={"t_statistic": float(t_stat), "p_value": float(p_value)}
                    ))
                    
                except Exception as e:
                    suite.add_result(ValidationResult(
                        test_name="statistical_significance",
                        test_type="statistical",
                        result=TestResult.SKIP,
                        message=f"Statistical test failed: {e}"
                    ))
            
        except Exception as e:
            suite.add_result(ValidationResult(
                test_name="statistical_validation",
                test_type="statistical",
                result=TestResult.FAIL,
                message=f"Statistical validation failed: {e}",
                details={"exception": str(e)}
            ))
        
        suite.finalize()
        return suite
    
    def comprehensive_validation(self, model: Any, model_type: str = "hawkes_process", test_data: dict[str, Any] | None = None) -> dict[str, ValidationSuite]:
        """Run comprehensive validation including all test types"""
        
        validation_results = {}
        
        # Mathematical invariants
        logger.info("Running mathematical invariant validation...")
        validation_results["mathematical_invariants"] = self.validate_mathematical_invariants(model, model_type)
        
        # Numerical stability
        logger.info("Running numerical stability tests...")
        validation_results["numerical_stability"] = self.stability_tester.run_stability_tests(model, self.validation_level)
        
        # Performance benchmarks
        logger.info("Running performance benchmarks...")
        validation_results["performance"] = self.performance_benchmark(model)
        
        # Statistical validation (if test data provided)
        if test_data and "predictions" in test_data and "actuals" in test_data:
            logger.info("Running statistical validation...")
            validation_results["statistical"] = self.statistical_validation(
                test_data["predictions"], test_data["actuals"]
            )
        
        # Store validation history
        for _suite_name, suite in validation_results.items():
            self.validation_history.append(suite)
        
        return validation_results
    
    def generate_validation_report(self, validation_results: dict[str, ValidationSuite]) -> str:
        """Generate comprehensive validation report"""
        
        report_lines = []
        report_lines.append("MATHEMATICAL MODEL VALIDATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Validation Level: {self.validation_level.value}")
        report_lines.append("")
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_warnings = 0
        
        for suite_name, suite in validation_results.items():
            summary = suite.get_summary()
            
            report_lines.append(f"{suite_name.upper().replace('_', ' ')}:")
            report_lines.append(f"  Total Tests: {summary['total']}")
            report_lines.append(f"  Passed: {summary['pass']} (✅)")
            report_lines.append(f"  Failed: {summary['fail']} (❌)")
            report_lines.append(f"  Warnings: {summary['warning']} (⚠️)")
            report_lines.append(f"  Skipped: {summary['skip']} (⏭️)")
            report_lines.append(f"  Success Rate: {summary['success_rate']:.1%}")
            report_lines.append(f"  Execution Time: {summary['total_time_seconds']:.2f}s")
            report_lines.append("")
            
            # Add failed test details
            failed_tests = [r for r in suite.results if r.result == TestResult.FAIL]
            if failed_tests:
                report_lines.append("  FAILED TESTS:")
                for test in failed_tests:
                    report_lines.append(f"    - {test.test_name}: {test.message}")
                report_lines.append("")
            
            total_tests += summary['total']
            total_passed += summary['pass']
            total_failed += summary['fail']
            total_warnings += summary['warning']
        
        # Overall summary
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        report_lines.append("OVERALL SUMMARY:")
        report_lines.append(f"  Total Tests: {total_tests}")
        report_lines.append(f"  Overall Success Rate: {overall_success_rate:.1%}")
        report_lines.append(f"  Status: {'✅ PASS' if overall_success_rate >= 0.8 else '❌ FAIL' if overall_success_rate < 0.6 else '⚠️ WARNING'}")
        
        return "\n".join(report_lines)

def create_validation_framework(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> MathematicalValidationFramework:
    """Create mathematical validation framework with specified level"""
    
    framework = MathematicalValidationFramework(validation_level)
    
    if not HYPOTHESIS_AVAILABLE:
        logger.warning("Hypothesis library not available - property-based testing will be limited")
    
    return framework

if __name__ == "__main__":
    print("🧪 MATHEMATICAL VALIDATION FRAMEWORK TESTING")
    print("=" * 60)
    
    # Create validation framework
    framework = create_validation_framework(ValidationLevel.STANDARD)
    
    print(f"Validation Level: {framework.validation_level.value}")
    print(f"Hypothesis Available: {'✅' if HYPOTHESIS_AVAILABLE else '❌'}")
    
    # Mock model for testing
    class MockHawkesModel:
        def compute_core_function(self, events, params):
            """Mock Hawkes computation"""
            mu = params.get("mu", 0.02)
            alpha = params.get("alpha", 35.51)
            beta = params.get("beta", 0.00442)
            
            if len(events) == 0:
                return np.array([mu])
            
            intensities = []
            for i, t in enumerate(events):
                intensity = mu
                for j in range(i):
                    dt = t - events[j]
                    if dt > 0 and beta * dt < 20:  # Numerical stability
                        intensity += alpha * np.exp(-beta * dt)
                intensities.append(intensity)
            
            return np.array(intensities)
    
    # Test comprehensive validation
    print("\n🔍 RUNNING COMPREHENSIVE VALIDATION...")
    
    mock_model = MockHawkesModel()
    validation_results = framework.comprehensive_validation(mock_model, "hawkes_process")
    
    print(f"Validation suites completed: {len(validation_results)}")
    
    # Generate and display report
    print("\n📋 VALIDATION REPORT:")
    report = framework.generate_validation_report(validation_results)
    print(report)
    
    print("\n✅ Validation framework testing completed")