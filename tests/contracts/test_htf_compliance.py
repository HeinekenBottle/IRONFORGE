"""
HTF Compliance Tests
===================

Tests to ensure HTF (High Timeframe) compliance with Golden Invariant:
- HTF Rule: Last-closed candle data only, no intra-candle usage
- Regression tests to prevent future violations
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List

from ironforge.contracts import HTFComplianceValidator, ContractViolationError


class TestHTFComplianceValidator:
    """Test HTF compliance validation."""
    
    def test_valid_htf_last_closed_only(self):
        """Test validation passes for last-closed HTF data."""
        valid_htf_data = {
            "htf_15m_close": 18500.0,
            "htf_1h_close": 18520.0,
            "htf_4h_close": 18480.0,
            "htf_daily_close": 18450.0,
            "last_closed_candle": True,
            "htf_context": {
                "15m_last_close": 18500.0,
                "1h_last_close": 18520.0,
                "closed_candle_data": True
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(valid_htf_data)
    
    def test_invalid_htf_intra_candle_patterns(self):
        """Test validation fails for intra-candle HTF patterns."""
        forbidden_patterns = [
            {"intra_candle_htf": 18500.0},
            {"current_candle_data": 18500.0},
            {"live_candle_htf": 18500.0},
            {"real_time_htf": 18500.0},
            {"streaming_htf": 18500.0},
            {"tick_data_htf": 18500.0},
        ]
        
        for invalid_data in forbidden_patterns:
            with pytest.raises(ContractViolationError, match="HTF compliance violation"):
                HTFComplianceValidator.validate_htf_usage(invalid_data)
    
    def test_invalid_htf_nested_patterns(self):
        """Test validation fails for nested intra-candle patterns."""
        invalid_nested_data = {
            "htf_context": {
                "15m_data": {
                    "current_candle": 18500.0,  # Forbidden pattern
                    "last_close": 18480.0
                }
            }
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(invalid_nested_data)
    
    def test_invalid_htf_string_values(self):
        """Test validation fails for forbidden patterns in string values."""
        invalid_string_data = {
            "htf_source": "real_time_feed",  # Forbidden pattern in value
            "data_type": "intra_candle_htf"  # Forbidden pattern in value
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(invalid_string_data)
    
    def test_complex_nested_structure(self):
        """Test validation on complex nested structures."""
        complex_data = {
            "session_data": {
                "events": [
                    {
                        "type": "Expansion",
                        "htf_context": {
                            "15m_close": 18500.0,
                            "1h_close": 18520.0
                        }
                    }
                ],
                "metadata": {
                    "htf_features": {
                        "f45_sv_m15_z": 0.5,  # Valid HTF feature
                        "f46_sv_h1_z": 0.3,
                        "last_closed_only": True
                    }
                }
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(complex_data)


class TestHTFFeatureCompliance:
    """Test HTF feature compliance (f45-f50)."""
    
    def test_valid_htf_features(self):
        """Test valid HTF features f45-f50."""
        valid_htf_features = {
            "features": {
                "f45_sv_m15_z": 0.5,      # Valid: last-closed M15 SV
                "f46_sv_h1_z": 0.3,       # Valid: last-closed H1 SV
                "f47_barpos_m15": 0.7,    # Valid: last-closed M15 bar position
                "f48_barpos_h1": 0.2,     # Valid: last-closed H1 bar position
                "f49_dist_daily_mid": 0.1, # Valid: distance to daily mid
                "f50_htf_regime": 1.0      # Valid: HTF regime indicator
            },
            "htf_mode": "last_closed_only"
        }
        
        assert HTFComplianceValidator.validate_htf_usage(valid_htf_features)
    
    def test_invalid_htf_features_intra_candle(self):
        """Test invalid HTF features with intra-candle data."""
        invalid_htf_features = {
            "features": {
                "f45_sv_m15_current": 0.5,  # Invalid: current candle
                "f46_sv_h1_live": 0.3,      # Invalid: live data
                "f47_barpos_m15_real_time": 0.7  # Invalid: real-time
            }
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(invalid_htf_features)


class TestHTFRegressionPrevention:
    """Regression tests to prevent future HTF violations."""
    
    def test_session_data_htf_compliance(self):
        """Test session data for HTF compliance."""
        # Sample session data structure
        session_data = {
            "session_id": "NY_2025-08-28",
            "events": [
                {
                    "type": "Expansion",
                    "timestamp": "2025-08-28T14:30:00",
                    "htf_context": {
                        "15m_close": 18500.0,
                        "1h_close": 18520.0,
                        "data_source": "last_closed_candles"
                    }
                }
            ],
            "metadata": {
                "htf_enabled": True,
                "htf_features": ["f45", "f46", "f47", "f48", "f49", "f50"],
                "htf_rule": "last_closed_only"
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(session_data)
    
    def test_graph_builder_htf_compliance(self):
        """Test graph builder HTF compliance."""
        graph_config = {
            "htf_config": {
                "enabled": True,
                "timeframes": ["15m", "1h", "4h"],
                "data_source": "closed_candles_only",
                "features": {
                    "f45_sv_m15_z": "last_closed",
                    "f46_sv_h1_z": "last_closed",
                    "f47_barpos_m15": "last_closed",
                    "f48_barpos_h1": "last_closed",
                    "f49_dist_daily_mid": "last_closed",
                    "f50_htf_regime": "last_closed"
                }
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(graph_config)
    
    def test_discovery_pipeline_htf_compliance(self):
        """Test discovery pipeline HTF compliance."""
        pipeline_config = {
            "discovery": {
                "htf_mode": "enabled",
                "feature_adapter": {
                    "htf_enabled": True,
                    "node_dim": 51,  # f0-f50
                    "htf_features": "last_closed_only"
                }
            },
            "tgat": {
                "enhanced": True,
                "htf_context": "closed_candles"
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(pipeline_config)
    
    def test_oracle_htf_compliance(self):
        """Test Oracle integration HTF compliance."""
        oracle_config = {
            "oracle": {
                "htf_context": True,
                "features": {
                    "htf_sv_15m": "last_closed",
                    "htf_sv_1h": "last_closed",
                    "htf_regime": "last_closed"
                },
                "training": {
                    "htf_data_source": "closed_candles_only"
                }
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(oracle_config)


class TestHTFComplianceIntegration:
    """Integration tests for HTF compliance across components."""
    
    def test_end_to_end_htf_compliance(self):
        """Test end-to-end HTF compliance in pipeline."""
        pipeline_data = {
            "config": {
                "htf_enabled": True,
                "htf_rule": "last_closed_only"
            },
            "session_data": {
                "htf_context": {
                    "15m_close": 18500.0,
                    "1h_close": 18520.0,
                    "4h_close": 18480.0
                }
            },
            "features": {
                "f45_sv_m15_z": 0.5,
                "f46_sv_h1_z": 0.3,
                "f47_barpos_m15": 0.7,
                "f48_barpos_h1": 0.2,
                "f49_dist_daily_mid": 0.1,
                "f50_htf_regime": 1.0
            },
            "discovery": {
                "htf_mode": "last_closed",
                "temporal_data": "closed_candles"
            },
            "scoring": {
                "htf_weights": "last_closed_based"
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(pipeline_data)
    
    def test_htf_compliance_with_real_session_structure(self):
        """Test HTF compliance with realistic session structure."""
        realistic_session = {
            "session_id": "NY_2025-08-28",
            "symbol": "NQ",
            "timeframe": "M5",
            "events": [
                {
                    "type": "Expansion",
                    "timestamp": "2025-08-28T14:30:00",
                    "price": 18500.0,
                    "features": {
                        # Standard features f0-f44
                        **{f"f{i}": 0.1 * i for i in range(45)},
                        # HTF features f45-f50 (last-closed only)
                        "f45_sv_m15_z": 0.5,
                        "f46_sv_h1_z": 0.3,
                        "f47_barpos_m15": 0.7,
                        "f48_barpos_h1": 0.2,
                        "f49_dist_daily_mid": 0.1,
                        "f50_htf_regime": 1.0
                    },
                    "htf_context": {
                        "15m_last_close": 18495.0,
                        "1h_last_close": 18520.0,
                        "4h_last_close": 18480.0,
                        "daily_last_close": 18450.0
                    }
                }
            ],
            "metadata": {
                "htf_mode": "enabled",
                "htf_rule": "last_closed_only",
                "feature_dimensions": {
                    "nodes": 51,  # f0-f50
                    "edges": 20   # e0-e19
                }
            }
        }
        
        assert HTFComplianceValidator.validate_htf_usage(realistic_session)


class TestHTFViolationDetection:
    """Test detection of specific HTF violations."""
    
    @pytest.mark.parametrize("violation_pattern", [
        "intra_candle",
        "current_candle", 
        "live_candle",
        "real_time",
        "streaming",
        "tick_data"
    ])
    def test_detect_specific_violations(self, violation_pattern):
        """Test detection of specific violation patterns."""
        violation_data = {
            f"htf_{violation_pattern}": 18500.0,
            "context": {
                f"{violation_pattern}_data": True
            }
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(violation_data)
    
    def test_case_insensitive_detection(self):
        """Test case-insensitive violation detection."""
        violation_data = {
            "HTF_INTRA_CANDLE": 18500.0,
            "Current_Candle_Data": True,
            "REAL_TIME_HTF": 18500.0
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(violation_data)
    
    def test_deep_nested_violation_detection(self):
        """Test detection of violations in deeply nested structures."""
        deep_nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "htf_data": {
                                "source": "intra_candle_feed"  # Violation at depth 5
                            }
                        }
                    }
                }
            }
        }
        
        with pytest.raises(ContractViolationError, match="HTF compliance violation"):
            HTFComplianceValidator.validate_htf_usage(deep_nested_data)
