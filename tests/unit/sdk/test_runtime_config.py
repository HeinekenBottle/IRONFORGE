"""
Comprehensive tests for runtime configuration system
Tests STRICT vs COMPAT modes, acceleration detection, and audit logging
"""

import json
import os
import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from ironforge.sdk.runtime_config import (
    RuntimeMode, AccelStatus, AccelerationState, RuntimeConfig, 
    AuditRunEntry, AccelerationDetector, RuntimeModeEnforcer,
    initialize_runtime_system
)


class TestRuntimeMode:
    """Test RuntimeMode enum"""
    
    def test_runtime_mode_values(self):
        """Test RuntimeMode enum has correct values"""
        assert RuntimeMode.STRICT == "strict"
        assert RuntimeMode.COMPAT == "compat"
    
    def test_runtime_mode_string_conversion(self):
        """Test string conversion works correctly"""
        assert str(RuntimeMode.STRICT) == "strict"
        assert str(RuntimeMode.COMPAT) == "compat"


class TestAccelerationState:
    """Test AccelerationState dataclass"""
    
    def test_default_state(self):
        """Test default acceleration state"""
        state = AccelerationState()
        assert state.sdpa == AccelStatus.MISSING
        assert state.flash == AccelStatus.MISSING  
        assert state.amp == AccelStatus.OFF
        assert state.cdc == AccelStatus.OFF
        assert state.is_degraded() is True
    
    def test_optimal_state(self):
        """Test optimal acceleration state"""
        state = AccelerationState(
            sdpa=AccelStatus.USED,
            flash=AccelStatus.USED,
            amp=AccelStatus.USED,
            cdc=AccelStatus.USED
        )
        assert not state.is_degraded()
        assert len(state.get_degradation_reasons()) == 0
    
    def test_partial_degradation(self):
        """Test partial degradation detection"""
        state = AccelerationState(
            sdpa=AccelStatus.USED,
            flash=AccelStatus.MISSING,
            amp=AccelStatus.USED,
            cdc=AccelStatus.OFF
        )
        assert state.is_degraded()
        reasons = state.get_degradation_reasons()
        assert "Flash Attention backend unavailable" in reasons
        assert "Compile Dynamic Cache disabled" in reasons
        assert len(reasons) == 2


class TestRuntimeConfig:
    """Test RuntimeConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RuntimeConfig()
        assert config.mode == RuntimeMode.STRICT
        assert config.enable_sdpa is True
        assert config.enable_flash_attention is True
        assert config.enable_amp is True
        assert config.enable_cdc is True
        assert config.operation_timeout_seconds == 300
        assert config.enable_audit_logging is True
    
    def test_from_env_defaults(self):
        """Test environment variable defaults"""
        with patch.dict(os.environ, {}, clear=True):
            config = RuntimeConfig.from_env()
            assert config.mode == RuntimeMode.STRICT
            assert config.enable_sdpa is True
    
    def test_from_env_overrides(self):
        """Test environment variable overrides"""
        env_vars = {
            "IRONFORGE_RUNTIME_MODE": "compat",
            "IRONFORGE_ENABLE_SDPA": "false", 
            "IRONFORGE_ENABLE_FLASH": "false",
            "IRONFORGE_TIMEOUT": "600",
            "IRONFORGE_MEMORY_LIMIT_GB": "8.0"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = RuntimeConfig.from_env()
            assert config.mode == RuntimeMode.COMPAT
            assert config.enable_sdpa is False
            assert config.enable_flash_attention is False
            assert config.operation_timeout_seconds == 600
            assert config.memory_limit_gb == 8.0
    
    def test_from_yaml(self):
        """Test YAML configuration loading"""
        yaml_content = """
runtime:
  mode: compat
  enable_sdpa: false
  enable_flash_attention: true
  operation_timeout_seconds: 450
  memory_limit_gb: 16.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                config = RuntimeConfig.from_yaml(f.name)
                assert config.mode == RuntimeMode.COMPAT
                assert config.enable_sdpa is False
                assert config.enable_flash_attention is True
                assert config.operation_timeout_seconds == 450
                assert config.memory_limit_gb == 16.0
            finally:
                os.unlink(f.name)


class TestAccelerationDetector:
    """Test acceleration component detection"""
    
    def test_detect_sdpa_available(self):
        """Test SDPA detection when available"""
        # SDPA should be available in PyTorch >= 2.0
        status = AccelerationDetector.detect_sdpa()
        # Can't assert exact status as it depends on PyTorch version
        assert status in [AccelStatus.USED, AccelStatus.MISSING]
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_detect_flash_no_cuda(self, mock_cuda):
        """Test Flash Attention detection without CUDA"""
        status = AccelerationDetector.detect_flash_attention()
        assert status == AccelStatus.MISSING
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_detect_amp_no_cuda(self, mock_cuda):
        """Test AMP detection without CUDA"""
        status = AccelerationDetector.detect_amp()
        assert status == AccelStatus.MISSING
    
    def test_detect_cdc(self):
        """Test CDC detection"""
        status = AccelerationDetector.detect_cdc()
        # CDC should be available if torch.compile exists (PyTorch >= 2.0)
        if hasattr(torch, 'compile'):
            assert status == AccelStatus.USED
        else:
            assert status == AccelStatus.MISSING
    
    def test_detect_all(self):
        """Test detecting all acceleration components"""
        state = AccelerationDetector.detect_all()
        assert isinstance(state, AccelerationState)
        assert state.sdpa in [AccelStatus.USED, AccelStatus.MISSING]
        assert state.flash in [AccelStatus.USED, AccelStatus.MISSING]
        assert state.amp in [AccelStatus.USED, AccelStatus.MISSING, AccelStatus.OFF]
        assert state.cdc in [AccelStatus.USED, AccelStatus.MISSING]


class TestRuntimeModeEnforcer:
    """Test RuntimeModeEnforcer behavior"""
    
    def test_strict_mode_enforcement_success(self):
        """Test STRICT mode passes with optimal acceleration"""
        config = RuntimeConfig(mode=RuntimeMode.STRICT)
        enforcer = RuntimeModeEnforcer(config)
        
        # Create optimal state
        accel_state = AccelerationState(
            sdpa=AccelStatus.USED,
            flash=AccelStatus.USED,
            amp=AccelStatus.USED,
            cdc=AccelStatus.USED
        )
        
        # Should not raise any exception
        enforcer.enforce_acceleration_requirements(accel_state)
    
    def test_strict_mode_enforcement_failure(self):
        """Test STRICT mode fails with missing acceleration"""
        config = RuntimeConfig(mode=RuntimeMode.STRICT)
        enforcer = RuntimeModeEnforcer(config)
        
        # Create degraded state
        accel_state = AccelerationState(
            sdpa=AccelStatus.MISSING,
            flash=AccelStatus.MISSING,
            amp=AccelStatus.MISSING,
            cdc=AccelStatus.MISSING
        )
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            enforcer.enforce_acceleration_requirements(accel_state)
        
        error_msg = str(exc_info.value)
        assert "STRICT mode acceleration requirements not met" in error_msg
        assert "SDPA" in error_msg
        assert "Flash Attention" in error_msg
    
    def test_compat_mode_allows_degradation(self):
        """Test COMPAT mode allows degraded performance"""
        config = RuntimeConfig(mode=RuntimeMode.COMPAT)
        enforcer = RuntimeModeEnforcer(config)
        
        # Create degraded state
        accel_state = AccelerationState(
            sdpa=AccelStatus.MISSING,
            flash=AccelStatus.MISSING,
            amp=AccelStatus.MISSING,
            cdc=AccelStatus.MISSING
        )
        
        # Should not raise exception (just log warnings)
        enforcer.enforce_acceleration_requirements(accel_state)
    
    def test_create_audit_entry(self):
        """Test audit entry creation"""
        config = RuntimeConfig()
        enforcer = RuntimeModeEnforcer(config)
        
        accel_state = AccelerationState(
            sdpa=AccelStatus.USED,
            flash=AccelStatus.MISSING,
            amp=AccelStatus.USED,
            cdc=AccelStatus.OFF
        )
        
        audit_entry = enforcer.create_audit_entry(accel_state)
        
        assert isinstance(audit_entry, AuditRunEntry)
        assert audit_entry.mode == "strict"
        assert audit_entry.sdpa == "used"
        assert audit_entry.flash == "missing"
        assert audit_entry.amp == "used"
        assert audit_entry.cdc == "off"
        assert audit_entry.degraded is True
        assert len(audit_entry.reasons) > 0
        assert audit_entry.pytorch_version == torch.__version__
    
    def test_save_audit_entry(self):
        """Test audit entry persistence"""
        config = RuntimeConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            audit_path = f.name
        
        try:
            config.audit_file_path = audit_path
            enforcer = RuntimeModeEnforcer(config)
            
            accel_state = AccelerationState(
                sdpa=AccelStatus.USED,
                flash=AccelStatus.USED,
                amp=AccelStatus.USED,
                cdc=AccelStatus.USED
            )
            
            audit_entry = enforcer.create_audit_entry(accel_state)
            enforcer.save_audit_entry(audit_entry)
            
            # Verify file was created and contains correct data
            assert os.path.exists(audit_path)
            
            with open(audit_path, 'r') as f:
                data = json.load(f)
            
            assert "format_version" in data
            assert "runs" in data
            assert len(data["runs"]) == 1
            
            run = data["runs"][0]
            assert run["mode"] == "strict"
            assert run["degraded"] is False
            
        finally:
            if os.path.exists(audit_path):
                os.unlink(audit_path)


@pytest.mark.strict_mode
class TestStrictModeIntegration:
    """Integration tests for STRICT mode"""
    
    def test_strict_mode_initialization(self):
        """Test STRICT mode system initialization"""
        with patch.dict(os.environ, {"IRONFORGE_RUNTIME_MODE": "strict"}):
            config, accel_state, enforcer = initialize_runtime_system()
            
            assert config.mode == RuntimeMode.STRICT
            assert isinstance(accel_state, AccelerationState)
            assert isinstance(enforcer, RuntimeModeEnforcer)
    
    @patch('ironforge.sdk.runtime_config.AccelerationDetector.detect_sdpa')
    def test_strict_mode_fails_missing_sdpa(self, mock_detect_sdpa):
        """Test STRICT mode fails when SDPA is missing"""
        mock_detect_sdpa.return_value = AccelStatus.MISSING
        
        with patch.dict(os.environ, {"IRONFORGE_RUNTIME_MODE": "strict"}):
            with pytest.raises(RuntimeError) as exc_info:
                initialize_runtime_system()
            
            assert "SDPA" in str(exc_info.value)
            assert "STRICT mode" in str(exc_info.value)


@pytest.mark.compat_mode  
class TestCompatModeIntegration:
    """Integration tests for COMPAT mode"""
    
    def test_compat_mode_initialization(self):
        """Test COMPAT mode system initialization"""
        with patch.dict(os.environ, {"IRONFORGE_RUNTIME_MODE": "compat"}):
            config, accel_state, enforcer = initialize_runtime_system()
            
            assert config.mode == RuntimeMode.COMPAT
            assert isinstance(accel_state, AccelerationState) 
            assert isinstance(enforcer, RuntimeModeEnforcer)
    
    @patch('ironforge.sdk.runtime_config.AccelerationDetector.detect_all')
    def test_compat_mode_allows_degradation(self, mock_detect_all):
        """Test COMPAT mode allows degraded acceleration"""
        # Mock completely degraded state
        mock_detect_all.return_value = AccelerationState(
            sdpa=AccelStatus.MISSING,
            flash=AccelStatus.MISSING,
            amp=AccelStatus.MISSING, 
            cdc=AccelStatus.MISSING
        )
        
        with patch.dict(os.environ, {"IRONFORGE_RUNTIME_MODE": "compat"}):
            # Should not raise exception
            config, accel_state, enforcer = initialize_runtime_system()
            
            assert config.mode == RuntimeMode.COMPAT
            assert accel_state.is_degraded()


@pytest.mark.timeout(300)  # 5 minute timeout for slow tests
@pytest.mark.slow
class TestAuditLedgerParity:
    """Tests for audit ledger and parity checking"""
    
    def test_audit_ledger_format(self):
        """Test audit ledger maintains correct format"""
        config = RuntimeConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            audit_path = f.name
        
        try:
            config.audit_file_path = audit_path
            enforcer = RuntimeModeEnforcer(config)
            
            # Create multiple audit entries
            for i in range(3):
                accel_state = AccelerationState(
                    sdpa=AccelStatus.USED if i % 2 == 0 else AccelStatus.MISSING,
                    flash=AccelStatus.USED,
                    amp=AccelStatus.USED,
                    cdc=AccelStatus.USED
                )
                
                audit_entry = enforcer.create_audit_entry(accel_state)
                enforcer.save_audit_entry(audit_entry)
            
            # Verify ledger format
            with open(audit_path, 'r') as f:
                data = json.load(f)
            
            assert data["format_version"] == "1.0"
            assert len(data["runs"]) == 3
            
            # Verify each entry has required fields
            required_fields = [
                "mode", "start_time", "wall_time", "peak_mem",
                "sdpa", "flash", "amp", "cdc", "degraded", "reasons"
            ]
            
            for run in data["runs"]:
                for field in required_fields:
                    assert field in run, f"Missing field: {field}"
                
                # Verify field types
                assert isinstance(run["degraded"], bool)
                assert isinstance(run["reasons"], list)
                assert isinstance(run["wall_time"], (int, float))
                assert isinstance(run["peak_mem"], (int, float))
            
        finally:
            if os.path.exists(audit_path):
                os.unlink(audit_path)
    
    def test_parity_diff_calculation(self):
        """Test parity difference calculation for CI validation"""
        # This would be expanded to test actual TGAT output parity
        # For now, test the audit entry comparison logic
        
        entry1 = AuditRunEntry(
            mode="strict",
            start_time="2024-01-01 12:00:00",
            wall_time=10.5,
            peak_mem=2.1,
            sdpa="used",
            flash="used", 
            amp="used",
            cdc="used",
            degraded=False
        )
        
        entry2 = AuditRunEntry(
            mode="compat",
            start_time="2024-01-01 12:00:10", 
            wall_time=12.3,
            peak_mem=2.3,
            sdpa="missing",
            flash="missing",
            amp="missing",
            cdc="missing",
            degraded=True,
            reasons=["SDPA unavailable", "Flash Attention unavailable"]
        )
        
        # Convert to dicts
        dict1 = entry1.to_dict()
        dict2 = entry2.to_dict()
        
        # Verify structure
        assert dict1["degraded"] is False
        assert dict2["degraded"] is True
        assert len(dict2["reasons"]) == 2
        
        # Performance regression check (wall_time difference)
        perf_diff = dict2["wall_time"] - dict1["wall_time"]
        assert perf_diff > 0  # COMPAT mode should be slower


if __name__ == "__main__":
    pytest.main([__file__, "-v"])