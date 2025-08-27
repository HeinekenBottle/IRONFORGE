"""
Runtime Configuration System for STRICT vs COMPAT execution modes
Handles acceleration component availability and fallback behavior
"""

from __future__ import annotations

import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


class RuntimeMode(str, Enum):
    """Runtime execution modes"""
    STRICT = "strict"
    COMPAT = "compat"


class AccelStatus(str, Enum):
    """Acceleration component status"""
    USED = "used"
    MISSING = "missing" 
    OFF = "off"


@dataclass
class AccelerationState:
    """Current state of acceleration components"""
    sdpa: AccelStatus = AccelStatus.MISSING
    flash: AccelStatus = AccelStatus.MISSING
    amp: AccelStatus = AccelStatus.OFF
    cdc: AccelStatus = AccelStatus.OFF  # Compile Dynamic Cache
    
    def is_degraded(self) -> bool:
        """Check if any acceleration is missing or off when it should be available"""
        return (
            self.sdpa == AccelStatus.MISSING or
            self.flash == AccelStatus.MISSING or
            self.amp == AccelStatus.OFF or
            self.cdc == AccelStatus.OFF
        )
    
    def get_degradation_reasons(self) -> List[str]:
        """Get list of degradation reasons"""
        reasons = []
        if self.sdpa == AccelStatus.MISSING:
            reasons.append("SDPA (Scaled Dot Product Attention) unavailable")
        if self.flash == AccelStatus.MISSING:
            reasons.append("Flash Attention backend unavailable")
        if self.amp == AccelStatus.OFF:
            reasons.append("Automatic Mixed Precision disabled")
        if self.cdc == AccelStatus.OFF:
            reasons.append("Compile Dynamic Cache disabled")
        return reasons


@dataclass
class RuntimeConfig:
    """Runtime configuration for STRICT vs COMPAT modes"""
    
    # Execution mode
    mode: RuntimeMode = RuntimeMode.STRICT
    
    # Acceleration preferences (what we want to use)
    enable_sdpa: bool = True
    enable_flash_attention: bool = True
    enable_amp: bool = True
    enable_cdc: bool = True
    
    # Timeout and performance settings
    operation_timeout_seconds: int = 300
    memory_limit_gb: Optional[float] = None
    
    # Audit settings
    enable_audit_logging: bool = True
    audit_file_path: str = "audit_run.json"
    
    # Fallback behavior in COMPAT mode
    allow_cpu_fallback: bool = True
    allow_fp32_fallback: bool = True
    
    @classmethod
    def from_env(cls) -> RuntimeConfig:
        """Create configuration from environment variables"""
        mode = RuntimeMode(os.getenv("IRONFORGE_RUNTIME_MODE", "strict"))
        
        return cls(
            mode=mode,
            enable_sdpa=_env_bool("IRONFORGE_ENABLE_SDPA", True),
            enable_flash_attention=_env_bool("IRONFORGE_ENABLE_FLASH", True),
            enable_amp=_env_bool("IRONFORGE_ENABLE_AMP", True),
            enable_cdc=_env_bool("IRONFORGE_ENABLE_CDC", True),
            operation_timeout_seconds=int(os.getenv("IRONFORGE_TIMEOUT", "300")),
            memory_limit_gb=_env_float("IRONFORGE_MEMORY_LIMIT_GB", None),
            enable_audit_logging=_env_bool("IRONFORGE_ENABLE_AUDIT", True),
            audit_file_path=os.getenv("IRONFORGE_AUDIT_FILE", "audit_run.json"),
            allow_cpu_fallback=_env_bool("IRONFORGE_ALLOW_CPU_FALLBACK", True),
            allow_fp32_fallback=_env_bool("IRONFORGE_ALLOW_FP32_FALLBACK", True),
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path | str) -> RuntimeConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        runtime_data = data.get("runtime", {})
        
        return cls(
            mode=RuntimeMode(runtime_data.get("mode", "strict")),
            enable_sdpa=runtime_data.get("enable_sdpa", True),
            enable_flash_attention=runtime_data.get("enable_flash_attention", True),
            enable_amp=runtime_data.get("enable_amp", True),
            enable_cdc=runtime_data.get("enable_cdc", True),
            operation_timeout_seconds=runtime_data.get("operation_timeout_seconds", 300),
            memory_limit_gb=runtime_data.get("memory_limit_gb"),
            enable_audit_logging=runtime_data.get("enable_audit_logging", True),
            audit_file_path=runtime_data.get("audit_file_path", "audit_run.json"),
            allow_cpu_fallback=runtime_data.get("allow_cpu_fallback", True),
            allow_fp32_fallback=runtime_data.get("allow_fp32_fallback", True),
        )


@dataclass
class AuditRunEntry:
    """Single audit run entry for the ledger"""
    
    # Basic run info
    mode: str
    start_time: str
    wall_time: float
    peak_mem: float
    
    # Acceleration status
    sdpa: str
    flash: str
    amp: str
    cdc: str
    
    # Degradation info
    degraded: bool
    reasons: List[str] = field(default_factory=list)
    
    # Additional metadata
    pytorch_version: str = ""
    cuda_available: bool = False
    device_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "mode": self.mode,
            "start_time": self.start_time,
            "wall_time": self.wall_time,
            "peak_mem": self.peak_mem,
            "sdpa": self.sdpa,
            "flash": self.flash,
            "amp": self.amp,
            "cdc": self.cdc,
            "degraded": self.degraded,
            "reasons": self.reasons,
            "pytorch_version": self.pytorch_version,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
        }


class AccelerationDetector:
    """Detects available acceleration components and their status"""
    
    @staticmethod
    def detect_sdpa() -> AccelStatus:
        """Detect SDPA availability"""
        try:
            from torch.nn.functional import scaled_dot_product_attention
            # Test with small tensors
            q = torch.randn(1, 1, 4, 8)
            k = torch.randn(1, 1, 4, 8)
            v = torch.randn(1, 1, 4, 8)
            _ = scaled_dot_product_attention(q, k, v)
            return AccelStatus.USED
        except (ImportError, RuntimeError) as e:
            logger.debug(f"SDPA detection failed: {e}")
            return AccelStatus.MISSING
    
    @staticmethod
    def detect_flash_attention() -> AccelStatus:
        """Detect Flash Attention backend availability"""
        if not torch.cuda.is_available():
            return AccelStatus.MISSING
            
        try:
            # Check if flash attention backend is available
            # This is PyTorch >= 2.0 specific
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                # Try to enable and test
                torch.backends.cuda.enable_flash_sdp(True)
                return AccelStatus.USED
            else:
                return AccelStatus.MISSING
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Flash Attention detection failed: {e}")
            return AccelStatus.MISSING
    
    @staticmethod
    def detect_amp() -> AccelStatus:
        """Detect AMP availability"""
        if not torch.cuda.is_available():
            return AccelStatus.MISSING
            
        try:
            from torch.cuda.amp import autocast, GradScaler
            
            # Test AMP with small computation
            with autocast():
                x = torch.randn(2, 2, device='cuda')
                _ = torch.mm(x, x)
            return AccelStatus.USED
        except (ImportError, RuntimeError) as e:
            logger.debug(f"AMP detection failed: {e}")
            return AccelStatus.MISSING
    
    @staticmethod
    def detect_cdc() -> AccelStatus:
        """Detect Compile Dynamic Cache availability"""
        try:
            # Check if torch.compile is available (PyTorch >= 2.0)
            if hasattr(torch, 'compile'):
                return AccelStatus.USED
            else:
                return AccelStatus.MISSING
        except Exception as e:
            logger.debug(f"CDC detection failed: {e}")
            return AccelStatus.MISSING
    
    @classmethod
    def detect_all(cls) -> AccelerationState:
        """Detect all acceleration components"""
        return AccelerationState(
            sdpa=cls.detect_sdpa(),
            flash=cls.detect_flash_attention(),
            amp=cls.detect_amp(),
            cdc=cls.detect_cdc(),
        )


class RuntimeModeEnforcer:
    """Enforces STRICT vs COMPAT mode behavior"""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
    
    def enforce_acceleration_requirements(self, accel_state: AccelerationState) -> None:
        """Enforce acceleration requirements based on mode"""
        
        if self.config.mode == RuntimeMode.STRICT:
            self._enforce_strict_mode(accel_state)
        else:
            self._log_compat_mode_degradation(accel_state)
    
    def _enforce_strict_mode(self, accel_state: AccelerationState) -> None:
        """Enforce STRICT mode - fail fast on missing acceleration"""
        
        errors = []
        
        if self.config.enable_sdpa and accel_state.sdpa == AccelStatus.MISSING:
            errors.append(
                "SDPA (Scaled Dot Product Attention) is required but unavailable. "
                "Please upgrade to PyTorch >= 2.0 or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        if self.config.enable_flash_attention and accel_state.flash == AccelStatus.MISSING:
            errors.append(
                "Flash Attention is required but unavailable. "
                "Please ensure CUDA is available and PyTorch has Flash Attention support, "
                "or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        if self.config.enable_amp and accel_state.amp == AccelStatus.MISSING:
            errors.append(
                "Automatic Mixed Precision is required but unavailable. "
                "Please ensure CUDA is available, or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        if self.config.enable_cdc and accel_state.cdc == AccelStatus.MISSING:
            errors.append(
                "Compile Dynamic Cache is required but unavailable. "
                "Please upgrade to PyTorch >= 2.0 with torch.compile support, "
                "or set IRONFORGE_RUNTIME_MODE=compat"
            )
        
        if errors:
            error_msg = "STRICT mode acceleration requirements not met:\n" + "\n".join(f"  - {e}" for e in errors)
            raise RuntimeError(error_msg)
    
    def _log_compat_mode_degradation(self, accel_state: AccelerationState) -> None:
        """Log degradation warnings in COMPAT mode"""
        
        if accel_state.is_degraded():
            reasons = accel_state.get_degradation_reasons()
            logger.warning("COMPAT mode: Running with degraded performance")
            for reason in reasons:
                logger.warning(f"  - {reason}")
    
    def create_audit_entry(self, accel_state: AccelerationState) -> AuditRunEntry:
        """Create audit entry for this run"""
        
        wall_time = time.time() - self.start_time
        current_memory = self._get_memory_usage()
        peak_mem = current_memory  # Simplified - could track peak during run
        
        return AuditRunEntry(
            mode=self.config.mode.value,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            wall_time=wall_time,
            peak_mem=peak_mem,
            sdpa=accel_state.sdpa.value,
            flash=accel_state.flash.value,
            amp=accel_state.amp.value,
            cdc=accel_state.cdc.value,
            degraded=accel_state.is_degraded(),
            reasons=accel_state.get_degradation_reasons(),
            pytorch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        )
    
    def save_audit_entry(self, audit_entry: AuditRunEntry) -> None:
        """Save audit entry to ledger file"""
        
        if not self.config.enable_audit_logging:
            return
        
        audit_path = Path(self.config.audit_file_path)
        
        # Load existing entries or create empty list
        entries = []
        if audit_path.exists():
            try:
                with open(audit_path, 'r') as f:
                    data = json.load(f)
                    entries = data.get("runs", [])
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid audit file format, creating new: {audit_path}")
        
        # Add new entry
        entries.append(audit_entry.to_dict())
        
        # Write back to file
        audit_data = {
            "format_version": "1.0",
            "runs": entries
        }
        
        with open(audit_path, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        logger.info(f"Audit entry saved to {audit_path}")
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in GB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)  # Convert to GB


def _env_bool(key: str, default: bool) -> bool:
    """Parse boolean from environment variable"""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    else:
        return default


def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    """Parse float from environment variable"""
    value = os.getenv(key)
    if value:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}")
    return default


def initialize_runtime_system() -> tuple[RuntimeConfig, AccelerationState, RuntimeModeEnforcer]:
    """Initialize the complete runtime system"""
    
    # Load configuration (env vars take precedence over files)
    config = RuntimeConfig.from_env()
    
    # Detect acceleration capabilities
    accel_state = AccelerationDetector.detect_all()
    
    # Create enforcer
    enforcer = RuntimeModeEnforcer(config)
    
    # Enforce requirements
    enforcer.enforce_acceleration_requirements(accel_state)
    
    logger.info(f"Runtime mode: {config.mode.value}")
    logger.info(f"Acceleration state: SDPA={accel_state.sdpa.value}, "
               f"Flash={accel_state.flash.value}, AMP={accel_state.amp.value}, CDC={accel_state.cdc.value}")
    
    if accel_state.is_degraded():
        logger.info(f"Degraded performance mode active")
    
    return config, accel_state, enforcer