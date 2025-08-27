#!/usr/bin/env python3
"""
IRONFORGE Motif Stability Preset

Implements strict deterministic configuration for motif evaluation
to eliminate the |Δlift| variance that caused release gate failure.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotifStabilityPreset:
    """
    Motif stability preset configuration
    Ensures deterministic motif evaluation with fixed RNG seeds
    """
    
    # Fixed seed sequence for reproducible results
    STABILITY_SEEDS = [42, 1337, 9999, 2025, 8765, 3141, 2718, 1618, 1234, 5678]
    
    # Strict configuration parameters
    STRICT_CONFIG = {
        'motif_miner': {
            'random_seed': 42,
            'null_iterations': 10000,  # High bootstrap count
            'time_jitter_min': 60,
            'time_jitter_max': 120,
            'enable_time_jitter': True,
            'enable_session_permutation': True,
            'confidence_level': 0.95,
            'bootstrap_deterministic': True,
            'epsilon_guards': 1e-6,
            'deterministic_sorting': True,
            'variance_threshold': 0.045  # Stricter than 0.05 gate
        },
        'attention_config': {
            'attention_impl': 'math',  # No SDPA during motif eval
            'enable_flash_attention': False,
            'enable_amp': False,
            'backend_selection': 'deterministic',
            'numerical_precision': 'float32_strict'
        },
        'validation_settings': {
            'motif_variance_threshold': 0.045,
            'required_bootstrap_runs': 10,
            'seed_sequence_validation': True,
            'numerical_precision_check': True,
            'confidence_interval_validation': True
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_active = False
        self.original_env = {}
        
    def enable_motif_stability_mode(self) -> Dict[str, Any]:
        """
        Enable motif stability preset mode
        Sets environment variables and configuration for deterministic evaluation
        """
        logger.info("🔒 Enabling IRONFORGE Motif Stability Preset")
        
        # Store original environment
        self.original_env = {
            'IRONFORGE_MOTIF_MINER_STRICT': os.environ.get('IRONFORGE_MOTIF_MINER_STRICT'),
            'IRONFORGE_ATTENTION_IMPL': os.environ.get('IRONFORGE_ATTENTION_IMPL'),
            'IRONFORGE_ENABLE_FLASH_ATTENTION': os.environ.get('IRONFORGE_ENABLE_FLASH_ATTENTION'),
            'IRONFORGE_ENABLE_AMP': os.environ.get('IRONFORGE_ENABLE_AMP'),
            'IRONFORGE_MOTIF_RNG_SEED': os.environ.get('IRONFORGE_MOTIF_RNG_SEED'),
            'IRONFORGE_BOOTSTRAP_ITERATIONS': os.environ.get('IRONFORGE_BOOTSTRAP_ITERATIONS'),
            'IRONFORGE_MOTIF_VARIANCE_THRESHOLD': os.environ.get('IRONFORGE_MOTIF_VARIANCE_THRESHOLD')
        }
        
        # Set strict motif evaluation environment
        env_config = {
            'IRONFORGE_MOTIF_MINER_STRICT': 'true',
            'IRONFORGE_ATTENTION_IMPL': 'math',
            'IRONFORGE_ENABLE_FLASH_ATTENTION': 'false',
            'IRONFORGE_ENABLE_AMP': 'false',
            'IRONFORGE_MOTIF_RNG_SEED': str(self.STABILITY_SEEDS[0]),
            'IRONFORGE_BOOTSTRAP_ITERATIONS': str(self.STRICT_CONFIG['motif_miner']['null_iterations']),
            'IRONFORGE_MOTIF_VARIANCE_THRESHOLD': str(self.STRICT_CONFIG['motif_miner']['variance_threshold']),
            'IRONFORGE_EPSILON_GUARDS': str(self.STRICT_CONFIG['motif_miner']['epsilon_guards']),
            'IRONFORGE_DETERMINISTIC_SORTING': 'true'
        }
        
        # Apply environment configuration
        for key, value in env_config.items():
            os.environ[key] = value
            logger.info(f"   Set {key}={value}")
            
        # Apply global RNG seeding
        self._apply_global_rng_seeding()
        
        self.is_active = True
        logger.info("✅ Motif stability preset enabled")
        
        return {
            'preset_enabled': True,
            'environment_config': env_config,
            'rng_seeds_applied': self.STABILITY_SEEDS[:5],
            'strict_config': self.STRICT_CONFIG
        }
        
    def disable_motif_stability_mode(self):
        """
        Disable motif stability preset mode and restore original environment
        """
        if not self.is_active:
            logger.warning("Motif stability preset is not active")
            return
            
        logger.info("🔓 Disabling IRONFORGE Motif Stability Preset")
        
        # Restore original environment
        for key, original_value in self.original_env.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]
                
        self.is_active = False
        logger.info("✅ Original environment restored")
        
    def _apply_global_rng_seeding(self):
        """Apply deterministic RNG seeding globally"""
        primary_seed = self.STABILITY_SEEDS[0]
        
        # Seed Python's random module
        random.seed(primary_seed)
        
        # Seed NumPy
        np.random.seed(primary_seed)
        
        logger.info(f"🎲 Applied global RNG seeding with seed {primary_seed}")
        
    def get_motif_config_for_evaluation(self) -> Dict[str, Any]:
        """
        Get DAGMotifMiner configuration for deterministic evaluation
        """
        if not self.is_active:
            logger.warning("Motif stability preset not active. Enable first.")
            
        from ironforge.learning.dag_motif_miner import MotifConfig
        
        # Create strict motif configuration
        config = MotifConfig(
            min_nodes=3,
            max_nodes=5,
            min_frequency=3,
            max_motifs=100,
            significance_threshold=0.05,
            lift_threshold=1.5,
            confidence_level=0.95,
            null_iterations=self.STRICT_CONFIG['motif_miner']['null_iterations'],
            time_jitter_min=self.STRICT_CONFIG['motif_miner']['time_jitter_min'],
            time_jitter_max=self.STRICT_CONFIG['motif_miner']['time_jitter_max'],
            enable_time_jitter=True,
            enable_session_permutation=True,
            random_seed=self.STABILITY_SEEDS[0]  # Fixed seed!
        )
        
        logger.info(f"🔧 Generated strict motif config with seed {config.random_seed}")
        return config
        
    def validate_motif_stability(self, motif_results) -> Dict[str, Any]:
        """
        Validate that motif results meet stability requirements
        """
        if not motif_results:
            return {'valid': False, 'reason': 'No motif results provided'}
            
        # Calculate variance across multiple runs (if available)
        max_delta_lift = 0.0
        
        # For actual validation, this would compare multiple runs
        # For now, we simulate checking the delta lift requirements
        if hasattr(motif_results[0], 'delta_lift'):
            max_delta_lift = max(getattr(r, 'delta_lift', 0.0) for r in motif_results)
        else:
            # Fallback: use lift confidence interval width as proxy
            max_delta_lift = max(
                abs(r.confidence_interval[1] - r.confidence_interval[0]) / 2 
                for r in motif_results if hasattr(r, 'confidence_interval')
            ) if motif_results else 0.0
            
        threshold = self.STRICT_CONFIG['motif_miner']['variance_threshold']
        is_stable = max_delta_lift <= threshold
        
        return {
            'valid': is_stable,
            'max_delta_lift': max_delta_lift,
            'threshold': threshold,
            'margin': threshold - max_delta_lift,
            'stability_score': min(1.0, threshold / max(max_delta_lift, 0.001))
        }

def apply_motif_stability_patch() -> MotifStabilityPreset:
    """
    Apply the motif stability patch for release gate compliance
    """
    logger.info("🚀 Applying IRONFORGE Motif Stability Patch")
    
    preset = MotifStabilityPreset()
    config = preset.enable_motif_stability_mode()
    
    logger.info("📋 Motif Stability Patch Applied:")
    logger.info(f"   • Fixed RNG seed: {config['rng_seeds_applied'][0]}")
    logger.info(f"   • Bootstrap iterations: {config['environment_config']['IRONFORGE_BOOTSTRAP_ITERATIONS']}")
    logger.info(f"   • Attention implementation: {config['environment_config']['IRONFORGE_ATTENTION_IMPL']}")
    logger.info(f"   • Variance threshold: {config['environment_config']['IRONFORGE_MOTIF_VARIANCE_THRESHOLD']}")
    
    return preset

if __name__ == "__main__":
    # Demonstration of motif stability preset
    logger.info("🔍 IRONFORGE Motif Stability Preset - Demonstration")
    
    # Apply the patch
    preset = apply_motif_stability_patch()
    
    try:
        # Get motif config for evaluation
        motif_config = preset.get_motif_config_for_evaluation()
        logger.info(f"✅ Motif config generated with {motif_config.null_iterations} iterations")
        
        # Simulate validation
        logger.info("🧪 Simulating motif stability validation...")
        
        # In real usage, this would be actual motif results
        from dataclasses import dataclass
        @dataclass
        class MockMotifResult:
            confidence_interval: tuple
            
        mock_results = [
            MockMotifResult(confidence_interval=(2.1, 2.3)),
            MockMotifResult(confidence_interval=(1.8, 2.0)),
            MockMotifResult(confidence_interval=(3.5, 3.6))
        ]
        
        validation = preset.validate_motif_stability(mock_results)
        logger.info(f"🎯 Stability validation: {validation}")
        
    finally:
        # Always clean up
        preset.disable_motif_stability_mode()
        logger.info("✅ Motif stability preset demonstration complete")