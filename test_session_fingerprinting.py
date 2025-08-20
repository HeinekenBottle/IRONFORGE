#!/usr/bin/env python3
"""
Test Session Fingerprinting System
Validates the feature vector extraction meets Stage 1 success criteria
"""

import logging
from pathlib import Path
import numpy as np
from ironforge.learning.session_fingerprinting import (
    SessionFingerprintExtractor, 
    SessionFingerprintConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_standard_vs_robust_scaler():
    """A/B test: StandardScaler vs RobustScaler"""
    logger.info("=== A/B TEST: StandardScaler vs RobustScaler ===")
    
    # Find adapted enhanced session files
    data_dir = Path("data/adapted")
    session_files = list(data_dir.glob("adapted_enhanced_rel_*.json"))
    
    if not session_files:
        logger.error("No adapted enhanced session files found")
        return None, None
    
    logger.info(f"Found {len(session_files)} session files")
    
    results = {}
    
    for scaler_type in ["standard", "robust"]:
        logger.info(f"\nTesting {scaler_type} scaler...")
        
        # Configure extractor
        config = SessionFingerprintConfig.default()
        config.scaler_type = scaler_type
        extractor = SessionFingerprintExtractor(config)
        
        # Extract fingerprints
        fingerprints = extractor.extract_batch_fingerprints(session_files[:10])  # Limit for testing
        
        if not fingerprints:
            logger.warning(f"No valid fingerprints extracted for {scaler_type}")
            continue
            
        # Fit and transform
        extractor.fit_scaler(fingerprints)
        scaled_fingerprints = extractor.transform_fingerprints(fingerprints)
        
        # Validate
        validation = extractor.validate_fingerprints(scaled_fingerprints)
        validation['scaler_type'] = scaler_type
        results[scaler_type] = validation
        
        logger.info(f"{scaler_type} results: {validation}")
    
    return results


def test_tempo_methods():
    """A/B test: std of 1st diff vs MAD-based tempo proxy"""
    logger.info("\n=== A/B TEST: Tempo Methods ===")
    
    # Find adapted enhanced session files
    data_dir = Path("data/adapted") 
    session_files = list(data_dir.glob("adapted_enhanced_rel_*.json"))
    
    if not session_files:
        logger.error("No adapted enhanced session files found")
        return None, None
    
    results = {}
    
    for tempo_method in ["std_diff", "mad_based"]:
        logger.info(f"\nTesting {tempo_method} tempo method...")
        
        # Configure extractor
        config = SessionFingerprintConfig.default()
        config.tempo_method = tempo_method
        extractor = SessionFingerprintExtractor(config)
        
        # Extract fingerprints
        fingerprints = extractor.extract_batch_fingerprints(session_files[:10])  # Limit for testing
        
        if not fingerprints:
            logger.warning(f"No valid fingerprints extracted for {tempo_method}")
            continue
            
        # Validate
        validation = extractor.validate_fingerprints(fingerprints)
        validation['tempo_method'] = tempo_method
        results[tempo_method] = validation
        
        logger.info(f"{tempo_method} results: {validation}")
    
    return results


def test_repeatability():
    """Test that repeated runs produce identical results"""
    logger.info("\n=== REPEATABILITY TEST ===")
    
    # Find adapted enhanced session files
    data_dir = Path("data/adapted")
    session_files = list(data_dir.glob("adapted_enhanced_rel_*.json"))[:5]  # Small subset
    
    if not session_files:
        logger.error("No adapted enhanced session files found")
        return False
    
    # Extract fingerprints twice
    extractor = SessionFingerprintExtractor()
    
    fingerprints_1 = extractor.extract_batch_fingerprints(session_files)
    fingerprints_2 = extractor.extract_batch_fingerprints(session_files)
    
    if len(fingerprints_1) != len(fingerprints_2):
        logger.error("Different number of fingerprints extracted in repeated runs")
        return False
    
    # Compare feature vectors
    for fp1, fp2 in zip(fingerprints_1, fingerprints_2):
        if fp1.session_id != fp2.session_id:
            logger.error(f"Session ID mismatch: {fp1.session_id} vs {fp2.session_id}")
            return False
            
        if not np.allclose(fp1.feature_vector, fp2.feature_vector, atol=1e-10):
            logger.error(f"Feature vectors differ for session {fp1.session_id}")
            logger.error(f"Vector 1: {fp1.feature_vector}")
            logger.error(f"Vector 2: {fp2.feature_vector}")
            return False
    
    logger.info(f"✓ Repeatability test passed for {len(fingerprints_1)} sessions")
    return True


def main():
    """Run all tests and report results"""
    logger.info("Starting Session Fingerprinting Validation Tests")
    
    # Test A/B comparisons
    scaler_results = test_standard_vs_robust_scaler()
    tempo_results = test_tempo_methods()
    
    # Test repeatability
    repeatability_ok = test_repeatability()
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    if scaler_results:
        logger.info("Scaler Comparison:")
        for scaler_type, results in scaler_results.items():
            logger.info(f"  {scaler_type}: dim={results['vector_dimension']}, valid={results['valid']}")
    
    if tempo_results:
        logger.info("Tempo Method Comparison:")
        for method, results in tempo_results.items():
            logger.info(f"  {method}: dim={results['vector_dimension']}, valid={results['valid']}")
    
    logger.info(f"Repeatability: {'✓ PASS' if repeatability_ok else '✗ FAIL'}")
    
    # Success criteria check
    success_criteria_met = True
    
    if scaler_results:
        for results in scaler_results.values():
            if not results.get('dimension_in_range', False):
                logger.warning(f"Vector dimension {results['vector_dimension']} not in range [20-32]")
                success_criteria_met = False
            if not results.get('no_nan_inf', False):
                logger.warning("NaN/Inf values detected")
                success_criteria_met = False
    
    if not repeatability_ok:
        success_criteria_met = False
    
    logger.info(f"\nStage 1 Success Criteria: {'✓ MET' if success_criteria_met else '✗ NOT MET'}")
    
    return success_criteria_met


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)