#!/usr/bin/env python3
"""
Demo: Online Classifier Integration
Demonstrates how to integrate the online classifier with discovery/report pipeline
"""

import logging
import json
from pathlib import Path
from ironforge.learning.online_session_classifier import create_online_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_discovery_pipeline_integration():
    """Demonstrate integration with discovery pipeline"""
    logger.info("=== DISCOVERY PIPELINE INTEGRATION DEMO ===")
    
    # Step 1: Create classifier (OFF by default)
    classifier = create_online_classifier(
        enabled=True,  # Would be controlled by CLI flag
        model_path=Path("models/session_fingerprints/v1.0.2"),
        completion_threshold=30.0,
        distance_metric="euclidean",
        confidence_method="softmax"
    )
    
    # Step 2: Simulate discovery pipeline reaching 30% completion
    sample_session_file = Path("data/adapted/adapted_enhanced_rel_NY_AM_Lvl-1_2025_07_29.json")
    if not sample_session_file.exists():
        logger.error("Sample session file not found")
        return False
    
    with open(sample_session_file, 'r') as f:
        session_data = json.load(f)
    
    # Step 3: Check if we're at 30% completion (would be done by discovery engine)
    events = session_data.get("events", [])
    total_events = len(events)
    events_at_30pct = int(total_events * 0.3)
    
    logger.info(f"Session has {total_events} total events")
    logger.info(f"30% checkpoint reached at event {events_at_30pct}")
    
    # Step 4: Classify partial session
    session_id = "NY_AM_2025-07-29"
    prediction = classifier.classify_partial_session(
        session_data,
        session_id,
        target_completion_pct=30.0
    )
    
    if prediction:
        logger.info(f"✓ Classification successful:")
        logger.info(f"  Session ID: {prediction.session_id}")
        logger.info(f"  Archetype: {prediction.archetype_id}")
        logger.info(f"  Confidence: {prediction.confidence:.3f}")
        logger.info(f"  Predicted volatility: {prediction.predicted_volatility_class}")
        logger.info(f"  HTF regime: {prediction.predicted_dominant_htf_regime}")
        logger.info(f"  Session completion: {prediction.pct_session_seen:.1f}%")
        
        # Step 5: Write sidecar to run directory
        run_dir = Path("demo_run_output")
        sidecar_path = classifier.write_sidecar(prediction, run_dir)
        logger.info(f"✓ Sidecar written: {sidecar_path}")
        
        # Step 6: Show sidecar content
        with open(sidecar_path, 'r') as f:
            sidecar_data = json.load(f)
        
        logger.info("✓ Sidecar contains:")
        for key in ["session_id", "archetype_id", "confidence", "predicted_stats"]:
            if key in sidecar_data:
                logger.info(f"  {key}: {sidecar_data[key]}")
        
        return True
    else:
        logger.error("Classification failed")
        return False


def demo_flag_controlled_operation():
    """Demonstrate flag-controlled operation"""
    logger.info("\n=== FLAG-CONTROLLED OPERATION DEMO ===")
    
    # Demonstrate OFF by default
    classifier_off = create_online_classifier(enabled=False)
    logger.info(f"Classifier disabled: enabled={classifier_off.config.enabled}")
    logger.info(f"Artifacts loaded: {classifier_off.is_loaded}")
    
    # Demonstrate how it would work when enabled
    logger.info("\nWhen enabled:")
    classifier_on = create_online_classifier(
        enabled=True,
        model_path=Path("models/session_fingerprints/v1.0.2")
    )
    logger.info(f"Classifier enabled: enabled={classifier_on.config.enabled}")
    logger.info(f"Artifacts loaded: {classifier_on.is_loaded}")
    logger.info(f"K-means clusters: {classifier_on.kmeans_model.n_clusters}")
    
    return True


def demo_confidence_documentation():
    """Document confidence score calculation"""
    logger.info("\n=== CONFIDENCE SCORE DOCUMENTATION ===")
    
    logger.info("Confidence Definition:")
    logger.info("1. Inverse Distance Method (default):")
    logger.info("   confidence = 1 / (1 + normalized_distance)")
    logger.info("   where normalized_distance = distance_to_centroid / max_distance_in_cluster_set")
    logger.info("   Range: (0, 1], higher is better")
    
    logger.info("\n2. Softmax Method:")
    logger.info("   confidence = softmax(-distances)[closest_cluster_index]")
    logger.info("   where softmax provides probability distribution over all clusters")
    logger.info("   Range: [0, 1], sum over all clusters = 1")
    
    # Demonstrate with actual calculation
    classifier = create_online_classifier(
        enabled=True,
        model_path=Path("models/session_fingerprints/v1.0.2"),
        confidence_method="inverse_distance"
    )
    
    sample_session_file = Path("data/adapted/adapted_enhanced_rel_NY_AM_Lvl-1_2025_07_29.json")
    if sample_session_file.exists():
        with open(sample_session_file, 'r') as f:
            session_data = json.load(f)
        
        # Get partial fingerprint and distances
        partial_data = classifier.extract_partial_session_data(session_data, 30.0)
        if partial_data:
            fingerprint = classifier.compute_partial_fingerprint(partial_data)
            distances = classifier.compute_distances_to_centroids(fingerprint)
            closest_idx = np.argmin(distances)
            
            logger.info(f"\nExample calculation:")
            logger.info(f"  Distances to centroids: {distances}")
            logger.info(f"  Closest cluster: {closest_idx} (distance: {distances[closest_idx]:.3f})")
            
            # Show both confidence methods
            conf_inverse = classifier.compute_confidence(distances, closest_idx)
            
            classifier.config.confidence_method = "softmax"
            conf_softmax = classifier.compute_confidence(distances, closest_idx)
            
            logger.info(f"  Inverse distance confidence: {conf_inverse:.3f}")
            logger.info(f"  Softmax confidence: {conf_softmax:.3f}")
    
    return True


def main():
    """Run integration demos"""
    logger.info("Online Classifier Integration Demonstration")
    logger.info("="*60)
    
    results = []
    
    # Demo components
    results.append(demo_flag_controlled_operation())
    results.append(demo_discovery_pipeline_integration()) 
    results.append(demo_confidence_documentation())
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION DEMO SUMMARY")
    logger.info("="*60)
    
    all_successful = all(results)
    
    logger.info("Demo Components:")
    components = [
        "Flag-controlled operation",
        "Discovery pipeline integration", 
        "Confidence score documentation"
    ]
    
    for component, success in zip(components, results):
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {component}: {status}")
    
    logger.info(f"\nIntegration Pattern:")
    logger.info(f"  1. Check if classifier is enabled (flag)")
    logger.info(f"  2. Load artifacts once at startup")
    logger.info(f"  3. At 30% session completion checkpoint:")
    logger.info(f"     - Extract partial session data")
    logger.info(f"     - Compute partial fingerprint")
    logger.info(f"     - Classify and assign confidence")
    logger.info(f"     - Write session_fingerprint.json sidecar")
    logger.info(f"  4. Continue normal discovery/report processing")
    
    logger.info(f"\nDemo Status: {'✅ COMPLETE' if all_successful else '❌ INCOMPLETE'}")
    
    return all_successful


if __name__ == "__main__":
    import numpy as np  # Needed for demo_confidence_documentation
    success = main()
    exit(0 if success else 1)