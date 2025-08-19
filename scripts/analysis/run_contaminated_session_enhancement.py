#!/usr/bin/env python3
"""
Run Phase 2 Enhancement on Contaminated Sessions
==============================================

Target the 33 contaminated TGAT-ready sessions that need decontamination.
"""

import json
import logging

from phase2_feature_pipeline_enhancement import FeaturePipelineEnhancer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_contaminated_sessions():
    """Get list of contaminated sessions that need enhancement."""
    with open('data_quality_assessment.json') as f:
        data = json.load(f)
    
    contaminated_sessions = []
    for session in data['session_assessments']:
        if (session['quality_score'] == 82.0 and 
            session['tgat_readiness'] and
            'Default energy density value (0.5)' in session['issues'] and
            'Default HTF carryover strength (0.3)' in session['issues']):
            contaminated_sessions.append(session['file'])
    
    return contaminated_sessions

def main():
    """Run enhancement on contaminated sessions."""
    logger.info("Targeting contaminated TGAT-ready sessions for enhancement")
    
    enhancer = FeaturePipelineEnhancer()
    contaminated_sessions = get_contaminated_sessions()
    
    logger.info(f"Found {len(contaminated_sessions)} contaminated sessions")
    
    # Process in batches of 10
    batch_size = 10
    total_enhanced = 0
    
    for i in range(0, len(contaminated_sessions), batch_size):
        batch = contaminated_sessions[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} sessions")
        
        for session_filename in batch:
            try:
                result = enhancer.enhance_session(session_filename)
                
                if result.get('status') == 'enhanced':
                    total_enhanced += 1
                    logger.info(f"✅ Enhanced {session_filename}: "
                              f"{result['pre_authenticity_score']:.1f}% → "
                              f"{result['post_authenticity_score']:.1f}% "
                              f"(+{result['improvement']:.1f}%)")
                elif result.get('status') == 'already_authentic':
                    logger.info(f"⚪ {session_filename} already authentic ({result['pre_authenticity_score']:.1f}%)")
                else:
                    logger.error(f"❌ Error with {session_filename}: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Exception processing {session_filename}: {e}")
    
    print(f"\n{'='*60}")
    print("CONTAMINATED SESSION ENHANCEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total Sessions Processed: {len(contaminated_sessions)}")
    print(f"Successfully Enhanced: {total_enhanced}")
    print(f"Enhancement Rate: {total_enhanced/len(contaminated_sessions)*100:.1f}%")

if __name__ == "__main__":
    main()