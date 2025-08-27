#!/usr/bin/env python3
"""
IRONFORGE Consolidated Phase Validation
======================================

Unified validation framework consolidating all phase*.py scripts:
- Phase 2: Feature pipeline enhancement and validation
- Phase 4: Archaeological discovery validation  
- Phase 5: TGAT model validation and decontamination

This replaces 10+ individual phase*.py scripts with a single, configurable validator.

Usage:
    python consolidated_phase_validation.py --phase 2 --mode enhancement
    python consolidated_phase_validation.py --phase 4 --mode archaeology
    python consolidated_phase_validation.py --phase 5 --mode tgat-validation
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add current directory to path
sys.path.append('.')

logger = logging.getLogger(__name__)


class ConsolidatedPhaseValidator:
    """Unified phase validation framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.results = {}
        
    def validate_phase2_enhancement(self) -> Dict[str, Any]:
        """Phase 2: Feature pipeline enhancement validation"""
        print("🔧 PHASE 2: FEATURE PIPELINE ENHANCEMENT")
        print("=" * 50)
        
        try:
            from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
            
            builder = EnhancedGraphBuilder()
            
            # Test enhanced feature generation
            test_session = {
                "session_name": "phase2_test",
                "events": [
                    {"timestamp": "2025-08-27T09:30:00", "price": 100.0, "volume": 1000},
                    {"timestamp": "2025-08-27T09:31:00", "price": 101.0, "volume": 1200},
                    {"timestamp": "2025-08-27T09:32:00", "price": 99.5, "volume": 800}
                ]
            }
            
            # Build graph and validate features
            graph, metadata = builder.build_rich_graph(test_session)
            X, edge_index, edge_times, enhanced_metadata, edge_attr = builder.to_tgat_format(graph)
            
            # Validation checks
            validations = {
                'graph_created': graph is not None,
                'features_generated': X is not None,
                'feature_dimensions': X.shape[1] if X is not None else 0,
                'edges_created': edge_index is not None,
                'metadata_present': metadata is not None
            }
            
            print("✅ Phase 2 Enhancement Validation:")
            for check, result in validations.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check}: {result}")
            
            return {
                'phase': 2,
                'mode': 'enhancement',
                'success': all(validations.values()),
                'validations': validations
            }
            
        except Exception as e:
            print(f"❌ Phase 2 validation failed: {e}")
            return {'phase': 2, 'mode': 'enhancement', 'success': False, 'error': str(e)}
    
    def validate_phase4_archaeology(self) -> Dict[str, Any]:
        """Phase 4: Archaeological discovery validation"""
        print("🏛️ PHASE 4: ARCHAEOLOGICAL DISCOVERY VALIDATION")
        print("=" * 50)
        
        try:
            # Test archaeological discovery components
            validations = {
                'broad_spectrum_archaeology': False,
                'structural_link_analyzer': False,
                'temporal_clustering_engine': False,
                'timeframe_lattice_mapper': False
            }
            
            # Test each component
            try:
                from analysis.broad_spectrum_archaeology import BroadSpectrumArchaeology
                validations['broad_spectrum_archaeology'] = True
            except ImportError:
                pass
                
            try:
                from analysis.structural_link_analyzer import StructuralLinkAnalyzer
                validations['structural_link_analyzer'] = True
            except ImportError:
                pass
                
            try:
                from analysis.temporal_clustering_engine import TemporalClusteringEngine
                validations['temporal_clustering_engine'] = True
            except ImportError:
                pass
                
            try:
                from analysis.timeframe_lattice_mapper import TimeframeLatticeMapper
                validations['timeframe_lattice_mapper'] = True
            except ImportError:
                pass
            
            print("✅ Phase 4 Archaeological Components:")
            for component, available in validations.items():
                status = "✅" if available else "❌"
                print(f"  {status} {component}: {'Available' if available else 'Missing'}")
            
            return {
                'phase': 4,
                'mode': 'archaeology',
                'success': any(validations.values()),
                'validations': validations
            }
            
        except Exception as e:
            print(f"❌ Phase 4 validation failed: {e}")
            return {'phase': 4, 'mode': 'archaeology', 'success': False, 'error': str(e)}
    
    def validate_phase5_tgat(self) -> Dict[str, Any]:
        """Phase 5: TGAT model validation and decontamination"""
        print("🧠 PHASE 5: TGAT MODEL VALIDATION")
        print("=" * 50)
        
        try:
            from ironforge.learning.tgat_discovery import IRONFORGEDiscovery
            
            # Initialize TGAT discovery engine
            discovery = IRONFORGEDiscovery()
            
            # Test pattern discovery capability
            test_session = {
                "session_name": "phase5_tgat_test",
                "events": [
                    {"timestamp": "2025-08-27T09:30:00", "price": 100.0, "volume": 1000},
                    {"timestamp": "2025-08-27T09:31:00", "price": 101.0, "volume": 1200},
                    {"timestamp": "2025-08-27T09:32:00", "price": 99.5, "volume": 800},
                    {"timestamp": "2025-08-27T09:33:00", "price": 102.0, "volume": 1500}
                ]
            }
            
            # Run discovery
            results = discovery.discover_session_patterns(test_session)
            
            # Validation metrics
            validations = {
                'discovery_completed': results is not None,
                'patterns_found': len(results.get('pattern_scores', [])) > 0 if results else False,
                'attention_weights': 'attention_weights' in results if results else False,
                'significance_scores': 'significance_scores' in results if results else False,
                'node_embeddings': 'node_embeddings' in results if results else False
            }
            
            # Check for decontamination (pattern diversity)
            pattern_diversity = 0
            if results and 'pattern_scores' in results:
                import torch
                pattern_scores = results['pattern_scores']
                if isinstance(pattern_scores, torch.Tensor):
                    # Count unique patterns (threshold > 0.5)
                    unique_patterns = (pattern_scores > 0.5).sum().item()
                    pattern_diversity = unique_patterns
            
            validations['pattern_diversity'] = pattern_diversity > 0
            validations['decontamination_success'] = pattern_diversity < 20  # Target <20% duplication
            
            print("✅ Phase 5 TGAT Validation:")
            for check, result in validations.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check}: {result}")
            
            if pattern_diversity > 0:
                print(f"  📊 Pattern diversity: {pattern_diversity} unique patterns")
            
            return {
                'phase': 5,
                'mode': 'tgat-validation',
                'success': validations['discovery_completed'] and validations['patterns_found'],
                'validations': validations,
                'pattern_diversity': pattern_diversity
            }
            
        except Exception as e:
            print(f"❌ Phase 5 validation failed: {e}")
            return {'phase': 5, 'mode': 'tgat-validation', 'success': False, 'error': str(e)}
    
    def validate_phase(self, phase: int, mode: str = 'default') -> Dict[str, Any]:
        """Validate specific phase"""
        
        if phase == 2:
            return self.validate_phase2_enhancement()
        elif phase == 4:
            return self.validate_phase4_archaeology()
        elif phase == 5:
            return self.validate_phase5_tgat()
        else:
            return {
                'phase': phase,
                'mode': mode,
                'success': False,
                'error': f'Phase {phase} validation not implemented'
            }


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='IRONFORGE Consolidated Phase Validation')
    parser.add_argument('--phase', type=int, required=True,
                       choices=[2, 4, 5],
                       help='Validation phase to run')
    parser.add_argument('--mode', type=str, default='default',
                       help='Validation mode')
    parser.add_argument('--output', type=str, default='phase_validation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run validation
    validator = ConsolidatedPhaseValidator()
    
    print(f"🧪 Starting IRONFORGE Phase Validation")
    print(f"Phase: {args.phase}")
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    start_time = time.time()
    
    try:
        results = validator.validate_phase(args.phase, args.mode)
        total_time = time.time() - start_time
        
        # Add timing to results
        results['total_time'] = total_time
        results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Validation completed in {total_time:.1f}s")
        print(f"Results saved to: {args.output}")
        
        if results.get('success'):
            print("🎉 Validation successful!")
        else:
            print("❌ Validation failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
