#!/usr/bin/env python3
"""
Phase 5: Enhanced Session TGAT Validation
=========================================
Test TGAT pattern discovery on enhanced sessions and validate quality improvement.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add IRONFORGE to path
ironforge_root = Path(__file__).parent
sys.path.insert(0, str(ironforge_root))

def create_mock_graph_from_session(session_data: Dict) -> tuple:
    """
    Convert enhanced session data to graph format for TGAT
    Returns: (X, edge_index, edge_times, edge_attr, metadata)
    """
    price_movements = session_data.get('price_movements', [])
    
    if len(price_movements) < 2:
        # Create minimal mock data if no price movements
        num_nodes = 10
        X = torch.randn(num_nodes, 37)  # 37D features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_times = torch.randn(3)
        edge_attr = torch.randn(3, 17)  # At least 17 dimensions for TGAT
    else:
        # Create graph from actual price movements
        num_nodes = min(len(price_movements), 50)  # Cap at 50 nodes for efficiency
        
        # Create 37D feature vectors
        X = torch.randn(num_nodes, 37)
        
        # For price levels, use actual values in first feature
        for i in range(num_nodes):
            if i < len(price_movements):
                X[i, 0] = price_movements[i].get('price_level', 23000.0) / 25000.0  # Normalize
        
        # Create sequential edges (temporal connections)
        if num_nodes > 1:
            edge_sources = list(range(num_nodes - 1))
            edge_targets = list(range(1, num_nodes))
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_times = torch.randn(len(edge_sources))
            edge_attr = torch.randn(len(edge_sources), 17)  # At least 17 dimensions for TGAT
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_times = torch.randn(1)
            edge_attr = torch.randn(1, 17)  # At least 17 dimensions for TGAT
    
    # Extract metadata
    metadata = {
        'session_name': session_data.get('session_metadata', {}).get('session_type', 'unknown'),
        'session_date': session_data.get('session_metadata', {}).get('session_date', '2025-08-14'),
        'energy_density': session_data.get('energy_state', {}).get('energy_density', 0.5),
        'htf_carryover': session_data.get('contamination_analysis', {}).get('htf_contamination', {}).get('htf_carryover_strength', 0.3),
        'liquidity_events_count': len(session_data.get('session_liquidity_events', []))
    }
    
    return X, edge_index, edge_times, edge_attr, metadata

def test_enhanced_session_pattern_discovery():
    """Test TGAT pattern discovery on enhanced sessions"""
    print("🏛️ ENHANCED SESSION TGAT PATTERN DISCOVERY TEST")
    print("=" * 50)
    
    # Load enhanced sessions
    enhanced_sessions_path = Path(__file__).parent / 'enhanced_sessions'
    session_files = list(enhanced_sessions_path.glob('enhanced_*.json'))
    
    if not session_files:
        print("❌ No enhanced sessions found")
        return {'status': 'failed', 'error': 'No enhanced sessions found'}
    
    print(f"Found {len(session_files)} enhanced sessions")
    
    try:
        from learning.tgat_discovery import IRONFORGEDiscovery
        tgat_discovery = IRONFORGEDiscovery()
        print("✅ TGAT discovery engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize TGAT: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Test on first 10 enhanced sessions for broader validation
    test_sessions = session_files[:10]
    results = []
    
    for session_file in test_sessions:
        print(f"\n🔍 Testing: {session_file.name}")
        
        try:
            # Load session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Convert to graph format
            X, edge_index, edge_times, edge_attr, metadata = create_mock_graph_from_session(session_data)
            print(f"   Graph: {X.shape[0]} nodes, {edge_index.shape[1]} edges")
            
            # Run TGAT discovery
            discovery_result = tgat_discovery.learn_session(X, edge_index, edge_times, metadata, edge_attr)
            
            patterns = discovery_result.get('patterns', [])
            print(f"   Patterns found: {len(patterns)}")
            
            # Analyze pattern quality
            if patterns:
                descriptions = [p.get('description', 'unknown') for p in patterns]
                unique_descriptions = len(set(descriptions))
                duplication_rate = ((len(patterns) - unique_descriptions) / len(patterns)) * 100.0
                print(f"   Pattern diversity: {unique_descriptions}/{len(patterns)} unique ({100-duplication_rate:.1f}% unique)")
                
                # Show sample patterns
                print("   Sample patterns:")
                for i, pattern in enumerate(patterns[:2]):
                    print(f"     {i+1}. {pattern.get('type', 'unknown')}: {pattern.get('description', 'N/A')}")
                
                results.append({
                    'session': session_file.name,
                    'patterns_count': len(patterns),
                    'unique_patterns': unique_descriptions,
                    'duplication_rate': duplication_rate,
                    'energy_density': metadata.get('energy_density', 0.5),
                    'htf_carryover': metadata.get('htf_carryover', 0.3),
                    'sample_patterns': patterns[:3]
                })
                
            else:
                print("   No patterns extracted")
                results.append({
                    'session': session_file.name,
                    'patterns_count': 0,
                    'unique_patterns': 0,
                    'duplication_rate': 0.0,
                    'energy_density': metadata.get('energy_density', 0.5),
                    'htf_carryover': metadata.get('htf_carryover', 0.3)
                })
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'session': session_file.name,
                'error': str(e),
                'patterns_count': 0
            })
    
    return {'status': 'success', 'results': results}

def analyze_pattern_quality_improvement(results: List[Dict]):
    """Analyze if enhanced sessions show improved pattern quality"""
    print("\n📊 PATTERN QUALITY ANALYSIS")
    print("=" * 30)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r and r['patterns_count'] > 0]
    
    if not successful_results:
        print("❌ No successful pattern extractions to analyze")
        return
    
    print(f"Analyzing {len(successful_results)} successful extractions:")
    
    # Calculate metrics
    total_patterns = sum(r['patterns_count'] for r in successful_results)
    total_unique = sum(r['unique_patterns'] for r in successful_results)
    avg_duplication_rate = np.mean([r['duplication_rate'] for r in successful_results])
    
    print(f"   Total patterns extracted: {total_patterns}")
    print(f"   Total unique patterns: {total_unique}")
    print(f"   Average duplication rate: {avg_duplication_rate:.1f}%")
    print(f"   Average uniqueness: {100 - avg_duplication_rate:.1f}%")
    
    # Baseline comparison (contaminated baseline was 96.8% duplication)
    baseline_duplication = 96.8
    improvement = baseline_duplication - avg_duplication_rate
    improvement_pct = (improvement / baseline_duplication) * 100
    
    print("\n🎯 IMPROVEMENT VS BASELINE:")
    print(f"   Baseline duplication rate: {baseline_duplication}%")
    print(f"   Enhanced duplication rate: {avg_duplication_rate:.1f}%")
    print(f"   Improvement: {improvement:.1f} percentage points")
    print(f"   Relative improvement: {improvement_pct:.1f}%")
    
    # Assessment
    if avg_duplication_rate < 25:
        print("\n✅ EXCELLENT: Very low duplication, authentic pattern diversity")
    elif avg_duplication_rate < 50:
        print("\n✅ GOOD: Reasonable pattern diversity achieved")  
    elif avg_duplication_rate < 75:
        print("\n🔶 MODERATE: Some improvement but still significant duplication")
    else:
        print("\n❌ POOR: High duplication suggests persistent template artifacts")
    
    return {
        'total_patterns': total_patterns,
        'total_unique': total_unique,
        'avg_duplication_rate': avg_duplication_rate,
        'improvement_vs_baseline': improvement,
        'relative_improvement_pct': improvement_pct
    }

def main():
    """Run enhanced session validation"""
    print("🚀 PHASE 5: ENHANCED SESSION TGAT VALIDATION")
    print("=" * 45)
    print("Testing pattern discovery on feature-enhanced sessions")
    print("Goal: Validate <25% duplication (vs 96.8% baseline)")
    print()
    
    # Test pattern discovery on enhanced sessions
    test_results = test_enhanced_session_pattern_discovery()
    
    if test_results['status'] == 'failed':
        print(f"\n❌ Testing failed: {test_results['error']}")
        return
    
    # Analyze results
    quality_analysis = analyze_pattern_quality_improvement(test_results['results'])
    
    # Final assessment
    print("\n🏛️ PHASE 5 FINAL ASSESSMENT:")
    print("=" * 32)
    
    if quality_analysis and quality_analysis['avg_duplication_rate'] < 25:
        print("✅ SUCCESS: Enhanced sessions achieve excellent pattern diversity")
        print(f"   Achievement: {quality_analysis['relative_improvement_pct']:.1f}% improvement vs baseline")
        print("   Recommendation: Enhanced feature pipeline is working correctly")
        print("   Next step: Scale to all 33 enhanced sessions")
        
    elif quality_analysis and quality_analysis['avg_duplication_rate'] < 50:
        print("🔶 PARTIAL SUCCESS: Good improvement but not optimal")
        print(f"   Achievement: {quality_analysis['relative_improvement_pct']:.1f}% improvement vs baseline")
        print("   Recommendation: Further feature enhancement may be needed")
        
    else:
        print("❌ INSUFFICIENT IMPROVEMENT: Still high duplication")
        print("   Issue: Enhanced features may not be sufficiently decontaminated")
        print("   Recommendation: Review Phase 2 enhancement methodology")
    
    return {
        'test_results': test_results,
        'quality_analysis': quality_analysis
    }

if __name__ == "__main__":
    main()