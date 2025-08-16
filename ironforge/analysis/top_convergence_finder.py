#!/usr/bin/env python3
"""
Find Top Multi-Timeframe Convergence Nodes
==========================================

Identifies the most significant multi-timeframe convergence points with lower thresholds
to discover actual nodes where 1m/5m/15m/1h+ converge into something significant.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def find_top_convergence_nodes():
    """Find top convergence nodes with detailed analysis"""
    
    sessions_path = Path('/Users/jack/IRONFORGE/enhanced_sessions_with_relativity')
    session_files = list(sessions_path.glob('*.json'))
    
    print(f"🔗 Analyzing {len(session_files)} sessions for top convergence nodes...")
    
    convergence_nodes = []
    
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Extract key metrics
            metadata = session_data.get('session_metadata', {})
            session_date = metadata.get('session_date', 'unknown')
            session_type = metadata.get('session_type', 'unknown').upper()
            
            # HTF contamination analysis
            contamination = session_data.get('contamination_analysis', {})
            htf_data = contamination.get('htf_contamination', {})
            inheritance_data = contamination.get('cross_session_inheritance', {})
            
            htf_carryover = htf_data.get('htf_carryover_strength', 0.0)
            cross_inheritance = htf_data.get('cross_session_inheritance', 0.0)
            
            # Energy metrics
            energy_density = session_data.get('energy_density', 0.0)
            
            # Phase transitions
            phase_transitions = contamination.get('phase_transitions', 0)
            
            # Temporal flow analysis
            temporal_flow = session_data.get('temporal_flow_analysis', {})
            energy_accumulation = temporal_flow.get('energy_accumulation', {})
            momentum_strength = temporal_flow.get('temporal_momentum_strength', 0.0)
            
            # FPFVG interactions count
            total_fpfvg_interactions = 0
            if isinstance(inheritance_data, dict):
                for key, value in inheritance_data.items():
                    if 'fpfvg_interactions' in key and isinstance(value, (int, float)):
                        total_fpfvg_interactions += value
            
            # Cascade events
            cascade_events = session_data.get('cascade_events', [])
            
            # Price movements activity
            price_movements = session_data.get('price_movements', [])
            high_momentum_events = [p for p in price_movements if abs(p.get('price_momentum', 0)) > 0.5]
            
            # Calculate convergence score
            convergence_score = (
                htf_carryover * 0.3 +
                cross_inheritance * 0.2 +
                energy_density * 0.2 +
                min(total_fpfvg_interactions / 10, 1.0) * 0.15 +
                min(len(cascade_events) / 5, 1.0) * 0.1 +
                min(len(high_momentum_events) / 20, 1.0) * 0.05
            )
            
            if convergence_score > 0.4:  # Lower threshold for discovery
                
                node = {
                    'node_id': f"{session_date}_{session_type}",
                    'session_date': session_date,
                    'session_type': session_type,
                    'convergence_score': convergence_score,
                    'htf_carryover_strength': htf_carryover,
                    'cross_session_inheritance': cross_inheritance,
                    'energy_density': energy_density,
                    'phase_transitions': phase_transitions,
                    'momentum_strength': momentum_strength,
                    'total_fpfvg_interactions': total_fpfvg_interactions,
                    'cascade_events_count': len(cascade_events),
                    'high_momentum_events': len(high_momentum_events),
                    'total_price_events': len(price_movements),
                    'energy_accumulation': energy_accumulation,
                    'historical_influences': [k for k, v in htf_data.items() if v is True and 'influence' in k],
                    'session_file': session_file.name
                }
                
                convergence_nodes.append(node)
        
        except Exception as e:
            print(f"  Error processing {session_file.name}: {e}")
            continue
    
    # Sort by convergence score
    convergence_nodes.sort(key=lambda x: x['convergence_score'], reverse=True)
    
    print(f"\n🎯 Found {len(convergence_nodes)} convergence nodes with score > 0.4")
    print(f"📊 Top 10 Convergence Nodes:\n")
    
    for i, node in enumerate(convergence_nodes[:10]):
        print(f"{i+1:2d}. {node['node_id']} (Score: {node['convergence_score']:.3f})")
        print(f"    HTF Carryover: {node['htf_carryover_strength']:.3f}")
        print(f"    Cross-Session: {node['cross_session_inheritance']:.3f}")
        print(f"    Energy Density: {node['energy_density']:.3f}")
        print(f"    FPFVG Interactions: {node['total_fpfvg_interactions']}")
        print(f"    Cascade Events: {node['cascade_events_count']}")
        print(f"    High Momentum Events: {node['high_momentum_events']}")
        print(f"    Historical Influences: {len(node['historical_influences'])}")
        print()
    
    # Save detailed analysis
    output_path = '/Users/jack/IRONFORGE/analysis/top_convergence_nodes.json'
    
    report = {
        'analysis_metadata': {
            'analysis_type': 'top_convergence_discovery',
            'sessions_analyzed': len(session_files),
            'nodes_found': len(convergence_nodes),
            'min_convergence_threshold': 0.4
        },
        'top_nodes': convergence_nodes[:20],  # Top 20 nodes
        'convergence_summary': {
            'highest_score': convergence_nodes[0]['convergence_score'] if convergence_nodes else 0,
            'highest_htf_carryover': max([n['htf_carryover_strength'] for n in convergence_nodes]) if convergence_nodes else 0,
            'avg_convergence_score': np.mean([n['convergence_score'] for n in convergence_nodes]) if convergence_nodes else 0,
            'nodes_with_energy_accumulation': len([n for n in convergence_nodes if n['energy_accumulation']]),
            'nodes_with_cascade_events': len([n for n in convergence_nodes if n['cascade_events_count'] > 0]),
            'session_type_distribution': dict(Counter([n['session_type'] for n in convergence_nodes]))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"💾 Detailed analysis saved to: {output_path}")
    
    # Return the top 3 nodes for detailed examination
    return convergence_nodes[:3]

if __name__ == "__main__":
    from collections import Counter
    top_nodes = find_top_convergence_nodes()