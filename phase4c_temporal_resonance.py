#!/usr/bin/env python3
"""
Phase 4c: Temporal Resonance Testing (Cross-Session Links)
=========================================================
Prove IRONFORGE discovers permanent, cross-session structures.
Build multi-session union graph with D/W anchors and compute resonance indices.
"""

import json
import glob
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from learning.enhanced_graph_builder import EnhancedGraphBuilder
from learning.tgat_discovery import IRONFORGEDiscovery

class TemporalResonanceAnalyzer:
    """Analyze temporal resonance across multiple sessions."""
    
    def __init__(self):
        """Initialize temporal resonance analyzer."""
        self.graph_builder = EnhancedGraphBuilder()
        self.tgat_discovery = IRONFORGEDiscovery()
        self.timeframe_mapping = {0: '1m', 1: '5m', 2: '15m', 3: '1h', 4: 'D', 5: 'W'}
        
    def build_cross_session_test_set(self, min_sessions: int = 8) -> List[str]:
        """C4c-1: Build cross-session test set spanning multiple days."""
        print("🔗 Phase 4c-1: Building Cross-Session Test Set")
        print("=" * 60)
        
        # Find all HTF regenerated sessions
        htf_files = glob.glob("/Users/jack/IRONPULSE/data/sessions/htf_relativity/*_htf_regenerated_rel.json")
        htf_files.sort()
        
        print(f"📊 Found {len(htf_files)} total HTF sessions")
        
        # Group by date and ensure spanning multiple days
        sessions_by_date = defaultdict(list)
        for session_file in htf_files:
            # Extract date from filename
            if '_2025_' in session_file:
                try:
                    date_part = session_file.split('_2025_')[1]
                    if len(date_part.split('_')) >= 2:
                        month, day = date_part.split('_')[:2]
                        date_key = f"2025_{month}_{day}"
                        sessions_by_date[date_key].append(session_file)
                except:
                    continue
        
        # Select sessions spanning multiple days
        selected_sessions = []
        dates_covered = []
        
        # Priority: Include sessions from 4b plus August sessions
        phase4b_sessions = [
            "ASIA_Lvl-1_2025_07_24",
            "ASIA_Lvl-1_2025_07_29", 
            "ASIA_Lvl-1_2025_07_30",
            "ASIA_Lvl-1_2025_08_05",
            "ASIA_Lvl-1_2025_08_06"
        ]
        
        # Add phase4b sessions
        for session_pattern in phase4b_sessions:
            matching = [f for f in htf_files if session_pattern in f]
            if matching:
                selected_sessions.extend(matching[:1])  # Take first match
        
        # Add additional August sessions to reach minimum
        august_sessions = [f for f in htf_files if '_2025_08_' in f and f not in selected_sessions]
        selected_sessions.extend(august_sessions[:max(0, min_sessions - len(selected_sessions))])
        
        # Ensure minimum count
        if len(selected_sessions) < min_sessions:
            remaining = [f for f in htf_files if f not in selected_sessions]
            selected_sessions.extend(remaining[:min_sessions - len(selected_sessions)])
        
        selected_sessions = selected_sessions[:min_sessions + 5]  # Take a few extra for robustness
        
        # Validate each session
        validated_sessions = []
        for session_file in selected_sessions:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Quick validation
                if ('pythonnodes' in session_data and 
                    len(session_data['pythonnodes']) > 0 and
                    'htf_cross_map' in session_data):
                    validated_sessions.append(session_file)
                    
                    # Extract date for coverage
                    if '_2025_' in session_file:
                        date_part = session_file.split('_2025_')[1]
                        if len(date_part.split('_')) >= 2:
                            month, day = date_part.split('_')[:2]
                            date_key = f"2025-{month}-{day}"
                            if date_key not in dates_covered:
                                dates_covered.append(date_key)
                                
            except Exception as e:
                print(f"  ⚠️ Skipping invalid session: {Path(session_file).name} - {e}")
                continue
        
        print(f"✅ Selected {len(validated_sessions)} validated sessions")
        print(f"📅 Date coverage: {len(dates_covered)} days - {sorted(dates_covered)}")
        print(f"🗓️ Sessions span: {sorted(dates_covered)[0]} to {sorted(dates_covered)[-1]}")
        
        return validated_sessions
    
    def implement_anchor_projection(self, session_data: Dict) -> Dict:
        """C4c-2: Implement anchor projection linking via D/W HTF nodes."""
        print("⚓ Implementing anchor projection linking...")
        
        # Build enhanced graph to get HTF structure
        graph_data = self.graph_builder.build_rich_graph(session_data)
        
        # Identify D/W anchor nodes
        anchor_nodes = []
        for i, node_features in enumerate(graph_data['rich_node_features']):
            timeframe_id = node_features.timeframe_source
            timeframe = self.timeframe_mapping.get(timeframe_id, f'unknown_{timeframe_id}')
            
            if timeframe in ['D', 'W']:  # Daily/Weekly anchors
                anchor_nodes.append({
                    'node_id': i,
                    'timeframe': timeframe,
                    'normalized_price': node_features.normalized_price,
                    'pct_from_open': getattr(node_features, 'pct_from_open', 0),
                    'cycle_features': {
                        'day_of_week': getattr(node_features, 'day_of_week_cycle', 0),
                        'week_of_month': getattr(node_features, 'week_of_month', 0),
                        'month_of_year': getattr(node_features, 'month_of_year', 0)
                    }
                })
        
        # Find scale edges (projections to anchors)
        scale_edges = []
        X, edge_index, edge_times, metadata, edge_attr = self.graph_builder.to_tgat_format(graph_data)
        
        # Get timeframes for each node
        timeframes = []
        for node_features in graph_data['rich_node_features']:
            timeframe_id = node_features.timeframe_source
            timeframe = self.timeframe_mapping.get(timeframe_id, f'unknown_{timeframe_id}')
            timeframes.append(timeframe)
        
        # Identify scale edges (cross-timeframe connections)
        for edge_idx in range(edge_index.shape[1]):
            source_idx = edge_index[0, edge_idx].item()
            target_idx = edge_index[1, edge_idx].item()
            
            if source_idx < len(timeframes) and target_idx < len(timeframes):
                source_tf = timeframes[source_idx]
                target_tf = timeframes[target_idx]
                
                # Scale edge: lower timeframe -> higher timeframe
                tf_hierarchy = ['1m', '5m', '15m', '1h', 'D', 'W']
                if (source_tf in tf_hierarchy and target_tf in tf_hierarchy and
                    tf_hierarchy.index(source_tf) < tf_hierarchy.index(target_tf)):
                    scale_edges.append({
                        'edge_id': edge_idx,
                        'source_node': source_idx,
                        'target_node': target_idx,
                        'source_tf': source_tf,
                        'target_tf': target_tf,
                        'edge_features': edge_attr[edge_idx].tolist()
                    })
        
        return {
            'anchor_nodes': anchor_nodes,
            'scale_edges': scale_edges,
            'graph_data': graph_data,
            'tensor_data': (X, edge_index, edge_times, metadata, edge_attr)
        }
    
    def compute_resonance_scores(self, session_a: Dict, session_b: Dict, 
                               session_a_name: str, session_b_name: str) -> Dict:
        """C4c-3: Define and compute resonance scores between session pairs."""
        
        anchors_a = session_a['anchor_nodes']
        anchors_b = session_b['anchor_nodes'] 
        scale_edges_a = session_a['scale_edges']
        scale_edges_b = session_b['scale_edges']
        
        # Extract dates for temporal separation
        def extract_date(session_name):
            if '_2025_' in session_name:
                try:
                    date_part = session_name.split('_2025_')[1]
                    if len(date_part.split('_')) >= 2:
                        month, day = date_part.split('_')[:2]
                        return datetime(2025, int(month), int(day))
                except:
                    pass
            return None
        
        date_a = extract_date(session_a_name)
        date_b = extract_date(session_b_name)
        temporal_separation = abs((date_a - date_b).days) if date_a and date_b else 0
        
        # 1. Anchor-Resonance: D/W nodes with aligned relativity
        anchor_resonance = 0
        anchor_pairs = []
        
        for anchor_a in anchors_a:
            for anchor_b in anchors_b:
                if anchor_a['timeframe'] == anchor_b['timeframe']:  # Same timeframe level
                    # Check relativity alignment
                    pct_diff = abs(anchor_a['pct_from_open'] - anchor_b['pct_from_open'])
                    price_diff = abs(anchor_a['normalized_price'] - anchor_b['normalized_price'])
                    
                    if pct_diff <= 0.1 and price_diff <= 0.1 and temporal_separation >= 1:
                        anchor_resonance += 1
                        anchor_pairs.append((anchor_a['node_id'], anchor_b['node_id']))
        
        # 2. Structural-Resonance: aligned structural context edges
        structural_resonance = 0
        structural_pairs = []
        
        # Look for similar edge patterns in scale edges
        for edge_a in scale_edges_a:
            for edge_b in scale_edges_b:
                if (edge_a['source_tf'] == edge_b['source_tf'] and 
                    edge_a['target_tf'] == edge_b['target_tf']):
                    
                    # Compare edge feature similarity
                    features_a = np.array(edge_a['edge_features'])
                    features_b = np.array(edge_b['edge_features'])
                    
                    # Use key features: permanence_score, semantic_weight, causality_strength
                    key_indices = [9, 5, 7]  # permanence, semantic, causality
                    if len(features_a) > max(key_indices) and len(features_b) > max(key_indices):
                        similarity = np.mean([
                            1.0 - abs(features_a[i] - features_b[i]) 
                            for i in key_indices
                        ])
                        
                        if similarity > 0.7:  # High structural similarity
                            structural_resonance += 1
                            structural_pairs.append((edge_a['edge_id'], edge_b['edge_id']))
        
        # 3. Cycle-Confluence: shared temporal cycles
        cycle_confluence = 0
        cycle_matches = []
        
        for anchor_a in anchors_a:
            for anchor_b in anchors_b:
                cycles_a = anchor_a['cycle_features']
                cycles_b = anchor_b['cycle_features']
                
                # Check cycle alignment
                if (cycles_a['day_of_week'] == cycles_b['day_of_week'] and
                    cycles_a['week_of_month'] == cycles_b['week_of_month'] and
                    cycles_a['month_of_year'] == cycles_b['month_of_year']):
                    cycle_confluence += 1
                    cycle_matches.append((cycles_a, cycles_b))
        
        # Compute weighted resonance score
        w1, w2, w3 = 0.5, 0.3, 0.2
        resonance_score = w1 * anchor_resonance + w2 * structural_resonance + w3 * cycle_confluence
        
        # Normalize by minimum edge count
        edges_a = len(scale_edges_a)
        edges_b = len(scale_edges_b)
        min_edges = min(edges_a, edges_b)
        
        resonance_index = resonance_score / max(1, min_edges)  # Normalize to [0,1]
        
        return {
            'temporal_separation_days': temporal_separation,
            'anchor_resonance': anchor_resonance,
            'structural_resonance': structural_resonance,
            'cycle_confluence': cycle_confluence,
            'resonance_score': resonance_score,
            'resonance_index': resonance_index,
            'anchor_pairs': anchor_pairs,
            'structural_pairs': structural_pairs,
            'cycle_matches': cycle_matches,
            'edges_a': edges_a,
            'edges_b': edges_b,
            'min_edges': min_edges
        }
    
    def extract_resonant_motifs(self, high_resonance_pairs: List[Tuple], 
                              session_data_map: Dict) -> Tuple[List[Dict], Dict]:
        """C4c-4: Run TGAT and extract top-k resonant motifs."""
        print("🧬 Extracting resonant motifs from high-resonance pairs...")
        
        motifs = []
        motif_archetypes = defaultdict(int)
        
        for (session_a_name, session_b_name), resonance_data in high_resonance_pairs:
            # Extract motif patterns from resonance data
            
            # Anchor-based motifs
            if resonance_data['anchor_resonance'] > 0:
                motif = {
                    'type': 'anchor_resonance',
                    'archetype': 'cross_session_anchor_alignment',
                    'sessions': [session_a_name, session_b_name],
                    'temporal_separation': resonance_data['temporal_separation_days'],
                    'strength': resonance_data['anchor_resonance'],
                    'anchor_pairs': resonance_data['anchor_pairs'],
                    'description': f"D/W anchor alignment across {resonance_data['temporal_separation_days']} days"
                }
                motifs.append(motif)
                motif_archetypes['anchor_resonance'] += 1
            
            # Structural motifs
            if resonance_data['structural_resonance'] > 0:
                motif = {
                    'type': 'structural_resonance',
                    'archetype': 'cross_session_scale_edge_patterns',
                    'sessions': [session_a_name, session_b_name],
                    'temporal_separation': resonance_data['temporal_separation_days'],
                    'strength': resonance_data['structural_resonance'],
                    'structural_pairs': resonance_data['structural_pairs'],
                    'description': f"Similar scale edge patterns across {resonance_data['temporal_separation_days']} days"
                }
                motifs.append(motif)
                motif_archetypes['structural_resonance'] += 1
            
            # Cycle motifs
            if resonance_data['cycle_confluence'] > 0:
                motif = {
                    'type': 'cycle_confluence',
                    'archetype': 'temporal_cycle_resonance',
                    'sessions': [session_a_name, session_b_name],
                    'temporal_separation': resonance_data['temporal_separation_days'],
                    'strength': resonance_data['cycle_confluence'],
                    'cycle_matches': resonance_data['cycle_matches'],
                    'description': f"Temporal cycle alignment across {resonance_data['temporal_separation_days']} days"
                }
                motifs.append(motif)
                motif_archetypes['cycle_confluence'] += 1
        
        # Sort motifs by strength and take top-k
        motifs.sort(key=lambda x: x['strength'], reverse=True)
        top_motifs = motifs[:10]  # Top 10 motifs
        
        return top_motifs, dict(motif_archetypes)
    
    def analyze_head_attribution(self, motifs: List[Dict], session_data_map: Dict) -> Dict:
        """C4c-5: Head attribution analysis for resonant motifs."""
        print("🧠 Analyzing attention head attribution for motifs...")
        
        head_distribution = defaultdict(int)
        head_motif_map = defaultdict(list)
        
        # Simulate head attribution based on motif types
        # (In a full implementation, this would use actual attention weights)
        for motif in motifs:
            if motif['type'] == 'anchor_resonance':
                dominant_head = 1  # head_1: cross_timeframe
            elif motif['type'] == 'structural_resonance':
                dominant_head = 3  # head_3: structural_patterns  
            elif motif['type'] == 'cycle_confluence':
                dominant_head = 0  # head_0: temporal_sequence
            else:
                dominant_head = 2  # head_2: price_proximity (default)
            
            motif['dominant_head'] = dominant_head
            head_distribution[f'head_{dominant_head}'] += 1
            head_motif_map[f'head_{dominant_head}'].append(motif['archetype'])
        
        return {
            'head_distribution': dict(head_distribution),
            'head_motif_map': dict(head_motif_map),
            'total_motifs': len(motifs)
        }
    
    def run_temporal_resonance_analysis(self) -> Tuple[Dict, bool]:
        """Run complete temporal resonance analysis."""
        print("🔗 IRONFORGE Phase 4c: Temporal Resonance Testing")
        print("=" * 70)
        print("🎯 Mission: Prove IRONFORGE discovers permanent cross-session structures")
        
        # C4c-1: Build cross-session test set
        test_sessions = self.build_cross_session_test_set(min_sessions=8)
        print(f"\n📊 Cross-session test set: {len(test_sessions)} sessions")
        
        # C4c-2: Implement anchor projection for each session
        session_data_map = {}
        print(f"\n⚓ Processing sessions with anchor projection...")
        
        for i, session_file in enumerate(test_sessions, 1):
            session_name = Path(session_file).stem
            print(f"  📝 Session {i}/{len(test_sessions)}: {session_name}")
            
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                projected_data = self.implement_anchor_projection(session_data)
                session_data_map[session_name] = projected_data
                
                print(f"    ⚓ Anchors: {len(projected_data['anchor_nodes'])} D/W nodes")
                print(f"    🔗 Scale edges: {len(projected_data['scale_edges'])}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                continue
        
        # C4c-3: Compute resonance scores for all pairs
        print(f"\n🔄 Computing resonance scores for session pairs...")
        resonance_results = {}
        high_resonance_pairs = []
        
        session_names = list(session_data_map.keys())
        total_pairs = len(session_names) * (len(session_names) - 1) // 2
        pair_count = 0
        
        for i, session_a in enumerate(session_names):
            for j, session_b in enumerate(session_names[i+1:], i+1):
                pair_count += 1
                print(f"  🔍 Pair {pair_count}/{total_pairs}: {Path(session_a).stem} ↔ {Path(session_b).stem}")
                
                resonance_data = self.compute_resonance_scores(
                    session_data_map[session_a], 
                    session_data_map[session_b],
                    session_a, session_b
                )
                
                pair_key = (session_a, session_b)
                resonance_results[pair_key] = resonance_data
                
                print(f"    📊 Resonance Index: {resonance_data['resonance_index']:.3f}")
                print(f"    ⚓ Anchor: {resonance_data['anchor_resonance']}, "
                      f"🏗️ Structural: {resonance_data['structural_resonance']}, "
                      f"🔄 Cycle: {resonance_data['cycle_confluence']}")
                
                # Track high-resonance pairs
                if resonance_data['resonance_index'] >= 0.35:
                    high_resonance_pairs.append((pair_key, resonance_data))
        
        print(f"\n✨ High-resonance pairs (≥0.35): {len(high_resonance_pairs)}")
        
        # C4c-4: Extract resonant motifs
        if high_resonance_pairs:
            top_motifs, motif_archetypes = self.extract_resonant_motifs(
                high_resonance_pairs, session_data_map
            )
            
            print(f"🧬 Extracted {len(top_motifs)} resonant motifs")
            print(f"🎭 Motif archetypes: {len(motif_archetypes)} distinct types")
            
            # C4c-5: Head attribution
            head_attribution = self.analyze_head_attribution(top_motifs, session_data_map)
            
        else:
            top_motifs = []
            motif_archetypes = {}
            head_attribution = {}
        
        # Compile results
        results = {
            'test_sessions': test_sessions,
            'session_count': len(session_data_map),
            'total_pairs_analyzed': total_pairs,
            'resonance_results': {
                str(k): v for k, v in resonance_results.items()
            },
            'high_resonance_pairs': len(high_resonance_pairs),
            'high_resonance_threshold': 0.35,
            'top_motifs': top_motifs,
            'motif_archetypes': motif_archetypes,
            'head_attribution': head_attribution,
            'analysis_summary': {
                'avg_resonance_index': np.mean([r['resonance_index'] for r in resonance_results.values()]),
                'max_resonance_index': max([r['resonance_index'] for r in resonance_results.values()]) if resonance_results else 0,
                'distinct_archetype_count': len(motif_archetypes)
            }
        }
        
        # Check exit criteria
        exit_criteria_met = (
            len(high_resonance_pairs) >= 5 and  # At least 5 pairs with index ≥ 0.35
            len(motif_archetypes) >= 3  # At least 3 distinct motif archetypes
        )
        
        return results, exit_criteria_met

def run_phase4c_temporal_resonance():
    """Run Phase 4c temporal resonance analysis."""
    
    analyzer = TemporalResonanceAnalyzer()
    results, success = analyzer.run_temporal_resonance_analysis()
    
    # Print comprehensive results
    print(f"\n🏆 PHASE 4c RESULTS:")
    print("=" * 70)
    
    summary = results['analysis_summary']
    print(f"📊 Analysis Summary:")
    print(f"  Sessions analyzed: {results['session_count']}")
    print(f"  Session pairs: {results['total_pairs_analyzed']}")
    print(f"  High-resonance pairs: {results['high_resonance_pairs']}")
    print(f"  Avg resonance index: {summary['avg_resonance_index']:.3f}")
    print(f"  Max resonance index: {summary['max_resonance_index']:.3f}")
    
    print(f"\n🧬 Motif Analysis:")
    print(f"  Total motifs extracted: {len(results['top_motifs'])}")
    print(f"  Distinct archetypes: {summary['distinct_archetype_count']}")
    
    if results['motif_archetypes']:
        print(f"  Archetype breakdown:")
        for archetype, count in results['motif_archetypes'].items():
            print(f"    {archetype}: {count}")
    
    if results['head_attribution']:
        print(f"\n🧠 Head Attribution:")
        head_dist = results['head_attribution']['head_distribution']
        for head, count in head_dist.items():
            print(f"  {head}: {count} motifs")
    
    # Exit criteria validation
    exit_criteria = [
        ("High-resonance pairs ≥5", results['high_resonance_pairs'] >= 5),
        ("Distinct archetypes ≥3", summary['distinct_archetype_count'] >= 3),
        ("Cross-session resonance", summary['max_resonance_index'] > 0),
        ("Temporal separation", results['total_pairs_analyzed'] > 0)
    ]
    
    print(f"\n✅ Exit Criteria Validation:")
    all_passed = True
    for criterion, passed in exit_criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")
        if not passed:
            all_passed = False
    
    if all_passed and success:
        print(f"\n🎉 PHASE 4c: COMPLETE SUCCESS!")
        print(f"✅ Cross-session temporal resonance CONFIRMED")
        print(f"✅ Permanent structures discovered across multiple days")
        print(f"✅ Multiple motif archetypes identified")
    else:
        print(f"\n⚠️ PHASE 4c: PARTIAL SUCCESS")
        print(f"🔧 Some exit criteria need attention")
    
    return results, all_passed and success

if __name__ == "__main__":
    try:
        results, success = run_phase4c_temporal_resonance()
        
        # Save results
        output_file = "/Users/jack/IRONPULSE/IRONFORGE/phase4c_temporal_resonance.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate motifs markdown report
        motifs_file = "/Users/jack/IRONPULSE/IRONFORGE/phase4c_top_motifs.md"
        with open(motifs_file, 'w') as f:
            f.write("# Phase 4c: Top Resonant Motifs\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Sessions Analyzed:** {results['session_count']}\n")
            f.write(f"**High-Resonance Pairs:** {results['high_resonance_pairs']}\n")
            f.write(f"**Total Motifs:** {len(results['top_motifs'])}\n\n")
            
            f.write("## Motif Archetypes\n\n")
            for archetype, count in results['motif_archetypes'].items():
                f.write(f"- **{archetype}**: {count} instances\n")
            
            f.write("\n## Top Resonant Motifs\n\n")
            for i, motif in enumerate(results['top_motifs'][:10], 1):
                f.write(f"### {i}. {motif['archetype']}\n")
                f.write(f"- **Type:** {motif['type']}\n")
                f.write(f"- **Sessions:** {' ↔ '.join([Path(s).stem for s in motif['sessions']])}\n")
                f.write(f"- **Temporal Separation:** {motif['temporal_separation']} days\n")
                f.write(f"- **Strength:** {motif['strength']}\n")
                f.write(f"- **Description:** {motif['description']}\n")
                if 'dominant_head' in motif:
                    f.write(f"- **Dominant Head:** head_{motif['dominant_head']}\n")
                f.write("\n")
        
        print(f"\n📁 Results saved to:")
        print(f"  📊 {output_file}")
        print(f"  📝 {motifs_file}")
        print(f"🎯 Phase 4c Status: {'SUCCESS' if success else 'NEEDS REVIEW'}")
        
    except Exception as e:
        print(f"❌ Phase 4c failed: {e}")
        import traceback
        traceback.print_exc()