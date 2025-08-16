#!/usr/bin/env python3
"""
IRONFORGE Semantic Market Events Analyzer
=========================================

Analyzes semantic features and market events within range clusters to provide:
1. Types of liquidity events (FVG types, sweep patterns, etc.)
2. Specific market structures present at each level
3. Order flow characteristics and patterns  
4. Session liquidity event signatures
5. HTF structural elements that define each cluster

Focuses on deep archaeological intelligence of market behavior at each range level.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import re
from datetime import datetime
from pathlib import Path
import logging

class SemanticMarketEventsAnalyzer:
    """
    Analyzes semantic features and market events within range clusters
    """
    
    def __init__(self, patterns_file: str = None):
        self.logger = logging.getLogger('semantic_market_events_analyzer')
        
        # Load discovered patterns
        if patterns_file is None:
            patterns_file = '/Users/jack/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json'
        
        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)
        
        # Initialize analysis structures
        self.range_clusters = self._extract_range_clusters()
        self.semantic_analysis = {}
        
        print(f"🔍 Semantic Market Events Analyzer initialized")
        print(f"  Patterns loaded: {len(self.patterns)}")
        print(f"  Range clusters: {list(self.range_clusters.keys())}")
    
    def _extract_range_clusters(self) -> Dict[str, List[Tuple[int, Dict]]]:
        """Extract and organize patterns by range levels"""
        range_clusters = defaultdict(list)
        
        for i, pattern in enumerate(self.patterns):
            desc = pattern.get('description', '')
            range_match = re.search(r'(\d+\.?\d*)% of range', desc)
            
            if range_match:
                range_pct = float(range_match.group(1))
                
                # Group into range buckets  
                if 15 <= range_pct < 25:
                    range_clusters['20%'].append((i, pattern))
                elif 35 <= range_pct < 45:
                    range_clusters['40%'].append((i, pattern))
                elif 55 <= range_pct < 65:
                    range_clusters['60%'].append((i, pattern))
                elif 75 <= range_pct < 85:
                    range_clusters['80%'].append((i, pattern))
        
        return range_clusters
    
    def analyze_liquidity_events(self) -> Dict:
        """Analyze liquidity events by range level"""
        print("💧 Analyzing liquidity events by range level...")
        
        liquidity_analysis = {}
        
        for range_level, patterns_list in self.range_clusters.items():
            print(f"  Analyzing {range_level} liquidity events...")
            
            analysis = self._analyze_range_liquidity_events(range_level, patterns_list)
            liquidity_analysis[range_level] = analysis
        
        return liquidity_analysis
    
    def _analyze_range_liquidity_events(self, range_level: str, patterns_list: List[Tuple[int, Dict]]) -> Dict:
        """Analyze liquidity events for a specific range level"""
        
        # Extract liquidity-related features
        fvg_events = []
        sweep_events = []
        pd_array_events = []
        consolidation_events = []
        expansion_events = []
        redelivery_events = []
        
        # Analyze constant features (these were filtered out, meaning they were present)
        liquidity_signatures = {
            'fvg_redelivery_flag': 0,
            'pd_array_interaction_flag': 0,
            'liq_sweep_flag': 0,
            'consolidation_flag': 0,
            'expansion_phase_flag': 0
        }
        
        event_types_found = []
        energy_states = []
        pattern_strengths = []
        
        for idx, pattern in patterns_list:
            semantic_context = pattern.get('semantic_context', {})
            
            # Event types
            events = semantic_context.get('event_types', [])
            event_types_found.extend(events)
            
            # Constant features analysis (liquidity events that were filtered)
            const_ctx = semantic_context.get('constant_features_context', {})
            const_names = const_ctx.get('constant_names', [])
            
            for liquidity_event in liquidity_signatures.keys():
                if liquidity_event in const_names:
                    liquidity_signatures[liquidity_event] += 1
            
            # Structural context for order flow
            structural = semantic_context.get('structural_context', {})
            energy_state = structural.get('energy_state', {})
            pattern_strength = structural.get('pattern_strength', 0)
            
            if energy_state:
                energy_states.append(energy_state)
            if pattern_strength:
                pattern_strengths.append(pattern_strength)
        
        # Calculate liquidity event frequencies
        total_patterns = len(patterns_list)
        
        liquidity_event_analysis = {
            'total_patterns': total_patterns,
            'fvg_events': {
                'count': liquidity_signatures['fvg_redelivery_flag'],
                'frequency': liquidity_signatures['fvg_redelivery_flag'] / total_patterns,
                'types': ['redelivery', 'first_presented', 'continuation']
            },
            'sweep_events': {
                'count': liquidity_signatures['liq_sweep_flag'],
                'frequency': liquidity_signatures['liq_sweep_flag'] / total_patterns,
                'types': ['buy_side_sweep', 'sell_side_sweep', 'double_sweep']
            },
            'pd_array_events': {
                'count': liquidity_signatures['pd_array_interaction_flag'],
                'frequency': liquidity_signatures['pd_array_interaction_flag'] / total_patterns,
                'types': ['premium_rejection', 'discount_acceptance', 'equilibrium_test']
            },
            'consolidation_events': {
                'count': liquidity_signatures['consolidation_flag'],
                'frequency': liquidity_signatures['consolidation_flag'] / total_patterns,
                'types': ['range_consolidation', 'accumulation', 'distribution']
            },
            'expansion_events': {
                'count': liquidity_signatures['expansion_phase_flag'],
                'frequency': liquidity_signatures['expansion_phase_flag'] / total_patterns,
                'types': ['breakout_expansion', 'momentum_expansion', 'volatility_expansion']
            },
            'semantic_events': dict(Counter(event_types_found)),
            'order_flow_characteristics': {
                'avg_pattern_strength': np.mean(pattern_strengths) if pattern_strengths else 0,
                'energy_events': len(energy_states),
                'energy_density': len(energy_states) / total_patterns
            }
        }
        
        return liquidity_event_analysis
    
    def analyze_market_structures(self) -> Dict:
        """Analyze market structures present at each range level"""
        print("🏗️ Analyzing market structures by range level...")
        
        structure_analysis = {}
        
        for range_level, patterns_list in self.range_clusters.items():
            print(f"  Analyzing {range_level} market structures...")
            
            analysis = self._analyze_range_market_structures(range_level, patterns_list)
            structure_analysis[range_level] = analysis
        
        return structure_analysis
    
    def _analyze_range_market_structures(self, range_level: str, patterns_list: List[Tuple[int, Dict]]) -> Dict:
        """Analyze market structures for a specific range level"""
        
        regime_labels = []
        structural_dominance = []
        temporal_dominance = []
        price_preferences = []
        regime_confidence = []
        
        archaeological_significance = []
        permanence_scores = []
        generalizability_scores = []
        
        for idx, pattern in patterns_list:
            # Regime characteristics
            regime_chars = pattern.get('regime_characteristics', {})
            
            regime_label = regime_chars.get('regime_label', '')
            if regime_label:
                regime_labels.append(regime_label)
            
            structural_dom = regime_chars.get('structural_dominance', '')
            if structural_dom:
                structural_dominance.append(structural_dom)
            
            temporal_dom = regime_chars.get('temporal_dominance', '')
            if temporal_dom:
                temporal_dominance.append(temporal_dom)
            
            price_pref = regime_chars.get('price_range_preference', '')
            if price_pref:
                price_preferences.append(price_pref)
            
            confidence = regime_chars.get('confidence', 0)
            if confidence:
                regime_confidence.append(confidence)
            
            # Archaeological significance
            arch_sig = pattern.get('archaeological_significance', {})
            if arch_sig:
                archaeological_significance.append(arch_sig.get('overall_significance', 0))
                permanence_scores.append(arch_sig.get('permanence_score', 0))
                generalizability_scores.append(arch_sig.get('generalizability_score', 0))
        
        structure_analysis = {
            'regime_distribution': dict(Counter(regime_labels)),
            'structural_dominance_patterns': dict(Counter(structural_dominance)),
            'temporal_dominance_patterns': dict(Counter(temporal_dominance)),
            'price_range_preferences': dict(Counter(price_preferences)),
            'regime_confidence': {
                'avg_confidence': np.mean(regime_confidence) if regime_confidence else 0,
                'high_confidence_rate': len([c for c in regime_confidence if c > 0.8]) / len(regime_confidence) if regime_confidence else 0
            },
            'archaeological_characteristics': {
                'avg_significance': np.mean(archaeological_significance) if archaeological_significance else 0,
                'avg_permanence': np.mean(permanence_scores) if permanence_scores else 0,
                'avg_generalizability': np.mean(generalizability_scores) if generalizability_scores else 0,
                'high_significance_rate': len([s for s in archaeological_significance if s > 0.5]) / len(archaeological_significance) if archaeological_significance else 0
            }
        }
        
        return structure_analysis
    
    def analyze_order_flow_characteristics(self) -> Dict:
        """Analyze order flow characteristics and patterns"""
        print("💹 Analyzing order flow characteristics by range level...")
        
        order_flow_analysis = {}
        
        for range_level, patterns_list in self.range_clusters.items():
            print(f"  Analyzing {range_level} order flow...")
            
            analysis = self._analyze_range_order_flow(range_level, patterns_list)
            order_flow_analysis[range_level] = analysis
        
        return order_flow_analysis
    
    def _analyze_range_order_flow(self, range_level: str, patterns_list: List[Tuple[int, Dict]]) -> Dict:
        """Analyze order flow for a specific range level"""
        
        pattern_strengths = []
        coherence_scores = []
        stability_scores = []
        confidence_scores = []
        node_counts = []
        time_spans = []
        
        energy_characteristics = []
        liquidity_environments = []
        
        for idx, pattern in patterns_list:
            # Pattern strength and coherence
            structural = pattern.get('semantic_context', {}).get('structural_context', {})
            pattern_strength = structural.get('pattern_strength', 0)
            if pattern_strength:
                pattern_strengths.append(pattern_strength)
            
            # Coherence and stability
            coherence = pattern.get('coherence_score', 0)
            if coherence:
                coherence_scores.append(coherence)
            
            stability = pattern.get('stability_score', 0)
            if stability:
                stability_scores.append(stability)
            
            confidence = pattern.get('confidence', 0)
            if confidence:
                confidence_scores.append(confidence)
            
            # Structural characteristics
            node_count = pattern.get('node_count', 0)
            if node_count:
                node_counts.append(node_count)
            
            time_span = pattern.get('time_span_hours', 0)
            if time_span is not None:
                time_spans.append(time_span)
            
            # Energy and liquidity
            energy_state = structural.get('energy_state', {})
            if energy_state:
                energy_characteristics.append(energy_state)
            
            liq_env = structural.get('liquidity_environment', [])
            if liq_env:
                liquidity_environments.extend(liq_env)
        
        order_flow_analysis = {
            'pattern_strength_metrics': {
                'avg_strength': np.mean(pattern_strengths) if pattern_strengths else 0,
                'max_strength': np.max(pattern_strengths) if pattern_strengths else 0,
                'min_strength': np.min(pattern_strengths) if pattern_strengths else 0,
                'strong_patterns_rate': len([s for s in pattern_strengths if s > 0.7]) / len(pattern_strengths) if pattern_strengths else 0
            },
            'coherence_metrics': {
                'avg_coherence': np.mean(coherence_scores) if coherence_scores else 0,
                'coherence_std': np.std(coherence_scores) if coherence_scores else 0,
                'high_coherence_rate': len([c for c in coherence_scores if c > 5.0]) / len(coherence_scores) if coherence_scores else 0
            },
            'stability_metrics': {
                'avg_stability': np.mean(stability_scores) if stability_scores else 0,
                'stability_consistency': 1.0 - np.std(stability_scores) if len(stability_scores) > 1 else 1.0,
                'high_stability_rate': len([s for s in stability_scores if s > 0.9]) / len(stability_scores) if stability_scores else 0
            },
            'structural_characteristics': {
                'avg_node_count': np.mean(node_counts) if node_counts else 0,
                'avg_time_span': np.mean(time_spans) if time_spans else 0,
                'time_span_std': np.std(time_spans) if time_spans else 0
            },
            'energy_signatures': {
                'energy_events_count': len(energy_characteristics),
                'energy_density': len(energy_characteristics) / len(patterns_list)
            },
            'liquidity_flow_patterns': {
                'liquidity_events_count': len(liquidity_environments),
                'liquidity_density': len(liquidity_environments) / len(patterns_list),
                'unique_liquidity_types': len(set(liquidity_environments)) if liquidity_environments else 0
            }
        }
        
        return order_flow_analysis
    
    def analyze_htf_structural_elements(self) -> Dict:
        """Analyze HTF structural elements that define each cluster"""
        print("🔗 Analyzing HTF structural elements by range level...")
        
        htf_analysis = {}
        
        for range_level, patterns_list in self.range_clusters.items():
            print(f"  Analyzing {range_level} HTF elements...")
            
            analysis = self._analyze_range_htf_elements(range_level, patterns_list)
            htf_analysis[range_level] = analysis
        
        return htf_analysis
    
    def _analyze_range_htf_elements(self, range_level: str, patterns_list: List[Tuple[int, Dict]]) -> Dict:
        """Analyze HTF structural elements for a specific range level"""
        
        relationship_types = []
        anchor_timeframes = []
        relativity_types = []
        
        # HTF-specific constant features
        htf_features = []
        cross_tf_confluence = 0
        temporal_echo_strength = 0
        scaling_factors = 0
        temporal_stability = 0
        
        confidence_scores = []
        temporal_contexts = []
        
        for idx, pattern in patterns_list:
            # Relationship analysis
            semantic_context = pattern.get('semantic_context', {})
            rel_type = semantic_context.get('relationship_type', '')
            if rel_type:
                relationship_types.append(rel_type)
            
            # Timeframe analysis
            anchor_tf = pattern.get('anchor_timeframe', '')
            if anchor_tf:
                anchor_timeframes.append(anchor_tf)
            
            relativity_type = pattern.get('relativity_type', '')
            if relativity_type:
                relativity_types.append(relativity_type)
            
            # HTF constant features (these were filtered out = present)
            const_ctx = semantic_context.get('constant_features_context', {})
            const_names = const_ctx.get('constant_names', [])
            
            htf_feature_names = ['cross_tf_confluence', 'temporal_echo_strength', 'scaling_factor', 'temporal_stability']
            
            for htf_feature in htf_feature_names:
                if htf_feature in const_names:
                    if htf_feature == 'cross_tf_confluence':
                        cross_tf_confluence += 1
                    elif htf_feature == 'temporal_echo_strength':
                        temporal_echo_strength += 1
                    elif htf_feature == 'scaling_factor':
                        scaling_factors += 1
                    elif htf_feature == 'temporal_stability':
                        temporal_stability += 1
            
            # Temporal context analysis
            temporal_ctx = pattern.get('temporal_context', {})
            if temporal_ctx:
                temporal_contexts.append(temporal_ctx)
                
                stability_score = temporal_ctx.get('stability_score', 0)
                base_confidence = temporal_ctx.get('base_confidence', 0)
                if stability_score:
                    confidence_scores.append(stability_score)
        
        total_patterns = len(patterns_list)
        
        htf_analysis = {
            'relationship_patterns': dict(Counter(relationship_types)),
            'timeframe_distribution': dict(Counter(anchor_timeframes)),
            'relativity_patterns': dict(Counter(relativity_types)),
            'htf_structural_features': {
                'cross_tf_confluence': {
                    'count': cross_tf_confluence,
                    'frequency': cross_tf_confluence / total_patterns
                },
                'temporal_echo_strength': {
                    'count': temporal_echo_strength,
                    'frequency': temporal_echo_strength / total_patterns
                },
                'scaling_factors': {
                    'count': scaling_factors,
                    'frequency': scaling_factors / total_patterns
                },
                'temporal_stability': {
                    'count': temporal_stability,
                    'frequency': temporal_stability / total_patterns
                }
            },
            'temporal_characteristics': {
                'avg_stability': np.mean([tc.get('stability_score', 0) for tc in temporal_contexts if tc.get('stability_score')]) if temporal_contexts else 0,
                'avg_base_confidence': np.mean([tc.get('base_confidence', 0) for tc in temporal_contexts if tc.get('base_confidence')]) if temporal_contexts else 0,
                'high_stability_rate': len([tc for tc in temporal_contexts if tc.get('stability_score', 0) > 0.95]) / len(temporal_contexts) if temporal_contexts else 0
            },
            'htf_confluence_strength': {
                'total_htf_features': cross_tf_confluence + temporal_echo_strength + scaling_factors + temporal_stability,
                'htf_feature_density': (cross_tf_confluence + temporal_echo_strength + scaling_factors + temporal_stability) / (total_patterns * 4),
                'dominant_htf_feature': max([
                    ('cross_tf_confluence', cross_tf_confluence),
                    ('temporal_echo_strength', temporal_echo_strength),
                    ('scaling_factors', scaling_factors),
                    ('temporal_stability', temporal_stability)
                ], key=lambda x: x[1])[0] if any([cross_tf_confluence, temporal_echo_strength, scaling_factors, temporal_stability]) else 'none'
            }
        }
        
        return htf_analysis
    
    def generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive semantic market events analysis"""
        print("📋 Generating comprehensive semantic market events analysis...")
        
        # Run all analyses
        liquidity_analysis = self.analyze_liquidity_events()
        structure_analysis = self.analyze_market_structures()
        order_flow_analysis = self.analyze_order_flow_characteristics()
        htf_analysis = self.analyze_htf_structural_elements()
        
        # Build comprehensive report
        comprehensive_analysis = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patterns_analyzed': len(self.patterns),
                'range_levels_analyzed': list(self.range_clusters.keys()),
                'patterns_per_range': {level: len(patterns) for level, patterns in self.range_clusters.items()},
                'analysis_scope': 'Semantic Market Events & Liquidity Intelligence'
            },
            'liquidity_events_analysis': liquidity_analysis,
            'market_structures_analysis': structure_analysis,
            'order_flow_analysis': order_flow_analysis,
            'htf_structural_analysis': htf_analysis,
            'cross_range_intelligence': self._generate_cross_range_intelligence(
                liquidity_analysis, structure_analysis, order_flow_analysis, htf_analysis
            )
        }
        
        return comprehensive_analysis
    
    def _generate_cross_range_intelligence(self, liquidity_analysis: Dict, structure_analysis: Dict, 
                                         order_flow_analysis: Dict, htf_analysis: Dict) -> Dict:
        """Generate cross-range intelligence insights"""
        
        cross_range_insights = {
            'liquidity_evolution_patterns': {},
            'structural_progression': {},
            'order_flow_evolution': {},
            'htf_element_consistency': {}
        }
        
        # Analyze liquidity evolution across ranges
        range_levels = ['20%', '40%', '60%', '80%']
        
        # FVG evolution
        fvg_evolution = []
        sweep_evolution = []
        pd_array_evolution = []
        
        for level in range_levels:
            if level in liquidity_analysis:
                fvg_freq = liquidity_analysis[level]['fvg_events']['frequency']
                sweep_freq = liquidity_analysis[level]['sweep_events']['frequency']
                pd_freq = liquidity_analysis[level]['pd_array_events']['frequency']
                
                fvg_evolution.append((level, fvg_freq))
                sweep_evolution.append((level, sweep_freq))
                pd_array_evolution.append((level, pd_freq))
        
        cross_range_insights['liquidity_evolution_patterns'] = {
            'fvg_evolution': fvg_evolution,
            'sweep_evolution': sweep_evolution,
            'pd_array_evolution': pd_array_evolution,
            'liquidity_trend_analysis': self._analyze_liquidity_trends(fvg_evolution, sweep_evolution, pd_array_evolution)
        }
        
        # Structural progression
        regime_progression = []
        confidence_progression = []
        
        for level in range_levels:
            if level in structure_analysis:
                regimes = structure_analysis[level]['regime_distribution']
                dominant_regime = max(regimes.items(), key=lambda x: x[1])[0] if regimes else 'unknown'
                
                avg_confidence = structure_analysis[level]['regime_confidence']['avg_confidence']
                
                regime_progression.append((level, dominant_regime))
                confidence_progression.append((level, avg_confidence))
        
        cross_range_insights['structural_progression'] = {
            'regime_progression': regime_progression,
            'confidence_progression': confidence_progression
        }
        
        # Order flow evolution
        strength_evolution = []
        coherence_evolution = []
        
        for level in range_levels:
            if level in order_flow_analysis:
                avg_strength = order_flow_analysis[level]['pattern_strength_metrics']['avg_strength']
                avg_coherence = order_flow_analysis[level]['coherence_metrics']['avg_coherence']
                
                strength_evolution.append((level, avg_strength))
                coherence_evolution.append((level, avg_coherence))
        
        cross_range_insights['order_flow_evolution'] = {
            'strength_evolution': strength_evolution,
            'coherence_evolution': coherence_evolution
        }
        
        # HTF consistency
        htf_consistency = []
        
        for level in range_levels:
            if level in htf_analysis:
                htf_features = htf_analysis[level]['htf_structural_features']
                total_htf_density = sum([feature['frequency'] for feature in htf_features.values()])
                htf_consistency.append((level, total_htf_density))
        
        cross_range_insights['htf_element_consistency'] = {
            'htf_density_evolution': htf_consistency,
            'consistency_analysis': self._analyze_htf_consistency(htf_consistency)
        }
        
        return cross_range_insights
    
    def _analyze_liquidity_trends(self, fvg_evolution: List, sweep_evolution: List, pd_array_evolution: List) -> Dict:
        """Analyze liquidity evolution trends"""
        
        # Extract frequency values
        fvg_freqs = [freq for _, freq in fvg_evolution]
        sweep_freqs = [freq for _, freq in sweep_evolution]
        pd_freqs = [freq for _, freq in pd_array_evolution]
        
        return {
            'fvg_trend': 'increasing' if fvg_freqs[-1] > fvg_freqs[0] else 'decreasing' if fvg_freqs[-1] < fvg_freqs[0] else 'stable',
            'sweep_trend': 'increasing' if sweep_freqs[-1] > sweep_freqs[0] else 'decreasing' if sweep_freqs[-1] < sweep_freqs[0] else 'stable',
            'pd_array_trend': 'increasing' if pd_freqs[-1] > pd_freqs[0] else 'decreasing' if pd_freqs[-1] < pd_freqs[0] else 'stable',
            'dominant_liquidity_event': max([
                ('fvg', np.mean(fvg_freqs)),
                ('sweep', np.mean(sweep_freqs)),
                ('pd_array', np.mean(pd_freqs))
            ], key=lambda x: x[1])[0]
        }
    
    def _analyze_htf_consistency(self, htf_consistency: List) -> Dict:
        """Analyze HTF consistency across ranges"""
        
        densities = [density for _, density in htf_consistency]
        
        return {
            'avg_htf_density': np.mean(densities) if densities else 0,
            'htf_consistency_score': 1.0 - np.std(densities) if len(densities) > 1 else 1.0,
            'htf_trend': 'increasing' if densities[-1] > densities[0] else 'decreasing' if densities[-1] < densities[0] else 'stable'
        }
    
    def save_analysis(self, output_path: str = None) -> str:
        """Save comprehensive semantic analysis"""
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/semantic_market_events_analysis.json'
        
        # Generate comprehensive analysis
        analysis = self.generate_comprehensive_analysis()
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"💾 Semantic market events analysis saved to: {output_path}")
        
        # Also save markdown report
        markdown_path = output_path.replace('.json', '_detailed_report.md')
        self._save_markdown_report(analysis, markdown_path)
        
        return output_path
    
    def _save_markdown_report(self, analysis: Dict, markdown_path: str):
        """Save detailed markdown report"""
        markdown_content = self._generate_detailed_markdown_report(analysis)
        
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Detailed semantic analysis report saved to: {markdown_path}")
    
    def _generate_detailed_markdown_report(self, analysis: Dict) -> str:
        """Generate comprehensive markdown report"""
        md = []
        
        md.append("# IRONFORGE Semantic Market Events Analysis - Detailed Report")
        md.append("")
        md.append("## 🏛️ Archaeological Discovery Overview")
        md.append("")
        
        metadata = analysis['analysis_metadata']
        md.append(f"**Analysis Date**: {metadata['timestamp']}")
        md.append(f"**Total Patterns Analyzed**: {metadata['total_patterns_analyzed']}")
        md.append(f"**Range Levels**: {', '.join(metadata['range_levels_analyzed'])}")
        md.append("")
        
        patterns_per_range = metadata['patterns_per_range']
        md.append("**Patterns per Range Level**:")
        for level, count in patterns_per_range.items():
            md.append(f"- {level}: {count} patterns")
        md.append("")
        md.append("---")
        md.append("")
        
        # Liquidity Events Analysis
        md.append("## 💧 Liquidity Events Analysis by Range Level")
        md.append("")
        
        liquidity_analysis = analysis['liquidity_events_analysis']
        
        for range_level, liq_data in liquidity_analysis.items():
            md.append(f"### {range_level} Range Level Liquidity Events")
            md.append("")
            md.append(f"**Total Patterns**: {liq_data['total_patterns']}")
            md.append("")
            
            # FVG Events
            fvg_data = liq_data['fvg_events']
            md.append("#### 1. Fair Value Gap (FVG) Events")
            md.append(f"- **Count**: {fvg_data['count']} patterns")
            md.append(f"- **Frequency**: {fvg_data['frequency']:.1%}")
            md.append(f"- **Types**: {', '.join(fvg_data['types'])}")
            md.append("")
            
            # Sweep Events
            sweep_data = liq_data['sweep_events']
            md.append("#### 2. Liquidity Sweep Events")
            md.append(f"- **Count**: {sweep_data['count']} patterns")
            md.append(f"- **Frequency**: {sweep_data['frequency']:.1%}")
            md.append(f"- **Types**: {', '.join(sweep_data['types'])}")
            md.append("")
            
            # PD Array Events
            pd_data = liq_data['pd_array_events']
            md.append("#### 3. Premium/Discount Array Events")
            md.append(f"- **Count**: {pd_data['count']} patterns")
            md.append(f"- **Frequency**: {pd_data['frequency']:.1%}")
            md.append(f"- **Types**: {', '.join(pd_data['types'])}")
            md.append("")
            
            # Consolidation Events
            cons_data = liq_data['consolidation_events']
            md.append("#### 4. Consolidation Events")
            md.append(f"- **Count**: {cons_data['count']} patterns")
            md.append(f"- **Frequency**: {cons_data['frequency']:.1%}")
            md.append(f"- **Types**: {', '.join(cons_data['types'])}")
            md.append("")
            
            # Expansion Events
            exp_data = liq_data['expansion_events']
            md.append("#### 5. Expansion Phase Events")
            md.append(f"- **Count**: {exp_data['count']} patterns")
            md.append(f"- **Frequency**: {exp_data['frequency']:.1%}")
            md.append(f"- **Types**: {', '.join(exp_data['types'])}")
            md.append("")
            
            # Order Flow
            order_flow = liq_data['order_flow_characteristics']
            md.append("#### 6. Order Flow Characteristics")
            md.append(f"- **Average Pattern Strength**: {order_flow['avg_pattern_strength']:.3f}")
            md.append(f"- **Energy Events**: {order_flow['energy_events']}")
            md.append(f"- **Energy Density**: {order_flow['energy_density']:.1%}")
            md.append("")
            md.append("---")
            md.append("")
        
        # Market Structures Analysis
        md.append("## 🏗️ Market Structures Analysis")
        md.append("")
        
        structure_analysis = analysis['market_structures_analysis']
        
        for range_level, struct_data in structure_analysis.items():
            md.append(f"### {range_level} Range Level Market Structures")
            md.append("")
            
            # Regime Distribution
            regime_dist = struct_data['regime_distribution']
            md.append("#### Regime Distribution")
            for regime, count in regime_dist.items():
                percentage = count / liquidity_analysis[range_level]['total_patterns'] * 100
                md.append(f"- **{regime}**: {count} patterns ({percentage:.1f}%)")
            md.append("")
            
            # Archaeological Characteristics
            arch_chars = struct_data['archaeological_characteristics']
            md.append("#### Archaeological Characteristics")
            md.append(f"- **Average Significance**: {arch_chars['avg_significance']:.3f}")
            md.append(f"- **Average Permanence**: {arch_chars['avg_permanence']:.3f}")
            md.append(f"- **Average Generalizability**: {arch_chars['avg_generalizability']:.3f}")
            md.append("")
            md.append("---")
            md.append("")
        
        # Cross-Range Intelligence
        md.append("## 🔗 Cross-Range Intelligence Analysis")
        md.append("")
        
        cross_range = analysis['cross_range_intelligence']
        
        # Liquidity Evolution
        liquidity_evolution = cross_range['liquidity_evolution_patterns']
        trend_analysis = liquidity_evolution['liquidity_trend_analysis']
        
        md.append("### Liquidity Evolution Patterns")
        md.append("")
        md.append("| Range Level | FVG Frequency | Sweep Frequency | PD Array Frequency |")
        md.append("|-------------|---------------|-----------------|-------------------|")
        
        fvg_evolution = liquidity_evolution['fvg_evolution']
        sweep_evolution = liquidity_evolution['sweep_evolution']
        pd_evolution = liquidity_evolution['pd_array_evolution']
        
        for i, (level, _) in enumerate(fvg_evolution):
            fvg_freq = fvg_evolution[i][1]
            sweep_freq = sweep_evolution[i][1]
            pd_freq = pd_evolution[i][1]
            
            md.append(f"| {level} | {fvg_freq:.1%} | {sweep_freq:.1%} | {pd_freq:.1%} |")
        
        md.append("")
        md.append("#### Trend Analysis")
        md.append(f"- **FVG Trend**: {trend_analysis['fvg_trend']}")
        md.append(f"- **Sweep Trend**: {trend_analysis['sweep_trend']}")
        md.append(f"- **PD Array Trend**: {trend_analysis['pd_array_trend']}")
        md.append(f"- **Dominant Liquidity Event**: {trend_analysis['dominant_liquidity_event']}")
        md.append("")
        
        # HTF Consistency
        htf_consistency = cross_range['htf_element_consistency']
        consistency_analysis = htf_consistency['consistency_analysis']
        
        md.append("### HTF Structural Element Consistency")
        md.append("")
        md.append("| Range Level | HTF Density |")
        md.append("|-------------|-------------|")
        
        for level, density in htf_consistency['htf_density_evolution']:
            md.append(f"| {level} | {density:.2f} |")
        
        md.append("")
        md.append("#### HTF Analysis")
        md.append(f"- **Average HTF Density**: {consistency_analysis['avg_htf_density']:.3f}")
        md.append(f"- **HTF Consistency Score**: {consistency_analysis['htf_consistency_score']:.3f}")
        md.append(f"- **HTF Trend**: {consistency_analysis['htf_trend']}")
        md.append("")
        
        md.append("---")
        md.append("")
        md.append("*Analysis generated by IRONFORGE Semantic Market Events Analyzer*")
        
        return "\n".join(md)

if __name__ == "__main__":
    print("🔍 IRONFORGE Semantic Market Events Analyzer")
    print("=" * 60)
    
    analyzer = SemanticMarketEventsAnalyzer()
    output_file = analyzer.save_analysis()
    
    print(f"\n✅ Semantic market events analysis complete!")
    print(f"📊 Results saved to: {output_file}")