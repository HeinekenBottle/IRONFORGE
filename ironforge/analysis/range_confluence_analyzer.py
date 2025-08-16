#!/usr/bin/env python3
"""
IRONFORGE Range Position Confluence Analyzer
==========================================

Analyzes range position confluence patterns by cluster, providing detailed insights on:
1. Session phase timing for each range level
2. Range completion rates and probabilities  
3. HTF confluence characteristics
4. Session energy signatures and velocity patterns
5. Cross-session continuation probabilities

Focuses on 20%, 40%, 60% range levels with comprehensive behavioral analysis.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import re
from datetime import datetime, timedelta
from pathlib import Path
import logging

class RangeConfluenceAnalyzer:
    """
    Analyzes range position confluence patterns with detailed behavioral metrics
    """
    
    def __init__(self, patterns_file: str = None, links_file: str = None):
        self.logger = logging.getLogger('range_confluence_analyzer')
        
        # Load discovered patterns
        if patterns_file is None:
            patterns_file = '/Users/jack/IRONFORGE/IRONFORGE/preservation/discovered_patterns.json'
        
        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)
        
        # Load pattern links analysis
        if links_file is None:
            links_file = '/Users/jack/IRONFORGE/analysis/pattern_links_analysis.json'
            
        with open(links_file, 'r') as f:
            self.links_analysis = json.load(f)
        
        # Initialize analysis structures
        self.range_clusters = self._extract_range_clusters()
        self.session_timeline = self._build_session_timeline()
        self.range_analysis = {}
        
        print(f"🎯 Range Confluence Analyzer initialized")
        print(f"  Patterns loaded: {len(self.patterns)}")
        print(f"  Range clusters identified: {len(self.range_clusters)}")
    
    def _extract_range_clusters(self) -> Dict[str, List]:
        """Extract range position confluence clusters from link analysis"""
        range_clusters = {}
        
        # Get structural clusters focused on range positions
        for cluster in self.links_analysis['pattern_clusters']:
            if cluster['cluster_type'] == 'structural_similarity':
                cluster_id = cluster['cluster_id']
                
                # Extract range level from cluster ID
                if 'range_position_confluence' in cluster_id:
                    range_match = re.search(r'(\d+)pct', cluster_id)
                    if range_match:
                        range_level = int(range_match.group(1))
                        range_clusters[f"{range_level}%"] = {
                            'cluster_data': cluster,
                            'pattern_ids': cluster['pattern_ids'],
                            'pattern_indices': [int(pid.split('_')[1]) for pid in cluster['pattern_ids']]
                        }
        
        return range_clusters
    
    def _build_session_timeline(self) -> Dict:
        """Build session timeline from enhanced sessions"""
        timeline = {}
        session_path = Path('/Users/jack/IRONFORGE/enhanced_sessions_with_relativity')
        
        if session_path.exists():
            for session_file in session_path.glob('*.json'):
                filename = session_file.name
                match = re.search(r'enhanced_rel_(\w+)_Lvl-1_(\d{4})_(\d{2})_(\d{2})', filename)
                
                if match:
                    session_type, year, month, day = match.groups()
                    date_str = f"{year}-{month}-{day}"
                    
                    if date_str not in timeline:
                        timeline[date_str] = {}
                    timeline[date_str][session_type] = {
                        'filename': filename,
                        'filepath': str(session_file)
                    }
        
        return timeline
    
    def analyze_range_confluence_patterns(self) -> Dict:
        """Main analysis method for range confluence patterns"""
        print("🏗️ Analyzing Range Position Confluence Patterns...")
        
        results = {}
        
        # Analyze each range level
        for range_level, cluster_data in self.range_clusters.items():
            print(f"\n📊 Analyzing {range_level} Range Level...")
            
            analysis = self._analyze_single_range_level(range_level, cluster_data)
            results[range_level] = analysis
            
            print(f"  ✅ {range_level} analysis complete: {len(cluster_data['pattern_indices'])} patterns")
        
        # Cross-range analysis
        results['cross_range_analysis'] = self._analyze_cross_range_relationships()
        
        return results
    
    def _analyze_single_range_level(self, range_level: str, cluster_data: Dict) -> Dict:
        """Analyze a single range level comprehensively"""
        pattern_indices = cluster_data['pattern_indices']
        patterns = [self.patterns[i] for i in pattern_indices]
        
        analysis = {
            'range_level': range_level,
            'total_patterns': len(patterns),
            'session_phase_timing': self._analyze_session_phase_timing(patterns),
            'range_completion_rates': self._analyze_range_completion_rates(patterns, range_level),
            'htf_confluence_characteristics': self._analyze_htf_confluence(patterns),
            'session_energy_signatures': self._analyze_session_energy_signatures(patterns),
            'velocity_patterns': self._analyze_velocity_patterns(patterns),
            'cross_session_continuation': self._analyze_cross_session_continuation(patterns, pattern_indices)
        }
        
        return analysis
    
    def _analyze_session_phase_timing(self, patterns: List[Dict]) -> Dict:
        """Analyze when range levels occur during session phases"""
        phase_timing = {
            'phase_distribution': defaultdict(int),
            'session_position_stats': [],
            'timing_characteristics': {}
        }
        
        for pattern in patterns:
            phase_info = pattern.get('phase_information', {})
            
            # Phase distribution
            primary_phase = phase_info.get('primary_phase', 'unknown')
            phase_timing['phase_distribution'][primary_phase] += 1
            
            # Session position statistics
            session_position = phase_info.get('session_position', 0)
            if session_position > 0:
                phase_timing['session_position_stats'].append(session_position)
        
        # Calculate timing statistics
        if phase_timing['session_position_stats']:
            positions = phase_timing['session_position_stats']
            phase_timing['timing_characteristics'] = {
                'avg_session_position': np.mean(positions),
                'median_session_position': np.median(positions),
                'position_std': np.std(positions),
                'early_occurrence_rate': len([p for p in positions if p < 2.0]) / len(positions),
                'mid_occurrence_rate': len([p for p in positions if 2.0 <= p < 4.0]) / len(positions),
                'late_occurrence_rate': len([p for p in positions if p >= 4.0]) / len(positions)
            }
        
        # Convert to regular dict for JSON serialization
        phase_timing['phase_distribution'] = dict(phase_timing['phase_distribution'])
        
        return phase_timing
    
    def _analyze_range_completion_rates(self, patterns: List[Dict], range_level: str) -> Dict:
        """Analyze range completion rates and probabilities"""
        completion_analysis = {
            'current_level': range_level,
            'completion_probability_80pct': 0.0,
            'completion_probability_100pct': 0.0,
            'session_success_metrics': {},
            'range_progression_analysis': {}
        }
        
        # Extract range percentages from patterns
        range_percentages = []
        session_completions = []
        
        for pattern in patterns:
            desc = pattern.get('description', '')
            range_match = re.search(r'(\d+\.?\d*)% of range', desc)
            if range_match:
                range_pct = float(range_match.group(1))
                range_percentages.append(range_pct)
        
        if range_percentages:
            current_range = float(range_level.replace('%', ''))
            
            # Calculate completion probabilities based on pattern analysis
            # This is a heuristic analysis based on range position patterns
            
            # Patterns at this level that continue beyond
            higher_range_patterns = [p for p in range_percentages if p > current_range]
            
            # Estimate completion rates
            if range_percentages:
                completion_80_count = len([p for p in range_percentages if p >= 80.0])
                completion_100_count = len([p for p in range_percentages if p >= 100.0])
                
                # Calculate probabilities based on pattern continuation
                completion_analysis['completion_probability_80pct'] = completion_80_count / len(range_percentages)
                completion_analysis['completion_probability_100pct'] = completion_100_count / len(range_percentages)
                
                # Range progression analysis
                completion_analysis['range_progression_analysis'] = {
                    'avg_range_reached': np.mean(range_percentages),
                    'max_range_reached': np.max(range_percentages),
                    'min_range_reached': np.min(range_percentages),
                    'range_std': np.std(range_percentages),
                    'progression_beyond_current': len(higher_range_patterns) / len(range_percentages)
                }
        
        # Session success metrics based on phase significance
        phase_significances = []
        for pattern in patterns:
            phase_info = pattern.get('phase_information', {})
            sig = phase_info.get('phase_significance', 0)
            if sig > 0:
                phase_significances.append(sig)
        
        if phase_significances:
            completion_analysis['session_success_metrics'] = {
                'avg_phase_significance': np.mean(phase_significances),
                'high_significance_rate': len([s for s in phase_significances if s > 0.8]) / len(phase_significances),
                'success_consistency': 1.0 - np.std(phase_significances)
            }
        
        return completion_analysis
    
    def _analyze_htf_confluence(self, patterns: List[Dict]) -> Dict:
        """Analyze HTF confluence characteristics at range level"""
        htf_analysis = {
            'confluence_strength': {},
            'timeframe_interactions': {},
            'structural_characteristics': {}
        }
        
        # Analyze confluence from descriptions and context
        confluence_indicators = []
        timeframe_mentions = []
        
        for pattern in patterns:
            desc = pattern.get('description', '')
            
            # Look for HTF confluence indicators
            if 'HTF confluence' in desc:
                confluence_indicators.append(pattern)
            
            # Extract timeframe information
            timeframe = pattern.get('anchor_timeframe', '1m')
            timeframe_mentions.append(timeframe)
            
            # Analyze structural context
            structural_context = pattern.get('semantic_context', {}).get('structural_context', {})
            pattern_strength = structural_context.get('pattern_strength', 0)
            
            if pattern_strength > 0:
                htf_analysis['confluence_strength']['pattern_strength'] = htf_analysis['confluence_strength'].get('pattern_strength', [])
                htf_analysis['confluence_strength']['pattern_strength'].append(pattern_strength)
        
        # Calculate confluence statistics
        if confluence_indicators:
            htf_analysis['confluence_strength']['confluence_rate'] = len(confluence_indicators) / len(patterns)
        else:
            htf_analysis['confluence_strength']['confluence_rate'] = 0.0
        
        # Timeframe interaction analysis
        timeframe_counts = Counter(timeframe_mentions)
        htf_analysis['timeframe_interactions'] = dict(timeframe_counts)
        
        # Structural characteristics
        if 'pattern_strength' in htf_analysis['confluence_strength']:
            strengths = htf_analysis['confluence_strength']['pattern_strength']
            htf_analysis['structural_characteristics'] = {
                'avg_confluence_strength': np.mean(strengths),
                'strong_confluence_rate': len([s for s in strengths if s > 0.7]) / len(strengths),
                'confluence_consistency': 1.0 - np.std(strengths) if len(strengths) > 1 else 1.0
            }
        
        return htf_analysis
    
    def _analyze_session_energy_signatures(self, patterns: List[Dict]) -> Dict:
        """Analyze session energy signatures and characteristics"""
        energy_analysis = {
            'energy_distribution': {},
            'session_characteristics': {},
            'energy_patterns': {}
        }
        
        # Extract energy-related information from patterns
        energy_states = []
        liquidity_environments = []
        session_phases = []
        
        for pattern in patterns:
            semantic_context = pattern.get('semantic_context', {})
            structural_context = semantic_context.get('structural_context', {})
            
            # Energy state analysis
            energy_state = structural_context.get('energy_state', {})
            if energy_state:
                energy_states.append(energy_state)
            
            # Liquidity environment
            liquidity_env = structural_context.get('liquidity_environment', [])
            if liquidity_env:
                liquidity_environments.extend(liquidity_env)
            
            # Session phase energy
            phase_info = pattern.get('phase_information', {})
            phase_sig = phase_info.get('phase_significance', 0)
            if phase_sig > 0:
                session_phases.append({
                    'phase': phase_info.get('primary_phase', 'unknown'),
                    'significance': phase_sig,
                    'position': phase_info.get('session_position', 0)
                })
        
        # Energy distribution analysis
        if energy_states:
            # Analyze energy state patterns (simplified for available data)
            energy_analysis['energy_distribution'] = {
                'total_energy_events': len(energy_states),
                'energy_event_rate': len(energy_states) / len(patterns)
            }
        
        # Session characteristics from phases
        if session_phases:
            phases = [sp['phase'] for sp in session_phases]
            significances = [sp['significance'] for sp in session_phases]
            
            energy_analysis['session_characteristics'] = {
                'dominant_phase': Counter(phases).most_common(1)[0][0] if phases else 'unknown',
                'avg_phase_significance': np.mean(significances),
                'high_energy_phase_rate': len([s for s in significances if s > 0.8]) / len(significances),
                'energy_consistency': 1.0 - np.std(significances) if len(significances) > 1 else 1.0
            }
        
        # Liquidity environment analysis
        if liquidity_environments:
            energy_analysis['energy_patterns'] = {
                'liquidity_events': len(liquidity_environments),
                'liquidity_event_rate': len(liquidity_environments) / len(patterns)
            }
        
        return energy_analysis
    
    def _analyze_velocity_patterns(self, patterns: List[Dict]) -> Dict:
        """Analyze velocity patterns and movement characteristics"""
        velocity_analysis = {
            'temporal_velocity': {},
            'structural_velocity': {},
            'progression_patterns': {}
        }
        
        # Extract temporal movement patterns
        session_positions = []
        phase_transitions = []
        
        for pattern in patterns:
            phase_info = pattern.get('phase_information', {})
            session_pos = phase_info.get('session_position', 0)
            
            if session_pos > 0:
                session_positions.append(session_pos)
            
            # Analyze phase characteristics for velocity inference
            phase = phase_info.get('primary_phase', 'unknown')
            phase_sig = phase_info.get('phase_significance', 0)
            
            if phase and phase_sig > 0:
                phase_transitions.append({
                    'phase': phase,
                    'significance': phase_sig,
                    'position': session_pos
                })
        
        # Temporal velocity analysis
        if session_positions:
            velocity_analysis['temporal_velocity'] = {
                'avg_session_position': np.mean(session_positions),
                'position_velocity_spread': np.std(session_positions),
                'early_velocity_patterns': len([p for p in session_positions if p < 2.0]) / len(session_positions),
                'late_velocity_patterns': len([p for p in session_positions if p > 4.0]) / len(session_positions)
            }
        
        # Structural velocity from pattern progression
        range_progressions = []
        for pattern in patterns:
            desc = pattern.get('description', '')
            range_match = re.search(r'(\d+\.?\d*)% of range', desc)
            if range_match:
                range_pct = float(range_match.group(1))
                range_progressions.append(range_pct)
        
        if range_progressions:
            velocity_analysis['structural_velocity'] = {
                'avg_range_position': np.mean(range_progressions),
                'range_velocity_spread': np.std(range_progressions),
                'velocity_consistency': 1.0 - (np.std(range_progressions) / np.mean(range_progressions)) if np.mean(range_progressions) > 0 else 0
            }
        
        # Progression pattern analysis
        if phase_transitions:
            phases = [pt['phase'] for pt in phase_transitions]
            significances = [pt['significance'] for pt in phase_transitions]
            
            velocity_analysis['progression_patterns'] = {
                'dominant_progression_phase': Counter(phases).most_common(1)[0][0] if phases else 'unknown',
                'avg_progression_significance': np.mean(significances),
                'high_velocity_rate': len([s for s in significances if s > 0.8]) / len(significances)
            }
        
        return velocity_analysis
    
    def _analyze_cross_session_continuation(self, patterns: List[Dict], pattern_indices: List[int]) -> Dict:
        """Analyze cross-session continuation probabilities"""
        continuation_analysis = {
            'continuation_probability': 0.0,
            'session_persistence': {},
            'evolution_tracking': {},
            'predictive_metrics': {}
        }
        
        # Find evolution links for these patterns
        evolution_links = []
        for link in self.links_analysis['pattern_links']:
            if link['type'] == 'cross_session_evolution':
                source_idx = int(link['source'].split('_')[1])
                target_idx = int(link['target'].split('_')[1])
                
                if source_idx in pattern_indices or target_idx in pattern_indices:
                    evolution_links.append(link)
        
        # Calculate continuation probability
        if pattern_indices:
            patterns_with_evolution = set()
            for link in evolution_links:
                source_idx = int(link['source'].split('_')[1])
                target_idx = int(link['target'].split('_')[1])
                if source_idx in pattern_indices:
                    patterns_with_evolution.add(source_idx)
                if target_idx in pattern_indices:
                    patterns_with_evolution.add(target_idx)
            
            continuation_analysis['continuation_probability'] = len(patterns_with_evolution) / len(pattern_indices)
        
        # Session persistence analysis
        session_names = [pattern.get('session_name', 'unknown') for pattern in patterns]
        session_distribution = Counter(session_names)
        
        continuation_analysis['session_persistence'] = {
            'unique_sessions': len(session_distribution),
            'avg_patterns_per_session': np.mean(list(session_distribution.values())) if session_distribution else 0,
            'max_session_persistence': max(session_distribution.values()) if session_distribution else 0,
            'session_distribution': dict(session_distribution)
        }
        
        # Evolution tracking
        if evolution_links:
            evolution_stages = [link.get('evolution_stage', 'unknown') for link in evolution_links]
            evolution_strengths = [link.get('strength', 0) for link in evolution_links if link.get('strength')]
            
            continuation_analysis['evolution_tracking'] = {
                'total_evolution_links': len(evolution_links),
                'evolution_stage_distribution': dict(Counter(evolution_stages)),
                'avg_evolution_strength': np.mean(evolution_strengths) if evolution_strengths else 0,
                'strong_evolution_rate': len([s for s in evolution_strengths if s > 0.8]) / len(evolution_strengths) if evolution_strengths else 0
            }
        
        # Predictive metrics based on structural similarity
        structural_links = []
        for link in self.links_analysis['pattern_links']:
            if 'structural' in link['type'] or 'range' in link['type']:
                source_idx = int(link['source'].split('_')[1])
                target_idx = int(link['target'].split('_')[1])
                
                if source_idx in pattern_indices or target_idx in pattern_indices:
                    structural_links.append(link)
        
        if structural_links:
            structural_strengths = [link.get('structural_similarity', 0) for link in structural_links if link.get('structural_similarity')]
            
            continuation_analysis['predictive_metrics'] = {
                'structural_consistency': np.mean(structural_strengths) if structural_strengths else 0,
                'high_consistency_rate': len([s for s in structural_strengths if s > 0.8]) / len(structural_strengths) if structural_strengths else 0,
                'predictive_reliability': len(structural_links) / len(pattern_indices) if pattern_indices else 0
            }
        
        return continuation_analysis
    
    def _analyze_cross_range_relationships(self) -> Dict:
        """Analyze relationships between different range levels"""
        cross_analysis = {
            'range_level_transitions': {},
            'progression_patterns': {},
            'confluence_interactions': {}
        }
        
        # Find links between different range levels
        range_transitions = []
        for link in self.links_analysis['pattern_links']:
            if link['type'] in ['range_position_similarity', 'structural_similarity']:
                source_idx = int(link['source'].split('_')[1])
                target_idx = int(link['target'].split('_')[1])
                
                source_pattern = self.patterns[source_idx]
                target_pattern = self.patterns[target_idx]
                
                # Extract range levels
                source_range = self._extract_range_from_pattern(source_pattern)
                target_range = self._extract_range_from_pattern(target_pattern)
                
                if source_range and target_range and source_range != target_range:
                    range_transitions.append({
                        'from_range': source_range,
                        'to_range': target_range,
                        'strength': link.get('strength', 0),
                        'link_type': link['type']
                    })
        
        # Analyze transitions
        if range_transitions:
            transition_pairs = [(rt['from_range'], rt['to_range']) for rt in range_transitions]
            transition_counts = Counter(transition_pairs)
            
            cross_analysis['range_level_transitions'] = {
                'total_transitions': len(range_transitions),
                'unique_transition_pairs': len(transition_counts),
                'most_common_transitions': dict(transition_counts.most_common(5)),
                'avg_transition_strength': np.mean([rt['strength'] for rt in range_transitions if rt['strength'] > 0])
            }
        
        return cross_analysis
    
    def _extract_range_from_pattern(self, pattern: Dict) -> Optional[float]:
        """Extract range percentage from pattern"""
        desc = pattern.get('description', '')
        range_match = re.search(r'(\d+\.?\d*)% of range', desc)
        return float(range_match.group(1)) if range_match else None
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive range confluence analysis report"""
        print("📋 Generating Comprehensive Range Confluence Analysis Report...")
        
        # Run main analysis
        analysis_results = self.analyze_range_confluence_patterns()
        
        # Build comprehensive report
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patterns_analyzed': len(self.patterns),
                'range_levels_analyzed': list(self.range_clusters.keys()),
                'analysis_scope': 'Range Position Confluence Behavioral Analysis'
            },
            'range_level_analysis': analysis_results,
            'summary_insights': self._generate_summary_insights(analysis_results),
            'predictive_intelligence': self._generate_predictive_intelligence(analysis_results)
        }
        
        return report
    
    def _generate_summary_insights(self, analysis_results: Dict) -> Dict:
        """Generate summary insights across all range levels"""
        insights = {
            'key_findings': [],
            'behavioral_patterns': {},
            'confluence_strength_ranking': {},
            'session_timing_patterns': {}
        }
        
        # Analyze patterns across range levels
        for range_level, analysis in analysis_results.items():
            if range_level != 'cross_range_analysis':
                # Key findings for each level
                completion_80 = analysis.get('range_completion_rates', {}).get('completion_probability_80pct', 0)
                continuation_prob = analysis.get('cross_session_continuation', {}).get('continuation_probability', 0)
                
                insights['key_findings'].append({
                    'range_level': range_level,
                    'completion_80_probability': completion_80,
                    'cross_session_continuation': continuation_prob,
                    'pattern_count': analysis.get('total_patterns', 0)
                })
                
                # Behavioral patterns
                phase_timing = analysis.get('session_phase_timing', {})
                dominant_phase = Counter(phase_timing.get('phase_distribution', {})).most_common(1)
                if dominant_phase:
                    insights['behavioral_patterns'][range_level] = {
                        'dominant_phase': dominant_phase[0][0],
                        'phase_concentration': dominant_phase[0][1] / analysis.get('total_patterns', 1)
                    }
                
                # Confluence strength
                htf_confluence = analysis.get('htf_confluence_characteristics', {})
                confluence_rate = htf_confluence.get('confluence_strength', {}).get('confluence_rate', 0)
                insights['confluence_strength_ranking'][range_level] = confluence_rate
        
        return insights
    
    def _generate_predictive_intelligence(self, analysis_results: Dict) -> Dict:
        """Generate predictive intelligence from analysis"""
        predictive = {
            'range_completion_predictions': {},
            'session_timing_predictions': {},
            'continuation_predictions': {},
            'confluence_predictions': {}
        }
        
        for range_level, analysis in analysis_results.items():
            if range_level != 'cross_range_analysis':
                # Range completion predictions
                completion_rates = analysis.get('range_completion_rates', {})
                predictive['range_completion_predictions'][range_level] = {
                    'probability_80pct': completion_rates.get('completion_probability_80pct', 0),
                    'probability_100pct': completion_rates.get('completion_probability_100pct', 0),
                    'avg_range_reached': completion_rates.get('range_progression_analysis', {}).get('avg_range_reached', 0)
                }
                
                # Continuation predictions
                continuation = analysis.get('cross_session_continuation', {})
                predictive['continuation_predictions'][range_level] = {
                    'continuation_probability': continuation.get('continuation_probability', 0),
                    'evolution_strength': continuation.get('evolution_tracking', {}).get('avg_evolution_strength', 0)
                }
        
        return predictive
    
    def save_analysis(self, output_path: str = None) -> str:
        """Save comprehensive range confluence analysis"""
        if output_path is None:
            output_path = '/Users/jack/IRONFORGE/analysis/range_confluence_analysis.json'
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Range confluence analysis saved to: {output_path}")
        
        # Also save markdown report
        markdown_path = output_path.replace('.json', '_report.md')
        self._save_markdown_report(report, markdown_path)
        
        return output_path
    
    def _save_markdown_report(self, report: Dict, markdown_path: str):
        """Save markdown formatted report"""
        markdown_content = self._generate_markdown_report(report)
        
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Markdown report saved to: {markdown_path}")
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown formatted report"""
        md = []
        md.append("# IRONFORGE Range Position Confluence Analysis Report")
        md.append("")
        md.append(f"**Analysis Date**: {report['analysis_metadata']['timestamp']}")
        md.append(f"**Patterns Analyzed**: {report['analysis_metadata']['total_patterns_analyzed']}")
        md.append(f"**Range Levels**: {', '.join(report['analysis_metadata']['range_levels_analyzed'])}")
        md.append("")
        md.append("---")
        md.append("")
        
        # Summary insights
        md.append("## 🎯 Key Findings Summary")
        md.append("")
        
        summary = report.get('summary_insights', {})
        key_findings = summary.get('key_findings', [])
        
        for finding in key_findings:
            range_level = finding['range_level']
            md.append(f"### {range_level} Range Level")
            md.append(f"- **Pattern Count**: {finding['pattern_count']}")
            md.append(f"- **80%+ Completion Probability**: {finding['completion_80_probability']:.1%}")
            md.append(f"- **Cross-Session Continuation**: {finding['cross_session_continuation']:.1%}")
            md.append("")
        
        # Detailed analysis for each range level
        md.append("---")
        md.append("")
        md.append("## 📊 Detailed Range Level Analysis")
        md.append("")
        
        range_analysis = report.get('range_level_analysis', {})
        
        for range_level, analysis in range_analysis.items():
            if range_level != 'cross_range_analysis':
                md.append(f"### {range_level} Range Position Analysis")
                md.append("")
                
                # Session phase timing
                phase_timing = analysis.get('session_phase_timing', {})
                md.append("#### 1. Session Phase Timing")
                phase_dist = phase_timing.get('phase_distribution', {})
                timing_chars = phase_timing.get('timing_characteristics', {})
                
                for phase, count in phase_dist.items():
                    percentage = count / analysis.get('total_patterns', 1) * 100
                    md.append(f"- **{phase}**: {count} patterns ({percentage:.1f}%)")
                
                if timing_chars:
                    md.append(f"- **Average Session Position**: {timing_chars.get('avg_session_position', 0):.2f}")
                    md.append(f"- **Early Occurrence Rate**: {timing_chars.get('early_occurrence_rate', 0):.1%}")
                    md.append(f"- **Late Occurrence Rate**: {timing_chars.get('late_occurrence_rate', 0):.1%}")
                
                md.append("")
                
                # Range completion rates
                completion = analysis.get('range_completion_rates', {})
                md.append("#### 2. Range Completion Analysis")
                md.append(f"- **80%+ Completion Probability**: {completion.get('completion_probability_80pct', 0):.1%}")
                md.append(f"- **100% Completion Probability**: {completion.get('completion_probability_100pct', 0):.1%}")
                
                progression = completion.get('range_progression_analysis', {})
                if progression:
                    md.append(f"- **Average Range Reached**: {progression.get('avg_range_reached', 0):.1f}%")
                    md.append(f"- **Progression Beyond Current Level**: {progression.get('progression_beyond_current', 0):.1%}")
                
                md.append("")
                
                # HTF confluence
                htf = analysis.get('htf_confluence_characteristics', {})
                md.append("#### 3. HTF Confluence Characteristics")
                confluence_rate = htf.get('confluence_strength', {}).get('confluence_rate', 0)
                md.append(f"- **HTF Confluence Rate**: {confluence_rate:.1%}")
                
                structural = htf.get('structural_characteristics', {})
                if structural:
                    md.append(f"- **Average Confluence Strength**: {structural.get('avg_confluence_strength', 0):.2f}")
                    md.append(f"- **Strong Confluence Rate**: {structural.get('strong_confluence_rate', 0):.1%}")
                
                md.append("")
                
                # Session energy signatures
                energy = analysis.get('session_energy_signatures', {})
                md.append("#### 4. Session Energy Signatures")
                
                session_chars = energy.get('session_characteristics', {})
                if session_chars:
                    md.append(f"- **Dominant Phase**: {session_chars.get('dominant_phase', 'unknown')}")
                    md.append(f"- **Average Phase Significance**: {session_chars.get('avg_phase_significance', 0):.2f}")
                    md.append(f"- **High Energy Phase Rate**: {session_chars.get('high_energy_phase_rate', 0):.1%}")
                
                md.append("")
                
                # Velocity patterns
                velocity = analysis.get('velocity_patterns', {})
                md.append("#### 5. Velocity Patterns")
                
                temporal_vel = velocity.get('temporal_velocity', {})
                if temporal_vel:
                    md.append(f"- **Average Session Position**: {temporal_vel.get('avg_session_position', 0):.2f}")
                    md.append(f"- **Early Velocity Patterns**: {temporal_vel.get('early_velocity_patterns', 0):.1%}")
                    md.append(f"- **Late Velocity Patterns**: {temporal_vel.get('late_velocity_patterns', 0):.1%}")
                
                md.append("")
                
                # Cross-session continuation
                continuation = analysis.get('cross_session_continuation', {})
                md.append("#### 6. Cross-Session Continuation Probabilities")
                md.append(f"- **Continuation Probability**: {continuation.get('continuation_probability', 0):.1%}")
                
                persistence = continuation.get('session_persistence', {})
                if persistence:
                    md.append(f"- **Unique Sessions**: {persistence.get('unique_sessions', 0)}")
                    md.append(f"- **Average Patterns per Session**: {persistence.get('avg_patterns_per_session', 0):.1f}")
                
                evolution = continuation.get('evolution_tracking', {})
                if evolution:
                    md.append(f"- **Evolution Links**: {evolution.get('total_evolution_links', 0)}")
                    md.append(f"- **Average Evolution Strength**: {evolution.get('avg_evolution_strength', 0):.2f}")
                
                md.append("")
                md.append("---")
                md.append("")
        
        # Predictive intelligence
        md.append("## 🔮 Predictive Intelligence Summary")
        md.append("")
        
        predictive = report.get('predictive_intelligence', {})
        completion_preds = predictive.get('range_completion_predictions', {})
        
        md.append("### Range Completion Predictions")
        md.append("")
        md.append("| Range Level | 80%+ Completion | 100% Completion | Avg Range Reached |")
        md.append("|-------------|-----------------|------------------|-------------------|")
        
        for range_level, pred in completion_preds.items():
            md.append(f"| {range_level} | {pred.get('probability_80pct', 0):.1%} | {pred.get('probability_100pct', 0):.1%} | {pred.get('avg_range_reached', 0):.1f}% |")
        
        md.append("")
        
        # Continuation predictions
        continuation_preds = predictive.get('continuation_predictions', {})
        if continuation_preds:
            md.append("### Cross-Session Continuation Predictions")
            md.append("")
            md.append("| Range Level | Continuation Probability | Evolution Strength |")
            md.append("|-------------|--------------------------|-------------------|")
            
            for range_level, pred in continuation_preds.items():
                md.append(f"| {range_level} | {pred.get('continuation_probability', 0):.1%} | {pred.get('evolution_strength', 0):.2f} |")
        
        md.append("")
        md.append("---")
        md.append("")
        md.append("*Analysis generated by IRONFORGE Range Confluence Analyzer*")
        
        return "\n".join(md)

if __name__ == "__main__":
    print("🎯 IRONFORGE Range Position Confluence Analyzer")
    print("=" * 60)
    
    analyzer = RangeConfluenceAnalyzer()
    output_file = analyzer.save_analysis()
    
    print(f"\n✅ Range confluence analysis complete!")
    print(f"📊 Results saved to: {output_file}")