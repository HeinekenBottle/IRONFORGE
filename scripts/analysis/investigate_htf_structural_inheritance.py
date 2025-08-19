#!/usr/bin/env python3
"""
RANK 6: Cross-Timeframe Structural Inheritance Investigation
===========================================================
Investigating how events inherit structural properties from higher timeframes,
creating detectable "HTF DNA" in 1-minute events.
"""

import glob
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

def extract_htf_ltf_relationships():
    """Extract HTF context and corresponding LTF event characteristics"""
    
    print("🏗️ EXTRACTING HTF-LTF STRUCTURAL RELATIONSHIPS")
    print("=" * 60)
    
    # Load preserved graphs with rich HTF context
    graph_files = glob.glob("/Users/jack/IRONPULSE/IRONFORGE/IRONFORGE/preservation/full_graph_store/*2025_08*.pkl")
    
    htf_ltf_data = []
    htf_context_stats = {'sessions_analyzed': 0, 'htf_features_found': 0, 'ltf_events_found': 0}
    
    print(f"🔍 Analyzing {len(graph_files)} sessions for HTF→LTF inheritance...")
    
    for graph_file in graph_files:
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            session_name = Path(graph_file).stem.replace('_graph_', '_').split('_202')[0]
            rich_features = graph_data.get('rich_node_features', [])
            htf_context_stats['sessions_analyzed'] += 1
            
            for i, feature in enumerate(rich_features):
                # Extract LTF (1m) event characteristics
                ltf_characteristics = {}
                
                # Event flags
                ltf_characteristics['expansion_flag'] = getattr(feature, 'expansion_phase_flag', 0.0)
                ltf_characteristics['consolidation_flag'] = getattr(feature, 'consolidation_flag', 0.0)
                ltf_characteristics['liq_sweep_flag'] = getattr(feature, 'liq_sweep_flag', 0.0)
                ltf_characteristics['reversal_flag'] = getattr(feature, 'reversal_flag', 0.0)
                ltf_characteristics['retracement_flag'] = getattr(feature, 'retracement_flag', 0.0)
                ltf_characteristics['fvg_redelivery_flag'] = getattr(feature, 'fvg_redelivery_flag', 0.0)
                
                # LTF event properties
                ltf_characteristics['time_minutes'] = getattr(feature, 'time_minutes', 0.0)
                ltf_characteristics['session_position'] = getattr(feature, 'session_position', 0.0)
                ltf_characteristics['normalized_price'] = getattr(feature, 'normalized_price', 0.0)
                ltf_characteristics['volatility_window'] = getattr(feature, 'volatility_window', 0.0)
                ltf_characteristics['energy_state'] = getattr(feature, 'energy_state', 0.0)
                
                # Price movement characteristics
                ltf_characteristics['pct_from_open'] = getattr(feature, 'pct_from_open', 0.0)
                ltf_characteristics['pct_from_high'] = getattr(feature, 'pct_from_high', 0.0)
                ltf_characteristics['pct_from_low'] = getattr(feature, 'pct_from_low', 0.0)
                
                # HTF context features
                htf_context = {}
                
                # HTF relationship features
                htf_context['price_to_HTF_ratio'] = getattr(feature, 'price_to_HTF_ratio', 0.0)
                htf_context['cross_tf_confluence'] = getattr(feature, 'cross_tf_confluence', 0.0)
                htf_context['timeframe_rank'] = getattr(feature, 'timeframe_rank', 0.0)
                htf_context['timeframe_source'] = getattr(feature, 'timeframe_source', 0.0)
                
                # HTF structural features
                htf_context['structural_importance'] = getattr(feature, 'structural_importance', 0.0)
                htf_context['fisher_regime'] = getattr(feature, 'fisher_regime', 0.0)
                htf_context['session_character'] = getattr(feature, 'session_character', 0.0)
                
                # Check if we have meaningful HTF context
                htf_signal_strength = sum(abs(v) for v in htf_context.values() if isinstance(v, int | float))
                
                # Check if we have meaningful LTF events
                ltf_event_present = any(ltf_characteristics[f] > 0.0 for f in [
                    'expansion_flag', 'consolidation_flag', 'liq_sweep_flag', 
                    'reversal_flag', 'retracement_flag', 'fvg_redelivery_flag'
                ])
                
                if htf_signal_strength > 0.1 and ltf_event_present:  # Threshold for meaningful data
                    htf_context_stats['htf_features_found'] += 1
                    htf_context_stats['ltf_events_found'] += 1
                    
                    # Create combined record
                    combined_record = {
                        'session': session_name,
                        'node_index': i,
                        'time_minutes': ltf_characteristics['time_minutes'],
                        'session_position': ltf_characteristics['session_position'],
                        
                        # LTF characteristics
                        **{f'ltf_{k}': v for k, v in ltf_characteristics.items()},
                        
                        # HTF context
                        **{f'htf_{k}': v for k, v in htf_context.items()},
                        
                        # Derived features
                        'htf_signal_strength': htf_signal_strength,
                        'dominant_ltf_event': get_dominant_event(ltf_characteristics)
                    }
                    
                    htf_ltf_data.append(combined_record)
        
        except Exception as e:
            print(f"  ⚠️ Error processing {Path(graph_file).stem}: {e}")
    
    print(f"✅ Extracted {len(htf_ltf_data)} HTF-LTF relationship records")
    print(f"📊 Sessions analyzed: {htf_context_stats['sessions_analyzed']}")
    print(f"🏗️ HTF features found: {htf_context_stats['htf_features_found']}")
    print(f"⚡ LTF events found: {htf_context_stats['ltf_events_found']}")
    
    return htf_ltf_data, htf_context_stats

def get_dominant_event(ltf_characteristics):
    """Determine the dominant LTF event type"""
    event_flags = {
        'expansion': ltf_characteristics.get('expansion_flag', 0.0),
        'consolidation': ltf_characteristics.get('consolidation_flag', 0.0),
        'liq_sweep': ltf_characteristics.get('liq_sweep_flag', 0.0),
        'reversal': ltf_characteristics.get('reversal_flag', 0.0),
        'retracement': ltf_characteristics.get('retracement_flag', 0.0),
        'fvg_redelivery': ltf_characteristics.get('fvg_redelivery_flag', 0.0)
    }
    
    # Return event type with highest flag value
    max_event = max(event_flags.items(), key=lambda x: x[1])
    return max_event[0] if max_event[1] > 0.0 else 'none'

def analyze_htf_ltf_correlations(htf_ltf_data):
    """Analyze correlations between HTF context and LTF event characteristics"""
    
    print("\n📊 HTF-LTF CORRELATION ANALYSIS")
    print("-" * 50)
    
    if not htf_ltf_data:
        print("❌ No HTF-LTF data available for correlation analysis")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(htf_ltf_data)
    
    # Define HTF features and LTF characteristics to correlate
    htf_features = [col for col in df.columns if col.startswith('htf_')]
    ltf_features = [col for col in df.columns if col.startswith('ltf_') and 'flag' not in col]
    
    print(f"🔍 Analyzing correlations between {len(htf_features)} HTF features and {len(ltf_features)} LTF characteristics")
    
    correlation_results = {}
    
    # Calculate correlations
    for htf_feature in htf_features:
        for ltf_feature in ltf_features:
            htf_values = df[htf_feature].dropna()
            ltf_values = df[ltf_feature].dropna()
            
            if len(htf_values) > 10 and len(ltf_values) > 10:  # Minimum sample size
                # Align indices
                common_indices = htf_values.index.intersection(ltf_values.index)
                if len(common_indices) > 10:
                    htf_aligned = htf_values.loc[common_indices]
                    ltf_aligned = ltf_values.loc[common_indices]
                    
                    # Calculate both Pearson and Spearman correlations
                    try:
                        pearson_corr, pearson_p = pearsonr(htf_aligned, ltf_aligned)
                        spearman_corr, spearman_p = spearmanr(htf_aligned, ltf_aligned)
                        
                        correlation_results[f"{htf_feature}→{ltf_feature}"] = {
                            'pearson_corr': pearson_corr,
                            'pearson_p': pearson_p,
                            'spearman_corr': spearman_corr,
                            'spearman_p': spearman_p,
                            'sample_size': len(common_indices),
                            'htf_feature': htf_feature,
                            'ltf_feature': ltf_feature
                        }
                    except:
                        continue
    
    # Report significant correlations
    print("\n🔥 SIGNIFICANT HTF→LTF CORRELATIONS (p < 0.05):")
    
    significant_correlations = []
    for relationship, stats in correlation_results.items():
        if stats['pearson_p'] < 0.05 and abs(stats['pearson_corr']) > 0.1:
            significant_correlations.append((relationship, stats))
    
    # Sort by correlation strength
    significant_correlations.sort(key=lambda x: abs(x[1]['pearson_corr']), reverse=True)
    
    for i, (relationship, stats) in enumerate(significant_correlations[:10]):
        print(f"   #{i+1}: {relationship}")
        print(f"       Pearson r: {stats['pearson_corr']:.3f} (p={stats['pearson_p']:.4f})")
        print(f"       Spearman ρ: {stats['spearman_corr']:.3f} (p={stats['spearman_p']:.4f})")
        print(f"       Sample size: {stats['sample_size']}")
    
    if not significant_correlations:
        print("   ❌ No significant correlations found (p < 0.05, |r| > 0.1)")
    
    return correlation_results

def discover_htf_inheritance_rules(htf_ltf_data):
    """Discover rules for how HTF states influence LTF event characteristics"""
    
    print("\n🧬 DISCOVERING HTF DNA INHERITANCE RULES")
    print("-" * 50)
    
    if not htf_ltf_data:
        print("❌ No HTF-LTF data available for rule discovery")
        return {}
    
    df = pd.DataFrame(htf_ltf_data)
    
    # Define HTF state categories
    inheritance_rules = {}
    
    # Rule 1: HTF Timeframe Rank influence
    print("\n📊 RULE 1: HTF Timeframe Rank → LTF Event Properties")
    
    # Categorize by timeframe rank
    high_tf_rank = df[df['htf_timeframe_rank'] > df['htf_timeframe_rank'].median()]
    low_tf_rank = df[df['htf_timeframe_rank'] <= df['htf_timeframe_rank'].median()]
    
    if len(high_tf_rank) > 5 and len(low_tf_rank) > 5:
        # Compare volatility
        high_tf_volatility = high_tf_rank['ltf_volatility_window'].mean()
        low_tf_volatility = low_tf_rank['ltf_volatility_window'].mean()
        
        # Compare energy state
        high_tf_energy = high_tf_rank['ltf_energy_state'].mean()
        low_tf_energy = low_tf_rank['ltf_energy_state'].mean()
        
        print(f"   High TF Rank events ({len(high_tf_rank)}):")
        print(f"      Avg volatility: {high_tf_volatility:.3f}")
        print(f"      Avg energy: {high_tf_energy:.3f}")
        print(f"   Low TF Rank events ({len(low_tf_rank)}):")
        print(f"      Avg volatility: {low_tf_volatility:.3f}")
        print(f"      Avg energy: {low_tf_energy:.3f}")
        
        volatility_diff = abs(high_tf_volatility - low_tf_volatility)
        energy_diff = abs(high_tf_energy - low_tf_energy)
        
        if volatility_diff > 0.1:
            inheritance_rules['timeframe_rank_volatility'] = {
                'rule': f"Higher TF rank → {'higher' if high_tf_volatility > low_tf_volatility else 'lower'} LTF volatility",
                'difference': volatility_diff,
                'significance': 'strong' if volatility_diff > 0.2 else 'moderate'
            }
        
        if energy_diff > 0.1:
            inheritance_rules['timeframe_rank_energy'] = {
                'rule': f"Higher TF rank → {'higher' if high_tf_energy > low_tf_energy else 'lower'} LTF energy",
                'difference': energy_diff,
                'significance': 'strong' if energy_diff > 0.2 else 'moderate'
            }
    
    # Rule 2: HTF Cross-TF Confluence influence
    print("\n📊 RULE 2: HTF Cross-TF Confluence → LTF Event Strength")
    
    high_confluence = df[df['htf_cross_tf_confluence'] > df['htf_cross_tf_confluence'].median()]
    low_confluence = df[df['htf_cross_tf_confluence'] <= df['htf_cross_tf_confluence'].median()]
    
    if len(high_confluence) > 5 and len(low_confluence) > 5:
        # Analyze event flag strengths
        high_conf_expansion = high_confluence['ltf_expansion_flag'].mean()
        low_conf_expansion = low_confluence['ltf_expansion_flag'].mean()
        
        high_conf_liq_sweep = high_confluence['ltf_liq_sweep_flag'].mean()
        low_conf_liq_sweep = low_confluence['ltf_liq_sweep_flag'].mean()
        
        print(f"   High Confluence events ({len(high_confluence)}):")
        print(f"      Avg expansion strength: {high_conf_expansion:.3f}")
        print(f"      Avg liq_sweep strength: {high_conf_liq_sweep:.3f}")
        print(f"   Low Confluence events ({len(low_confluence)}):")
        print(f"      Avg expansion strength: {low_conf_expansion:.3f}")
        print(f"      Avg liq_sweep strength: {low_conf_liq_sweep:.3f}")
        
        expansion_diff = abs(high_conf_expansion - low_conf_expansion)
        liq_sweep_diff = abs(high_conf_liq_sweep - low_conf_liq_sweep)
        
        if expansion_diff > 0.05:
            inheritance_rules['confluence_expansion'] = {
                'rule': f"Higher HTF confluence → {'stronger' if high_conf_expansion > low_conf_expansion else 'weaker'} expansion events",
                'difference': expansion_diff,
                'significance': 'strong' if expansion_diff > 0.1 else 'moderate'
            }
        
        if liq_sweep_diff > 0.02:
            inheritance_rules['confluence_liq_sweep'] = {
                'rule': f"Higher HTF confluence → {'stronger' if high_conf_liq_sweep > low_conf_liq_sweep else 'weaker'} liq_sweep events",
                'difference': liq_sweep_diff,
                'significance': 'strong' if liq_sweep_diff > 0.05 else 'moderate'
            }
    
    # Rule 3: HTF Price Ratio influence on LTF directional bias
    print("\n📊 RULE 3: HTF Price Ratio → LTF Directional Characteristics")
    
    high_price_ratio = df[df['htf_price_to_HTF_ratio'] > 1.0]  # Above HTF reference
    low_price_ratio = df[df['htf_price_to_HTF_ratio'] < 1.0]   # Below HTF reference
    
    if len(high_price_ratio) > 5 and len(low_price_ratio) > 5:
        # Analyze price movement characteristics
        high_ratio_from_high = high_price_ratio['ltf_pct_from_high'].mean()
        low_ratio_from_high = low_price_ratio['ltf_pct_from_high'].mean()
        
        high_ratio_from_low = high_price_ratio['ltf_pct_from_low'].mean()
        low_ratio_from_low = low_price_ratio['ltf_pct_from_low'].mean()
        
        print(f"   High HTF Price Ratio events ({len(high_price_ratio)}):")
        print(f"      Avg % from high: {high_ratio_from_high:.1f}%")
        print(f"      Avg % from low: {high_ratio_from_low:.1f}%")
        print(f"   Low HTF Price Ratio events ({len(low_price_ratio)}):")
        print(f"      Avg % from high: {low_ratio_from_high:.1f}%")
        print(f"      Avg % from low: {low_ratio_from_low:.1f}%")
        
        from_high_diff = abs(high_ratio_from_high - low_ratio_from_high)
        from_low_diff = abs(high_ratio_from_low - low_ratio_from_low)
        
        if from_high_diff > 5.0:  # 5% difference threshold
            inheritance_rules['price_ratio_high_bias'] = {
                'rule': f"Higher HTF price ratio → events occur {'closer to' if high_ratio_from_high < low_ratio_from_high else 'further from'} session highs",
                'difference': from_high_diff,
                'significance': 'strong' if from_high_diff > 10.0 else 'moderate'
            }
        
        if from_low_diff > 5.0:
            inheritance_rules['price_ratio_low_bias'] = {
                'rule': f"Higher HTF price ratio → events occur {'closer to' if high_ratio_from_low < low_ratio_from_low else 'further from'} session lows",
                'difference': from_low_diff,
                'significance': 'strong' if from_low_diff > 10.0 else 'moderate'
            }
    
    # Report discovered rules
    print("\n🧬 DISCOVERED HTF DNA INHERITANCE RULES:")
    
    if inheritance_rules:
        for rule_name, rule_data in inheritance_rules.items():
            significance = rule_data['significance']
            difference = rule_data['difference']
            rule_text = rule_data['rule']
            
            print(f"   🔗 {rule_name.upper()}:")
            print(f"      Rule: {rule_text}")
            print(f"      Strength: {significance} (diff: {difference:.3f})")
    else:
        print("   ❌ No strong inheritance rules discovered")
    
    return inheritance_rules

def analyze_event_type_htf_preferences(htf_ltf_data):
    """Analyze HTF context preferences for different LTF event types"""
    
    print("\n🎯 EVENT TYPE HTF PREFERENCES")
    print("-" * 50)
    
    if not htf_ltf_data:
        print("❌ No HTF-LTF data available for event preference analysis")
        return {}
    
    df = pd.DataFrame(htf_ltf_data)
    
    # Group by dominant event type
    event_preferences = {}
    
    for event_type in ['expansion', 'consolidation', 'liq_sweep', 'reversal', 'retracement', 'fvg_redelivery']:
        event_data = df[df['dominant_ltf_event'] == event_type]
        
        if len(event_data) > 5:  # Minimum sample size
            preferences = {
                'count': len(event_data),
                'avg_htf_timeframe_rank': event_data['htf_timeframe_rank'].mean(),
                'avg_htf_cross_tf_confluence': event_data['htf_cross_tf_confluence'].mean(),
                'avg_htf_price_ratio': event_data['htf_price_to_HTF_ratio'].mean(),
                'avg_htf_structural_importance': event_data['htf_structural_importance'].mean(),
                'avg_htf_fisher_regime': event_data['htf_fisher_regime'].mean(),
                'avg_htf_signal_strength': event_data['htf_signal_strength'].mean()
            }
            
            event_preferences[event_type] = preferences
            
            print(f"\n📊 {event_type.upper()} Events ({preferences['count']} occurrences):")
            print(f"   HTF Timeframe Rank: {preferences['avg_htf_timeframe_rank']:.3f}")
            print(f"   HTF Cross-TF Confluence: {preferences['avg_htf_cross_tf_confluence']:.3f}")
            print(f"   HTF Price Ratio: {preferences['avg_htf_price_ratio']:.3f}")
            print(f"   HTF Structural Importance: {preferences['avg_htf_structural_importance']:.3f}")
            print(f"   HTF Fisher Regime: {preferences['avg_htf_fisher_regime']:.3f}")
            print(f"   HTF Signal Strength: {preferences['avg_htf_signal_strength']:.3f}")
    
    return event_preferences

def test_htf_coherence_hypothesis(htf_ltf_data):
    """Test the specific hypothesis that discovered patterns align across timeframes"""
    
    print("\n🧪 TESTING HTF COHERENCE HYPOTHESIS")
    print("=" * 60)
    print("Hypothesis: Discovered patterns show multi-timeframe coherence")
    
    if not htf_ltf_data:
        print("❌ No HTF-LTF data available for coherence testing")
        return {}
    
    df = pd.DataFrame(htf_ltf_data)
    
    coherence_results = {}
    
    # Test 1: HTF-LTF Event Alignment
    print("\n📊 TEST 1: HTF-LTF Event Alignment")
    
    # High HTF signal strength should correlate with stronger LTF events
    high_htf_signal = df[df['htf_signal_strength'] > df['htf_signal_strength'].quantile(0.75)]
    low_htf_signal = df[df['htf_signal_strength'] < df['htf_signal_strength'].quantile(0.25)]
    
    if len(high_htf_signal) > 5 and len(low_htf_signal) > 5:
        # Compare LTF event strengths
        high_htf_expansion = high_htf_signal['ltf_expansion_flag'].mean()
        low_htf_expansion = low_htf_signal['ltf_expansion_flag'].mean()
        
        high_htf_volatility = high_htf_signal['ltf_volatility_window'].mean()
        low_htf_volatility = low_htf_signal['ltf_volatility_window'].mean()
        
        expansion_coherence = high_htf_expansion > low_htf_expansion
        volatility_coherence = high_htf_volatility > low_htf_volatility
        
        coherence_results['event_alignment'] = {
            'expansion_coherence': expansion_coherence,
            'volatility_coherence': volatility_coherence,
            'expansion_diff': high_htf_expansion - low_htf_expansion,
            'volatility_diff': high_htf_volatility - low_htf_volatility
        }
        
        print(f"   High HTF signal → Higher LTF expansion: {'✅' if expansion_coherence else '❌'}")
        print(f"   High HTF signal → Higher LTF volatility: {'✅' if volatility_coherence else '❌'}")
        print(f"   Expansion difference: {high_htf_expansion - low_htf_expansion:.3f}")
        print(f"   Volatility difference: {high_htf_volatility - low_htf_volatility:.3f}")
    
    # Test 2: Cross-TF Confluence Validation
    print("\n📊 TEST 2: Cross-TF Confluence Validation")
    
    high_confluence = df[df['htf_cross_tf_confluence'] > df['htf_cross_tf_confluence'].quantile(0.75)]
    low_confluence = df[df['htf_cross_tf_confluence'] < df['htf_cross_tf_confluence'].quantile(0.25)]
    
    if len(high_confluence) > 5 and len(low_confluence) > 5:
        # High confluence should lead to more decisive LTF events
        high_conf_events = (high_confluence['ltf_expansion_flag'] + 
                           high_confluence['ltf_liq_sweep_flag'] + 
                           high_confluence['ltf_reversal_flag']).mean()
        
        low_conf_events = (low_confluence['ltf_expansion_flag'] + 
                          low_confluence['ltf_liq_sweep_flag'] + 
                          low_confluence['ltf_reversal_flag']).mean()
        
        confluence_coherence = high_conf_events > low_conf_events
        
        coherence_results['confluence_validation'] = {
            'coherence': confluence_coherence,
            'high_confluence_strength': high_conf_events,
            'low_confluence_strength': low_conf_events,
            'difference': high_conf_events - low_conf_events
        }
        
        print(f"   High confluence → Stronger LTF events: {'✅' if confluence_coherence else '❌'}")
        print(f"   High confluence event strength: {high_conf_events:.3f}")
        print(f"   Low confluence event strength: {low_conf_events:.3f}")
        print(f"   Difference: {high_conf_events - low_conf_events:.3f}")
    
    # Test 3: Structural Importance Inheritance
    print("\n📊 TEST 3: Structural Importance Inheritance")
    
    high_structural = df[df['htf_structural_importance'] > df['htf_structural_importance'].quantile(0.75)]
    low_structural = df[df['htf_structural_importance'] < df['htf_structural_importance'].quantile(0.25)]
    
    if len(high_structural) > 5 and len(low_structural) > 5:
        # High structural importance should correlate with more significant LTF price movements
        high_struct_price_movement = (abs(high_structural['ltf_pct_from_open']) + 
                                     high_structural['ltf_volatility_window']).mean()
        
        low_struct_price_movement = (abs(low_structural['ltf_pct_from_open']) + 
                                    low_structural['ltf_volatility_window']).mean()
        
        structural_coherence = high_struct_price_movement > low_struct_price_movement
        
        coherence_results['structural_inheritance'] = {
            'coherence': structural_coherence,
            'high_structural_movement': high_struct_price_movement,
            'low_structural_movement': low_struct_price_movement,
            'difference': high_struct_price_movement - low_struct_price_movement
        }
        
        print(f"   High structural importance → Larger LTF movements: {'✅' if structural_coherence else '❌'}")
        print(f"   High structural movement: {high_struct_price_movement:.3f}")
        print(f"   Low structural movement: {low_struct_price_movement:.3f}")
        print(f"   Difference: {high_struct_price_movement - low_struct_price_movement:.3f}")
    
    # Overall coherence assessment
    coherence_tests_passed = sum(1 for test in coherence_results.values() 
                                if test.get('coherence', False) or 
                                   test.get('expansion_coherence', False) or 
                                   test.get('volatility_coherence', False))
    
    total_tests = len(coherence_results) + sum(1 for test in coherence_results.values() 
                                              if 'expansion_coherence' in test)
    
    coherence_rate = (coherence_tests_passed / total_tests * 100) if total_tests > 0 else 0
    
    print("\n🎯 OVERALL HTF COHERENCE ASSESSMENT:")
    print(f"   Tests passed: {coherence_tests_passed}/{total_tests}")
    print(f"   Coherence rate: {coherence_rate:.1f}%")
    
    if coherence_rate >= 75:
        print("   ✅ STRONG multi-timeframe coherence detected")
    elif coherence_rate >= 50:
        print("   ⚠️ MODERATE multi-timeframe coherence detected")
    else:
        print("   ❌ WEAK multi-timeframe coherence detected")
    
    coherence_results['overall_assessment'] = {
        'tests_passed': coherence_tests_passed,
        'total_tests': total_tests,
        'coherence_rate': coherence_rate,
        'assessment': 'strong' if coherence_rate >= 75 else 'moderate' if coherence_rate >= 50 else 'weak'
    }
    
    return coherence_results

def create_htf_inheritance_visualization(htf_ltf_data, correlation_results, inheritance_rules, coherence_results):
    """Create comprehensive visualization of HTF-LTF inheritance patterns"""
    
    print("\n📈 CREATING HTF INHERITANCE VISUALIZATION")
    print("-" * 50)
    
    if not htf_ltf_data:
        print("❌ No data for visualization")
        return
    
    df = pd.DataFrame(htf_ltf_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Timeframe Structural Inheritance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: HTF Signal Strength vs LTF Event Strength
    ax1 = axes[0, 0]
    
    ltf_event_strength = (df['ltf_expansion_flag'] + df['ltf_consolidation_flag'] + 
                         df['ltf_liq_sweep_flag'] + df['ltf_reversal_flag'])
    
    scatter = ax1.scatter(df['htf_signal_strength'], ltf_event_strength, 
                         alpha=0.6, s=30, c=df['ltf_volatility_window'], cmap='viridis')
    ax1.set_xlabel('HTF Signal Strength')
    ax1.set_ylabel('LTF Event Strength')
    ax1.set_title('HTF Signal vs LTF Event Strength')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Volatility')
    
    # Plot 2: Cross-TF Confluence vs Event Types
    ax2 = axes[0, 1]
    
    event_types = ['expansion', 'consolidation', 'liq_sweep', 'reversal', 'retracement']
    confluence_by_type = []
    
    for event_type in event_types:
        event_data = df[df['dominant_ltf_event'] == event_type]
        if len(event_data) > 0:
            confluence_by_type.append(event_data['htf_cross_tf_confluence'].mean())
        else:
            confluence_by_type.append(0)
    
    bars = ax2.bar(event_types, confluence_by_type, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Average HTF Cross-TF Confluence')
    ax2.set_title('HTF Confluence by LTF Event Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, confluence_by_type, strict=False):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Price Ratio vs Price Position
    ax3 = axes[0, 2]
    
    ax3.scatter(df['htf_price_to_HTF_ratio'], df['ltf_pct_from_high'], 
               alpha=0.6, s=30, label='% from High')
    ax3.scatter(df['htf_price_to_HTF_ratio'], df['ltf_pct_from_low'], 
               alpha=0.6, s=30, label='% from Low')
    ax3.set_xlabel('HTF Price to HTF Ratio')
    ax3.set_ylabel('LTF Price Position (%)')
    ax3.set_title('HTF Price Ratio vs LTF Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation heatmap (top correlations)
    ax4 = axes[1, 0]
    
    if correlation_results:
        # Get top 10 correlations
        top_correlations = sorted(correlation_results.items(), 
                                key=lambda x: abs(x[1]['pearson_corr']), reverse=True)[:10]
        
        if top_correlations:
            relationships = [rel[0].replace('htf_', '').replace('ltf_', '').replace('→', '→') for rel, _ in top_correlations]
            correlations = [stats['pearson_corr'] for _, stats in top_correlations]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(relationships))
            bars = ax4.barh(y_pos, correlations, color=['red' if c < 0 else 'blue' for c in correlations], alpha=0.7)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([rel[:20] + '...' if len(rel) > 20 else rel for rel in relationships], fontsize=8)
            ax4.set_xlabel('Correlation Coefficient')
            ax4.set_title('Top HTF→LTF Correlations')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 5: Inheritance rules strength
    ax5 = axes[1, 1]
    
    if inheritance_rules:
        rule_names = list(inheritance_rules.keys())
        rule_strengths = [rule['difference'] for rule in inheritance_rules.values()]
        rule_colors = ['red' if rule['significance'] == 'strong' else 'orange' 
                      for rule in inheritance_rules.values()]
        
        bars = ax5.bar(rule_names, rule_strengths, color=rule_colors, alpha=0.7)
        ax5.set_ylabel('Rule Strength (Difference)')
        ax5.set_title('HTF DNA Inheritance Rules')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        import matplotlib.patches as mpatches
        strong_patch = mpatches.Patch(color='red', label='Strong')
        moderate_patch = mpatches.Patch(color='orange', label='Moderate')
        ax5.legend(handles=[strong_patch, moderate_patch])
    else:
        ax5.text(0.5, 0.5, 'No Inheritance\nRules Discovered', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('HTF DNA Inheritance Rules')
    
    # Plot 6: Coherence test results
    ax6 = axes[1, 2]
    
    if coherence_results and 'overall_assessment' in coherence_results:
        assessment = coherence_results['overall_assessment']
        tests_passed = assessment['tests_passed']
        total_tests = assessment['total_tests']
        coherence_rate = assessment['coherence_rate']
        
        # Create pie chart of test results
        passed_tests = tests_passed
        failed_tests = total_tests - tests_passed
        
        if total_tests > 0:
            labels = ['Passed', 'Failed']
            sizes = [passed_tests, failed_tests]
            colors = ['green', 'red']
            
            wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                              startangle=90)
            ax6.set_title(f'HTF Coherence Tests\n({coherence_rate:.1f}% coherence)')
        else:
            ax6.text(0.5, 0.5, 'No Coherence\nTests Available', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('HTF Coherence Tests')
    else:
        ax6.text(0.5, 0.5, 'No Coherence\nTests Available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('HTF Coherence Tests')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = '/Users/jack/IRONPULSE/IRONFORGE/htf_structural_inheritance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    return output_path

def main():
    """Main HTF structural inheritance investigation"""
    
    print("🏗️ CROSS-TIMEFRAME STRUCTURAL INHERITANCE INVESTIGATION")
    print("=" * 80)
    print("Investigating how events inherit structural properties from higher timeframes")
    print("Creating detectable 'HTF DNA' in 1-minute events")
    print("=" * 80)
    
    # Step 1: Extract HTF-LTF relationships
    htf_ltf_data, htf_context_stats = extract_htf_ltf_relationships()
    if not htf_ltf_data:
        print("❌ Cannot proceed without HTF-LTF relationship data")
        return
    
    # Step 2: Analyze correlations
    correlation_results = analyze_htf_ltf_correlations(htf_ltf_data)
    
    # Step 3: Discover inheritance rules
    inheritance_rules = discover_htf_inheritance_rules(htf_ltf_data)
    
    # Step 4: Analyze event type preferences
    analyze_event_type_htf_preferences(htf_ltf_data)
    
    # Step 5: Test coherence hypothesis
    coherence_results = test_htf_coherence_hypothesis(htf_ltf_data)
    
    # Step 6: Create visualization
    viz_path = create_htf_inheritance_visualization(htf_ltf_data, correlation_results, 
                                                   inheritance_rules, coherence_results)
    
    # Summary
    print("\n🏗️ HTF STRUCTURAL INHERITANCE SUMMARY")
    print("=" * 60)
    
    total_relationships = len(htf_ltf_data)
    significant_correlations = len([r for r in correlation_results.values() 
                                  if r['pearson_p'] < 0.05 and abs(r['pearson_corr']) > 0.1])
    inheritance_rules_count = len(inheritance_rules)
    
    print("📊 Analysis Results:")
    print(f"   HTF-LTF relationships analyzed: {total_relationships}")
    print(f"   Significant correlations found: {significant_correlations}")
    print(f"   HTF DNA inheritance rules: {inheritance_rules_count}")
    
    # Key discoveries
    if coherence_results and 'overall_assessment' in coherence_results:
        assessment = coherence_results['overall_assessment']
        coherence_rate = assessment['coherence_rate']
        coherence_level = assessment['assessment']
        
        print("\n🧬 HTF DNA INHERITANCE:")
        print(f"   Multi-timeframe coherence: {coherence_rate:.1f}% ({coherence_level})")
        
        if coherence_rate >= 75:
            print("   ✅ STRONG evidence of HTF structural inheritance")
        elif coherence_rate >= 50:
            print("   ⚠️ MODERATE evidence of HTF structural inheritance")
        else:
            print("   ❌ WEAK evidence of HTF structural inheritance")
    
    # Top inheritance rules
    if inheritance_rules:
        print("\n🔗 TOP HTF DNA INHERITANCE RULES:")
        sorted_rules = sorted(inheritance_rules.items(), 
                            key=lambda x: x[1]['difference'], reverse=True)
        
        for i, (_rule_name, rule_data) in enumerate(sorted_rules[:3]):
            print(f"   #{i+1}: {rule_data['rule']}")
            print(f"       Strength: {rule_data['significance']} (diff: {rule_data['difference']:.3f})")
    
    # Top correlations
    if correlation_results:
        print("\n📊 TOP HTF→LTF CORRELATIONS:")
        top_correlations = sorted(correlation_results.items(), 
                                key=lambda x: abs(x[1]['pearson_corr']), reverse=True)
        
        for i, (relationship, stats) in enumerate(top_correlations[:3]):
            print(f"   #{i+1}: {relationship}")
            print(f"       Correlation: {stats['pearson_corr']:.3f} (p={stats['pearson_p']:.4f})")
    
    print("\n✅ RANK 6 INVESTIGATION COMPLETE")
    print(f"💾 Results visualization: {viz_path}")
    
    if inheritance_rules_count > 0 or significant_correlations > 0:
        print("\n🚀 BREAKTHROUGH: HTF DNA detected in LTF events!")
        print("   Events inherit measurable structural properties from higher timeframes.")
        print("   This validates multi-timeframe coherence in discovered patterns.")
    else:
        print("\n📝 Result: Limited HTF structural inheritance detected.")
        print("   LTF events appear more independent of HTF context than expected.")

if __name__ == "__main__":
    main()