#!/usr/bin/env python3
"""
Analyze NY PM session patterns from IRONFORGE discoveries
"""
import re
import glob
import pickle
from collections import defaultdict, Counter
from datetime import datetime

def analyze_nypm_patterns():
    """Analyze discovered patterns specifically for NY PM sessions"""
    
    # Get all NY PM sessions
    graph_files = glob.glob('/Users/jack/IRONPULSE/IRONFORGE/preservation/full_graph_store/NY_PM*.pkl')
    
    pm_analysis = {
        'sessions': [],
        'time_patterns': defaultdict(list),
        'price_clusters': defaultdict(int),
        'cross_timeframe_density': defaultdict(int),
        'cascade_sequences': []
    }
    
    print("🔍 Analyzing NY PM Cross-Timeframe Pattern Discovery...")
    
    for graph_file in graph_files:
        session_name = graph_file.split('/')[-1].replace('.pkl', '')
        date = re.search(r'(\d{4}_\d{2}_\d{2})', session_name).group(1) if re.search(r'(\d{4}_\d{2}_\d{2})', session_name) else 'unknown'
        
        pm_analysis['sessions'].append({
            'name': session_name,
            'date': date,
            'file': graph_file
        })
        
        try:
            with open(graph_file, 'rb') as f:
                graph_data = pickle.load(f)
            
            # Analyze scale edges (cross-timeframe links)
            scale_edges = graph_data.get('edges', {}).get('scale', [])
            
            print(f"\n📊 {session_name} ({date}):")
            print(f"   Scale edges discovered: {len(scale_edges)}")
            
            # Time-based pattern analysis
            time_clusters = defaultdict(list)
            price_levels = []
            
            for edge in scale_edges:
                if edge.get('tf_source') == '1m':  # 1m source events
                    # Extract price from source node
                    rich_features = graph_data.get('rich_node_features', [])
                    if edge.get('source', 0) < len(rich_features):
                        feature_str = str(rich_features[edge.get('source', 0)])
                        price_match = re.search(r"'price_level': ([0-9.]+)", feature_str)
                        time_match = re.search(r"'formatted_time': '([^']+)'", feature_str)
                        
                        if price_match and time_match:
                            price = float(price_match.group(1))
                            time_str = time_match.group(1)
                            target_tf = edge.get('tf_target', 'unknown')
                            
                            # Cluster by hour for time patterns
                            hour = time_str.split(':')[0] if ':' in time_str else 'unknown'
                            time_clusters[hour].append({
                                'price': price,
                                'time': time_str,
                                'target_tf': target_tf
                            })
                            
                            price_levels.append(price)
                            pm_analysis['cross_timeframe_density'][target_tf] += 1
            
            # Analyze key time periods
            key_times = ['13', '14', '15', '16']  # 1 PM, 2 PM, 3 PM, 4 PM
            for hour in key_times:
                if hour in time_clusters:
                    patterns = time_clusters[hour]
                    print(f"   {hour}:XX hour - {len(patterns)} cross-TF links discovered")
                    
                    # Look for price clustering
                    prices = [p['price'] for p in patterns]
                    if prices:
                        price_range = max(prices) - min(prices)
                        avg_price = sum(prices) / len(prices)
                        print(f"     Price range: {min(prices):.0f} - {max(prices):.0f} (avg: {avg_price:.0f})")
                        
                        # Target timeframe analysis
                        tf_targets = Counter([p['target_tf'] for p in patterns])
                        for tf, count in tf_targets.most_common(3):
                            print(f"     -> {tf}: {count} links")
            
            # Look for cascade sequences (same price across multiple timeframes)
            if price_levels:
                price_counter = Counter([int(p/10)*10 for p in price_levels])  # Round to nearest 10
                high_frequency_prices = [(price, count) for price, count in price_counter.most_common(5) if count >= 4]
                
                if high_frequency_prices:
                    print(f"   🎯 High-frequency price levels:")
                    for price, count in high_frequency_prices:
                        print(f"     {price}s level: {count} cross-TF connections")
                        pm_analysis['cascade_sequences'].append({
                            'session': session_name,
                            'date': date,
                            'price_cluster': price,
                            'connection_count': count
                        })
        
        except Exception as e:
            print(f"   ⚠️ Error analyzing {session_name}: {e}")
    
    # Summary analysis
    print(f"\n🏆 NY PM Cross-Timeframe Discovery Summary:")
    print(f"Sessions analyzed: {len(pm_analysis['sessions'])}")
    
    total_cross_tf = sum(pm_analysis['cross_timeframe_density'].values())
    print(f"Total cross-timeframe links: {total_cross_tf}")
    
    print(f"\nTarget timeframe distribution:")
    for tf, count in sorted(pm_analysis['cross_timeframe_density'].items()):
        percentage = (count / total_cross_tf) * 100 if total_cross_tf > 0 else 0
        print(f"  {tf}: {count} links ({percentage:.1f}%)")
    
    # High-impact cascade sequences
    if pm_analysis['cascade_sequences']:
        print(f"\n🎯 Discovered Cascade Sequences (4+ cross-TF connections):")
        for seq in sorted(pm_analysis['cascade_sequences'], key=lambda x: x['connection_count'], reverse=True):
            print(f"  {seq['date']} - {seq['price_cluster']}s level: {seq['connection_count']} connections")
    
    return pm_analysis

if __name__ == "__main__":
    analysis = analyze_nypm_patterns()