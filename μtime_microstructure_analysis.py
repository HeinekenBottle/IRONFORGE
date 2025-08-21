#!/usr/bin/env python3
"""
μTime — ±5m Microstructure Around Anchors Analysis
Real IRONFORGE implementation using actual data structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class IRONFORGEMicroTimeAnalyzer:
    def __init__(self, data_dir="/Users/jack/IRONFORGE/data/shards/NQ_M5"):
        self.data_dir = Path(data_dir)
        self.sessions_data = {}
        self.anchors = []
        self.micro_events = []
        
    def load_sessions(self):
        """Load all available IRONFORGE session data"""
        print("🔍 Loading IRONFORGE session data...")
        session_dirs = list(self.data_dir.glob("shard_*"))
        
        sessions_loaded = 0
        for session_dir in session_dirs:
            try:
                session_name = session_dir.name.replace("shard_", "")
                nodes_file = session_dir / "nodes.parquet"
                edges_file = session_dir / "edges.parquet"
                
                if nodes_file.exists():
                    nodes = pd.read_parquet(nodes_file)
                    edges = pd.read_parquet(edges_file) if edges_file.exists() else pd.DataFrame()
                    
                    # Convert timestamp to ET (timestamps are in milliseconds)
                    if 't' in nodes.columns:
                        nodes['timestamp_et'] = pd.to_datetime(nodes['t'], unit='ms')
                        nodes['hour'] = nodes['timestamp_et'].dt.hour
                        nodes['minute'] = nodes['timestamp_et'].dt.minute
                        nodes['time_str'] = nodes['timestamp_et'].dt.strftime('%H:%M')
                        
                        self.sessions_data[session_name] = {
                            'nodes': nodes,
                            'edges': edges,
                            'session_type': session_name.split('_')[0],
                            'date': session_name.split('_')[1] if '_' in session_name else 'unknown'
                        }
                        sessions_loaded += 1
                        print(f"  ✅ {session_name}: {len(nodes)} nodes, {len(edges)} edges")
                    
            except Exception as e:
                print(f"  ❌ {session_dir.name}: {str(e)}")
        
        print(f"📊 Loaded {sessions_loaded} sessions total")
        return sessions_loaded > 0
    
    def build_anchors(self):
        """Build comprehensive anchor list per session with ET timestamps"""
        print("\n🎯 Building anchor list...")
        
        for session_name, data in self.sessions_data.items():
            nodes = data['nodes']
            session_type = data['session_type']
            
            try:
                if 'price' in nodes.columns and len(nodes) > 0:
                    # Basic session statistics
                    session_high = nodes['price'].max()
                    session_low = nodes['price'].min()
                    session_range = session_high - session_low
                    
                    if session_range <= 0:
                        continue
                        
                    # Find session H/L taken times
                    high_row = nodes.loc[nodes['price'].idxmax()]
                    low_row = nodes.loc[nodes['price'].idxmin()]
                    
                    anchors = {
                        'session_name': session_name,
                        'session_type': session_type,
                        'session_H_taken': high_row['timestamp_et'],
                        'session_L_taken': low_row['timestamp_et'],
                        'session_high': session_high,
                        'session_low': session_low,
                        'session_range': session_range,
                        'priorDay_H': session_high,  # Simplified - would need prior day data
                        'priorDay_L': session_low,   # Simplified - would need prior day data
                    }
                    
                    # Theory B events (40%, 60%, 80% zones)
                    theory_b_events = []
                    for pct, zone_name in [(0.4, "40%"), (0.6, "60%"), (0.8, "80%")]:
                        target_level = session_low + (session_range * pct)
                        # Find closest price point to zone level
                        closest_idx = (nodes['price'] - target_level).abs().idxmin()
                        closest_row = nodes.loc[closest_idx]
                        
                        theory_b_events.append({
                            'zone': zone_name,
                            'timestamp_et': closest_row['timestamp_et'],
                            'price': closest_row['price'],
                            'precision': abs(closest_row['price'] - target_level)
                        })
                    
                    anchors['theory_b_events'] = theory_b_events
                    
                    # Session time deciles (10% through 90% completion)
                    time_deciles = {}
                    for decile in range(10, 100, 10):
                        idx = int(len(nodes) * decile / 100)
                        if idx < len(nodes):
                            time_deciles[f'{decile}%'] = nodes.iloc[idx]['timestamp_et']
                    anchors['time_deciles'] = time_deciles
                    
                    # Session range deciles (10% through 90% of price range)
                    range_deciles = {}
                    for decile in range(10, 100, 10):
                        target_price = session_low + (session_range * decile / 100)
                        closest_idx = (nodes['price'] - target_price).abs().idxmin()
                        range_deciles[f'{decile}%'] = nodes.loc[closest_idx]['timestamp_et']
                    anchors['range_deciles'] = range_deciles
                    
                    self.anchors.append(anchors)
                    
            except Exception as e:
                print(f"  ❌ Error processing {session_name}: {e}")
        
        print(f"⚓ Built anchors for {len(self.anchors)} sessions")
    
    def detect_micro_events(self, nodes, window_start, window_end):
        """Detect micro events in the given time window"""
        window_nodes = nodes[
            (nodes['timestamp_et'] >= window_start) & 
            (nodes['timestamp_et'] <= window_end)
        ].copy()
        
        if len(window_nodes) < 2:
            return []
        
        events = []
        
        # Calculate price movements
        price_changes = window_nodes['price'].diff()
        price_range = window_nodes['price'].max() - window_nodes['price'].min()
        
        # Detect displacement bars (large price moves)
        mean_change = abs(price_changes).mean()
        large_moves = abs(price_changes) > (mean_change * 2)
        
        if large_moves.any():
            events.append('displacement_bar')
        
        # Detect potential liquidity sweeps (price hits extremes then reverses)
        if len(window_nodes) >= 3:
            prices = window_nodes['price'].values
            # Simple pattern: high -> low -> recovery or low -> high -> decline
            if (max(prices[:len(prices)//2]) > max(prices[len(prices)//2:]) and 
                min(prices[len(prices)//2:]) < min(prices[:len(prices)//2])):
                events.append('liquidity_sweep')
        
        # Detect expansions (sustained directional movement)
        if price_range > mean_change * 3:
            first_half_avg = window_nodes.iloc[:len(window_nodes)//2]['price'].mean()
            second_half_avg = window_nodes.iloc[len(window_nodes)//2:]['price'].mean()
            if abs(second_half_avg - first_half_avg) > price_range * 0.3:
                events.append('expansion_phase')
        
        # Detect potential FVG creation (gaps in price action)
        if len(window_nodes) >= 3:
            # Look for price gaps
            price_diff_threshold = window_nodes['price'].std() * 1.5
            gaps = abs(price_changes) > price_diff_threshold
            if gaps.any():
                events.append('FVG_create')
        
        return events
    
    def scan_microstructure(self):
        """Scan ±5m windows around anchors for micro events"""
        print("\n🔬 Scanning microstructure around anchors...")
        
        all_minutes = defaultdict(int)
        anchor_event_pairs = defaultdict(int)
        sequences_15m = defaultdict(int)
        session_buckets = defaultdict(lambda: defaultdict(int))
        special_1435_events = 0
        
        total_anchors_scanned = 0
        
        for anchor_data in self.anchors:
            session_name = anchor_data['session_name']
            session_type = anchor_data['session_type']
            
            if session_name not in self.sessions_data:
                continue
                
            nodes = self.sessions_data[session_name]['nodes']
            
            # Collect all anchor timestamps
            anchor_times = []
            
            # Session H/L
            anchor_times.extend([
                ('session_H_taken', anchor_data['session_H_taken']),
                ('session_L_taken', anchor_data['session_L_taken'])
            ])
            
            # Theory B events
            for tb_event in anchor_data['theory_b_events']:
                anchor_times.append((f"TheoryB_{tb_event['zone']}", tb_event['timestamp_et']))
            
            # Time and range deciles
            for decile, timestamp in anchor_data['time_deciles'].items():
                anchor_times.append((f"time_decile_{decile}", timestamp))
            
            for decile, timestamp in anchor_data['range_deciles'].items():
                anchor_times.append((f"range_decile_{decile}", timestamp))
            
            # Scan ±5m around each anchor
            for anchor_type, anchor_time in anchor_times:
                if pd.isna(anchor_time):
                    continue
                    
                try:
                    total_anchors_scanned += 1
                    
                    # ±5m window
                    window_start = anchor_time - timedelta(minutes=5)
                    window_end = anchor_time + timedelta(minutes=5)
                    
                    # Find events in window
                    window_nodes = nodes[
                        (nodes['timestamp_et'] >= window_start) & 
                        (nodes['timestamp_et'] <= window_end)
                    ]
                    
                    # Count by minute
                    for _, node in window_nodes.iterrows():
                        minute_str = node['time_str']
                        all_minutes[minute_str] += 1
                        session_buckets[session_type][minute_str] += 1
                        
                        # 14:35 ET ±3m analysis
                        hour, minute = node['hour'], node['minute']
                        if hour == 14 and 32 <= minute <= 38:
                            special_1435_events += 1
                    
                    # Detect micro events
                    micro_events = self.detect_micro_events(nodes, window_start, window_end)
                    
                    for event_type in micro_events:
                        pair_key = f"{anchor_type}→{event_type}"
                        anchor_event_pairs[pair_key] += 1
                        
                        # Look for sequences within next 15m
                        sequence_end = anchor_time + timedelta(minutes=15)
                        future_events = self.detect_micro_events(nodes, anchor_time, sequence_end)
                        
                        for future_event in future_events:
                            seq_key = f"{anchor_type}→{future_event}"
                            sequences_15m[seq_key] += 1
                    
                except Exception as e:
                    print(f"    ❌ Error scanning {anchor_type}: {e}")
        
        # Store results
        self.hot_minutes = dict(Counter(all_minutes).most_common(10))
        self.anchor_events = dict(Counter(anchor_event_pairs).most_common(10))
        self.sequences = dict(Counter(sequences_15m).most_common(5))
        self.session_buckets = dict(session_buckets)
        self.special_1435_count = special_1435_events
        
        print(f"🎯 Scanned {total_anchors_scanned} anchors across {len(self.anchors)} sessions")
        print(f"🔥 Found activity in {len(all_minutes)} unique minutes")
        print(f"⚡ Detected {len(anchor_event_pairs)} anchor→event patterns")
        print(f"🔗 Found {len(sequences_15m)} sequence patterns")
        print(f"🎯 14:35 ET ±3m events: {special_1435_events}")
    
    def calculate_wilson_ci_width(self, count, total):
        """Calculate Wilson confidence interval width"""
        if total == 0:
            return 100.0
        
        p = count / total
        z = 1.96  # 95% confidence
        
        denominator = 1 + (z**2 / total)
        center = (p + z**2 / (2 * total)) / denominator
        width = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        return width * 200  # Convert to percentage points (both sides)
    
    def create_markdown_tables(self):
        """Create the 4 required markdown tables"""
        print("\n📊 Creating analysis tables...")
        
        # Table 1: Hot Minutes (ET)
        tables = "# μTime Microstructure Analysis Results\n\n"
        tables += "## Hot Minutes (ET): Top 10 Minute Stamps\n\n"
        tables += "| Minute (ET) | Count | % of Total |\n"
        tables += "|-------------|-------|------------|\n"
        
        total_events = sum(self.hot_minutes.values()) if self.hot_minutes else 1
        for minute, count in list(self.hot_minutes.items())[:10]:
            pct = (count / total_events * 100)
            tables += f"| {minute} | {count} | {pct:.1f}% |\n"
        
        if not self.hot_minutes:
            tables += "| missing | 0 | 0.0% |\n"
        
        # Table 2: Anchor→Event (±5m)
        tables += "\n## Anchor→Event (±5m): Top 10 Pairs with Lift vs Baseline\n\n"
        tables += "| Anchor→Event | Count | Lift vs Baseline |\n"
        tables += "|---------------|-------|------------------|\n"
        
        # Calculate baseline (average occurrence rate)
        baseline = np.mean(list(self.anchor_events.values())) if self.anchor_events else 1.0
        
        for pair, count in list(self.anchor_events.items())[:10]:
            lift = count / baseline
            tables += f"| {pair} | {count} | {lift:.2f}x |\n"
        
        if not self.anchor_events:
            tables += "| missing | 0 | 0.00x |\n"
        
        # Table 3: Sequences (A→B within 15m)
        tables += "\n## Sequences (A→B within 15m): Top 5\n\n"
        tables += "| Sequence | Count | % of Sessions |\n"
        tables += "|----------|-------|---------------|\n"
        
        total_sessions = len(self.sessions_data)
        for seq, count in list(self.sequences.items())[:5]:
            pct = (count / total_sessions * 100) if total_sessions > 0 else 0
            tables += f"| {seq} | {count} | {pct:.1f}% |\n"
        
        if not self.sequences:
            tables += "| missing | 0 | 0.0% |\n"
        
        # Table 4: By Session Bucket
        tables += "\n## By Session Bucket (ASIA/LONDON/NY-AM/NY-PM): Counts & % + Wilson CI\n\n"
        tables += "| Session Type | Events | % Total | Wilson CI Width | Flags |\n"
        tables += "|--------------|--------|---------|----------------|-------|\n"
        
        total_all_events = sum(sum(bucket.values()) for bucket in self.session_buckets.values())
        
        for session_type, bucket in self.session_buckets.items():
            bucket_total = sum(bucket.values())
            pct = (bucket_total / total_all_events * 100) if total_all_events > 0 else 0
            
            ci_width = self.calculate_wilson_ci_width(bucket_total, total_all_events)
            
            flags = []
            if bucket_total < 5:
                flags.append("n<5")
            if ci_width > 30:
                flags.append("CI>30pp")
            
            flag_str = ", ".join(flags) if flags else "—"
            
            tables += f"| {session_type} | {bucket_total} | {pct:.1f}% | {ci_width:.1f}pp | {flag_str} |\n"
        
        if not self.session_buckets:
            tables += "| missing | 0 | 0.0% | 100.0pp | n<5, CI>30pp |\n"
        
        # 14:35 ET ±3m Analysis
        tables += f"\n## 14:35 ET ±3m Analysis\n\n"
        tables += f"**Occurrences**: {self.special_1435_count} events\n\n"
        special_pct = (self.special_1435_count / total_all_events * 100) if total_all_events > 0 else 0
        tables += f"**Percentage**: {special_pct:.1f}% of all microstructure events\n\n"
        
        return tables
    
    def generate_insights(self):
        """Generate 3 key one-liner insights"""
        insights = []
        
        # Biggest minute hotspot
        if self.hot_minutes:
            top_minute = list(self.hot_minutes.keys())[0]
            top_count = list(self.hot_minutes.values())[0]
            insights.append(f"**Biggest Hotspot**: {top_minute} ET with {top_count} microstructure events across all sessions")
        else:
            insights.append("**Biggest Hotspot**: missing - insufficient event data detected")
        
        # Strongest Anchor→Event lift
        if self.anchor_events:
            top_pair = list(self.anchor_events.keys())[0]
            top_count = list(self.anchor_events.values())[0]
            baseline = np.mean(list(self.anchor_events.values()))
            lift = top_count / baseline
            insights.append(f"**Strongest Lift**: {top_pair} with {lift:.2f}x baseline lift ({top_count} occurrences)")
        else:
            insights.append("**Strongest Lift**: missing - insufficient anchor→event pairs detected")
        
        # Session-specific pattern
        if self.session_buckets:
            dominant_session = max(self.session_buckets.keys(), 
                                 key=lambda x: sum(self.session_buckets[x].values()))
            session_count = sum(self.session_buckets[dominant_session].values())
            total_sessions = len([k for k in self.session_buckets.keys() if k == dominant_session])
            insights.append(f"**Session Pattern**: {dominant_session} session type dominates with {session_count} microstructure events")
        else:
            insights.append("**Session Pattern**: missing - insufficient session bucket data")
        
        return insights

def main():
    """Main analysis execution"""
    print("🚀 μTime — ±5m Microstructure Around Anchors (IRONFORGE)")
    print("=" * 65)
    
    analyzer = IRONFORGEMicroTimeAnalyzer()
    
    # Load session data
    if not analyzer.load_sessions():
        print("❌ No session data loaded. Terminating analysis.")
        return
    
    # Build anchor list
    analyzer.build_anchors()
    if not analyzer.anchors:
        print("❌ No anchors created. Terminating analysis.")
        return
    
    # Scan microstructure
    analyzer.scan_microstructure()
    
    # Generate results
    tables = analyzer.create_markdown_tables()
    insights = analyzer.generate_insights()
    
    # Output
    print("\n" + "=" * 65)
    print("📊 MICROSTRUCTURE ANALYSIS RESULTS")
    print("=" * 65)
    print(tables)
    
    print("## Key Insights\n")
    for insight in insights:
        print(f"- {insight}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"μtime_analysis_{timestamp}.md"
    
    with open(output_file, 'w') as f:
        f.write(tables)
        f.write("\n## Key Insights\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()