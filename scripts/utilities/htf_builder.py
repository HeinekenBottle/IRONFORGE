"""
HTF Builder Improved - Build sophisticated multi-timeframe data for IRONFORGE
Generates pythonnodes arrays and htf_cross_map for graph builder integration
"""
import glob
import json


class HTFBuilder:
    def __init__(self):
        self.timeframes = {
            '5m': 5,
            '15m': 15,
            '1h': 60,
            'D': 1440
        }
    
    def aggregate_session(self, session_path):
        """Build sophisticated HTF data with pythonnodes and cross_map"""
        with open(session_path) as f:
            session = json.load(f)
        
        if 'price_movements' not in session:
            return None
        
        movements = session['price_movements']
        if not movements:
            return None
            
        # Build pythonnodes for each timeframe
        pythonnodes = {}
        pythonnodes['1m'] = self._create_1m_nodes(movements)
        
        # Build HTF nodes  
        for tf_name, tf_minutes in self.timeframes.items():
            pythonnodes[tf_name] = self._build_htf_nodes(movements, tf_minutes)
        
        # Create cross-timeframe mapping
        htf_cross_map = self._build_cross_map(pythonnodes)
        
        # Add to session in expected format
        session['pythonnodes'] = pythonnodes
        session['htf_cross_map'] = htf_cross_map
        
        return session
    
    def _create_1m_nodes(self, movements):
        """Convert 1m price movements to node format"""
        nodes = []
        for i, movement in enumerate(movements):
            node = {
                'id': i,
                'timestamp': movement.get('timestamp', '00:00:00'),
                'price_level': movement.get('price_level', 0),
                'open': movement.get('price_level', 0),
                'high': movement.get('price_level', 0),
                'low': movement.get('price_level', 0), 
                'close': movement.get('price_level', 0),
                'event_type': movement.get('event_type', 'session_level'),
                'context': movement.get('context', ''),
                'contamination_risk': movement.get('contamination_risk', False),
                'meta': {
                    'coverage': 1,  # 1m coverage
                    'source': '1m_movement'
                }
            }
            nodes.append(node)
        return nodes

    def _build_htf_nodes(self, movements, period_minutes):
        """Build HTF nodes by aggregating 1m movements"""
        if not movements:
            return []
            
        nodes = []
        current_bucket = []
        node_id = 0
        
        for movement in movements:
            current_bucket.append(movement)
            
            # Create HTF node when bucket is full or at end
            if len(current_bucket) >= period_minutes or movement == movements[-1]:
                if current_bucket:
                    node = self._create_htf_node(current_bucket, node_id, period_minutes)
                    nodes.append(node)
                    node_id += 1
                    current_bucket = []
        
        return nodes
    
    def _create_htf_node(self, bucket, node_id, period_minutes):
        """Create single HTF node from bucket of 1m movements"""
        prices = [m.get('price_level', 0) for m in bucket]
        
        node = {
            'id': node_id,
            'timestamp': bucket[0].get('timestamp', '00:00:00'),
            'open': bucket[0].get('price_level', 0),
            'high': max(prices) if prices else 0,
            'low': min(prices) if prices else 0,
            'close': bucket[-1].get('price_level', 0),
            'timeframe': f'{period_minutes}m',
            'source_movements': len(bucket),
            
            # HTF pattern detection
            'pd_array': self._detect_pd_array(prices),
            'liquidity_sweep': self._detect_liquidity(prices),
            'fpfvg': self._detect_fpfvg(prices),
            
            'meta': {
                'coverage': len(bucket),  # How many 1m movements this covers
                'period_minutes': period_minutes,
                'price_range': max(prices) - min(prices) if prices else 0
            }
        }
        
        return node
    
    def _build_cross_map(self, pythonnodes):
        """Build cross-timeframe mapping for scale edges"""
        cross_map = {}
        
        # Build mapping from 1m to each HTF
        one_min_nodes = pythonnodes.get('1m', [])
        
        for tf_name in self.timeframes:
            htf_nodes = pythonnodes.get(tf_name, [])
            if not htf_nodes:
                continue
                
            mapping_key = f"1m_to_{tf_name}"
            cross_map[mapping_key] = {}
            
            # Map each 1m node to appropriate HTF parent
            for one_min_idx, one_min_node in enumerate(one_min_nodes):
                parent_htf_idx = self._find_parent_htf_node(
                    one_min_node, htf_nodes, self.timeframes[tf_name]
                )
                cross_map[mapping_key][str(one_min_idx)] = parent_htf_idx
                
        return cross_map
    
    def _find_parent_htf_node(self, one_min_node, htf_nodes, period_minutes):
        """Find which HTF node contains this 1m node"""
        one_min_time = self._parse_time(one_min_node.get('timestamp', '00:00:00'))
        
        for htf_idx, htf_node in enumerate(htf_nodes):
            htf_time = self._parse_time(htf_node.get('timestamp', '00:00:00'))
            
            # Check if 1m node falls within this HTF node's time window
            if htf_time <= one_min_time < (htf_time + period_minutes):
                return htf_idx
        
        # Default to first HTF node if no match found
        return 0 if htf_nodes else None
    
    def _parse_time(self, timestamp):
        """Convert HH:MM:SS to minutes since midnight"""
        try:
            if ':' in timestamp:
                parts = timestamp.split(':')
                return int(parts[0]) * 60 + int(parts[1])
            return 0
        except (ValueError, IndexError):
            return 0
    
    def _detect_pd_array(self, prices):
        """Detect PD Array formation in price sequence"""
        if len(prices) < 3:
            return None
            
        # Simple swing high/low detection
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                return {'type': 'swing_high', 'level': prices[i]}
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                return {'type': 'swing_low', 'level': prices[i]}
        return None
    
    def _detect_liquidity(self, prices):
        """Detect liquidity sweep"""
        if len(prices) < 2:
            return False
        # Simplified: detect price spike and reversal
        max_price = max(prices)
        max_idx = prices.index(max_price)
        if max_idx > 0 and max_idx < len(prices)-1:
            if prices[max_idx+1] < prices[max_idx-1]:
                return True
        return False
    
    def _detect_fpfvg(self, prices):
        """Detect Fair Value Gap"""
        if len(prices) < 3:
            return None
        # Simplified: gap between candles
        for i in range(1, len(prices)-1):
            gap_up = prices[i-1] < prices[i+1] and prices[i] > prices[i+1]
            gap_down = prices[i-1] > prices[i+1] and prices[i] < prices[i+1]
            if gap_up or gap_down:
                return {'gap': True, 'level': prices[i]}
        return None
    
    def process_all_sessions(self, input_dir='/Users/jack/IRONPULSE/data/sessions/level_1', 
                           output_dir='/Users/jack/IRONPULSE/data/sessions/htf_enhanced'):
        """Process all sessions and add sophisticated HTF data with pythonnodes"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        pattern = f'{input_dir}/**/*.json'
        files = glob.glob(pattern, recursive=True)
        processed = 0
        
        for filepath in files:
            print(f"Processing {os.path.basename(filepath)}...")
            enhanced = self.aggregate_session(filepath)
            
            if enhanced:
                output_path = os.path.join(
                    output_dir, 
                    os.path.basename(filepath).replace('.json', '_htf.json')
                )
                with open(output_path, 'w') as f:
                    json.dump(enhanced, f, indent=2)
                processed += 1
                
                # Print summary for first file
                if processed == 1:
                    pythonnodes = enhanced.get('pythonnodes', {})
                    cross_map = enhanced.get('htf_cross_map', {})
                    print("  Sample HTF data structure:")
                    for tf, nodes in pythonnodes.items():
                        print(f"    {tf}: {len(nodes)} nodes")
                    print(f"    Cross-mappings: {len(cross_map)} timeframe pairs")
        
        print(f"\n✅ Created HTF-enhanced data for {processed} sessions")
        print(f"📁 Output directory: {output_dir}")
        return processed

if __name__ == "__main__":
    builder = HTFBuilder()
    builder.process_all_sessions()
