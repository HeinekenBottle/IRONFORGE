"""
Timeframe Lattice Mapper
Pattern analysis component for timeframe relationships
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TimeframeLatticeMapper:
    """
    Maps discovered patterns across different timeframes
    Analyzes multi-timeframe pattern relationships
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Timeframe Lattice Mapper initialized")
    
    def map_timeframe_patterns(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map patterns across timeframes
        
        Args:
            session_data: Session data with multi-timeframe information
            
        Returns:
            Timeframe mapping results
        """
        try:
            session_name = session_data.get('session_name', 'unknown')
            self.logger.info(f"Mapping timeframe patterns for {session_name}")
            
            # Placeholder implementation
            results = {
                'session_name': session_name,
                'timeframe_mapping': 'not_yet_implemented',
                'status': 'placeholder'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Timeframe mapping failed: {e}")
            return {'error': str(e)}
    
    def map_session_lattice(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create global lattice mapping from Monthly → 1m timeframes
        
        Args:
            session_data: Enhanced session data
            
        Returns:
            Comprehensive lattice with nodes, edges, and hot zones
        """
        try:
            session_name = session_data.get('session_name', 'unknown')
            
            # Timeframes from highest to lowest resolution
            timeframes = ['Monthly', 'Weekly', 'Daily', '4H', '1H', '15m', '5m', '1m']
            
            # Extract events and price data from enhanced session format
            events = []
            
            # Convert session_liquidity_events to events
            liquidity_events = session_data.get('session_liquidity_events', [])
            for event in liquidity_events:
                events.append({
                    'timestamp': event.get('timestamp', '00:00:00'),
                    'price': event.get('price_level', 0),
                    'event_type': event.get('event_type', 'liquidity'),
                    'significance': event.get('intensity', 0.5),
                    'archaeological_zone': event.get('zone', 'unknown')
                })
            
            # Convert price_movements to events
            price_movements = session_data.get('price_movements', [])
            for movement in price_movements:
                events.append({
                    'timestamp': movement.get('timestamp', '00:00:00'),
                    'price': movement.get('price', 0),
                    'event_type': 'price_movement',
                    'significance': movement.get('intensity', 0.6),
                    'archaeological_zone': movement.get('zone', 'movement')
                })
            
            # Convert FPFVG interactions to events
            fpfvg = session_data.get('session_fpfvg', {})
            if fpfvg.get('fpfvg_present') and 'fpfvg_formation' in fpfvg:
                interactions = fpfvg['fpfvg_formation'].get('interactions', [])
                for interaction in interactions:
                    events.append({
                        'timestamp': interaction.get('interaction_time', '00:00:00'),
                        'price': interaction.get('price_level', 0),
                        'event_type': f"fpfvg_{interaction.get('interaction_type', 'unknown')}",
                        'significance': 0.8,  # FPFVG events are high significance
                        'archaeological_zone': 'fpfvg_zone'
                    })
            
            enhanced_features = session_data.get('energy_state', {})
            
            nodes = []
            edges = []
            hot_zones = []
            
            # Create nodes for each timeframe
            for tf_idx, timeframe in enumerate(timeframes):
                # Identify events relevant to this timeframe
                tf_events = self._filter_events_by_timeframe(events, timeframe)
                
                for event in tf_events:
                    node = {
                        'id': f"{timeframe}_{event.get('timestamp', tf_idx)}",
                        'timeframe': timeframe,
                        'timeframe_level': tf_idx,
                        'event_type': event.get('event_type', 'unknown'),
                        'price': event.get('price', 0),
                        'timestamp': event.get('timestamp'),
                        'significance': event.get('significance', 0.5),
                        'archaeological_zone': event.get('archaeological_zone')
                    }
                    nodes.append(node)
                    
                    # Check for hot zones (high significance events)
                    if event.get('significance', 0) > 0.7:
                        hot_zone = {
                            'timeframe': timeframe,
                            'price_level': event.get('price', 0),
                            'zone_type': event.get('archaeological_zone', 'unknown'),
                            'intensity': event.get('significance', 0),
                            'event_count': 1
                        }
                        hot_zones.append(hot_zone)
            
            # Create edges between timeframes (structural relationships)
            for i in range(len(timeframes) - 1):
                higher_tf = timeframes[i]
                lower_tf = timeframes[i + 1]
                
                # Find nodes in these timeframes
                higher_nodes = [n for n in nodes if n['timeframe'] == higher_tf]
                lower_nodes = [n for n in nodes if n['timeframe'] == lower_tf]
                
                # Create edges for related events
                for h_node in higher_nodes:
                    for l_node in lower_nodes:
                        # Connect if events are temporally and spatially related
                        price_diff = abs(h_node['price'] - l_node['price'])
                        if price_diff < 50:  # Within 50 points
                            edge = {
                                'from_node': h_node['id'],
                                'to_node': l_node['id'],
                                'from_timeframe': higher_tf,
                                'to_timeframe': lower_tf,
                                'relationship_type': 'temporal_cascade',
                                'strength': 1.0 - (price_diff / 100),  # Inverse distance
                                'price_proximity': price_diff
                            }
                            edges.append(edge)
            
            # Aggregate hot zones by price level
            aggregated_zones = self._aggregate_hot_zones(hot_zones)
            
            results = {
                'session_name': session_name,
                'timeframes_analyzed': timeframes,
                'nodes': nodes,
                'edges': edges,
                'hot_zones': aggregated_zones,
                'metrics': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'hot_zones_count': len(aggregated_zones),
                    'events_processed': len(events)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lattice mapping failed for {session_name}: {e}")
            return {'error': str(e), 'session_name': session_name}
    
    def _filter_events_by_timeframe(self, events: List[Dict], timeframe: str) -> List[Dict]:
        """Filter events relevant to specific timeframe"""
        # For now, return all events with timeframe weight
        filtered = []
        for event in events:
            # Add timeframe relevance scoring
            event_copy = event.copy()
            event_copy['timeframe_relevance'] = self._calculate_timeframe_relevance(event, timeframe)
            if event_copy['timeframe_relevance'] > 0.3:
                filtered.append(event_copy)
        return filtered
    
    def _calculate_timeframe_relevance(self, event: Dict, timeframe: str) -> float:
        """Calculate how relevant an event is to a specific timeframe"""
        # Basic relevance scoring based on event type and significance
        base_relevance = event.get('significance', 0.5)
        
        # Adjust based on timeframe
        tf_multipliers = {
            'Monthly': 0.3, 'Weekly': 0.4, 'Daily': 0.6, '4H': 0.7,
            '1H': 0.8, '15m': 0.9, '5m': 0.95, '1m': 1.0
        }
        
        multiplier = tf_multipliers.get(timeframe, 0.5)
        return min(base_relevance * multiplier, 1.0)
    
    def _aggregate_hot_zones(self, hot_zones: List[Dict]) -> List[Dict]:
        """Aggregate hot zones by exact price level and timeframe - NO ARTIFICIAL ROUNDING"""
        aggregated = {}
        
        for zone in hot_zones:
            # CRITICAL FIX: Preserve exact price levels - no artificial clustering
            price_level = zone['price_level']  # Keep original precision
            key = f"{zone['timeframe']}_{price_level}"
            
            if key in aggregated:
                aggregated[key]['intensity'] = max(aggregated[key]['intensity'], zone['intensity'])
                aggregated[key]['event_count'] += zone['event_count']
            else:
                aggregated[key] = zone.copy()
                aggregated[key]['price_level'] = price_level
        
        return list(aggregated.values())