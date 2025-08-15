"""
IRONFORGE Graph Builder
Converts session JSON to full-preservation TGAT graphs
"""
import json
import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import OneHotEncoder

class IRONFORGEGraphBuilder:
    """Transform session JSON into graphs preserving ALL information"""
    
    def __init__(self):
        self.event_encoder = OneHotEncoder(sparse=False)
        self.preserve_everything = True
        
    def build_graph(self, session_json: Dict[str, Any]) -> Dict:
        """
        Create multi-layer graph from session data
        
        Returns:
            graph with nodes, multi-type edges, and full metadata
        """
        # Extract all events - price movements, PD array interactions, FVG touches
        nodes = []
        node_id = 0
        
        # Price movement events
        if 'price_movements' in session_json:
            for pm in session_json['price_movements']:
                nodes.append({
                    'id': node_id,
                    'type': pm.get('movement_type', 'unknown'),
                    'time': self._parse_time(pm.get('timestamp', '00:00:00')),
                    'price': pm.get('price_level', 0),
                    'category': 'price_movement',
                    'raw': pm  # PRESERVE EVERYTHING
                })
                node_id += 1
        
        # PD Array interactions (liquidity, FVG)
        if 'session_fpfvg' in session_json:
            for interaction in session_json['session_fpfvg'].get('interactions', []):
                nodes.append({
                    'id': node_id,
                    'type': 'fpfvg_interaction',
                    'time': self._parse_time(interaction.get('time', '00:00:00')),
                    'price': interaction.get('price', 0),
                    'category': 'pd_array',
                    'raw': interaction
                })
                node_id += 1
        
        # Build edges
        edges = {
            'temporal': self._build_temporal_edges(nodes),
            'scale': self._build_scale_edges(nodes),
            'cascade': self._build_cascade_edges(nodes, session_json),
            'pd_array': self._build_pd_edges(nodes),
            'discovered': []  # Space for TGAT to find new patterns
        }
        
        # Full graph with metadata
        graph = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'session_id': session_json.get('timestamp', 'unknown'),
                'energy_state': session_json.get('energy_state', {}),
                'contamination': session_json.get('contamination_analysis', {}),
                'preserved_raw': session_json  # NEVER LOSE INFORMATION
            }
        }
        
        return graph
    
    def _parse_time(self, timestamp: str) -> float:
        """Convert HH:MM:SS to minutes from start"""
        try:
            parts = timestamp.split(':')
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60
        except:
            return 0.0
    
    def _build_temporal_edges(self, nodes: List) -> List:
        """Sequential time edges"""
        edges = []
        sorted_nodes = sorted(nodes, key=lambda x: x['time'])
        for i in range(len(sorted_nodes) - 1):
            edges.append({
                'source': sorted_nodes[i]['id'],
                'target': sorted_nodes[i+1]['id'],
                'time_delta': sorted_nodes[i+1]['time'] - sorted_nodes[i]['time'],
                'type': 'temporal'
            })
        return edges
    
    def _build_scale_edges(self, nodes: List) -> List:
        """Multi-scale connections (1m → 5m → 15m)"""
        edges = []
        # Group nodes by time buckets
        buckets = {1: {}, 5: {}, 15: {}}
        
        for node in nodes:
            for scale in [1, 5, 15]:
                bucket_id = int(node['time'] // scale)
                key = (scale, bucket_id)
                if key not in buckets[scale]:
                    buckets[scale][key] = []
                buckets[scale][key].append(node['id'])
        
        # Link across scales
        for scale in [1, 5]:
            next_scale = 5 if scale == 1 else 15
            for (s, bucket_id), node_ids in buckets[scale].items():
                parent_bucket = (next_scale, bucket_id // next_scale)
                if parent_bucket in buckets[next_scale]:
                    for source in node_ids:
                        for target in buckets[next_scale][parent_bucket]:
                            edges.append({
                                'source': source,
                                'target': target,
                                'scale_from': scale,
                                'scale_to': next_scale,
                                'type': 'scale'
                            })
        return edges
    
    def _build_cascade_edges(self, nodes: List, session: Dict) -> List:
        """Cascade sequence connections"""
        edges = []
        # Identify cascade patterns from your session data
        # This is where your cascade detection logic goes
        return edges
    
    def _build_pd_edges(self, nodes: List) -> List:
        """PD array relationship edges"""
        edges = []
        pd_nodes = [n for n in nodes if n['category'] == 'pd_array']
        price_nodes = [n for n in nodes if n['category'] == 'price_movement']
        
        # Link PD interactions to nearby price movements
        for pd in pd_nodes:
            for pm in price_nodes:
                time_diff = abs(pd['time'] - pm['time'])
                if time_diff < 5:  # Within 5 minutes
                    edges.append({
                        'source': pd['id'],
                        'target': pm['id'],
                        'time_delta': time_diff,
                        'type': 'pd_interaction'
                    })
        return edges
    
    def to_tgat_format(self, graph: Dict) -> tuple:
        """Convert to PyTorch Geometric Temporal format"""
        # Node features
        X = []
        for node in graph['nodes']:
            features = [
                node['time'],
                node['price'] / 100000.0,  # Normalize
                1.0 if node['category'] == 'price_movement' else 0.0,
                1.0 if node['category'] == 'pd_array' else 0.0
            ]
            X.append(features)
        X = torch.tensor(X, dtype=torch.float)
        
        # Edge indices and times
        all_edges = []
        edge_times = []
        for edge_type, edges in graph['edges'].items():
            if edges:  # Skip empty edge types
                for edge in edges:
                    all_edges.append([edge['source'], edge['target']])
                    edge_times.append(edge.get('time_delta', 0))
        
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t()
            edge_times = torch.tensor(edge_times, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_times = torch.zeros(0, dtype=torch.float)
        
        return X, edge_index, edge_times, graph['metadata']
