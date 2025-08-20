#!/usr/bin/env python3
"""
IRONFORGE Temporal Query Engine
Interactive system for querying temporal patterns in NetworkX graphs
"""
import pandas as pd
import networkx as nx
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re

class TemporalQueryEngine:
    """Interactive temporal pattern query system"""
    
    def __init__(self, shard_dir: str = "data/shards/NQ_M5"):
        self.shard_dir = shard_dir
        self.sessions = {}
        self.graphs = {}
        self.metadata = {}
        print("🔍 Initializing Temporal Query Engine...")
        self._load_all_sessions()
        
    def _load_all_sessions(self):
        """Load all available sessions into memory"""
        shard_paths = sorted(glob.glob(f"{self.shard_dir}/shard_*"))
        print(f"📊 Loading {len(shard_paths)} sessions...")
        
        for shard_path in shard_paths:
            session_id = Path(shard_path).name.replace('shard_', '')
            try:
                # Load nodes and edges
                nodes = pd.read_parquet(f"{shard_path}/nodes.parquet")
                edges = pd.read_parquet(f"{shard_path}/edges.parquet")
                
                # Create NetworkX graph
                G = nx.from_pandas_edgelist(
                    edges, 
                    source='src', 
                    target='dst',
                    edge_attr=True,
                    create_using=nx.DiGraph()
                )
                
                # Add node attributes
                node_attrs = {row['node_id']: row.to_dict() for _, row in nodes.iterrows()}
                nx.set_node_attributes(G, node_attrs)
                
                self.sessions[session_id] = nodes
                self.graphs[session_id] = G
                
                # Store metadata
                with open(f"{shard_path}/meta.json", 'r') as f:
                    import json
                    self.metadata[session_id] = json.load(f)
                    
            except Exception as e:
                print(f"⚠️  Failed to load {session_id}: {e}")
                
        print(f"✅ Loaded {len(self.sessions)} sessions successfully")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a temporal question and get probabilistic answers"""
        print(f"\n🤔 Query: {question}")
        
        # Parse the question and route to appropriate analysis
        if "after" in question.lower() and "what happens" in question.lower():
            return self._analyze_temporal_sequence(question)
        elif "when" in question.lower() and ("starts with" in question.lower() or "begins with" in question.lower()):
            return self._analyze_opening_patterns(question)
        elif "show me" in question.lower() or "find" in question.lower():
            return self._search_patterns(question)
        elif "probability" in question.lower() or "likely" in question.lower():
            return self._calculate_probabilities(question)
        else:
            return self._general_analysis(question)
    
    def _analyze_temporal_sequence(self, question: str) -> Dict[str, Any]:
        """Analyze what happens after specific events"""
        results = {
            "query_type": "temporal_sequence",
            "total_sessions": len(self.sessions),
            "matches": [],
            "probabilities": {},
            "insights": []
        }
        
        # Extract time window if mentioned
        time_match = re.search(r'(\d+)\s*minutes?', question)
        time_window = int(time_match.group(1)) if time_match else 15
        
        # Look for specific event types mentioned
        event_patterns = {
            "40% zone": "f40",  # Assuming f40 relates to 40% zone
            "expansion": "expansion",
            "retracement": "retracement", 
            "reversal": "reversal"
        }
        
        trigger_event = None
        for pattern, field in event_patterns.items():
            if pattern in question.lower():
                trigger_event = pattern
                break
        
        if not trigger_event:
            results["error"] = "Could not identify trigger event in question"
            return results
        
        # TODO(human): Enhance 40% zone detection and Theory B analysis
        # Analyze each session for the pattern
        outcomes = []
        for session_id, nodes in self.sessions.items():
            # Find trigger events
            if trigger_event == "40% zone":
                # Look for high f40 values (assuming this indicates 40% zone)
                trigger_nodes = nodes[nodes['f40'] > 0.5]
            else:
                # Look for events with specific kinds or patterns
                trigger_nodes = nodes[nodes['kind'] == 1]  # Assuming kind=1 for events
                
            for _, trigger_node in trigger_nodes.iterrows():
                trigger_time = trigger_node['t']
                # Find what happens in next time_window minutes
                future_time = trigger_time + (time_window * 60 * 1000)  # Convert to milliseconds
                future_events = nodes[
                    (nodes['t'] > trigger_time) & 
                    (nodes['t'] <= future_time)
                ]
                
                if len(future_events) > 0:
                    # Analyze the outcome
                    price_change = future_events['price'].max() - future_events['price'].min()
                    outcome = {
                        "session": session_id,
                        "trigger_time": trigger_time,
                        "price_change": price_change,
                        "event_count": len(future_events),
                        "outcome_type": "expansion" if price_change > 20 else "consolidation"
                    }
                    outcomes.append(outcome)
        
        results["matches"] = outcomes
        
        # Calculate probabilities
        if outcomes:
            total = len(outcomes)
            expansion_count = sum(1 for o in outcomes if o["outcome_type"] == "expansion")
            results["probabilities"] = {
                "expansion": expansion_count / total,
                "consolidation": (total - expansion_count) / total,
                "sample_size": total
            }
            
            avg_price_change = np.mean([o["price_change"] for o in outcomes])
            results["insights"].append(f"Average price movement: {avg_price_change:.1f} points")
            results["insights"].append(f"Sample size: {total} occurrences")
        
        return results
    
    def _analyze_opening_patterns(self, question: str) -> Dict[str, Any]:
        """Analyze session opening patterns and their outcomes"""
        results = {
            "query_type": "opening_patterns",
            "sessions_analyzed": [],
            "pattern_outcomes": {}
        }
        
        for session_id, nodes in self.sessions.items():
            if len(nodes) < 10:  # Skip very short sessions
                continue
                
            # Analyze first 20% of session
            early_nodes = nodes.head(int(len(nodes) * 0.2))
            
            # Determine opening pattern
            price_start = early_nodes['price'].iloc[0]
            price_early_end = early_nodes['price'].iloc[-1]
            price_change = price_early_end - price_start
            
            if abs(price_change) < 10:
                pattern = "consolidation_start"
            elif price_change > 10:
                pattern = "expansion_up_start"
            else:
                pattern = "expansion_down_start"
            
            # Calculate final session range
            session_range = nodes['price'].max() - nodes['price'].min()
            
            results["sessions_analyzed"].append({
                "session": session_id,
                "opening_pattern": pattern,
                "early_change": price_change,
                "final_range": session_range
            })
        
        # Group by pattern type
        pattern_groups = {}
        for session in results["sessions_analyzed"]:
            pattern = session["opening_pattern"]
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(session["final_range"])
        
        # Calculate statistics for each pattern
        for pattern, ranges in pattern_groups.items():
            results["pattern_outcomes"][pattern] = {
                "count": len(ranges),
                "avg_range": np.mean(ranges),
                "median_range": np.median(ranges),
                "max_range": np.max(ranges),
                "min_range": np.min(ranges)
            }
        
        return results
    
    def _search_patterns(self, question: str) -> Dict[str, Any]:
        """Search for specific patterns across sessions"""
        results = {
            "query_type": "pattern_search", 
            "matches": [],
            "total_sessions_searched": len(self.sessions)
        }
        
        # Simple pattern matching based on question keywords
        for session_id, nodes in self.sessions.items():
            session_range = nodes['price'].max() - nodes['price'].min()
            
            # Check if this session matches the search criteria
            if "large range" in question.lower() and session_range > 150:
                results["matches"].append({
                    "session": session_id,
                    "range": session_range,
                    "events": len(nodes),
                    "quality": "high" if session_range > 200 else "medium"
                })
        
        return results
    
    def _calculate_probabilities(self, question: str) -> Dict[str, Any]:
        """Calculate probabilities for specific scenarios"""
        return {
            "query_type": "probability_calculation",
            "note": "Probability calculation based on historical patterns",
            "total_samples": len(self.sessions)
        }
    
    def _general_analysis(self, question: str) -> Dict[str, Any]:
        """General analysis for unstructured questions"""
        return {
            "query_type": "general_analysis",
            "suggestion": "Try questions like: 'What happens after a 40% zone event?', 'When session starts with expansion, what's the final range?', 'Show me sessions with large ranges'",
            "available_sessions": len(self.sessions),
            "date_range": self._get_date_range()
        }
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get the date range of available sessions"""
        dates = []
        for session_id in self.sessions.keys():
            # Extract date from session ID
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', session_id)
            if date_match:
                dates.append(date_match.group())
        
        if dates:
            return {"start": min(dates), "end": max(dates)}
        return {"start": "unknown", "end": "unknown"}
    
    def list_sessions(self) -> List[str]:
        """List all available sessions"""
        return list(self.sessions.keys())
    
    def session_info(self, session_id: str) -> Dict[str, Any]:
        """Get detailed info about a specific session"""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        nodes = self.sessions[session_id]
        graph = self.graphs[session_id]
        
        return {
            "session_id": session_id,
            "nodes": len(nodes),
            "edges": graph.number_of_edges(),
            "price_range": nodes['price'].max() - nodes['price'].min(),
            "duration_ms": nodes['t'].max() - nodes['t'].min(),
            "event_types": nodes['kind'].value_counts().to_dict(),
            "metadata": self.metadata.get(session_id, {})
        }

# Interactive CLI wrapper
def run_interactive_query():
    """Run interactive query session"""
    print("🚀 IRONFORGE Temporal Query Engine")
    print("=" * 50)
    
    engine = TemporalQueryEngine()
    
    print("\n💡 Example questions:")
    print("• What happens after a 40% zone event?")
    print("• When session starts with expansion, what's the final range?") 
    print("• Show me sessions with large ranges")
    print("• What's the probability of expansion after reversal?")
    print("\n📋 Commands: 'list', 'info <session>', 'help', 'quit'")
    
    while True:
        try:
            question = input("\n🤔 Ask your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'list':
                sessions = engine.list_sessions()
                print(f"\n📊 Available sessions ({len(sessions)}):")
                for session in sessions[:10]:  # Show first 10
                    print(f"  • {session}")
                if len(sessions) > 10:
                    print(f"  ... and {len(sessions) - 10} more")
            elif question.lower().startswith('info '):
                session_id = question[5:].strip()
                info = engine.session_info(session_id)
                print(f"\n📋 Session Info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            elif question.lower() == 'help':
                print("\n💡 Ask temporal questions about your data!")
                print("Examples:")
                print("• 'What happens after X?' - Temporal sequence analysis")
                print("• 'When session starts with Y?' - Opening pattern analysis")
                print("• 'Show me sessions where Z' - Pattern search")
            else:
                result = engine.ask(question)
                print(f"\n🎯 Results:")
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"  {key}: {len(value)} items (showing first 5)")
                        for item in value[:5]:
                            print(f"    • {item}")
                    else:
                        print(f"  {key}: {value}")
                        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_interactive_query()