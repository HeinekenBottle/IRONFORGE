#!/usr/bin/env python3
"""
Cross-Session Liquidity Analysis for IRONFORGE
Analyzes relationships between liquidity consumption and subsequent session patterns
"""
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from temporal_query_engine import TemporalQueryEngine
import re

class CrossSessionAnalyzer:
    """Analyze cross-session liquidity dependencies and FPFVG patterns"""
    
    def __init__(self, engine: TemporalQueryEngine = None):
        if engine is None:
            print("🔍 Initializing Cross-Session Analyzer...")
            self.engine = TemporalQueryEngine()
        else:
            self.engine = engine
        
        self.session_sequence = self._build_session_sequence()
        print(f"✅ Loaded {len(self.session_sequence)} sessions for cross-session analysis")
        
    def _build_session_sequence(self) -> List[Dict[str, Any]]:
        """Build chronological sequence of sessions with metadata"""
        sessions = []
        
        for session_id in self.engine.sessions.keys():
            # Parse session type and date
            parts = session_id.split('_')
            if len(parts) >= 2:
                session_type = parts[0]
                date_str = parts[1]
                
                # Convert to datetime for sorting
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    sessions.append({
                        'session_id': session_id,
                        'type': session_type,
                        'date': date_obj,
                        'date_str': date_str
                    })
                except ValueError:
                    continue
        
        # Sort by date and session type priority
        session_priority = {
            'MIDNIGHT': 0, 'PREASIA': 1, 'ASIA': 2, 'PREMARKET': 3, 
            'LONDON': 4, 'LUNCH': 5, 'NYAM': 6, 'NY': 7, 'NYPM': 8
        }
        
        sessions.sort(key=lambda x: (x['date'], session_priority.get(x['type'], 99)))
        return sessions
    
    def analyze_fpfvg_redelivery_patterns(self) -> Dict[str, Any]:
        """Analyze FPFVG redelivery patterns and their cross-session dependencies"""
        print("\n🔍 Analyzing FPFVG (First Presented Fair Value Gap) redelivery patterns...")
        
        results = {
            "total_sessions": len(self.session_sequence),
            "fpfvg_sessions": [],
            "redelivery_patterns": {},
            "cross_session_dependencies": [],
            "liquidity_influences": {}
        }
        
        # Analyze each session for FPFVG patterns
        for i, session_info in enumerate(self.session_sequence):
            session_id = session_info['session_id']
            session_type = session_info['type']
            
            if session_id not in self.engine.sessions:
                continue
                
            nodes = self.engine.sessions[session_id]
            fpfvg_analysis = self._detect_fpfvg_patterns(nodes, session_id)
            
            if fpfvg_analysis['has_fpfvg']:
                results["fpfvg_sessions"].append({
                    "session_id": session_id,
                    "type": session_type,
                    "date": session_info['date_str'],
                    "fpfvg_data": fpfvg_analysis
                })
                
                # Look for cross-session influences (previous session)
                if i > 0:
                    prev_session = self.session_sequence[i-1]
                    dependency = self._analyze_cross_session_dependency(
                        prev_session['session_id'], 
                        session_id,
                        fpfvg_analysis
                    )
                    if dependency:
                        results["cross_session_dependencies"].append(dependency)
        
        # Analyze patterns by session type
        results["redelivery_patterns"] = self._analyze_redelivery_by_type(results["fpfvg_sessions"])
        results["liquidity_influences"] = self._analyze_liquidity_influences(results["cross_session_dependencies"])
        
        return results
    
    def _detect_fpfvg_patterns(self, nodes: pd.DataFrame, session_id: str) -> Dict[str, Any]:
        """Detect First Presented Fair Value Gap patterns in a session"""
        analysis = {
            "has_fpfvg": False,
            "fpfvg_events": [],
            "redelivery_events": [],
            "gap_characteristics": {},
            "liquidity_metrics": {}
        }
        
        # Look for gap patterns in price action
        if len(nodes) < 5:
            return analysis
            
        # Calculate price gaps and volume patterns
        price_diffs = nodes['price'].diff().abs()
        volume_proxy = nodes['kind'].value_counts().get(1, 0)  # Event density as volume proxy
        
        # Detect significant gaps (potential FPFVG)
        gap_threshold = price_diffs.quantile(0.9)  # Top 10% of price moves
        significant_gaps = nodes[price_diffs > gap_threshold]
        
        if len(significant_gaps) > 0:
            analysis["has_fpfvg"] = True
            
            for _, gap_event in significant_gaps.iterrows():
                gap_time = gap_event['t']
                gap_price = gap_event['price']
                
                # Look for redelivery (price returning to gap area)
                future_events = nodes[nodes['t'] > gap_time]
                if len(future_events) > 0:
                    gap_tolerance = gap_threshold * 0.5
                    redelivery_events = future_events[
                        abs(future_events['price'] - gap_price) <= gap_tolerance
                    ]
                    
                    if len(redelivery_events) > 0:
                        analysis["redelivery_events"].append({
                            "gap_time": gap_time,
                            "gap_price": gap_price,
                            "redelivery_time": redelivery_events.iloc[0]['t'],
                            "redelivery_price": redelivery_events.iloc[0]['price'],
                            "time_to_redelivery": redelivery_events.iloc[0]['t'] - gap_time
                        })
                
                analysis["fpfvg_events"].append({
                    "time": gap_time,
                    "price": gap_price,
                    "magnitude": price_diffs.loc[gap_event.name],
                    "context_features": [gap_event[f'f{i}'] for i in range(10)]  # First 10 features
                })
        
        # Calculate liquidity metrics
        analysis["liquidity_metrics"] = {
            "event_density": len(nodes) / ((nodes['t'].max() - nodes['t'].min()) / (60*1000)) if len(nodes) > 1 else 0,
            "price_volatility": price_diffs.std(),
            "range": nodes['price'].max() - nodes['price'].min(),
            "mean_price": nodes['price'].mean()
        }
        
        return analysis
    
    def _analyze_cross_session_dependency(self, prev_session_id: str, current_session_id: str, 
                                        current_fpfvg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze dependency between previous session liquidity and current FPFVG patterns"""
        if prev_session_id not in self.engine.sessions:
            return None
            
        prev_nodes = self.engine.sessions[prev_session_id]
        current_nodes = self.engine.sessions[current_session_id]
        
        # Analyze previous session liquidity consumption
        prev_liquidity = self._calculate_liquidity_metrics(prev_nodes)
        
        # Check if current session FPFVG characteristics correlate with previous liquidity
        if not current_fpfvg["has_fpfvg"]:
            return None
            
        dependency = {
            "previous_session": prev_session_id,
            "current_session": current_session_id,
            "prev_liquidity": prev_liquidity,
            "current_fpfvg_count": len(current_fpfvg["fpfvg_events"]),
            "current_redelivery_count": len(current_fpfvg["redelivery_events"]),
            "correlation_strength": 0.0
        }
        
        # Calculate correlation metrics
        if prev_liquidity["imbalance"] > 0.6 and len(current_fpfvg["redelivery_events"]) > 0:
            dependency["correlation_strength"] = 0.8
            dependency["pattern_type"] = "high_imbalance_redelivery"
        elif prev_liquidity["consumption_rate"] > 0.7 and current_fpfvg["has_fpfvg"]:
            dependency["correlation_strength"] = 0.6
            dependency["pattern_type"] = "high_consumption_fpfvg"
        
        return dependency if dependency["correlation_strength"] > 0.3 else None
    
    def _calculate_liquidity_metrics(self, nodes: pd.DataFrame) -> Dict[str, float]:
        """Calculate liquidity consumption metrics for a session"""
        if len(nodes) < 3:
            return {"imbalance": 0.0, "consumption_rate": 0.0, "efficiency": 0.0}
            
        # Use price volatility and event distribution as liquidity proxies
        price_range = nodes['price'].max() - nodes['price'].min()
        price_std = nodes['price'].std()
        event_count = len(nodes)
        
        # Calculate metrics
        imbalance = min(price_std / (price_range + 1e-6), 1.0)  # Normalized volatility
        consumption_rate = min(event_count / 100.0, 1.0)  # Normalized event density
        efficiency = 1.0 - (price_std / (price_range + 1e-6))  # How efficiently price moved
        
        return {
            "imbalance": imbalance,
            "consumption_rate": consumption_rate,
            "efficiency": max(efficiency, 0.0),
            "raw_metrics": {
                "range": price_range,
                "volatility": price_std,
                "events": event_count
            }
        }
    
    def _analyze_redelivery_by_type(self, fpfvg_sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze FPFVG redelivery patterns by session type"""
        patterns = {}
        
        for session_type in ['ASIA', 'LONDON', 'NY', 'LUNCH']:
            type_sessions = [s for s in fpfvg_sessions if s['type'] == session_type]
            if not type_sessions:
                continue
                
            total_fpfvg = sum(len(s['fpfvg_data']['fpfvg_events']) for s in type_sessions)
            total_redelivery = sum(len(s['fpfvg_data']['redelivery_events']) for s in type_sessions)
            
            patterns[session_type] = {
                "session_count": len(type_sessions),
                "fpfvg_events": total_fpfvg,
                "redelivery_events": total_redelivery,
                "redelivery_rate": total_redelivery / max(total_fpfvg, 1),
                "avg_sessions_with_fpfvg": len(type_sessions)
            }
        
        return patterns
    
    def _analyze_liquidity_influences(self, dependencies: List[Dict]) -> Dict[str, Any]:
        """Analyze how previous session liquidity influences current patterns"""
        if not dependencies:
            return {"no_dependencies": True}
            
        influences = {
            "total_dependencies": len(dependencies),
            "pattern_types": {},
            "correlation_distribution": [],
            "strongest_correlations": []
        }
        
        # Group by pattern type
        for dep in dependencies:
            pattern_type = dep.get("pattern_type", "unknown")
            if pattern_type not in influences["pattern_types"]:
                influences["pattern_types"][pattern_type] = []
            influences["pattern_types"][pattern_type].append(dep)
            
            influences["correlation_distribution"].append(dep["correlation_strength"])
        
        # Find strongest correlations
        sorted_deps = sorted(dependencies, key=lambda x: x["correlation_strength"], reverse=True)
        influences["strongest_correlations"] = sorted_deps[:5]
        
        # Calculate summary statistics
        if influences["correlation_distribution"]:
            influences["avg_correlation"] = np.mean(influences["correlation_distribution"])
            influences["max_correlation"] = np.max(influences["correlation_distribution"])
        
        return influences
    
    def query_cross_session_pattern(self, question: str) -> Dict[str, Any]:
        """Query cross-session patterns with natural language"""
        print(f"\n🤔 Cross-Session Query: {question}")
        
        if "am" in question.lower() and ("previous" in question.lower() or "prior" in question.lower()):
            return self._analyze_am_session_dependencies()
        elif "fpfvg" in question.lower() and "redelivery" in question.lower():
            return self.analyze_fpfvg_redelivery_patterns()
        elif "liquidity" in question.lower() and ("influence" in question.lower() or "affect" in question.lower()):
            return self._analyze_liquidity_cascade_effects()
        else:
            return {"error": "Query type not recognized", "suggestion": "Try: 'Do AM sessions depend on previous session liquidity?'"}
    
    def _analyze_am_session_dependencies(self) -> Dict[str, Any]:
        """Analyze whether AM sessions are influenced by previous session patterns"""
        print("🌅 Analyzing AM session dependencies...")
        
        am_sessions = [s for s in self.session_sequence if s['type'] in ['PREMARKET', 'LONDON']]
        dependencies = []
        
        for i, am_session in enumerate(am_sessions):
            # Find the previous session (likely evening/night session)
            prev_sessions = [s for s in self.session_sequence 
                           if s['date'] <= am_session['date'] and s['session_id'] != am_session['session_id']]
            
            if prev_sessions:
                prev_session = prev_sessions[-1]  # Most recent previous session
                
                # Analyze current AM session patterns
                am_nodes = self.engine.sessions.get(am_session['session_id'])
                if am_nodes is not None:
                    am_analysis = self._detect_fpfvg_patterns(am_nodes, am_session['session_id'])
                    
                    dependency = self._analyze_cross_session_dependency(
                        prev_session['session_id'],
                        am_session['session_id'],
                        am_analysis
                    )
                    
                    if dependency:
                        dependencies.append(dependency)
        
        return {
            "am_sessions_analyzed": len(am_sessions),
            "dependencies_found": len(dependencies),
            "dependency_rate": len(dependencies) / max(len(am_sessions), 1),
            "dependencies": dependencies,
            "summary": f"Found {len(dependencies)} cross-session dependencies in {len(am_sessions)} AM sessions"
        }
    
    def _analyze_liquidity_cascade_effects(self) -> Dict[str, Any]:
        """Analyze cascade effects of liquidity consumption across sessions"""
        print("🌊 Analyzing liquidity cascade effects...")
        
        cascades = []
        
        for i in range(1, len(self.session_sequence)):
            current = self.session_sequence[i]
            previous = self.session_sequence[i-1]
            
            if current['session_id'] in self.engine.sessions and previous['session_id'] in self.engine.sessions:
                current_nodes = self.engine.sessions[current['session_id']]
                previous_nodes = self.engine.sessions[previous['session_id']]
                
                prev_liquidity = self._calculate_liquidity_metrics(previous_nodes)
                current_liquidity = self._calculate_liquidity_metrics(current_nodes)
                
                # Look for cascade patterns
                if prev_liquidity["imbalance"] > 0.6 and current_liquidity["consumption_rate"] > 0.5:
                    cascades.append({
                        "previous_session": previous['session_id'],
                        "current_session": current['session_id'],
                        "prev_imbalance": prev_liquidity["imbalance"],
                        "current_consumption": current_liquidity["consumption_rate"],
                        "cascade_strength": prev_liquidity["imbalance"] * current_liquidity["consumption_rate"]
                    })
        
        return {
            "total_cascades": len(cascades),
            "cascades": sorted(cascades, key=lambda x: x["cascade_strength"], reverse=True),
            "avg_cascade_strength": np.mean([c["cascade_strength"] for c in cascades]) if cascades else 0
        }

def run_cross_session_analysis():
    """Interactive cross-session analysis"""
    print("🔗 IRONFORGE Cross-Session Liquidity Analyzer")
    print("=" * 55)
    
    analyzer = CrossSessionAnalyzer()
    
    print("\n💡 Example questions:")
    print("• Do AM sessions depend on previous session liquidity?")
    print("• Analyze FPFVG redelivery patterns across sessions")
    print("• How does liquidity consumption cascade between sessions?")
    
    while True:
        try:
            question = input("\n🤔 Ask about cross-session patterns: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Analysis complete!")
                break
            elif question.lower() == 'help':
                print("\n💡 Ask about relationships between sessions:")
                print("• Cross-session dependencies")
                print("• FPFVG redelivery patterns")
                print("• Liquidity cascade effects")
            else:
                result = analyzer.query_cross_session_pattern(question)
                print(f"\n🎯 Analysis Results:")
                
                # Pretty print results
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"  {key}: {len(value)} items (showing first 3)")
                        for item in value[:3]:
                            if isinstance(item, dict):
                                print(f"    • {item.get('session_id', str(item)[:50])}")
                            else:
                                print(f"    • {item}")
                    elif isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
                        
        except KeyboardInterrupt:
            print("\n👋 Analysis complete!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_cross_session_analysis()