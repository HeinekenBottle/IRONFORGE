#!/usr/bin/env python3
"""
Enhanced FPFVG Analysis for IRONFORGE
Uses actual feature variations and price action to detect First Presented Fair Value Gap patterns
"""
import pandas as pd
import numpy as np
from cross_session_analyzer import CrossSessionAnalyzer
from typing import Dict, List, Any, Tuple

class EnhancedFPFVGAnalyzer:
    """Enhanced analyzer for FPFVG patterns using meaningful features"""
    
    def __init__(self):
        self.base_analyzer = CrossSessionAnalyzer()
        self.feature_importance = self._calculate_feature_importance()
        
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate which features are most meaningful across all sessions"""
        print("🔍 Calculating feature importance across all sessions...")
        
        feature_importance = {}
        
        # Analyze feature variance across all sessions
        for i in range(45):  # f0 to f44
            feature_name = f'f{i}'
            total_variance = 0
            session_count = 0
            
            for session_id, nodes in self.base_analyzer.engine.sessions.items():
                if feature_name in nodes.columns:
                    variance = nodes[feature_name].var()
                    if not np.isnan(variance):
                        total_variance += variance
                        session_count += 1
            
            if session_count > 0:
                feature_importance[feature_name] = total_variance / session_count
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"✅ Top 10 most important features:")
        for feat, importance in sorted_features[:10]:
            print(f"  {feat}: {importance:.6f}")
            
        return dict(sorted_features)
    
    def analyze_fpfvg_liquidity_dependency(self, target_session_type: str = "LONDON") -> Dict[str, Any]:
        """
        Analyze FPFVG redelivery patterns in specific session types 
        and their dependency on previous session liquidity
        """
        print(f"\n🎯 Analyzing FPFVG patterns in {target_session_type} sessions...")
        
        results = {
            "target_session_type": target_session_type,
            "sessions_analyzed": 0,
            "fpfvg_patterns": [],
            "liquidity_dependencies": [],
            "statistical_summary": {}
        }
        
        # Get sessions of target type
        target_sessions = [s for s in self.base_analyzer.session_sequence 
                          if s['type'] == target_session_type]
        
        results["sessions_analyzed"] = len(target_sessions)
        
        for i, session_info in enumerate(target_sessions):
            session_id = session_info['session_id']
            
            if session_id not in self.base_analyzer.engine.sessions:
                continue
                
            # Analyze FPFVG patterns in current session
            current_nodes = self.base_analyzer.engine.sessions[session_id]
            fpfvg_analysis = self._enhanced_fpfvg_detection(current_nodes, session_id)
            
            if fpfvg_analysis['significant_patterns']:
                results["fpfvg_patterns"].append({
                    "session": session_id,
                    "date": session_info['date_str'],
                    "analysis": fpfvg_analysis
                })
                
                # Find previous session and analyze liquidity dependency
                previous_session = self._find_previous_session(session_info)
                if previous_session:
                    dependency = self._analyze_liquidity_dependency(
                        previous_session, session_id, fpfvg_analysis
                    )
                    if dependency:
                        results["liquidity_dependencies"].append(dependency)
        
        # Calculate statistical summary
        results["statistical_summary"] = self._calculate_statistical_summary(results)
        
        return results
    
    def _enhanced_fpfvg_detection(self, nodes: pd.DataFrame, session_id: str) -> Dict[str, Any]:
        """Enhanced FPFVG detection using meaningful features and price action"""
        analysis = {
            "session_id": session_id,
            "significant_patterns": False,
            "fpfvg_events": [],
            "redelivery_events": [],
            "feature_signals": {},
            "price_characteristics": {}
        }
        
        if len(nodes) < 10:  # Need minimum events for pattern detection
            return analysis
        
        # Use top important features for pattern detection
        top_features = list(self.feature_importance.keys())[:5]  # Top 5 features
        
        # Detect significant feature changes (potential FPFVG signals)
        for feature in top_features:
            if feature not in nodes.columns:
                continue
                
            feature_values = nodes[feature]
            if feature_values.var() > 0:
                # Look for sudden changes in feature values
                feature_diff = feature_values.diff().abs()
                threshold = feature_diff.quantile(0.85)  # Top 15% of changes
                
                significant_changes = nodes[feature_diff > threshold]
                if len(significant_changes) > 0:
                    analysis["feature_signals"][feature] = {
                        "significant_changes": len(significant_changes),
                        "max_change": feature_diff.max(),
                        "avg_change": feature_diff.mean()
                    }
        
        # Price-based FPFVG detection
        price_analysis = self._detect_price_gaps(nodes)
        analysis["price_characteristics"] = price_analysis
        
        # Combine feature and price signals to identify FPFVG
        if (len(analysis["feature_signals"]) >= 2 and 
            price_analysis["significant_gaps"] > 0):
            analysis["significant_patterns"] = True
            
            # Look for redelivery patterns
            redelivery_analysis = self._detect_redelivery_patterns(nodes, price_analysis)
            analysis["redelivery_events"] = redelivery_analysis
        
        return analysis
    
    def _detect_price_gaps(self, nodes: pd.DataFrame) -> Dict[str, Any]:
        """Detect significant price gaps that could be FVGs"""
        price_diffs = nodes['price'].diff().abs()
        gap_threshold = price_diffs.quantile(0.9)  # Top 10% of price moves
        
        significant_gaps = nodes[price_diffs > gap_threshold]
        
        return {
            "total_gaps": len(significant_gaps),
            "significant_gaps": len(significant_gaps),
            "avg_gap_size": price_diffs.mean(),
            "max_gap_size": price_diffs.max(),
            "gap_threshold": gap_threshold,
            "gap_events": significant_gaps[['t', 'price']].to_dict('records') if len(significant_gaps) > 0 else []
        }
    
    def _detect_redelivery_patterns(self, nodes: pd.DataFrame, price_analysis: Dict) -> List[Dict]:
        """Detect redelivery patterns for identified gaps"""
        redelivery_events = []
        
        for gap_event in price_analysis["gap_events"]:
            gap_time = gap_event['t']
            gap_price = gap_event['price']
            
            # Look for price returning to gap area (within tolerance)
            tolerance = price_analysis["gap_threshold"] * 0.3
            future_events = nodes[nodes['t'] > gap_time]
            
            if len(future_events) > 0:
                price_returns = future_events[
                    abs(future_events['price'] - gap_price) <= tolerance
                ]
                
                if len(price_returns) > 0:
                    first_return = price_returns.iloc[0]
                    redelivery_events.append({
                        "gap_time": gap_time,
                        "gap_price": gap_price,
                        "redelivery_time": first_return['t'],
                        "redelivery_price": first_return['price'],
                        "time_to_redelivery_ms": first_return['t'] - gap_time,
                        "price_accuracy": abs(first_return['price'] - gap_price)
                    })
        
        return redelivery_events
    
    def _find_previous_session(self, current_session_info: Dict) -> str:
        """Find the most relevant previous session for dependency analysis"""
        current_index = self.base_analyzer.session_sequence.index(current_session_info)
        
        # Look for previous session on same day or previous day
        for i in range(current_index - 1, -1, -1):
            prev_session = self.base_analyzer.session_sequence[i]
            
            # Priority: same day sessions, then previous day major sessions
            if (prev_session['date'] == current_session_info['date'] or
                (prev_session['date'] < current_session_info['date'] and 
                 prev_session['type'] in ['NY', 'NYPM', 'ASIA'])):
                return prev_session['session_id']
        
        return None
    
    def _analyze_liquidity_dependency(self, prev_session_id: str, current_session_id: str,
                                    current_fpfvg: Dict) -> Dict[str, Any]:
        """Analyze how previous session liquidity influences current FPFVG patterns"""
        if prev_session_id not in self.base_analyzer.engine.sessions:
            return None
            
        prev_nodes = self.base_analyzer.engine.sessions[prev_session_id]
        
        # Enhanced liquidity analysis using meaningful features
        prev_liquidity = self._calculate_enhanced_liquidity_metrics(prev_nodes)
        
        # Correlation analysis
        correlation_score = self._calculate_correlation_score(
            prev_liquidity, current_fpfvg
        )
        
        if correlation_score > 0.4:  # Significant correlation threshold
            return {
                "previous_session": prev_session_id,
                "current_session": current_session_id,
                "correlation_score": correlation_score,
                "prev_liquidity_metrics": prev_liquidity,
                "current_fpfvg_strength": len(current_fpfvg.get("redelivery_events", [])),
                "dependency_type": self._classify_dependency_type(prev_liquidity, current_fpfvg)
            }
        
        return None
    
    def _calculate_enhanced_liquidity_metrics(self, nodes: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhanced liquidity metrics using feature analysis"""
        metrics = {"session_quality": 0.0, "feature_intensity": 0.0, "liquidity_imbalance": 0.0}
        
        if len(nodes) < 5:
            return metrics
        
        # Use top features for liquidity calculation
        top_features = list(self.feature_importance.keys())[:3]
        
        feature_intensity = 0.0
        for feature in top_features:
            if feature in nodes.columns and nodes[feature].var() > 0:
                # Normalized feature intensity
                intensity = nodes[feature].std() / (abs(nodes[feature].mean()) + 1e-6)
                feature_intensity += min(intensity, 2.0)  # Cap extreme values
        
        # Price-based metrics
        price_range = nodes['price'].max() - nodes['price'].min()
        price_volatility = nodes['price'].std()
        event_density = len(nodes) / max((nodes['t'].max() - nodes['t'].min()) / (60*1000), 1)
        
        metrics.update({
            "session_quality": min(event_density / 10.0, 1.0),
            "feature_intensity": feature_intensity / len(top_features),
            "liquidity_imbalance": min(price_volatility / (price_range + 1e-6), 1.0),
            "raw_metrics": {
                "range": price_range,
                "volatility": price_volatility,
                "event_density": event_density,
                "events": len(nodes)
            }
        })
        
        return metrics
    
    def _calculate_correlation_score(self, prev_liquidity: Dict, current_fpfvg: Dict) -> float:
        """Calculate correlation between previous liquidity and current FPFVG patterns"""
        score = 0.0
        
        # High previous liquidity imbalance + current FPFVG redelivery
        if prev_liquidity.get("liquidity_imbalance", 0) > 0.6:
            if len(current_fpfvg.get("redelivery_events", [])) > 0:
                score += 0.4
        
        # High feature intensity + significant FPFVG patterns
        if prev_liquidity.get("feature_intensity", 0) > 0.5:
            if current_fpfvg.get("significant_patterns", False):
                score += 0.3
        
        # Session quality correlation
        if prev_liquidity.get("session_quality", 0) > 0.7:
            if len(current_fpfvg.get("redelivery_events", [])) > 1:  # Multiple redeliveries
                score += 0.3
        
        return min(score, 1.0)
    
    def _classify_dependency_type(self, prev_liquidity: Dict, current_fpfvg: Dict) -> str:
        """Classify the type of cross-session dependency"""
        if (prev_liquidity.get("liquidity_imbalance", 0) > 0.7 and 
            len(current_fpfvg.get("redelivery_events", [])) > 0):
            return "imbalance_redelivery"
        elif (prev_liquidity.get("feature_intensity", 0) > 0.6 and
              current_fpfvg.get("significant_patterns", False)):
            return "intensity_fpfvg"
        elif prev_liquidity.get("session_quality", 0) > 0.8:
            return "quality_cascade"
        else:
            return "general_correlation"
    
    def _calculate_statistical_summary(self, results: Dict) -> Dict[str, Any]:
        """Calculate statistical summary of the analysis"""
        total_sessions = results["sessions_analyzed"]
        fpfvg_sessions = len(results["fpfvg_patterns"])
        dependency_sessions = len(results["liquidity_dependencies"])
        
        summary = {
            "fpfvg_detection_rate": fpfvg_sessions / max(total_sessions, 1),
            "dependency_rate": dependency_sessions / max(fpfvg_sessions, 1),
            "total_redeliveries": 0,
            "avg_correlation_score": 0.0
        }
        
        # Count total redeliveries
        for pattern in results["fpfvg_patterns"]:
            summary["total_redeliveries"] += len(pattern["analysis"].get("redelivery_events", []))
        
        # Average correlation score
        if results["liquidity_dependencies"]:
            scores = [dep["correlation_score"] for dep in results["liquidity_dependencies"]]
            summary["avg_correlation_score"] = np.mean(scores)
        
        return summary

def analyze_fpfvg_question():
    """Answer the specific FPFVG redelivery question"""
    print("🔍 IRONFORGE Enhanced FPFVG Analysis")
    print("Investigating: Does FPFVG redelivery depend on previous session liquidity?")
    print("=" * 70)
    
    analyzer = EnhancedFPFVGAnalyzer()
    
    # Analyze different session types
    session_types = ["LONDON", "NY", "ASIA", "PREMARKET"]
    
    for session_type in session_types:
        print(f"\n📊 Analyzing {session_type} sessions...")
        result = analyzer.analyze_fpfvg_liquidity_dependency(session_type)
        
        print(f"Sessions analyzed: {result['sessions_analyzed']}")
        print(f"FPFVG detection rate: {result['statistical_summary']['fpfvg_detection_rate']:.1%}")
        print(f"Dependency rate: {result['statistical_summary']['dependency_rate']:.1%}")
        print(f"Avg correlation score: {result['statistical_summary']['avg_correlation_score']:.3f}")
        
        if result["liquidity_dependencies"]:
            print(f"Found {len(result['liquidity_dependencies'])} significant dependencies:")
            for dep in result["liquidity_dependencies"][:2]:  # Show top 2
                print(f"  {dep['previous_session']} → {dep['current_session']}")
                print(f"    Correlation: {dep['correlation_score']:.3f}, Type: {dep['dependency_type']}")

if __name__ == "__main__":
    analyze_fpfvg_question()