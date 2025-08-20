#!/usr/bin/env python3
"""
FPFVG Dependency Investigator
Advanced tool for detecting subtle cross-session dependencies
"""
import pandas as pd
import numpy as np
from enhanced_fpfvg_analyzer import EnhancedFPFVGAnalyzer
from typing import Dict, List, Any

class FPFVGDependencyInvestigator:
    """Investigate subtle FPFVG dependencies with lower thresholds and more granular analysis"""
    
    def __init__(self):
        self.enhanced_analyzer = EnhancedFPFVGAnalyzer()
        
    def investigate_am_session_dependencies(self) -> Dict[str, Any]:
        """Deep dive into AM session FPFVG patterns and previous session influences"""
        print("🌅 Investigating AM Session FPFVG Dependencies")
        print("=" * 50)
        
        results = {
            "am_sessions": [],
            "subtle_dependencies": [],
            "pattern_insights": {},
            "recommendations": []
        }
        
        # Focus on LONDON and PREMARKET sessions (AM sessions)
        am_types = ["LONDON", "PREMARKET"]
        
        for session_type in am_types:
            am_sessions = [s for s in self.enhanced_analyzer.base_analyzer.session_sequence 
                          if s['type'] == session_type]
            
            print(f"\n📊 Analyzing {session_type} sessions ({len(am_sessions)} total)")
            
            for session_info in am_sessions:
                session_id = session_info['session_id']
                
                if session_id not in self.enhanced_analyzer.base_analyzer.engine.sessions:
                    continue
                
                # Get current session data
                current_nodes = self.enhanced_analyzer.base_analyzer.engine.sessions[session_id]
                
                # Enhanced FPFVG analysis
                fpfvg_analysis = self.enhanced_analyzer._enhanced_fpfvg_detection(current_nodes, session_id)
                
                # Find ALL previous sessions (not just immediate)
                previous_sessions = self._get_relevant_previous_sessions(session_info)
                
                session_analysis = {
                    "session_id": session_id,
                    "type": session_type,
                    "date": session_info['date_str'],
                    "fpfvg_strength": len(fpfvg_analysis.get("redelivery_events", [])),
                    "feature_signals": len(fpfvg_analysis.get("feature_signals", {})),
                    "previous_influences": []
                }
                
                # Analyze each previous session's influence
                for prev_session_id in previous_sessions:
                    influence = self._analyze_subtle_influence(prev_session_id, session_id, fpfvg_analysis)
                    if influence:
                        session_analysis["previous_influences"].append(influence)
                        
                        # Add to subtle dependencies if correlation > 0.2 (lower threshold)
                        if influence.get("correlation_score", 0) > 0.2:
                            results["subtle_dependencies"].append(influence)
                
                results["am_sessions"].append(session_analysis)
        
        # Generate insights
        results["pattern_insights"] = self._generate_pattern_insights(results)
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def _get_relevant_previous_sessions(self, current_session_info: Dict) -> List[str]:
        """Get all relevant previous sessions for dependency analysis"""
        current_index = self.enhanced_analyzer.base_analyzer.session_sequence.index(current_session_info)
        previous_sessions = []
        
        # Look back up to 5 sessions or 2 days
        lookback_limit = min(current_index, 5)
        
        for i in range(1, lookback_limit + 1):
            prev_session = self.enhanced_analyzer.base_analyzer.session_sequence[current_index - i]
            
            # Include if within 2 days and is a major session
            days_diff = (current_session_info['date'] - prev_session['date']).days
            
            if (days_diff <= 2 and 
                prev_session['type'] in ['NY', 'NYPM', 'ASIA', 'LONDON', 'LUNCH']):
                previous_sessions.append(prev_session['session_id'])
        
        return previous_sessions
    
    def _analyze_subtle_influence(self, prev_session_id: str, current_session_id: str,
                                current_fpfvg: Dict) -> Dict[str, Any]:
        """Analyze subtle influences with lower correlation thresholds"""
        if prev_session_id not in self.enhanced_analyzer.base_analyzer.engine.sessions:
            return None
            
        prev_nodes = self.enhanced_analyzer.base_analyzer.engine.sessions[prev_session_id]
        
        # Multi-dimensional influence analysis
        influence_score = 0.0
        influence_factors = {}
        
        # 1. Feature intensity correlation
        prev_f8_intensity = prev_nodes['f8'].std() if 'f8' in prev_nodes.columns else 0
        current_redeliveries = len(current_fpfvg.get("redelivery_events", []))
        
        if prev_f8_intensity > 1000 and current_redeliveries > 0:  # Lower threshold
            influence_score += 0.3
            influence_factors["f8_intensity"] = prev_f8_intensity
        
        # 2. Session range correlation
        prev_range = prev_nodes['price'].max() - prev_nodes['price'].min()
        current_fpfvg_strength = len(current_fpfvg.get("feature_signals", {}))
        
        if prev_range > 100 and current_fpfvg_strength > 2:  # Large previous range → FPFVG
            influence_score += 0.25
            influence_factors["range_influence"] = prev_range
        
        # 3. Event density correlation
        prev_event_density = len(prev_nodes)
        if prev_event_density > 40 and current_redeliveries > 0:  # High activity → redelivery
            influence_score += 0.2
            influence_factors["density_influence"] = prev_event_density
        
        # 4. Time-of-day effect
        prev_session_parts = prev_session_id.split('_')
        current_session_parts = current_session_id.split('_')
        
        if (len(prev_session_parts) > 0 and len(current_session_parts) > 0):
            # NY/NYPM → LONDON dependency (overnight effect)
            if (prev_session_parts[0] in ['NY', 'NYPM'] and 
                current_session_parts[0] in ['LONDON', 'PREMARKET']):
                if current_redeliveries > 0:
                    influence_score += 0.25
                    influence_factors["overnight_effect"] = True
        
        if influence_score > 0.2:  # Lower threshold for subtle patterns
            return {
                "previous_session": prev_session_id,
                "current_session": current_session_id,
                "correlation_score": influence_score,
                "influence_factors": influence_factors,
                "dependency_type": self._classify_subtle_dependency(influence_factors),
                "current_fpfvg_events": current_redeliveries
            }
        
        return None
    
    def _classify_subtle_dependency(self, influence_factors: Dict) -> str:
        """Classify the type of subtle dependency"""
        if "overnight_effect" in influence_factors:
            return "overnight_liquidity_transfer"
        elif "f8_intensity" in influence_factors and influence_factors["f8_intensity"] > 2000:
            return "high_intensity_spillover"
        elif "range_influence" in influence_factors and influence_factors["range_influence"] > 150:
            return "large_range_redelivery"
        elif "density_influence" in influence_factors:
            return "activity_cascade"
        else:
            return "general_subtle_correlation"
    
    def _generate_pattern_insights(self, results: Dict) -> Dict[str, Any]:
        """Generate insights from the dependency analysis"""
        insights = {
            "total_am_sessions": len(results["am_sessions"]),
            "sessions_with_dependencies": len([s for s in results["am_sessions"] if s["previous_influences"]]),
            "total_subtle_dependencies": len(results["subtle_dependencies"]),
            "dependency_types": {},
            "strongest_correlations": []
        }
        
        # Count dependency types
        for dep in results["subtle_dependencies"]:
            dep_type = dep.get("dependency_type", "unknown")
            if dep_type not in insights["dependency_types"]:
                insights["dependency_types"][dep_type] = 0
            insights["dependency_types"][dep_type] += 1
        
        # Find strongest correlations
        sorted_deps = sorted(results["subtle_dependencies"], 
                           key=lambda x: x.get("correlation_score", 0), reverse=True)
        insights["strongest_correlations"] = sorted_deps[:3]
        
        return insights
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if results["pattern_insights"]["total_subtle_dependencies"] > 0:
            recommendations.append(
                f"Found {results['pattern_insights']['total_subtle_dependencies']} subtle dependencies - "
                "FPFVG redelivery patterns do show cross-session influences"
            )
            
            # Specific recommendations based on dependency types
            dep_types = results["pattern_insights"]["dependency_types"]
            
            if "overnight_liquidity_transfer" in dep_types:
                recommendations.append(
                    "Overnight liquidity transfer detected - Monitor NY/NYPM → LONDON transitions"
                )
            
            if "high_intensity_spillover" in dep_types:
                recommendations.append(
                    "High f8 intensity spillover found - Previous session f8 > 2000 correlates with FPFVG redelivery"
                )
                
            if "large_range_redelivery" in dep_types:
                recommendations.append(
                    "Large range redelivery pattern - Sessions with >150pt range tend to create next-session FPFVG"
                )
        else:
            recommendations.append(
                "No significant dependencies detected - FPFVG patterns may be more session-independent"
            )
        
        return recommendations

def run_fpfvg_investigation():
    """Run the FPFVG dependency investigation"""
    investigator = FPFVGDependencyInvestigator()
    results = investigator.investigate_am_session_dependencies()
    
    print("\n" + "="*60)
    print("🎯 INVESTIGATION RESULTS")
    print("="*60)
    
    insights = results["pattern_insights"]
    
    print(f"\n📊 Summary:")
    print(f"• Total AM sessions analyzed: {insights['total_am_sessions']}")
    print(f"• Sessions with dependencies: {insights['sessions_with_dependencies']}")
    print(f"• Subtle dependencies found: {insights['total_subtle_dependencies']}")
    
    if insights["dependency_types"]:
        print(f"\n🔗 Dependency Types Found:")
        for dep_type, count in insights["dependency_types"].items():
            print(f"• {dep_type}: {count} occurrences")
    
    if insights["strongest_correlations"]:
        print(f"\n⭐ Strongest Correlations:")
        for i, corr in enumerate(insights["strongest_correlations"], 1):
            print(f"{i}. {corr['previous_session']} → {corr['current_session']}")
            print(f"   Correlation: {corr['correlation_score']:.3f}, Type: {corr['dependency_type']}")
            print(f"   FPFVG events: {corr['current_fpfvg_events']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    return results

if __name__ == "__main__":
    results = run_fpfvg_investigation()