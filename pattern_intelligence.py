#!/usr/bin/env python3
"""
IRONFORGE Pattern Intelligence Layer
===================================
Advanced pattern classification, trending analysis, and relationship mapping.
Provides actionable intelligence from discovered patterns for daily trading insights.

Features:
- Pattern classification and taxonomies
- Temporal trend analysis across sessions
- Market regime pattern detection
- Real-time pattern matching and alerts
- Statistical significance testing
- Pattern performance tracking
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# IRONFORGE components
from ironforge_discovery_sdk import PatternAnalysis, CrossSessionLink, IRONFORGEDiscoverySDK


@dataclass
class PatternTrend:
    """Temporal trend in pattern occurrence"""
    pattern_type: str
    trend_strength: float  # -1 to 1, negative = decreasing, positive = increasing
    significance: float   # p-value from trend test
    occurrences_by_day: Dict[str, int]
    avg_confidence_trend: float
    description: str


@dataclass
class MarketRegime:
    """Identified market regime based on pattern clustering"""
    regime_id: str
    regime_name: str
    characteristic_patterns: List[str]
    sessions: List[str]
    start_date: str
    end_date: str
    confidence: float
    description: str


@dataclass
class PatternAlert:
    """Real-time pattern alert"""
    alert_id: str
    pattern_match: PatternAnalysis
    historical_matches: List[PatternAnalysis]
    confidence: float
    alert_type: str  # "strong_match", "regime_shift", "rare_pattern"
    timestamp: str
    description: str
    suggested_action: str


class PatternIntelligenceEngine:
    """
    Advanced pattern intelligence system for actionable trading insights
    
    Provides sophisticated analysis beyond basic pattern discovery:
    - Market regime identification
    - Pattern trend analysis
    - Real-time pattern matching
    - Performance tracking
    """
    
    def __init__(self, sdk: IRONFORGEDiscoverySDK):
        """Initialize with discovery SDK for pattern data access"""
        self.sdk = sdk
        self.pattern_trends: Dict[str, PatternTrend] = {}
        self.market_regimes: List[MarketRegime] = []
        self.pattern_alerts: List[PatternAlert] = []
        
        # Intelligence cache
        self.intelligence_cache = self.sdk.discovery_cache_path / "pattern_intelligence"
        self.intelligence_cache.mkdir(exist_ok=True)
        
        # Pattern taxonomy for classification
        self.pattern_taxonomy = {
            'temporal_structural': {
                'family': 'structural',
                'category': 'position_based',
                'trading_relevance': 'high',
                'typical_timeframe': 'intraday'
            },
            'htf_confluence': {
                'family': 'confluence',
                'category': 'multi_timeframe',
                'trading_relevance': 'very_high',
                'typical_timeframe': 'daily'
            },
            'scale_alignment': {
                'family': 'scale',
                'category': 'fractal',
                'trading_relevance': 'medium',
                'typical_timeframe': 'hourly'
            }
        }
    
    def analyze_pattern_trends(self, days_lookback: int = 30) -> Dict[str, PatternTrend]:
        """
        Analyze temporal trends in pattern occurrence
        
        Args:
            days_lookback: Number of days to analyze for trends
            
        Returns:
            Dictionary of pattern trends by type
        """
        print(f"📈 Analyzing pattern trends over {days_lookback} days")
        
        # Group patterns by type and date
        patterns_by_type_date = defaultdict(lambda: defaultdict(list))
        
        for pattern in self.sdk.pattern_database.values():
            try:
                pattern_date = datetime.fromisoformat(pattern.session_date).date()
                cutoff_date = datetime.now().date() - timedelta(days=days_lookback)
                
                if pattern_date >= cutoff_date:
                    date_str = pattern_date.isoformat()
                    patterns_by_type_date[pattern.pattern_type][date_str].append(pattern)
            except:
                continue
        
        # Calculate trends for each pattern type
        trends = {}
        
        for pattern_type, date_patterns in patterns_by_type_date.items():
            if len(date_patterns) < 3:  # Need minimum data for trend
                continue
            
            # Create time series
            dates = sorted(date_patterns.keys())
            occurrences = [len(date_patterns[date]) for date in dates]
            confidences = []
            
            for date in dates:
                avg_conf = np.mean([p.confidence for p in date_patterns[date]])
                confidences.append(avg_conf)
            
            # Calculate trend using linear regression
            x_vals = np.arange(len(dates))
            
            # Occurrence trend
            occ_slope, occ_intercept, occ_r, occ_p, _ = stats.linregress(x_vals, occurrences)
            
            # Confidence trend
            conf_slope, _, _, conf_p, _ = stats.linregress(x_vals, confidences)
            
            # Create trend object
            trend = PatternTrend(
                pattern_type=pattern_type,
                trend_strength=occ_slope / max(np.std(occurrences), 0.1),  # Normalized slope
                significance=occ_p,
                occurrences_by_day={date: len(date_patterns[date]) for date in dates},
                avg_confidence_trend=conf_slope,
                description=self._describe_trend(occ_slope, occ_p, conf_slope)
            )
            
            trends[pattern_type] = trend
        
        self.pattern_trends = trends
        
        # Cache results
        cache_file = self.intelligence_cache / f"pattern_trends_{datetime.now().strftime('%Y%m%d')}.json"
        with open(cache_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in trends.items()}, f, indent=2)
        
        print(f"✅ Analyzed trends for {len(trends)} pattern types")
        return trends
    
    def _describe_trend(self, slope: float, p_value: float, conf_slope: float) -> str:
        """Generate human-readable trend description"""
        if p_value > 0.05:
            return "No significant trend detected"
        
        direction = "increasing" if slope > 0 else "decreasing"
        strength = "strong" if abs(slope) > 1.0 else "moderate" if abs(slope) > 0.5 else "weak"
        
        conf_dir = "with improving confidence" if conf_slope > 0 else "with declining confidence"
        
        return f"{strength.capitalize()} {direction} trend {conf_dir}"
    
    def identify_market_regimes(self, min_sessions: int = 3) -> List[MarketRegime]:
        """
        Identify market regimes based on pattern clustering
        
        Args:
            min_sessions: Minimum sessions required to define a regime
            
        Returns:
            List of identified market regimes
        """
        print("🏛️ Identifying market regimes from pattern clusters")
        
        # Create feature matrix for clustering
        sessions_data = defaultdict(lambda: defaultdict(int))
        sessions_dates = {}
        
        for pattern in self.sdk.pattern_database.values():
            session = pattern.session_name
            sessions_data[session][pattern.pattern_type] += 1
            sessions_dates[session] = pattern.session_date
        
        # Convert to matrix
        session_names = list(sessions_data.keys())
        pattern_types = list(set(p.pattern_type for p in self.sdk.pattern_database.values()))
        
        feature_matrix = []
        for session in session_names:
            features = [sessions_data[session][ptype] for ptype in pattern_types]
            feature_matrix.append(features)
        
        if len(feature_matrix) < min_sessions:
            print(f"⚠️ Insufficient data for regime identification (need {min_sessions} sessions)")
            return []
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=min_sessions)
        clusters = clustering.fit_predict(scaled_features)
        
        # Create regime objects
        regimes = []
        unique_clusters = set(clusters)
        unique_clusters.discard(-1)  # Remove noise cluster
        
        for cluster_id in unique_clusters:
            cluster_sessions = [session_names[i] for i, c in enumerate(clusters) if c == cluster_id]
            
            if len(cluster_sessions) < min_sessions:
                continue
            
            # Find characteristic patterns for this regime
            pattern_counts = defaultdict(int)
            for session in cluster_sessions:
                for ptype in pattern_types:
                    pattern_counts[ptype] += sessions_data[session][ptype]
            
            # Top patterns by relative frequency
            total_patterns = sum(pattern_counts.values())
            char_patterns = sorted(
                [(ptype, count/total_patterns) for ptype, count in pattern_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Date range
            regime_dates = [sessions_dates[s] for s in cluster_sessions if s in sessions_dates]
            regime_dates.sort()
            start_date = regime_dates[0] if regime_dates else "unknown"
            end_date = regime_dates[-1] if regime_dates else "unknown"
            
            # Create regime
            regime = MarketRegime(
                regime_id=f"regime_{cluster_id}",
                regime_name=f"Pattern Regime {cluster_id}",
                characteristic_patterns=[ptype for ptype, _ in char_patterns],
                sessions=cluster_sessions,
                start_date=start_date,
                end_date=end_date,
                confidence=len(cluster_sessions) / len(session_names),
                description=f"Regime characterized by {', '.join(p[0] for p in char_patterns[:2])}"
            )
            
            regimes.append(regime)
        
        self.market_regimes = regimes
        
        # Cache results
        cache_file = self.intelligence_cache / f"market_regimes_{datetime.now().strftime('%Y%m%d')}.json"
        with open(cache_file, 'w') as f:
            json.dump([asdict(r) for r in regimes], f, indent=2)
        
        print(f"✅ Identified {len(regimes)} market regimes")
        return regimes
    
    def find_pattern_matches(self, target_pattern: PatternAnalysis, 
                           similarity_threshold: float = 0.8,
                           max_matches: int = 10) -> List[Tuple[PatternAnalysis, float]]:
        """
        Find historical patterns similar to target pattern
        
        Args:
            target_pattern: Pattern to find matches for
            similarity_threshold: Minimum similarity for matches
            max_matches: Maximum number of matches to return
            
        Returns:
            List of (matching_pattern, similarity_score) tuples
        """
        matches = []
        
        for pattern in self.sdk.pattern_database.values():
            if pattern.pattern_id == target_pattern.pattern_id:
                continue  # Skip self
            
            similarity = self._calculate_pattern_similarity(target_pattern, pattern)
            
            if similarity >= similarity_threshold:
                matches.append((pattern, similarity))
        
        # Sort by similarity and limit
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_matches]
    
    def _calculate_pattern_similarity(self, p1: PatternAnalysis, p2: PatternAnalysis) -> float:
        """Calculate similarity between patterns (same as SDK method)"""
        # Type similarity
        type_similarity = 1.0 if p1.pattern_type == p2.pattern_type else 0.0
        
        # Structural position similarity
        struct_diff = abs(p1.structural_position - p2.structural_position)
        struct_similarity = max(0.0, 1.0 - struct_diff)
        
        # Confidence similarity
        conf_diff = abs(p1.confidence - p2.confidence)
        conf_similarity = max(0.0, 1.0 - conf_diff)
        
        # Enhanced features similarity
        features_similarity = 0.0
        if p1.enhanced_features and p2.enhanced_features:
            energy_diff = abs(p1.enhanced_features.get('energy_density', 0) - p2.enhanced_features.get('energy_density', 0))
            htf_diff = abs(p1.enhanced_features.get('htf_carryover', 0) - p2.enhanced_features.get('htf_carryover', 0))
            features_similarity = max(0.0, 1.0 - (energy_diff + htf_diff) / 2.0)
        
        # Weighted average
        total_similarity = (
            type_similarity * 0.3 +
            struct_similarity * 0.3 +
            conf_similarity * 0.2 +
            features_similarity * 0.2
        )
        
        return total_similarity
    
    def generate_pattern_alerts(self, new_patterns: List[PatternAnalysis]) -> List[PatternAlert]:
        """
        Generate alerts for new patterns based on historical analysis
        
        Args:
            new_patterns: Newly discovered patterns to analyze
            
        Returns:
            List of pattern alerts
        """
        alerts = []
        
        for pattern in new_patterns:
            # Find historical matches
            matches = self.find_pattern_matches(pattern, similarity_threshold=0.7)
            
            if not matches:
                # Rare pattern alert
                alert = PatternAlert(
                    alert_id=f"rare_{pattern.pattern_id}",
                    pattern_match=pattern,
                    historical_matches=[],
                    confidence=pattern.confidence,
                    alert_type="rare_pattern",
                    timestamp=datetime.now().isoformat(),
                    description=f"Rare {pattern.pattern_type} pattern with no close historical matches",
                    suggested_action="Monitor for unique market conditions"
                )
                alerts.append(alert)
            
            elif len(matches) >= 3 and matches[0][1] >= 0.9:
                # Strong match alert
                historical_patterns = [match[0] for match in matches[:3]]
                
                alert = PatternAlert(
                    alert_id=f"strong_{pattern.pattern_id}",
                    pattern_match=pattern,
                    historical_matches=historical_patterns,
                    confidence=matches[0][1],
                    alert_type="strong_match",
                    timestamp=datetime.now().isoformat(),
                    description=f"Strong match to {len(matches)} historical patterns",
                    suggested_action="Check historical outcomes for this pattern type"
                )
                alerts.append(alert)
        
        self.pattern_alerts.extend(alerts)
        return alerts
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive pattern intelligence summary"""
        return {
            'pattern_trends_count': len(self.pattern_trends),
            'significant_trends': len([t for t in self.pattern_trends.values() if t.significance < 0.05]),
            'market_regimes_count': len(self.market_regimes),
            'pattern_alerts_count': len(self.pattern_alerts),
            'most_active_pattern_type': max(
                Counter(p.pattern_type for p in self.sdk.pattern_database.values()).items(),
                key=lambda x: x[1],
                default=("none", 0)
            )[0],
            'pattern_taxonomy_coverage': len(set(p.pattern_type for p in self.sdk.pattern_database.values()) 
                                           & set(self.pattern_taxonomy.keys())),
            'cache_location': str(self.intelligence_cache)
        }
    
    def generate_intelligence_report(self) -> str:
        """Generate comprehensive intelligence report"""
        summary = self.get_intelligence_summary()
        
        # Top trends
        significant_trends = [(k, v) for k, v in self.pattern_trends.items() if v.significance < 0.05]
        significant_trends.sort(key=lambda x: abs(x[1].trend_strength), reverse=True)
        
        report = f"""
🧠 IRONFORGE Pattern Intelligence Report
========================================
Generated: {datetime.now().isoformat()}

📊 Intelligence Overview:
- Pattern Trends Analyzed: {summary['pattern_trends_count']}
- Statistically Significant Trends: {summary['significant_trends']}
- Market Regimes Identified: {summary['market_regimes_count']}
- Active Pattern Alerts: {summary['pattern_alerts_count']}

🔥 Most Active Pattern Type: {summary['most_active_pattern_type']}

📈 Top Significant Trends:
"""
        
        for i, (pattern_type, trend) in enumerate(significant_trends[:3], 1):
            direction = "📈" if trend.trend_strength > 0 else "📉"
            report += f"{i}. {direction} {pattern_type}: {trend.description}\n"
        
        if self.market_regimes:
            report += f"\n🏛️ Market Regimes:\n"
            for regime in self.market_regimes:
                report += f"- {regime.regime_name}: {len(regime.sessions)} sessions ({regime.description})\n"
        
        if self.pattern_alerts:
            report += f"\n🚨 Recent Alerts:\n"
            recent_alerts = sorted(self.pattern_alerts, key=lambda x: x.timestamp, reverse=True)[:3]
            for alert in recent_alerts:
                report += f"- {alert.alert_type}: {alert.description}\n"
        
        report += f"\n💾 Intelligence Cache: {summary['cache_location']}\n"
        
        return report


# Practical workflow functions
def analyze_market_intelligence() -> Dict[str, Any]:
    """Complete market intelligence analysis workflow"""
    print("🧠 Starting comprehensive market intelligence analysis")
    
    # Initialize SDK and intelligence engine
    sdk = IRONFORGEDiscoverySDK()
    intel_engine = PatternIntelligenceEngine(sdk)
    
    # Run discovery if needed
    if not sdk.pattern_database:
        print("📊 Running pattern discovery first...")
        sdk.discover_all_sessions()
    
    # Analyze trends
    trends = intel_engine.analyze_pattern_trends()
    
    # Identify regimes
    regimes = intel_engine.identify_market_regimes()
    
    # Generate report
    report = intel_engine.generate_intelligence_report()
    print(report)
    
    return {
        'trends': trends,
        'regimes': regimes,
        'summary': intel_engine.get_intelligence_summary()
    }


def find_similar_patterns(session_name: str, pattern_index: int = 0) -> List[Tuple[PatternAnalysis, float]]:
    """Find patterns similar to a specific pattern from a session"""
    sdk = IRONFORGEDiscoverySDK()
    intel_engine = PatternIntelligenceEngine(sdk)
    
    # Find target pattern
    target_patterns = [p for p in sdk.pattern_database.values() if session_name in p.session_name]
    
    if not target_patterns:
        print(f"❌ No patterns found for session containing '{session_name}'")
        return []
    
    if pattern_index >= len(target_patterns):
        print(f"❌ Pattern index {pattern_index} not found (max: {len(target_patterns)-1})")
        return []
    
    target_pattern = target_patterns[pattern_index]
    
    # Find matches
    matches = intel_engine.find_pattern_matches(target_pattern)
    
    print(f"🎯 Target: {target_pattern.description}")
    print(f"✅ Found {len(matches)} similar patterns:")
    
    for i, (pattern, similarity) in enumerate(matches[:5], 1):
        print(f"{i}. {pattern.session_name}: {pattern.description} (similarity: {similarity:.3f})")
    
    return matches


if __name__ == "__main__":
    print("🧠 IRONFORGE Pattern Intelligence Engine")
    print("=" * 50)
    print("Available functions:")
    print("1. analyze_market_intelligence() - Complete intelligence analysis")
    print("2. find_similar_patterns('session_name') - Find similar patterns")
    print("3. PatternIntelligenceEngine(sdk) - Full intelligence engine")
    print("\nExample usage:")
    print("  results = analyze_market_intelligence()")
    print("  matches = find_similar_patterns('NY_PM')")