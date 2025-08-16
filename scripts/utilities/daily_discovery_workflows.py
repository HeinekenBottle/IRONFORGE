#!/usr/bin/env python3
"""
IRONFORGE Daily Discovery Workflows
===================================
Practical daily-use workflows for pattern discovery and market analysis.
Designed for systematic pattern hunting with actionable outputs.

Workflows:
1. Morning Market Analysis - Pre-session pattern review
2. Session Pattern Hunter - Real-time pattern discovery
3. Cross-Session Relationship Tracker - Multi-day pattern links
4. Regime Change Monitor - Market phase transitions
5. Pattern Performance Tracker - Historical pattern outcomes

Author: IRONFORGE Archaeological Discovery System
Date: August 14, 2025
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np

# IRONFORGE components
from ironforge_discovery_sdk import IRONFORGEDiscoverySDK, PatternAnalysis
from pattern_intelligence import PatternIntelligenceEngine, analyze_market_intelligence


@dataclass
class MarketAnalysis:
    """Daily market analysis result"""
    analysis_date: str
    session_patterns: Dict[str, List[PatternAnalysis]]
    dominant_pattern_types: List[str]
    pattern_strength_score: float
    cross_session_signals: List[str]
    regime_status: str
    trading_insights: List[str]
    confidence_level: str


@dataclass
class SessionDiscoveryResult:
    """Real-time session discovery result"""
    session_name: str
    discovery_timestamp: str
    patterns_found: List[PatternAnalysis]
    strength_indicators: Dict[str, float]
    historical_comparisons: List[str]
    immediate_insights: List[str]
    next_session_expectations: List[str]


class DailyDiscoveryWorkflows:
    """
    Production workflows for daily pattern discovery and market analysis
    
    Provides systematic, actionable workflows for:
    - Pre-market preparation
    - Real-time session analysis
    - Cross-session pattern tracking
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize daily discovery workflows"""
        self.sdk = IRONFORGEDiscoverySDK()
        self.intel_engine = PatternIntelligenceEngine(self.sdk)
        
        # Workflow cache
        self.workflow_cache = self.sdk.discovery_cache_path / "daily_workflows"
        self.workflow_cache.mkdir(exist_ok=True)
        
        # Session schedule for systematic analysis
        self.session_schedule = {
            'PREMARKET': {'start': '04:00', 'end': '09:30', 'importance': 'medium'},
            'NY_AM': {'start': '09:30', 'end': '12:00', 'importance': 'high'},
            'LUNCH': {'start': '12:00', 'end': '13:30', 'importance': 'low'},
            'NY_PM': {'start': '13:30', 'end': '16:00', 'importance': 'very_high'},
            'LONDON': {'start': '03:00', 'end': '12:00', 'importance': 'high'},
            'ASIA': {'start': '18:00', 'end': '03:00', 'importance': 'medium'}
        }
        
        print("📋 Daily Discovery Workflows initialized")
        print(f"💾 Workflow cache: {self.workflow_cache}")
    
    def morning_market_analysis(self, days_lookback: int = 7) -> MarketAnalysis:
        """
        Morning pre-market analysis workflow
        
        Analyzes recent patterns to prepare for the trading day:
        - Reviews patterns from last N days
        - Identifies dominant themes
        - Suggests session focus areas
        - Provides confidence assessment
        
        Args:
            days_lookback: Number of days to analyze
            
        Returns:
            Comprehensive morning analysis
        """
        print(f"🌅 Morning Market Analysis - Reviewing last {days_lookback} days")
        
        # Ensure pattern database is populated
        if not self.sdk.pattern_database:
            print("📊 Loading pattern database...")
            self.sdk.discover_all_sessions()
        
        # Filter patterns to recent days
        cutoff_date = datetime.now().date() - timedelta(days=days_lookback)
        recent_patterns = []
        
        for pattern in self.sdk.pattern_database.values():
            try:
                pattern_date = datetime.fromisoformat(pattern.session_date).date()
                if pattern_date >= cutoff_date:
                    recent_patterns.append(pattern)
            except:
                continue
        
        print(f"🔍 Analyzing {len(recent_patterns)} recent patterns")
        
        # Group patterns by session type
        session_patterns = {}
        for pattern in recent_patterns:
            session_type = self._extract_session_type(pattern.session_name)
            if session_type not in session_patterns:
                session_patterns[session_type] = []
            session_patterns[session_type].append(pattern)
        
        # Identify dominant pattern types
        pattern_type_counts = {}
        for pattern in recent_patterns:
            pattern_type_counts[pattern.pattern_type] = pattern_type_counts.get(pattern.pattern_type, 0) + 1
        
        dominant_patterns = sorted(pattern_type_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_pattern_types = [p[0] for p in dominant_patterns[:3]]
        
        # Calculate pattern strength score
        avg_confidence = np.mean([p.confidence for p in recent_patterns]) if recent_patterns else 0.0
        pattern_density = len(recent_patterns) / days_lookback  # patterns per day
        pattern_strength_score = (avg_confidence * 0.6 + min(pattern_density / 5.0, 1.0) * 0.4)
        
        # Generate cross-session signals
        cross_session_signals = self._identify_cross_session_signals(recent_patterns)
        
        # Determine regime status
        regime_status = self._assess_current_regime(recent_patterns)
        
        # Generate trading insights
        trading_insights = self._generate_trading_insights(
            session_patterns, dominant_pattern_types, pattern_strength_score
        )
        
        # Determine confidence level
        confidence_level = self._calculate_confidence_level(pattern_strength_score, len(recent_patterns))
        
        # Create analysis result
        analysis = MarketAnalysis(
            analysis_date=datetime.now().date().isoformat(),
            session_patterns=session_patterns,
            dominant_pattern_types=dominant_pattern_types,
            pattern_strength_score=pattern_strength_score,
            cross_session_signals=cross_session_signals,
            regime_status=regime_status,
            trading_insights=trading_insights,
            confidence_level=confidence_level
        )
        
        # Cache results
        cache_file = self.workflow_cache / f"morning_analysis_{datetime.now().strftime('%Y%m%d')}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'analysis_date': analysis.analysis_date,
                'dominant_pattern_types': analysis.dominant_pattern_types,
                'pattern_strength_score': analysis.pattern_strength_score,
                'cross_session_signals': analysis.cross_session_signals,
                'regime_status': analysis.regime_status,
                'trading_insights': analysis.trading_insights,
                'confidence_level': analysis.confidence_level,
                'patterns_analyzed': len(recent_patterns)
            }, f, indent=2)
        
        # Print analysis
        self._print_morning_analysis(analysis)
        
        return analysis
    
    def _extract_session_type(self, session_name: str) -> str:
        """Extract session type from session name"""
        session_name = session_name.upper()
        
        if 'NY_AM' in session_name or 'NYAM' in session_name:
            return 'NY_AM'
        elif 'NY_PM' in session_name or 'NYPM' in session_name:
            return 'NY_PM'
        elif 'LONDON' in session_name:
            return 'LONDON'
        elif 'ASIA' in session_name:
            return 'ASIA'
        elif 'LUNCH' in session_name:
            return 'LUNCH'
        elif 'PREMARKET' in session_name:
            return 'PREMARKET'
        elif 'MIDNIGHT' in session_name:
            return 'MIDNIGHT'
        else:
            return 'UNKNOWN'
    
    def _identify_cross_session_signals(self, patterns: List[PatternAnalysis]) -> List[str]:
        """Identify cross-session continuation signals"""
        signals = []
        
        # Group by date and session
        by_date_session = {}
        for pattern in patterns:
            date = pattern.session_date
            session = self._extract_session_type(pattern.session_name)
            key = f"{date}_{session}"
            
            if key not in by_date_session:
                by_date_session[key] = []
            by_date_session[key].append(pattern)
        
        # Look for continuation patterns
        dates = sorted(set(p.session_date for p in patterns))
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            
            # Check for pattern continuation across days
            current_patterns = [p for p in patterns if p.session_date == current_date]
            next_patterns = [p for p in patterns if p.session_date == next_date]
            
            if current_patterns and next_patterns:
                # Look for similar pattern types
                current_types = set(p.pattern_type for p in current_patterns)
                next_types = set(p.pattern_type for p in next_patterns)
                
                common_types = current_types & next_types
                if common_types:
                    signals.append(f"{', '.join(common_types)} patterns continuing from {current_date}")
        
        # Look for strength building
        if len(patterns) >= 3:
            recent_confidences = [p.confidence for p in patterns[-3:]]
            if all(recent_confidences[i] <= recent_confidences[i+1] for i in range(len(recent_confidences)-1)):
                signals.append("Pattern confidence strengthening over recent sessions")
        
        return signals
    
    def _assess_current_regime(self, patterns: List[PatternAnalysis]) -> str:
        """Assess current market regime based on recent patterns"""
        if not patterns:
            return "Insufficient data"
        
        # Analyze pattern distribution
        pattern_types = [p.pattern_type for p in patterns]
        type_counts = {ptype: pattern_types.count(ptype) for ptype in set(pattern_types)}
        
        # Determine regime based on dominant patterns
        dominant_type = max(type_counts, key=type_counts.get)
        dominant_ratio = type_counts[dominant_type] / len(patterns)
        
        if dominant_ratio >= 0.6:
            return f"Strong {dominant_type} regime"
        elif dominant_ratio >= 0.4:
            return f"Moderate {dominant_type} regime"
        else:
            return "Mixed pattern regime"
    
    def _generate_trading_insights(self, session_patterns: Dict[str, List[PatternAnalysis]], 
                                 dominant_types: List[str], strength_score: float) -> List[str]:
        """Generate actionable trading insights"""
        insights = []
        
        # Session-specific insights
        for session_type, patterns in session_patterns.items():
            if not patterns:
                continue
                
            avg_confidence = np.mean([p.confidence for p in patterns])
            importance = self.session_schedule.get(session_type, {}).get('importance', 'medium')
            
            if avg_confidence >= 0.7 and importance in ['high', 'very_high']:
                insights.append(f"Focus on {session_type} session - high confidence patterns ({avg_confidence:.2f})")
            elif len(patterns) >= 3:
                insights.append(f"{session_type} showing consistent pattern activity ({len(patterns)} patterns)")
        
        # Pattern type insights
        if dominant_types:
            insights.append(f"Primary pattern theme: {dominant_types[0]} patterns dominating")
            
            if 'temporal_structural' in dominant_types:
                insights.append("Watch for structural position entries - temporal patterns active")
            
            if 'htf_confluence' in dominant_types:
                insights.append("Higher timeframe alignment expected - confluence patterns present")
        
        # Strength insights
        if strength_score >= 0.8:
            insights.append("High pattern strength environment - follow systematic signals")
        elif strength_score <= 0.3:
            insights.append("Low pattern strength - use reduced position sizing")
        
        return insights
    
    def _calculate_confidence_level(self, strength_score: float, pattern_count: int) -> str:
        """Calculate overall confidence level"""
        if strength_score >= 0.7 and pattern_count >= 10:
            return "High"
        elif strength_score >= 0.5 and pattern_count >= 5:
            return "Medium"
        elif pattern_count >= 3:
            return "Low"
        else:
            return "Insufficient Data"
    
    def _print_morning_analysis(self, analysis: MarketAnalysis):
        """Print formatted morning analysis"""
        print("\n" + "="*60)
        print(f"🌅 MORNING MARKET ANALYSIS - {analysis.analysis_date}")
        print("="*60)
        
        print(f"\n📊 Pattern Overview:")
        print(f"   Strength Score: {analysis.pattern_strength_score:.2f}/1.0")
        print(f"   Confidence Level: {analysis.confidence_level}")
        print(f"   Current Regime: {analysis.regime_status}")
        
        print(f"\n🔥 Dominant Pattern Types:")
        for i, ptype in enumerate(analysis.dominant_pattern_types, 1):
            print(f"   {i}. {ptype}")
        
        if analysis.cross_session_signals:
            print(f"\n🔗 Cross-Session Signals:")
            for signal in analysis.cross_session_signals:
                print(f"   • {signal}")
        
        print(f"\n💡 Trading Insights:")
        for insight in analysis.trading_insights:
            print(f"   • {insight}")
        
        print(f"\n📈 Session Focus Areas:")
        for session_type, patterns in analysis.session_patterns.items():
            if patterns:
                importance = self.session_schedule.get(session_type, {}).get('importance', 'medium')
                avg_conf = np.mean([p.confidence for p in patterns])
                print(f"   {session_type}: {len(patterns)} patterns (confidence: {avg_conf:.2f}, importance: {importance})")
        
        print("\n" + "="*60)
    
    def hunt_session_patterns(self, session_name_filter: str = "NY_PM") -> SessionDiscoveryResult:
        """
        Real-time session pattern hunting workflow
        
        Focuses on discovering patterns in specific session type with
        immediate actionable insights.
        
        Args:
            session_name_filter: Filter for specific session type
            
        Returns:
            Real-time session discovery result
        """
        print(f"🎯 Session Pattern Hunter - Focusing on {session_name_filter}")
        
        # Find matching sessions
        session_files = list(self.sdk.enhanced_sessions_path.glob(f'*{session_name_filter}*.json'))
        
        if not session_files:
            print(f"❌ No sessions found matching '{session_name_filter}'")
            return SessionDiscoveryResult(
                session_name=session_name_filter,
                discovery_timestamp=datetime.now().isoformat(),
                patterns_found=[],
                strength_indicators={},
                historical_comparisons=[],
                immediate_insights=["No matching sessions found"],
                next_session_expectations=[]
            )
        
        # Use most recent session
        latest_session = max(session_files, key=lambda f: f.stat().st_mtime)
        print(f"🔍 Analyzing: {latest_session.name}")
        
        # Discover patterns
        patterns = self.sdk.discover_session_patterns(latest_session)
        
        if not patterns:
            print("❌ No patterns discovered in session")
            return SessionDiscoveryResult(
                session_name=latest_session.stem,
                discovery_timestamp=datetime.now().isoformat(),
                patterns_found=[],
                strength_indicators={'pattern_count': 0},
                historical_comparisons=[],
                immediate_insights=["No patterns found in this session"],
                next_session_expectations=[]
            )
        
        # Calculate strength indicators
        strength_indicators = {
            'pattern_count': len(patterns),
            'avg_confidence': np.mean([p.confidence for p in patterns]),
            'max_confidence': max(p.confidence for p in patterns),
            'pattern_types': len(set(p.pattern_type for p in patterns)),
            'temporal_span': max(p.time_span_hours for p in patterns) if patterns else 0.0
        }
        
        # Generate historical comparisons
        historical_comparisons = self._generate_historical_comparisons(patterns, session_name_filter)
        
        # Generate immediate insights
        immediate_insights = self._generate_immediate_insights(patterns, strength_indicators)
        
        # Generate expectations for next session
        next_session_expectations = self._generate_next_session_expectations(patterns)
        
        result = SessionDiscoveryResult(
            session_name=latest_session.stem,
            discovery_timestamp=datetime.now().isoformat(),
            patterns_found=patterns,
            strength_indicators=strength_indicators,
            historical_comparisons=historical_comparisons,
            immediate_insights=immediate_insights,
            next_session_expectations=next_session_expectations
        )
        
        # Print results
        self._print_session_discovery(result)
        
        return result
    
    def _generate_historical_comparisons(self, patterns: List[PatternAnalysis], 
                                       session_filter: str) -> List[str]:
        """Generate historical pattern comparisons"""
        comparisons = []
        
        # Compare to historical sessions of same type
        similar_sessions = [p for p in self.sdk.pattern_database.values() 
                          if session_filter.upper() in p.session_name.upper()]
        
        if len(similar_sessions) > len(patterns):
            historical_avg_confidence = np.mean([p.confidence for p in similar_sessions])
            current_avg_confidence = np.mean([p.confidence for p in patterns])
            
            if current_avg_confidence > historical_avg_confidence * 1.1:
                comparisons.append(f"Above average confidence vs historical {session_filter} sessions")
            elif current_avg_confidence < historical_avg_confidence * 0.9:
                comparisons.append(f"Below average confidence vs historical {session_filter} sessions")
            
            historical_pattern_count = len(similar_sessions) / len(set(p.session_name for p in similar_sessions))
            if len(patterns) > historical_pattern_count * 1.2:
                comparisons.append(f"Higher pattern density than typical {session_filter} sessions")
        
        return comparisons
    
    def _generate_immediate_insights(self, patterns: List[PatternAnalysis], 
                                   strength_indicators: Dict[str, float]) -> List[str]:
        """Generate immediate actionable insights"""
        insights = []
        
        # Pattern count insights
        if strength_indicators['pattern_count'] >= 5:
            insights.append("High pattern activity - multiple opportunities identified")
        elif strength_indicators['pattern_count'] >= 3:
            insights.append("Moderate pattern activity - selective opportunities")
        else:
            insights.append("Low pattern activity - wait for clearer signals")
        
        # Confidence insights
        if strength_indicators['avg_confidence'] >= 0.8:
            insights.append("High confidence patterns - execute with normal position sizing")
        elif strength_indicators['avg_confidence'] >= 0.6:
            insights.append("Moderate confidence - use reduced position sizing")
        else:
            insights.append("Low confidence patterns - paper trade or skip")
        
        # Pattern type diversity
        if strength_indicators['pattern_types'] >= 3:
            insights.append("Diverse pattern types - multiple setups available")
        
        # Temporal insights
        if strength_indicators['temporal_span'] > 2.0:
            insights.append("Extended temporal patterns - expect longer-term moves")
        
        # Specific pattern insights
        pattern_types = [p.pattern_type for p in patterns]
        if 'htf_confluence' in pattern_types:
            insights.append("HTF confluence present - alignment with higher timeframes")
        if 'temporal_structural' in pattern_types:
            insights.append("Structural patterns active - watch key levels")
        
        return insights
    
    def _generate_next_session_expectations(self, patterns: List[PatternAnalysis]) -> List[str]:
        """Generate expectations for next session"""
        expectations = []
        
        if not patterns:
            return ["Insufficient data for next session expectations"]
        
        # High confidence patterns suggest continuation
        high_conf_patterns = [p for p in patterns if p.confidence >= 0.7]
        if high_conf_patterns:
            expectations.append("Pattern continuation likely in next session")
        
        # Pattern types suggest next session characteristics
        pattern_types = [p.pattern_type for p in patterns]
        
        if 'temporal_structural' in pattern_types:
            expectations.append("Watch for structural continuation patterns")
        
        if 'htf_confluence' in pattern_types:
            expectations.append("HTF alignment may persist - maintain directional bias")
        
        # Strength-based expectations
        avg_confidence = np.mean([p.confidence for p in patterns])
        if avg_confidence >= 0.8:
            expectations.append("High probability environment continues")
        elif avg_confidence <= 0.4:
            expectations.append("Low probability environment - wait for improvement")
        
        return expectations
    
    def _print_session_discovery(self, result: SessionDiscoveryResult):
        """Print formatted session discovery results"""
        print("\n" + "="*60)
        print(f"🎯 SESSION PATTERN DISCOVERY - {result.session_name}")
        print("="*60)
        
        print(f"\n📊 Discovery Summary:")
        print(f"   Patterns Found: {result.strength_indicators['pattern_count']}")
        print(f"   Average Confidence: {result.strength_indicators['avg_confidence']:.2f}")
        print(f"   Pattern Types: {result.strength_indicators['pattern_types']}")
        print(f"   Max Temporal Span: {result.strength_indicators['temporal_span']:.1f}h")
        
        if result.patterns_found:
            print(f"\n🔍 Discovered Patterns:")
            for i, pattern in enumerate(result.patterns_found, 1):
                print(f"   {i}. {pattern.pattern_type}: {pattern.description} (conf: {pattern.confidence:.2f})")
        
        if result.historical_comparisons:
            print(f"\n📈 Historical Context:")
            for comp in result.historical_comparisons:
                print(f"   • {comp}")
        
        print(f"\n💡 Immediate Insights:")
        for insight in result.immediate_insights:
            print(f"   • {insight}")
        
        print(f"\n🔮 Next Session Expectations:")
        for expectation in result.next_session_expectations:
            print(f"   • {expectation}")
        
        print("\n" + "="*60)
    
    def track_pattern_performance(self, days_lookback: int = 14) -> Dict[str, Any]:
        """
        Track pattern performance over time
        
        Analyzes how patterns have performed historically to guide
        future pattern selection and confidence assessment.
        
        Args:
            days_lookback: Days to analyze for performance tracking
            
        Returns:
            Pattern performance analysis
        """
        print(f"📈 Pattern Performance Tracking - Last {days_lookback} days")
        
        # This is a framework for performance tracking
        # In a real implementation, this would connect to actual trade results
        
        performance_data = {
            'analysis_period_days': days_lookback,
            'patterns_analyzed': len(self.sdk.pattern_database),
            'performance_by_type': {},
            'confidence_correlation': 0.0,
            'temporal_performance': {},
            'session_performance': {}
        }
        
        print("📊 Performance tracking framework ready")
        print("   (Connect to actual trading results for complete analysis)")
        
        return performance_data


# Convenience functions for daily use
def morning_prep(days_back: int = 7) -> MarketAnalysis:
    """Quick morning market preparation"""
    workflows = DailyDiscoveryWorkflows()
    return workflows.morning_market_analysis(days_back)


def hunt_patterns(session: str = "NY_PM") -> SessionDiscoveryResult:
    """Quick pattern hunting for specific session"""
    workflows = DailyDiscoveryWorkflows()
    return workflows.hunt_session_patterns(session)


def full_market_intel() -> Dict[str, Any]:
    """Complete daily market intelligence workflow"""
    print("🧠 Running complete daily market intelligence workflow...")
    
    # Morning analysis
    workflows = DailyDiscoveryWorkflows()
    morning_analysis = workflows.morning_market_analysis()
    
    # Pattern intelligence
    intel_results = analyze_market_intelligence()
    
    # Session hunting for key sessions
    ny_pm_hunt = workflows.hunt_session_patterns("NY_PM")
    ny_am_hunt = workflows.hunt_session_patterns("NY_AM")
    
    return {
        'morning_analysis': morning_analysis,
        'intelligence_analysis': intel_results,
        'ny_pm_patterns': ny_pm_hunt,
        'ny_am_patterns': ny_am_hunt,
        'workflow_timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    print("📋 IRONFORGE Daily Discovery Workflows")
    print("=" * 50)
    print("Available workflows:")
    print("1. morning_prep() - Morning market preparation")
    print("2. hunt_patterns('NY_PM') - Session pattern hunting")
    print("3. full_market_intel() - Complete daily intelligence")
    print("4. DailyDiscoveryWorkflows() - Full workflow system")
    print("\nExample usage:")
    print("  analysis = morning_prep()")
    print("  patterns = hunt_patterns('NY_PM')")
    print("  intel = full_market_intel()")