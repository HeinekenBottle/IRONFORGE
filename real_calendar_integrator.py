#!/usr/bin/env python3
"""
Real Calendar Integrator - Phase 5
Integrate real economic calendar with enhanced statistical analysis
"""

import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from economic_calendar_loader import EconomicCalendarIntegrator
from enhanced_statistical_framework import EnhancedStatisticalAnalyzer, VolatilityCalculator

class RealCalendarProcessor:
    """Process and integrate real economic calendar with IRONFORGE sessions"""
    
    def __init__(self):
        self.integrator = EconomicCalendarIntegrator()
        self.analyzer = EnhancedStatisticalAnalyzer()
        self.vol_calculator = VolatilityCalculator()
        
    def process_sessions_with_real_calendar(self, calendar_path: str,
                                          enhanced_session_dir: str = "/Users/jack/IRONFORGE/data/day_news_enhanced") -> Dict:
        """Process all sessions with real economic calendar integration"""
        
        print(f"🔄 Processing sessions with real economic calendar...")
        
        # Load and integrate calendar
        integration_result = self.integrator.integrate_calendar_with_sessions(
            calendar_path, enhanced_session_dir
        )
        
        if "error" in integration_result:
            return integration_result
        
        # Process each enhanced session with real calendar data
        enhanced_files = glob.glob(f"{enhanced_session_dir}/day_news_*.json")
        processed_data = []
        
        for file_path in enhanced_files:
            session_result = self._process_session_with_real_calendar(file_path)
            if session_result:
                processed_data.extend(session_result)
        
        # Generate comprehensive analysis
        analysis_result = self._generate_comprehensive_analysis(processed_data)
        analysis_result["integration_stats"] = integration_result
        
        return analysis_result
    
    def _process_session_with_real_calendar(self, file_path: str) -> List[Dict]:
        """Process single session with real calendar integration"""
        
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                
            session_events = []
            events = session_data.get("events", [])
            session_info = session_data.get("session_info", {})
            
            for event in events:
                range_position = event.get('range_position', 0.5)
                
                # Process RD@40 events
                if abs(range_position - 0.40) <= 0.025:
                    # Get enhanced data fields
                    event_data = self._extract_enhanced_event_data(event, session_data)
                    
                    # Add real calendar context if available
                    real_news = event.get("real_news_context", {})
                    if real_news:
                        event_data.update({
                            "news_source": real_news.get("news_source", "unknown"),
                            "event_time_utc": real_news.get("event_time_utc", ""),
                            "session_time_et": real_news.get("session_time_et", ""),
                            "news_bucket": real_news.get("news_bucket", "quiet"),
                            "news_distance_mins": real_news.get("news_distance_mins", 999),
                            "session_overlap": real_news.get("session_overlap", False),
                            "event_name": real_news.get("event_name", ""),
                            "impact_level": real_news.get("impact_level", "low")
                        })
                    else:
                        # Use synthetic news context as fallback
                        news_context = event.get("news_context", {})
                        event_data.update({
                            "news_source": "synthetic",
                            "news_bucket": news_context.get("news_bucket", "quiet"),
                            "news_distance_mins": 999,
                            "session_overlap": False,
                            "impact_level": "low"
                        })
                    
                    # Calculate volatility multiplier
                    event_time = event.get("timestamp", "")
                    if event_time:
                        vol_multiplier = self.vol_calculator.calculate_volatility_multiplier(
                            session_data, event_time
                        )
                        event_data["volatility_multiplier"] = vol_multiplier
                    
                    session_events.append(event_data)
                    
            return session_events
            
        except Exception as e:
            print(f"⚠️ Error processing session {Path(file_path).name}: {e}")
            return []
    
    def _extract_enhanced_event_data(self, event: Dict, session_data: Dict) -> Dict:
        """Extract all enhanced data fields for analysis"""
        
        # Get session metadata
        session_info = session_data.get("session_info", {})
        day_profile = session_data.get("day_profile", {})
        
        # Base event data
        event_data = {
            "session_id": session_info.get("session_id", "unknown"),
            "session_type": session_info.get("session_type", "unknown"),
            "day_of_week": day_profile.get("day_of_week", "unknown"),
            "day_profile_name": day_profile.get("profile_name", "unknown"),
            "range_position": event.get("range_position", 0.5),
            "energy_density": event.get("energy_density", 0.5),
            "timestamp": event.get("timestamp", ""),
        }
        
        # Path classification (E1/E2/E3)
        event_data["path_classification"] = self._classify_path(event)
        
        # Time-to-event measurements
        event_data["time_to_60"] = event.get("time_to_60_pct")
        event_data["time_to_80"] = event.get("time_to_80_pct")
        
        # Additional E-fields
        event_data["mid_hit"] = event.get("mid_hit", False)
        event_data["second_rd"] = event.get("second_rd40", False)
        event_data["failure"] = event.get("failure_flag", False)
        event_data["f8_level"] = event.get("f8_level", 0.0)
        event_data["f8_slope"] = event.get("f8_slope", 0.0)
        event_data["gap_age_days"] = event.get("gap_age_days", 0)
        event_data["regime"] = event.get("regime", "unknown")
        event_data["h1_breakout"] = event.get("h1_breakout", False)
        
        return event_data
    
    def _classify_path(self, event: Dict) -> str:
        """Classify RD@40 event into E1/E2/E3 paths"""
        
        # Simple classification logic (can be enhanced)
        range_position = event.get("range_position", 0.5)
        energy_density = event.get("energy_density", 0.5)
        
        # Look for explicit classification first
        if "path_classification" in event:
            return event["path_classification"]
        
        # Default classification based on energy and position
        if energy_density > 0.7 and range_position < 0.45:
            return "e3_accel"
        elif energy_density < 0.3 and abs(range_position - 0.40) < 0.02:
            return "e2_mr"
        else:
            return "e1_cont"
    
    def _generate_comprehensive_analysis(self, processed_data: List[Dict]) -> Dict:
        """Generate comprehensive analysis with all specified analyses"""
        
        if not processed_data:
            return {"error": "No processed data available"}
        
        print(f"📊 Generating comprehensive analysis for {len(processed_data)} RD@40 events...")
        
        analysis_results = {}
        
        # 1. Day Table Analysis
        day_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "day_of_week", "e3_accel"
        )
        analysis_results["day_analysis"] = day_results
        
        # 2. News Table Analysis
        news_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "news_bucket", "e3_accel"
        )
        analysis_results["news_analysis"] = news_results
        
        # 3. Day × News Matrix
        # Create composite key for matrix analysis
        for item in processed_data:
            item["day_news_composite"] = f"{item['day_of_week']}_{item['news_bucket']}"
        
        matrix_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "day_news_composite", "e3_accel"
        )
        analysis_results["day_news_matrix"] = matrix_results
        
        # 4. f8_level Split
        for item in processed_data:
            item["f8_level_bucket"] = "high_f8" if item.get("f8_level", 0) > 0.5 else "low_f8"
        
        f8_level_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "f8_level_bucket", "e3_accel"
        )
        analysis_results["f8_level_split"] = f8_level_results
        
        # 5. f8_slope Split
        for item in processed_data:
            slope = item.get("f8_slope", 0)
            if slope > 0.1:
                item["f8_slope_bucket"] = "positive"
            elif slope < -0.1:
                item["f8_slope_bucket"] = "negative"
            else:
                item["f8_slope_bucket"] = "neutral"
        
        f8_slope_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "f8_slope_bucket", "e3_accel"
        )
        analysis_results["f8_slope_split"] = f8_slope_results
        
        # 6. Gap Age Split
        for item in processed_data:
            gap_age = item.get("gap_age_days", 0)
            item["gap_age_bucket"] = "fresh" if gap_age == 0 else "aged_1to3"
        
        gap_age_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "gap_age_bucket", "e3_accel"
        )
        analysis_results["gap_age_split"] = gap_age_results
        
        # 7. Session Overlap Split
        overlap_results = self.analyzer.analyze_slice_with_validation(
            processed_data, "session_overlap", "e3_accel"
        )
        analysis_results["session_overlap_split"] = overlap_results
        
        # 8. Volatility Analysis
        vol_data = [item for item in processed_data if "volatility_multiplier" in item]
        if vol_data:
            analysis_results["volatility_stats"] = self._analyze_volatility_multipliers(vol_data)
        
        # Generate summary insights
        analysis_results["summary_insights"] = self._generate_summary_insights(analysis_results)
        analysis_results["total_events"] = len(processed_data)
        analysis_results["processed_timestamp"] = datetime.now().isoformat()
        
        return analysis_results
    
    def _analyze_volatility_multipliers(self, vol_data: List[Dict]) -> Dict:
        """Analyze volatility multipliers by different splits"""
        
        import numpy as np
        
        vol_stats = {}
        
        # Overall volatility stats
        multipliers = [item["volatility_multiplier"] for item in vol_data]
        vol_stats["overall"] = {
            "mean": float(np.mean(multipliers)),
            "median": float(np.median(multipliers)),
            "std": float(np.std(multipliers)),
            "min": float(np.min(multipliers)),
            "max": float(np.max(multipliers))
        }
        
        # Volatility by news bucket
        news_vol = {}
        for bucket in set(item["news_bucket"] for item in vol_data):
            bucket_mults = [item["volatility_multiplier"] for item in vol_data 
                          if item["news_bucket"] == bucket]
            if bucket_mults:
                news_vol[bucket] = {
                    "mean": float(np.mean(bucket_mults)),
                    "count": len(bucket_mults)
                }
        vol_stats["by_news_bucket"] = news_vol
        
        return vol_stats
    
    def _generate_summary_insights(self, analysis_results: Dict) -> List[str]:
        """Generate summary insights from comprehensive analysis"""
        
        insights = []
        
        # Day analysis insights
        if "day_analysis" in analysis_results:
            day_results = analysis_results["day_analysis"]
            best_day = max(day_results.items(), key=lambda x: x[1].percentage)
            insights.append(f"Best day for E3_ACCEL: {best_day[0]} ({best_day[1].percentage:.1f}%)")
            
        # News analysis insights  
        if "news_analysis" in analysis_results:
            news_results = analysis_results["news_analysis"]
            best_news = max(news_results.items(), key=lambda x: x[1].percentage)
            insights.append(f"Best news context for E3_ACCEL: {best_news[0]} ({best_news[1].percentage:.1f}%)")
        
        # Volatility insights
        if "volatility_stats" in analysis_results:
            vol_stats = analysis_results["volatility_stats"]
            overall_mean = vol_stats["overall"]["mean"]
            insights.append(f"Average volatility multiplier: {overall_mean:.2f}x")
            
            # Compare news buckets
            if "by_news_bucket" in vol_stats:
                for bucket, stats in vol_stats["by_news_bucket"].items():
                    if stats["count"] > 2:  # Only report on buckets with sufficient data
                        insights.append(f"{bucket} volatility: {stats['mean']:.2f}x (n={stats['count']})")
        
        # Inconclusive flags
        total_inconclusive = 0
        for analysis_name, results in analysis_results.items():
            if isinstance(results, dict) and any(hasattr(v, 'inconclusive_flag') for v in results.values()):
                inconclusive_count = sum(1 for v in results.values() 
                                       if hasattr(v, 'inconclusive_flag') and v.inconclusive_flag)
                total_inconclusive += inconclusive_count
        
        if total_inconclusive > 0:
            insights.append(f"⚠️ {total_inconclusive} analysis slices marked inconclusive (CI width > 30pp)")
        
        return insights

def demo_real_calendar_integration():
    """Demonstrate real calendar integration with comprehensive analysis"""
    
    print("🧪 DEMO: Real Calendar Integration")
    print("=" * 50)
    
    processor = RealCalendarProcessor()
    
    # Process with sample calendar
    result = processor.process_sessions_with_real_calendar(
        "/Users/jack/IRONFORGE/data/economic_calendar/sample_calendar.csv"
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📊 Analysis Results:")
    print(f"Total events processed: {result.get('total_events', 0)}")
    
    # Display summary insights
    if "summary_insights" in result:
        print(f"\n💡 Summary Insights:")
        for insight in result["summary_insights"]:
            print(f"  • {insight}")
    
    # Display integration stats
    if "integration_stats" in result:
        stats = result["integration_stats"]
        print(f"\n📈 Integration Statistics:")
        print(f"  Processed sessions: {stats.get('processed_sessions', 0)}")
        print(f"  Updated events: {stats.get('updated_events', 0)}")
        print(f"  News bucket distribution: {stats.get('news_bucket_distribution', {})}")

if __name__ == "__main__":
    demo_real_calendar_integration()