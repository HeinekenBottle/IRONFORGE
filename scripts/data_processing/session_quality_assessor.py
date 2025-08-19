#!/usr/bin/env python3
"""
Level 1 Session Quality Assessment Tool
=====================================

Systematically analyzes Level 1 session data quality to identify artificial patterns
causing 96.8% duplication in TGAT model discoveries.

Scoring System (0-100):
- Complete (80-100): Ready for TGAT training
- Partial (50-79): Requires minor cleaning  
- Artificial (20-49): Extensive artificial data
- Unusable (0-19): Critical corruption
"""

import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionQualityAssessor:
    def __init__(self, data_path: str = "/Users/jack/IRONPULSE/data/sessions/level_1"):
        self.data_path = Path(data_path)
        self.quality_scores = {}
        self.detailed_analysis = {}
        
    def assess_session_quality(self, session_data: dict[str, Any], session_file: str) -> dict[str, Any]:
        """
        Comprehensive quality assessment for a single session.
        Returns score (0-100) and detailed analysis.
        """
        score = 0
        analysis = {
            "file": session_file,
            "quality_score": 0,
            "category": "",
            "issues": [],
            "strengths": [],
            "metadata_quality": 0,
            "temporal_quality": 0, 
            "price_data_quality": 0,
            "feature_authenticity": 0,
            "tgat_readiness": False
        }
        
        # 1. Metadata Quality Assessment (25 points)
        metadata_score = self._assess_metadata_quality(session_data, analysis)
        score += metadata_score
        analysis["metadata_quality"] = metadata_score
        
        # 2. Temporal Coherence Assessment (25 points)
        temporal_score = self._assess_temporal_coherence(session_data, analysis)
        score += temporal_score
        analysis["temporal_quality"] = temporal_score
        
        # 3. Price Data Quality Assessment (25 points)
        price_score = self._assess_price_data_quality(session_data, analysis)
        score += price_score
        analysis["price_data_quality"] = price_score
        
        # 4. Feature Authenticity Assessment (25 points)
        authenticity_score = self._assess_feature_authenticity(session_data, analysis)
        score += authenticity_score
        analysis["feature_authenticity"] = authenticity_score
        
        # Final score and category
        analysis["quality_score"] = score
        analysis["category"] = self._categorize_quality(score)
        analysis["tgat_readiness"] = score >= 70  # Threshold for training readiness
        
        return analysis
    
    def _assess_metadata_quality(self, session_data: dict[str, Any], analysis: dict[str, Any]) -> int:
        """Assess session metadata completeness and consistency (0-25 points)"""
        score = 0
        
        metadata = session_data.get("session_metadata", {})
        
        # Required fields present (10 points)
        required_fields = ["session_type", "session_date", "session_start", "session_end"]
        present_fields = sum(1 for field in required_fields if metadata.get(field) and metadata.get(field) != "")
        score += (present_fields / len(required_fields)) * 10
        
        if present_fields < len(required_fields):
            analysis["issues"].append(f"Missing metadata fields: {set(required_fields) - set(metadata.keys())}")
        else:
            analysis["strengths"].append("Complete metadata fields")
        
        # Date format consistency (5 points)
        session_date = metadata.get("session_date", "")
        if session_date and len(session_date) >= 10:  # YYYY-MM-DD format
            try:
                datetime.strptime(session_date, "%Y-%m-%d")
                score += 5
                analysis["strengths"].append("Valid date format")
            except ValueError:
                analysis["issues"].append(f"Invalid date format: {session_date}")
        else:
            analysis["issues"].append("Missing or invalid session_date")
        
        # Session duration consistency (5 points)
        duration = metadata.get("session_duration", 0)
        if isinstance(duration, int | float) and duration > 0:
            score += 5
            analysis["strengths"].append(f"Valid session duration: {duration} minutes")
        else:
            analysis["issues"].append("Missing or invalid session_duration")
        
        # Data completeness indicator (5 points)
        completeness = metadata.get("data_completeness", "")
        if completeness == "complete_session":
            score += 5
            analysis["strengths"].append("Marked as complete session")
        else:
            analysis["issues"].append(f"Incomplete session marker: {completeness}")
        
        return score
    
    def _assess_temporal_coherence(self, session_data: dict[str, Any], analysis: dict[str, Any]) -> int:
        """Assess temporal relationships and chronological ordering (0-25 points)"""
        score = 0
        
        metadata = session_data.get("session_metadata", {})
        price_movements = session_data.get("price_movements", [])
        
        # Session time range validity (10 points)
        start_time = metadata.get("session_start", "")
        end_time = metadata.get("session_end", "")
        
        if start_time and end_time:
            try:
                start = datetime.strptime(start_time, "%H:%M:%S")
                end = datetime.strptime(end_time, "%H:%M:%S")
                if end > start or (end < start and end.hour < 12):  # Handle overnight sessions
                    score += 10
                    analysis["strengths"].append("Valid session time range")
                else:
                    analysis["issues"].append("Invalid session time range")
            except ValueError:
                analysis["issues"].append("Invalid time format in session metadata")
        else:
            analysis["issues"].append("Missing session start/end times")
        
        # Price movement temporal consistency (10 points)
        if price_movements:
            timestamps = [pm.get("timestamp", "") for pm in price_movements if pm.get("timestamp")]
            non_empty_timestamps = [ts for ts in timestamps if ts and ts != ""]
            
            if len(non_empty_timestamps) >= len(price_movements) * 0.8:  # 80% have timestamps
                score += 5
                analysis["strengths"].append("Most price movements have timestamps")
            else:
                analysis["issues"].append(f"Missing timestamps: {len(price_movements) - len(non_empty_timestamps)} of {len(price_movements)}")
            
            # Check for chronological ordering
            valid_timestamps = []
            for ts in non_empty_timestamps:
                try:
                    if ":" in ts:
                        parsed = datetime.strptime(ts, "%H:%M:%S")
                        valid_timestamps.append(parsed)
                except ValueError:
                    continue
            
            if len(valid_timestamps) > 1:
                is_ordered = all(valid_timestamps[i] <= valid_timestamps[i+1] for i in range(len(valid_timestamps)-1))
                if is_ordered:
                    score += 5
                    analysis["strengths"].append("Chronologically ordered price movements")
                else:
                    analysis["issues"].append("Price movements not chronologically ordered")
        
        # Phase transition temporal consistency (5 points)
        phase_transitions = session_data.get("phase_transitions", [])
        if phase_transitions:
            phase_times = []
            for phase in phase_transitions:
                start = phase.get("start_time", "")
                end = phase.get("end_time", "")
                if start and end:
                    try:
                        start_dt = datetime.strptime(start, "%H:%M:%S")
                        end_dt = datetime.strptime(end, "%H:%M:%S")
                        if end_dt > start_dt:
                            phase_times.append((start_dt, end_dt))
                    except ValueError:
                        continue
            
            if phase_times:
                # Check for non-overlapping phases
                phase_times.sort()
                non_overlapping = all(phase_times[i][1] <= phase_times[i+1][0] for i in range(len(phase_times)-1))
                if non_overlapping:
                    score += 5
                    analysis["strengths"].append("Non-overlapping phase transitions")
                else:
                    analysis["issues"].append("Overlapping phase transitions detected")
        
        return score
    
    def _assess_price_data_quality(self, session_data: dict[str, Any], analysis: dict[str, Any]) -> int:
        """Assess price data consistency and realism (0-25 points)"""
        score = 0
        
        price_movements = session_data.get("price_movements", [])
        
        # Price level presence and validity (10 points)
        price_levels = []
        for pm in price_movements:
            price = pm.get("price_level") or pm.get("price")
            if price and isinstance(price, int | float) and price > 0:
                price_levels.append(price)
        
        if price_levels:
            if len(price_levels) >= len(price_movements) * 0.9:  # 90% valid prices
                score += 10
                analysis["strengths"].append(f"Valid price data: {len(price_levels)} of {len(price_movements)} movements")
            else:
                score += 5
                analysis["issues"].append(f"Some missing price levels: {len(price_levels)} of {len(price_movements)}")
        else:
            analysis["issues"].append("No valid price levels found")
        
        # Price range realism (10 points)
        if len(price_levels) > 1:
            price_range = max(price_levels) - min(price_levels)
            median_price = statistics.median(price_levels)
            
            # Check if prices are in realistic range (assume futures market ~20k-30k range)
            if 15000 < median_price < 35000:
                score += 5
                analysis["strengths"].append(f"Realistic price range: {median_price:.1f} median")
            else:
                analysis["issues"].append(f"Unrealistic price range: {median_price:.1f} median")
            
            # Check for reasonable volatility
            if price_range > 0 and price_range < median_price * 0.1:  # Less than 10% range
                score += 5
                analysis["strengths"].append(f"Reasonable volatility: {price_range:.1f} point range")
            elif price_range > 0:
                score += 2  # Some range is better than no range
                analysis["issues"].append(f"High volatility session: {price_range:.1f} point range")
            else:
                analysis["issues"].append("No price variation detected")
        
        # Structure identification quality (5 points)
        structures = session_data.get("structures_identified", {})
        fvgs = structures.get("fair_value_gaps", [])
        session_levels = structures.get("session_levels", [])
        
        if fvgs or session_levels:
            score += 5
            analysis["strengths"].append(f"Market structures identified: {len(fvgs)} FVGs, {len(session_levels)} levels")
        else:
            analysis["issues"].append("No market structures identified")
        
        return score
    
    def _assess_feature_authenticity(self, session_data: dict[str, Any], analysis: dict[str, Any]) -> int:
        """Assess whether features are genuine vs artificial/default (0-25 points)"""
        score = 0
        
        # Energy state authenticity (10 points)
        energy_state = session_data.get("energy_state", {})
        
        # Check for non-default energy values
        energy_density = energy_state.get("energy_density")
        if energy_density != 0.5:  # Default value
            score += 3
            analysis["strengths"].append(f"Non-default energy density: {energy_density}")
        else:
            analysis["issues"].append("Default energy density value (0.5)")
        
        total_accumulated = energy_state.get("total_accumulated", 0)
        session_duration = energy_state.get("session_duration", 0)
        if total_accumulated > 0 and session_duration > 0:
            score += 3
            analysis["strengths"].append(f"Calculated energy accumulation: {total_accumulated}")
        else:
            analysis["issues"].append("Missing or zero energy accumulation")
        
        # Check for realistic phase counts
        expansion_phases = energy_state.get("expansion_phases", 0)
        consolidation_phases = energy_state.get("consolidation_phases", 0)
        if expansion_phases > 0 and consolidation_phases > 0:
            score += 4
            analysis["strengths"].append(f"Multiple phases identified: {expansion_phases} expansion, {consolidation_phases} consolidation")
        else:
            analysis["issues"].append("Missing or unrealistic phase counts")
        
        # Contamination analysis authenticity (10 points)
        contamination = session_data.get("contamination_analysis", {})
        htf_contamination = contamination.get("htf_contamination", {})
        
        htf_strength = htf_contamination.get("htf_carryover_strength")
        if htf_strength and htf_strength != 0.3:  # Default value
            score += 5
            analysis["strengths"].append(f"Non-default HTF carryover: {htf_strength}")
        else:
            analysis["issues"].append("Default HTF carryover strength (0.3)")
        
        # Cross-session interaction presence
        cross_interaction = htf_contamination.get("immediate_cross_session_interaction")
        if isinstance(cross_interaction, bool):
            score += 5
            analysis["strengths"].append(f"Cross-session interaction analyzed: {cross_interaction}")
        else:
            analysis["issues"].append("Missing cross-session interaction analysis")
        
        # Liquidity events presence (5 points)
        liquidity_events = session_data.get("session_liquidity_events", [])
        if liquidity_events:
            score += 5
            analysis["strengths"].append(f"Liquidity events recorded: {len(liquidity_events)}")
        else:
            analysis["issues"].append("No liquidity events recorded")
        
        return score
    
    def _categorize_quality(self, score: int) -> str:
        """Categorize session based on quality score"""
        if score >= 80:
            return "Complete"
        elif score >= 50:
            return "Partial"
        elif score >= 20:
            return "Artificial"
        else:
            return "Unusable"
    
    def analyze_all_sessions(self) -> dict[str, Any]:
        """Analyze all sessions and return comprehensive quality report"""
        all_files = []
        
        # Collect all JSON files
        for root, _dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.json'):
                    all_files.append(os.path.join(root, file))
        
        results = {
            "analysis_date": datetime.now().isoformat(),
            "total_sessions": len(all_files),
            "quality_distribution": {"Complete": 0, "Partial": 0, "Artificial": 0, "Unusable": 0},
            "session_assessments": [],
            "summary_statistics": {},
            "tgat_ready_sessions": []
        }
        
        scores = []
        
        for file_path in all_files:
            try:
                with open(file_path) as f:
                    session_data = json.load(f)
                
                session_name = os.path.basename(file_path)
                assessment = self.assess_session_quality(session_data, session_name)
                
                results["session_assessments"].append(assessment)
                results["quality_distribution"][assessment["category"]] += 1
                scores.append(assessment["quality_score"])
                
                if assessment["tgat_readiness"]:
                    results["tgat_ready_sessions"].append(session_name)
                
                print(f"Analyzed {session_name}: {assessment['quality_score']}/100 ({assessment['category']})")
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
                results["session_assessments"].append({
                    "file": os.path.basename(file_path),
                    "quality_score": 0,
                    "category": "Error",
                    "issues": [f"Failed to parse: {str(e)}"],
                    "tgat_readiness": False
                })
        
        # Summary statistics
        if scores:
            results["summary_statistics"] = {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
                "tgat_ready_percentage": (len(results["tgat_ready_sessions"]) / len(scores)) * 100
            }
        
        return results

def main():
    """Run complete quality assessment on all Level 1 sessions"""
    print("🔍 Starting Level 1 Session Quality Assessment...")
    print("=" * 60)
    
    assessor = SessionQualityAssessor()
    results = assessor.analyze_all_sessions()
    
    # Save detailed results
    output_path = "/Users/jack/IRONPULSE/IRONFORGE/data_quality_assessment.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 QUALITY ASSESSMENT SUMMARY")
    print("=" * 60)
    print(f"Total Sessions Analyzed: {results['total_sessions']}")
    print("\nQuality Distribution:")
    for category, count in results["quality_distribution"].items():
        percentage = (count / results["total_sessions"]) * 100
        print(f"  {category:12}: {count:3} sessions ({percentage:5.1f}%)")
    
    if results["summary_statistics"]:
        stats = results["summary_statistics"]
        print("\nScore Statistics:")
        print(f"  Mean Score     : {stats['mean_score']:5.1f}/100")
        print(f"  Median Score   : {stats['median_score']:5.1f}/100")
        print(f"  Score Range    : {stats['min_score']:3.0f}-{stats['max_score']:3.0f}")
        print(f"  Std Deviation  : {stats['std_deviation']:5.1f}")
    
    print(f"\n🎯 TGAT Ready Sessions: {len(results['tgat_ready_sessions'])} ({results['summary_statistics'].get('tgat_ready_percentage', 0):.1f}%)")
    
    if results["tgat_ready_sessions"]:
        print("\nTGAT-Ready Sessions:")
        for session in results["tgat_ready_sessions"][:10]:  # Show first 10
            print(f"  ✅ {session}")
        if len(results["tgat_ready_sessions"]) > 10:
            print(f"  ... and {len(results['tgat_ready_sessions']) - 10} more")
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    print("\n🔍 Ready for Phase 2: TGAT Model Quality Recovery")

if __name__ == "__main__":
    main()