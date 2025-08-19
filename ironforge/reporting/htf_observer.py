#!/usr/bin/env python3
"""
HTF Context Observability
========================

Light observability for HTF context features with run-level summaries
and regime monitoring for archaeological discovery insights.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HTFRunSummary:
    """Summary statistics for HTF features across a discovery run"""

    run_id: str
    total_sessions: int
    total_zones: int
    avg_confidence: float

    # HTF-specific metrics
    regime_distribution: Dict[str, int]  # {consolidation, transition, expansion}
    htf_feature_coverage: Dict[str, float]  # Coverage % for each HTF feature
    barpos_coherence: float  # Temporal coherence metric
    sv_anomaly_rate: float  # Rate of SV anomalies detected

    # Archaeological discovery metrics
    theory_b_zones: int
    dimensional_anchor_rate: float
    discovery_density_by_regime: Dict[str, float]

    # Quality metrics
    temporal_integrity_score: float
    feature_completeness_score: float
    overall_quality_score: float


class HTFObserver:
    """Light observability for HTF context features"""

    def __init__(self, output_dir: str = "runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Regime name mapping
        self.regime_names = {0: "consolidation", 1: "transition", 2: "expansion"}

    def analyze_htf_run(
        self, run_id: str, shards_dir: str, zones_data: Optional[List[Dict]] = None
    ) -> HTFRunSummary:
        """Analyze HTF features and archaeological discovery for a run"""

        logger.info(f"Analyzing HTF run: {run_id}")

        shards_path = Path(shards_dir)

        # Collect HTF statistics across all sessions
        session_stats = []
        total_zones = 0
        total_confidence = 0.0
        regime_counts = defaultdict(int)
        feature_coverage = defaultdict(list)
        theory_b_zones = 0

        # Process each shard
        for shard_dir in shards_path.glob("shard_*"):
            if not shard_dir.is_dir():
                continue

            session_stat = self._analyze_session_htf(shard_dir)
            if session_stat:
                session_stats.append(session_stat)

                # Aggregate regime distribution
                for regime_code, count in session_stat.get("regime_counts", {}).items():
                    regime_name = self.regime_names.get(regime_code, f"unknown_{regime_code}")
                    regime_counts[regime_name] += count

                # Aggregate feature coverage
                for feature, coverage in session_stat.get("feature_coverage", {}).items():
                    feature_coverage[feature].append(coverage)

        # Process zones data if provided
        if zones_data:
            total_zones = len(zones_data)
            if zones_data:
                confidences = [z.get("confidence", 0.0) for z in zones_data]
                total_confidence = np.mean(confidences) if confidences else 0.0

                # Count Theory B zones
                theory_b_zones = len(
                    [z for z in zones_data if z.get("theoretical_basis") == "Theory B"]
                )

        # Calculate summary metrics
        total_sessions = len(session_stats)
        avg_confidence = total_confidence

        # Average feature coverage
        avg_feature_coverage = {}
        for feature, coverages in feature_coverage.items():
            avg_feature_coverage[feature] = np.mean(coverages) if coverages else 0.0

        # Calculate quality scores
        temporal_integrity_score = self._calculate_temporal_integrity_score(session_stats)
        feature_completeness_score = self._calculate_feature_completeness_score(
            avg_feature_coverage
        )
        overall_quality_score = (temporal_integrity_score + feature_completeness_score) / 2

        # Discovery density by regime
        discovery_density = self._calculate_discovery_density_by_regime(regime_counts, total_zones)

        # Create summary
        summary = HTFRunSummary(
            run_id=run_id,
            total_sessions=total_sessions,
            total_zones=total_zones,
            avg_confidence=avg_confidence,
            regime_distribution=dict(regime_counts),
            htf_feature_coverage=avg_feature_coverage,
            barpos_coherence=self._calculate_barpos_coherence(session_stats),
            sv_anomaly_rate=self._calculate_sv_anomaly_rate(session_stats),
            theory_b_zones=theory_b_zones,
            dimensional_anchor_rate=theory_b_zones / max(1, total_zones),
            discovery_density_by_regime=discovery_density,
            temporal_integrity_score=temporal_integrity_score,
            feature_completeness_score=feature_completeness_score,
            overall_quality_score=overall_quality_score,
        )

        # Save summary
        self._save_run_summary(summary)

        return summary

    def _analyze_session_htf(self, shard_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze HTF features for a single session"""

        nodes_file = shard_dir / "nodes.parquet"
        if not nodes_file.exists():
            return None

        try:
            import pandas as pd

            df = pd.read_parquet(nodes_file)

            # Check for HTF features
            htf_features = ["f45", "f46", "f47", "f48", "f49", "f50"]
            if not all(f in df.columns for f in htf_features):
                logger.warning(f"Missing HTF features in {shard_dir.name}")
                return None

            # Analyze regime distribution
            regime_counts = df["f50"].value_counts().to_dict()

            # Calculate feature coverage
            feature_coverage = {}
            feature_names = [
                "sv_m15_z",
                "sv_h1_z",
                "barpos_m15",
                "barpos_h1",
                "dist_daily_mid",
                "htf_regime",
            ]

            for feature, name in zip(htf_features, feature_names):
                non_nan_count = df[feature].notna().sum()
                coverage = non_nan_count / len(df) if len(df) > 0 else 0.0
                feature_coverage[name] = coverage

            # Calculate barpos statistics
            barpos_m15_values = df["f47"].dropna()
            barpos_h1_values = df["f48"].dropna()
            barpos_variance = np.var(list(barpos_m15_values) + list(barpos_h1_values))

            # SV anomaly detection (z-score > 1.5)
            sv_anomalies = 0
            for sv_feature in ["f45", "f46"]:
                sv_values = df[sv_feature].dropna()
                anomalies = (np.abs(sv_values) > 1.5).sum()
                sv_anomalies += anomalies

            return {
                "session_id": shard_dir.name,
                "node_count": len(df),
                "regime_counts": regime_counts,
                "feature_coverage": feature_coverage,
                "barpos_variance": barpos_variance,
                "sv_anomalies": sv_anomalies,
            }

        except Exception as e:
            logger.error(f"Error analyzing {shard_dir.name}: {e}")
            return None

    def _calculate_temporal_integrity_score(self, session_stats: List[Dict]) -> float:
        """Calculate temporal integrity score across sessions"""

        if not session_stats:
            return 0.0

        # Score based on barpos coherence and feature consistency
        barpos_variances = [s.get("barpos_variance", 1.0) for s in session_stats]
        avg_variance = np.mean(barpos_variances)

        # Lower variance = higher integrity (inverted scoring)
        integrity_score = 1.0 / (1.0 + avg_variance)

        return min(1.0, integrity_score)

    def _calculate_feature_completeness_score(self, feature_coverage: Dict[str, float]) -> float:
        """Calculate feature completeness score"""

        if not feature_coverage:
            return 0.0

        # Weight different features
        weights = {
            "sv_m15_z": 0.15,  # Lower weight (often NaN for single sessions)
            "sv_h1_z": 0.15,  # Lower weight (often NaN for single sessions)
            "barpos_m15": 0.25,  # High weight (should always be present)
            "barpos_h1": 0.25,  # High weight (should always be present)
            "dist_daily_mid": 0.1,  # Medium weight
            "htf_regime": 0.1,  # Medium weight
        }

        weighted_score = 0.0
        total_weight = 0.0

        for feature, coverage in feature_coverage.items():
            weight = weights.get(feature, 0.1)
            weighted_score += coverage * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _calculate_barpos_coherence(self, session_stats: List[Dict]) -> float:
        """Calculate overall barpos coherence metric"""

        if not session_stats:
            return 0.0

        variances = [s.get("barpos_variance", 1.0) for s in session_stats]
        avg_variance = np.mean(variances)

        # Convert variance to coherence (0-1 scale)
        coherence = 1.0 / (1.0 + avg_variance)
        return min(1.0, coherence)

    def _calculate_sv_anomaly_rate(self, session_stats: List[Dict]) -> float:
        """Calculate SV anomaly rate across sessions"""

        if not session_stats:
            return 0.0

        total_anomalies = sum(s.get("sv_anomalies", 0) for s in session_stats)
        total_nodes = sum(s.get("node_count", 0) for s in session_stats)

        return total_anomalies / total_nodes if total_nodes > 0 else 0.0

    def _calculate_discovery_density_by_regime(
        self, regime_counts: Dict[str, int], total_zones: int
    ) -> Dict[str, float]:
        """Calculate archaeological discovery density by regime"""

        total_events = sum(regime_counts.values())
        if total_events == 0:
            return {}

        density = {}
        for regime, count in regime_counts.items():
            regime_proportion = count / total_events
            expected_zones = regime_proportion * total_zones
            actual_density = expected_zones / count if count > 0 else 0.0
            density[regime] = actual_density

        return density

    def _save_run_summary(self, summary: HTFRunSummary) -> None:
        """Save run summary to JSON file"""

        output_file = self.output_dir / f"htf_summary_{summary.run_id}.json"

        try:
            with open(output_file, "w") as f:
                json.dump(asdict(summary), f, indent=2)

            logger.info(f"HTF summary saved: {output_file}")

        except Exception as e:
            logger.error(f"Error saving HTF summary: {e}")

    def generate_regime_ribbon_data(self, run_id: str) -> Dict[str, Any]:
        """Generate data for minidash regime ribbon visualization"""

        summary_file = self.output_dir / f"htf_summary_{run_id}.json"
        if not summary_file.exists():
            return {}

        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)

            # Prepare regime ribbon data
            regime_data = {
                "regime_distribution": summary.get("regime_distribution", {}),
                "discovery_density": summary.get("discovery_density_by_regime", {}),
                "quality_score": summary.get("overall_quality_score", 0.0),
                "htf_enabled": True,
                "total_zones": summary.get("total_zones", 0),
                "theory_b_zones": summary.get("theory_b_zones", 0),
            }

            return regime_data

        except Exception as e:
            logger.error(f"Error generating regime ribbon data: {e}")
            return {}

    def print_run_summary(self, summary: HTFRunSummary) -> None:
        """Print formatted run summary"""

        print(f"🏛️ HTF Run Summary: {summary.run_id}")
        print("=" * 50)
        print(f"📊 Sessions: {summary.total_sessions}")
        print(f"🎯 Zones: {summary.total_zones} (avg confidence: {summary.avg_confidence:.2f})")
        print(
            f"🏺 Theory B zones: {summary.theory_b_zones} ({summary.dimensional_anchor_rate:.1%})"
        )
        print()

        print("📈 Regime Distribution:")
        for regime, count in summary.regime_distribution.items():
            print(f"   {regime.title()}: {count} events")
        print()

        print("⚡ HTF Feature Coverage:")
        for feature, coverage in summary.htf_feature_coverage.items():
            print(f"   {feature}: {coverage:.1%}")
        print()

        print("🔍 Quality Metrics:")
        print(f"   Temporal Integrity: {summary.temporal_integrity_score:.2f}")
        print(f"   Feature Completeness: {summary.feature_completeness_score:.2f}")
        print(f"   Overall Quality: {summary.overall_quality_score:.2f}")
        print()

        print("🏛️ Archaeological Discovery:")
        for regime, density in summary.discovery_density_by_regime.items():
            print(f"   {regime.title()} density: {density:.3f}")


# Quick test function
def test_htf_observer():
    """Test HTF observer with sample data"""

    observer = HTFObserver()

    # Test with existing shards
    shards_dir = "/Users/jack/IRONFORGE/data/shards/NQ_M5"

    # Mock zones data
    mock_zones = [
        {"confidence": 0.85, "theoretical_basis": "Theory B"},
        {"confidence": 0.92, "theoretical_basis": "HTF Transition Theory"},
        {"confidence": 0.78, "theoretical_basis": "Theory B"},
    ]

    summary = observer.analyze_htf_run("test_run_001", shards_dir, mock_zones)
    observer.print_run_summary(summary)

    return summary


if __name__ == "__main__":
    test_htf_observer()
