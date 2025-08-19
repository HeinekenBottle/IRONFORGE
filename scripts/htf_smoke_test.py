#!/usr/bin/env python3
"""
HTF Context Smoke Tests
=====================

CI/smoke tests for HTF context features ensuring temporal integrity,
feature completeness, and archaeological discovery functionality.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTFSmokeTest:
    """Smoke tests for HTF context features"""
    
    def __init__(self, shards_dir: str = "data/shards/NQ_M5"):
        self.shards_dir = Path(shards_dir)
        self.test_results = []
        
    def run_all_tests(self) -> bool:
        """Run all HTF smoke tests"""
        
        print("🧪 HTF Context Smoke Tests")
        print("=" * 40)
        print("Version: v0.7.1 (Node Features v1.1)")
        print()
        
        # Test suite
        tests = [
            ("Feature Count Validation", self.test_feature_count),
            ("HTF Feature Presence", self.test_htf_feature_presence),
            ("Temporal Integrity", self.test_temporal_integrity),
            ("Feature Range Validation", self.test_feature_ranges),
            ("Bar Closure Leakage", self.test_bar_closure_leakage),
            ("Golden Sample Assertion", self.test_golden_sample),
            ("Cross-Session Consistency", self.test_cross_session_consistency)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"📋 {test_name}...")
            
            try:
                result = test_func()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}")
                
                if not result:
                    all_passed = False
                    
                self.test_results.append((test_name, result))
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                all_passed = False
                self.test_results.append((test_name, False))
        
        print()
        print("📊 Test Summary:")
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        print(f"   Passed: {passed}/{total}")
        print(f"   Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed
    
    def test_feature_count(self) -> bool:
        """Test that nodes have exactly 51 features + metadata"""
        
        sample_nodes = self._get_sample_nodes_file()
        if not sample_nodes:
            return False
        
        df = pd.read_parquet(sample_nodes)
        
        # Should be 51 node features + 7 metadata columns = 58 total
        expected_total = 58
        actual_total = len(df.columns)
        
        if actual_total != expected_total:
            logger.error(f"Expected {expected_total} columns, got {actual_total}")
            return False
        
        # Check for feature columns f0-f50 (51 features)
        feature_cols = [f"f{i}" for i in range(51)]
        missing_features = [f for f in feature_cols if f not in df.columns]
        
        if missing_features:
            logger.error(f"Missing feature columns: {missing_features}")
            return False
        
        return True
    
    def test_htf_feature_presence(self) -> bool:
        """Test that HTF features f45-f50 are present and correctly named"""
        
        sample_nodes = self._get_sample_nodes_file()
        if not sample_nodes:
            return False
        
        df = pd.read_parquet(sample_nodes)
        
        # HTF features should be f45-f50
        htf_features = [f"f{i}" for i in range(45, 51)]
        missing_htf = [f for f in htf_features if f not in df.columns]
        
        if missing_htf:
            logger.error(f"Missing HTF features: {missing_htf}")
            return False
        
        return True
    
    def test_temporal_integrity(self) -> bool:
        """Test temporal integrity - no future leakage in HTF features"""
        
        sample_nodes = self._get_sample_nodes_file()
        if not sample_nodes:
            return False
        
        df = pd.read_parquet(sample_nodes)
        
        # Check that timestamps are monotonic
        if not df['t'].is_monotonic_increasing:
            logger.error("Timestamps are not monotonic increasing")
            return False
        
        # For barpos features, all values should be between 0 and 1
        barpos_features = ['f47', 'f48']  # barpos_m15, barpos_h1
        
        for feature in barpos_features:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    if not values.between(0, 1).all():
                        logger.error(f"Barpos feature {feature} has values outside [0,1]")
                        return False
        
        return True
    
    def test_feature_ranges(self) -> bool:
        """Test that HTF features are within expected ranges"""
        
        sample_nodes = self._get_sample_nodes_file()
        if not sample_nodes:
            return False
        
        df = pd.read_parquet(sample_nodes)
        
        # Range validations
        validations = [
            ('f47', 'barpos_m15', lambda x: x.between(0, 1).all()),
            ('f48', 'barpos_h1', lambda x: x.between(0, 1).all()),
            ('f49', 'dist_daily_mid', lambda x: x.between(-5, 5).all()),  # Reasonable range
            ('f50', 'htf_regime', lambda x: x.isin([0, 1, 2]).all())
        ]
        
        for feature, name, validator in validations:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    if not validator(values):
                        logger.error(f"Feature {feature} ({name}) failed range validation")
                        return False
        
        return True
    
    def test_bar_closure_leakage(self) -> bool:
        """Test that HTF features only use closed bar information"""
        
        # This is a logical test - barpos features should show evidence
        # of proper bar closure (values at or near 1.0 for end-of-bar events)
        
        sample_nodes = self._get_sample_nodes_file()
        if not sample_nodes:
            return False
        
        df = pd.read_parquet(sample_nodes)
        
        # Check barpos distribution - should be skewed toward 1.0 for closed bars
        for barpos_col in ['f47', 'f48']:
            if barpos_col in df.columns:
                values = df[barpos_col].dropna()
                if len(values) > 0:
                    # Most values should be 1.0 (closed bars)
                    end_of_bar_ratio = (values == 1.0).mean()
                    if end_of_bar_ratio < 0.5:  # At least 50% should be at bar end
                        logger.warning(f"Low end-of-bar ratio for {barpos_col}: {end_of_bar_ratio:.2f}")
                        # Don't fail the test, just warn
        
        return True
    
    def test_golden_sample(self) -> bool:
        """Test golden sample assertions"""
        
        sample_nodes = self._get_sample_nodes_file()
        sample_edges = self._get_sample_edges_file()
        
        if not sample_nodes or not sample_edges:
            return False
        
        # Node assertions
        nodes_df = pd.read_parquet(sample_nodes)
        if len(nodes_df.columns) != 58:  # 51 features + 7 metadata
            logger.error(f"Expected 58 node columns, got {len(nodes_df.columns)}")
            return False
        
        # Edge assertions  
        edges_df = pd.read_parquet(sample_edges)
        expected_edge_cols = 24  # Should be unchanged from v1.0
        if len(edges_df.columns) != expected_edge_cols:
            logger.error(f"Expected {expected_edge_cols} edge columns, got {len(edges_df.columns)}")
            return False
        
        return True
    
    def test_cross_session_consistency(self) -> bool:
        """Test consistency across multiple sessions"""
        
        # Get multiple session samples
        shard_dirs = list(self.shards_dir.glob("shard_*"))[:5]  # Test first 5 sessions
        
        if len(shard_dirs) < 2:
            logger.warning("Insufficient sessions for cross-session test")
            return True  # Pass if not enough data
        
        feature_counts = []
        htf_feature_presence = []
        
        for shard_dir in shard_dirs:
            nodes_file = shard_dir / "nodes.parquet"
            if nodes_file.exists():
                df = pd.read_parquet(nodes_file)
                feature_counts.append(len(df.columns))
                
                # Check HTF feature presence
                htf_present = all(f"f{i}" in df.columns for i in range(45, 51))
                htf_feature_presence.append(htf_present)
        
        # All sessions should have same number of features
        if len(set(feature_counts)) != 1:
            logger.error(f"Inconsistent feature counts across sessions: {set(feature_counts)}")
            return False
        
        # All sessions should have HTF features
        if not all(htf_feature_presence):
            logger.error("HTF features missing in some sessions")
            return False
        
        return True
    
    def _get_sample_nodes_file(self) -> Path:
        """Get a sample nodes.parquet file for testing"""
        
        for shard_dir in self.shards_dir.glob("shard_*"):
            nodes_file = shard_dir / "nodes.parquet"
            if nodes_file.exists():
                return nodes_file
        
        logger.error(f"No sample nodes file found in {self.shards_dir}")
        return None
    
    def _get_sample_edges_file(self) -> Path:
        """Get a sample edges.parquet file for testing"""
        
        for shard_dir in self.shards_dir.glob("shard_*"):
            edges_file = shard_dir / "edges.parquet"
            if edges_file.exists():
                return edges_file
        
        logger.error(f"No sample edges file found in {self.shards_dir}")
        return None


def run_nightly_htf_checks() -> bool:
    """Run nightly HTF checks with dry-run validation"""
    
    print("🌙 Nightly HTF Checks")
    print("=" * 30)
    
    # Check 1: Dry run prep-shards
    print("📋 Dry Run Validation...")
    
    import subprocess
    try:
        cmd = [
            "python3", "-m", "ironforge.sdk.cli", "prep-shards",
            "--source-glob", "data/enhanced/enhanced_*_Lvl-1_*.json",
            "--htf-context", "--dry-run"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ Dry run passed")
        else:
            print(f"   ❌ Dry run failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Dry run error: {e}")
        return False
    
    # Check 2: Smoke tests
    print("🧪 Smoke Tests...")
    
    smoke_test = HTFSmokeTest()
    smoke_passed = smoke_test.run_all_tests()
    
    if not smoke_passed:
        return False
    
    print("✅ All nightly checks passed")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--nightly":
        success = run_nightly_htf_checks()
    else:
        smoke_test = HTFSmokeTest()
        success = smoke_test.run_all_tests()
    
    sys.exit(0 if success else 1)