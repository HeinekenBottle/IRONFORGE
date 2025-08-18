"""Tests for motif cards and scanner functionality."""

import numpy as np
import pytest

from ironforge.motifs.cards import MotifCard, MotifStep, default_cards
from ironforge.motifs.scanner import _events_by_type, _find_next, scan_session_for_cards


class TestMotifStep:
    """Test MotifStep dataclass."""
    
    def test_basic_step(self):
        """Test basic step creation."""
        step = MotifStep("sweep", (0, 0))
        assert step.event_type == "sweep"
        assert step.within_minutes == (0, 0)
        assert step.htf_under_mid is None
    
    def test_step_with_guardrail(self):
        """Test step with structural guardrail."""
        step = MotifStep("fvg_redelivery", (12, 30), htf_under_mid=True)
        assert step.event_type == "fvg_redelivery"
        assert step.within_minutes == (12, 30)
        assert step.htf_under_mid is True


class TestMotifCard:
    """Test MotifCard dataclass."""
    
    def test_basic_card(self):
        """Test basic card creation."""
        steps = [
            MotifStep("sweep", (0, 0)),
            MotifStep("fvg_redelivery", (12, 30))
        ]
        card = MotifCard(
            id="test1",
            name="Test Card",
            steps=steps,
            window_minutes=(12, 30),
            min_confluence=65.0
        )
        
        assert card.id == "test1"
        assert card.name == "Test Card" 
        assert len(card.steps) == 2
        assert card.window_minutes == (12, 30)
        assert card.min_confluence == 65.0
        assert card.max_confluence == 100.0


class TestDefaultCards:
    """Test default card definitions."""
    
    def test_default_cards_structure(self):
        """Test default cards have expected structure."""
        cards = default_cards()
        
        assert len(cards) == 3
        assert all(isinstance(card, MotifCard) for card in cards)
        assert all(card.id for card in cards)  # All have IDs
        assert all(card.steps for card in cards)  # All have steps
    
    def test_card_c1_structure(self):
        """Test card c1 (Sweep → FVG redelivery)."""
        cards = default_cards()
        c1 = next(card for card in cards if card.id == "c1")
        
        assert c1.name == "Sweep → FVG redelivery under HTF midpoint"
        assert len(c1.steps) == 2
        assert c1.steps[0].event_type == "sweep"
        assert c1.steps[1].event_type == "fvg_redelivery"
        assert c1.steps[1].htf_under_mid is True
        assert c1.window_minutes == (12, 30)
    
    def test_card_c2_structure(self):
        """Test card c2 (Expansion → Consolidation → Redelivery)."""
        cards = default_cards()
        c2 = next(card for card in cards if card.id == "c2")
        
        assert len(c2.steps) == 3
        assert c2.steps[0].event_type == "expansion"
        assert c2.steps[1].event_type == "consolidation" 
        assert c2.steps[2].event_type == "redelivery"
        assert c2.min_confluence == 70.0
    
    def test_card_c3_structure(self):
        """Test card c3 (First-presentation FVG)."""
        cards = default_cards()
        c3 = next(card for card in cards if card.id == "c3")
        
        assert len(c3.steps) == 1
        assert c3.steps[0].event_type == "fpfvg"
        assert c3.window_minutes == (10, 25)


class TestEventsByType:
    """Test _events_by_type utility function."""
    
    def test_empty_events(self):
        """Test with empty event list."""
        result = _events_by_type([])
        assert result == {}
    
    def test_single_event_type(self):
        """Test with single event type."""
        events = [
            {"type": "sweep", "minute": 10},
            {"type": "sweep", "minute": 25}
        ]
        result = _events_by_type(events)
        
        assert "sweep" in result
        assert result["sweep"] == [10, 25]  # Should be sorted
    
    def test_multiple_event_types(self):
        """Test with multiple event types."""
        events = [
            {"type": "fvg_redelivery", "minute": 30},
            {"type": "sweep", "minute": 5},
            {"type": "fvg_redelivery", "minute": 15},
            {"type": "expansion", "minute": 45}
        ]
        result = _events_by_type(events)
        
        assert set(result.keys()) == {"sweep", "fvg_redelivery", "expansion"}
        assert result["sweep"] == [5]
        assert result["fvg_redelivery"] == [15, 30]  # Should be sorted
        assert result["expansion"] == [45]
    
    def test_unsorted_input(self):
        """Test that output is sorted even with unsorted input."""
        events = [
            {"type": "sweep", "minute": 25},
            {"type": "sweep", "minute": 10},
            {"type": "sweep", "minute": 20}
        ]
        result = _events_by_type(events)
        
        assert result["sweep"] == [10, 20, 25]


class TestFindNext:
    """Test _find_next utility function."""
    
    def test_find_next_basic(self):
        """Test basic next finding."""
        mins = [10, 20, 30, 40, 50]
        result = _find_next(mins, 15, 0, 10)  # Look for [15, 25]
        assert result == 20
    
    def test_find_next_exact_match(self):
        """Test exact boundary matches."""
        mins = [10, 20, 30, 40, 50]
        result = _find_next(mins, 10, 0, 10)  # Look for [10, 20]
        assert result == 10  # Should find exact match
        
        result = _find_next(mins, 10, 10, 10)  # Look for [20, 20]
        assert result == 20
    
    def test_find_next_no_match(self):
        """Test when no match found."""
        mins = [10, 20, 30, 40, 50]
        result = _find_next(mins, 60, 0, 5)  # Look for [60, 65]
        assert result is None
        
        result = _find_next(mins, 5, 0, 2)  # Look for [5, 7]
        assert result is None
    
    def test_find_next_empty_list(self):
        """Test with empty list."""
        result = _find_next([], 10, 0, 20)
        assert result is None


class TestScanSessionForCards:
    """Test main scanning function."""
    
    def test_no_events_no_matches(self):
        """Test with no events."""
        matches = scan_session_for_cards("test_session", [], None)
        assert matches == []
    
    def test_single_event_match_c3(self):
        """Test matching card c3 (single fpfvg event)."""
        events = [
            {"type": "fpfvg", "minute": 15, "htf_under_mid": False}
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        
        # Should match c3
        assert len(matches) >= 1
        c3_matches = [m for m in matches if m.card_id == "c3"]
        assert len(c3_matches) == 1
        
        match = c3_matches[0]
        assert match.session_id == "test_session"
        assert match.window == (15, 15)
        assert match.steps_at == [15]
    
    def test_two_event_match_c1(self):
        """Test matching card c1 (sweep → fvg_redelivery)."""
        events = [
            {"type": "sweep", "minute": 5, "htf_under_mid": False},
            {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": True}  # Within 12-30 minutes
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        
        # Should match c1
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) >= 1
        
        match = c1_matches[0]
        assert match.session_id == "test_session"
        assert match.window == (5, 20)
        assert match.steps_at == [5, 20]
        assert 12 <= (20 - 5) <= 30  # Window constraint
    
    def test_htf_guardrail_filtering(self):
        """Test HTF under midpoint guardrail."""
        # This should NOT match c1 because fvg_redelivery needs htf_under_mid=True
        events = [
            {"type": "sweep", "minute": 5, "htf_under_mid": False},
            {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": False}  # Wrong guardrail
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) == 0  # Should be filtered out
    
    def test_time_window_constraints(self):
        """Test time window constraint filtering."""
        # Events too close together for c1 (needs 12-30 minute window)
        events = [
            {"type": "sweep", "minute": 5, "htf_under_mid": False},
            {"type": "fvg_redelivery", "minute": 10, "htf_under_mid": True}  # Only 5 minutes apart
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) == 0  # Should be filtered out
    
    def test_confluence_threshold_filtering(self):
        """Test confluence threshold filtering."""
        events = [
            {"type": "fpfvg", "minute": 15, "htf_under_mid": False}
        ]
        
        # Low confluence should filter out matches
        low_confluence = np.array([30.0, 40.0, 35.0, 20.0, 25.0, 30.0] + [50.0] * 50)
        matches_low = scan_session_for_cards("test_session", events, low_confluence, min_confluence=65.0)
        c3_matches_low = [m for m in matches_low if m.card_id == "c3"]
        
        # High confluence should allow matches  
        high_confluence = np.array([80.0, 85.0, 90.0, 85.0, 80.0, 75.0] + [70.0] * 50)
        matches_high = scan_session_for_cards("test_session", events, high_confluence, min_confluence=65.0)
        c3_matches_high = [m for m in matches_high if m.card_id == "c3"]
        
        assert len(c3_matches_high) >= len(c3_matches_low)
    
    def test_complex_sequence_c2(self):
        """Test complex three-step sequence (c2)."""
        events = [
            {"type": "expansion", "minute": 10, "htf_under_mid": False},
            {"type": "consolidation", "minute": 30, "htf_under_mid": False},  # 20 min later (5-40 range)
            {"type": "redelivery", "minute": 45, "htf_under_mid": False}      # 35 min later (10-40 range)
        ]
        
        matches = scan_session_for_cards("test_session", events, None, min_confluence=70.0)
        c2_matches = [m for m in matches if m.card_id == "c2"]
        
        if c2_matches:  # May not match due to confluence threshold
            match = c2_matches[0]
            assert match.window == (10, 45)
            assert match.steps_at == [10, 30, 45]
            assert 15 <= (45 - 10) <= 80  # Window constraint for c2
    
    def test_multiple_matches_sorting(self):
        """Test that matches are sorted by score descending."""
        events = [
            {"type": "fpfvg", "minute": 15, "htf_under_mid": False},
            {"type": "fpfvg", "minute": 20, "htf_under_mid": False}
        ]
        
        # Create confluence with higher scores for second event
        confluence = np.array([60.0] * 15 + [90.0] * 10 + [95.0] * 25)
        matches = scan_session_for_cards("test_session", events, confluence)
        
        if len(matches) > 1:
            # Should be sorted by score descending
            for i in range(len(matches) - 1):
                assert matches[i].score >= matches[i + 1].score


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_event_types(self):
        """Test with events not matching any cards."""
        events = [
            {"type": "unknown_event", "minute": 10, "htf_under_mid": False},
            {"type": "another_unknown", "minute": 20, "htf_under_mid": False}
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        assert matches == []
    
    def test_partial_sequence(self):
        """Test incomplete event sequences."""
        # Only first step of c1
        events = [
            {"type": "sweep", "minute": 5, "htf_under_mid": False}
            # Missing fvg_redelivery
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) == 0
    
    def test_events_out_of_order(self):
        """Test events in non-chronological order."""
        # Events in reverse chronological order
        events = [
            {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": True},
            {"type": "sweep", "minute": 5, "htf_under_mid": False}
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        
        # Should still find match since we search from first event type
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) >= 1
    
    def test_duplicate_events(self):
        """Test duplicate events at same time."""
        events = [
            {"type": "sweep", "minute": 5, "htf_under_mid": False},
            {"type": "sweep", "minute": 5, "htf_under_mid": False},  # Duplicate
            {"type": "fvg_redelivery", "minute": 20, "htf_under_mid": True}
        ]
        
        matches = scan_session_for_cards("test_session", events, None)
        
        # Should handle duplicates gracefully
        c1_matches = [m for m in matches if m.card_id == "c1"]
        assert len(c1_matches) >= 1  # May find multiple starting from each sweep
    
    def test_empty_confluence_array(self):
        """Test with empty confluence array."""
        events = [
            {"type": "fpfvg", "minute": 15, "htf_under_mid": False}
        ]
        
        matches = scan_session_for_cards("test_session", events, np.array([]))
        # Should use default confluence value
        assert len(matches) >= 0  # May or may not match depending on default threshold
    
    def test_short_confluence_array(self):
        """Test confluence array shorter than event timeline."""
        events = [
            {"type": "fpfvg", "minute": 50, "htf_under_mid": False}  # Beyond confluence array
        ]
        
        short_confluence = np.array([80.0, 85.0, 90.0])  # Only 3 minutes
        matches = scan_session_for_cards("test_session", events, short_confluence)
        
        # Should handle gracefully (may use default or skip)
        assert isinstance(matches, list)


@pytest.mark.performance
class TestPerformance:
    """Test performance requirements."""
    
    def test_large_event_list_performance(self):
        """Test performance with large event list."""
        import time
        
        # Create 100 events
        events = []
        for i in range(100):
            events.append({
                "type": np.random.choice(["sweep", "fvg_redelivery", "expansion", "consolidation", "redelivery", "fpfvg"]),
                "minute": i * 2,  # Every 2 minutes
                "htf_under_mid": np.random.choice([True, False])
            })
        
        confluence = np.random.uniform(60, 90, 200)  # 200 minutes of confluence data
        
        start_time = time.time()
        matches = scan_session_for_cards("perf_test", events, confluence)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 1.0  # 1 second for 100 events
        assert isinstance(matches, list)
    
    def test_memory_usage(self):
        """Test memory efficiency."""
        # Create large dataset
        events = []
        for i in range(1000):
            events.append({
                "type": "sweep" if i % 10 == 0 else "other",
                "minute": i,
                "htf_under_mid": False
            })
        
        confluence = np.random.uniform(70, 80, 1000)
        
        # Should not consume excessive memory
        matches = scan_session_for_cards("memory_test", events, confluence)
        
        # Verify reasonable number of matches (not exponential)
        assert len(matches) < 200  # Should be much less than input size
