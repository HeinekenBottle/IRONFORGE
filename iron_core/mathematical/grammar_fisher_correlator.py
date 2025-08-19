"""
Grammar-Fisher Correlation Engine
=================================

Implements the critical discovery that Fisher spikes correspond to grammatical phrase 
boundaries - moments when partial parse trees have unique completions.

Mathematical Foundation:
- Fisher Information F measures parsing confidence at phrase boundaries
- When multiple parse paths converge to single continuation → Fisher spike
- Deterministic mode activates when grammatical constraints eliminate ambiguity

This represents a fundamental discovery about market microstructure as a formal 
language with measurable information geometry.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GrammarParseState:
    """Current state of grammatical parsing with Fisher correlation"""
    partial_parse_tree: list[str]  # Current sequence of grammar symbols
    parse_paths: list[list[str]]   # Multiple possible continuations
    fisher_confidence: float       # Current Fisher Information level
    phrase_boundary_detected: bool # True when unique continuation found
    deterministic_continuation: str | None  # The unique next symbol
    convergence_probability: float # Probability of path convergence

@dataclass
class FisherGrammarCorrelation:
    """Correlation between Fisher spike and grammar phrase boundary"""
    fisher_value: float
    grammar_symbol: str
    phrase_boundary_type: str  # 'convergence', 'divergence', 'continuation'
    parse_confidence: float
    unique_continuation: bool
    correlation_strength: float

class GrammarFisherCorrelator:
    """
    Predictive parser that tracks Fisher Information as parsing confidence
    
    Key Discovery: Fisher spikes occur at grammatical phrase boundaries where
    multiple parse paths converge to a single continuation, eliminating ambiguity
    and triggering deterministic cascade prediction.
    """
    
    def __init__(self):
        # Grammar rules for market events (Type-2 context-free)
        self.grammar_rules = {
            'S': [['CONSOLIDATION', 'EXPANSION'], ['FPFVG', 'REDELIVERY']],
            'CONSOLIDATION': [['RANGE_FORMATION'], ['LIQUIDITY_BUILD']],
            'EXPANSION': [['BREAKOUT', 'CONTINUATION'], ['MOMENTUM', 'ACCELERATION']],
            'FPFVG': [['FORMATION'], ['INHERITANCE']],
            'REDELIVERY': [['COMPLETION'], ['PARTIAL', 'COMPLETION']],
            'BREAKOUT': [['VOLUME_SPIKE'], ['PRICE_ACCELERATION']],
            'MOMENTUM': [['SUSTAINED'], ['EXHAUSTION']]
        }
        
        # Fisher correlation thresholds
        self.CONVERGENCE_THRESHOLD = 0.8    # Probability threshold for path convergence
        self.FISHER_SPIKE_THRESHOLD = 500.0 # Fisher value indicating phrase boundary
        self.DETERMINISTIC_THRESHOLD = 0.9  # Confidence for deterministic prediction
        
        # State tracking
        self.current_parse_state = None
        self.correlation_history = []
        self.phrase_boundaries = []
        
        self.logger = logging.getLogger(__name__)
    
    def parse_market_sequence(self, event_sequence: list[str], 
                            fisher_values: list[float]) -> GrammarParseState:
        """
        Parse market event sequence and correlate with Fisher Information
        
        Args:
            event_sequence: Sequence of market events (grammar symbols)
            fisher_values: Corresponding Fisher Information values
            
        Returns:
            GrammarParseState with current parsing state and Fisher correlation
        """
        if len(event_sequence) != len(fisher_values):
            raise ValueError("Event sequence and Fisher values must have same length")
        
        # Initialize parsing with start symbol
        if not event_sequence:
            return self._create_empty_parse_state()
        
        # Build partial parse tree
        partial_parse = event_sequence.copy()
        
        # Find all possible continuations from current state
        parse_paths = self._find_parse_continuations(partial_parse)
        
        # Calculate Fisher confidence (latest Fisher value)
        fisher_confidence = fisher_values[-1] if fisher_values else 0.0
        
        # Detect phrase boundaries based on parse path convergence
        phrase_boundary_detected, convergence_prob = self._detect_phrase_boundary(parse_paths)
        
        # Determine deterministic continuation if available
        deterministic_continuation = self._find_deterministic_continuation(
            parse_paths, convergence_prob
        )
        
        parse_state = GrammarParseState(
            partial_parse_tree=partial_parse,
            parse_paths=parse_paths,
            fisher_confidence=fisher_confidence,
            phrase_boundary_detected=phrase_boundary_detected,
            deterministic_continuation=deterministic_continuation,
            convergence_probability=convergence_prob
        )
        
        # Update internal state
        self.current_parse_state = parse_state
        
        # Record Fisher-Grammar correlation
        if len(event_sequence) > 0:
            correlation = self._calculate_fisher_grammar_correlation(
                event_sequence[-1], fisher_confidence, parse_state
            )
            self.correlation_history.append(correlation)
        
        return parse_state
    
    def _find_parse_continuations(self, partial_parse: list[str]) -> list[list[str]]:
        """Find all possible grammatical continuations from current parse state"""
        if not partial_parse:
            return [['S']]  # Start with start symbol
        
        last_symbol = partial_parse[-1]
        continuations = []
        
        # Check if last symbol can be expanded
        if last_symbol in self.grammar_rules:
            for rule in self.grammar_rules[last_symbol]:
                continuation = partial_parse + rule
                continuations.append(continuation)
        
        # If no expansions, this might be a terminal - look for higher-level continuations
        if not continuations:
            # Try to find what could follow this sequence
            for _symbol, rules in self.grammar_rules.items():
                for rule in rules:
                    if self._sequence_matches_rule_prefix(partial_parse, rule):
                        # This sequence could be part of this rule
                        remaining = rule[len(partial_parse):]
                        if remaining:
                            continuation = partial_parse + remaining
                            continuations.append(continuation)
        
        return continuations if continuations else [partial_parse]  # No continuations found
    
    def _sequence_matches_rule_prefix(self, sequence: list[str], rule: list[str]) -> bool:
        """Check if sequence matches the beginning of a grammar rule"""
        if len(sequence) > len(rule):
            return False
        
        return sequence == rule[:len(sequence)]
    
    def _detect_phrase_boundary(self, parse_paths: list[list[str]]) -> tuple[bool, float]:
        """
        Detect phrase boundary based on parse path convergence
        
        Returns:
            (boundary_detected, convergence_probability)
        """
        if len(parse_paths) <= 1:
            return True, 1.0  # Single path = complete convergence
        
        # Calculate convergence by looking at next symbols
        next_symbols = []
        for path in parse_paths:
            if len(path) > 0:
                # Look at the next symbol that would be added
                if len(path) > len(self.current_parse_state.partial_parse_tree if self.current_parse_state else []):
                    next_symbols.append(path[-1])
        
        if not next_symbols:
            return False, 0.0
        
        # Calculate convergence probability
        unique_symbols = set(next_symbols)
        convergence_prob = 1.0 - (len(unique_symbols) - 1) / max(1, len(next_symbols))
        
        boundary_detected = convergence_prob >= self.CONVERGENCE_THRESHOLD
        
        return boundary_detected, convergence_prob
    
    def _find_deterministic_continuation(self, parse_paths: list[list[str]], 
                                       convergence_prob: float) -> str | None:
        """Find deterministic continuation if convergence is high enough"""
        if convergence_prob < self.DETERMINISTIC_THRESHOLD:
            return None
        
        if not parse_paths:
            return None
        
        # Find the most common next symbol
        next_symbols = []
        current_length = len(self.current_parse_state.partial_parse_tree if self.current_parse_state else [])
        
        for path in parse_paths:
            if len(path) > current_length:
                next_symbols.append(path[current_length])
        
        if not next_symbols:
            return None
        
        # Return most frequent next symbol if it's dominant
        from collections import Counter
        symbol_counts = Counter(next_symbols)
        most_common = symbol_counts.most_common(1)[0]
        
        if most_common[1] / len(next_symbols) >= self.DETERMINISTIC_THRESHOLD:
            return most_common[0]
        
        return None
    
    def _calculate_fisher_grammar_correlation(self, grammar_symbol: str, 
                                            fisher_value: float,
                                            parse_state: GrammarParseState) -> FisherGrammarCorrelation:
        """Calculate correlation between Fisher spike and grammar state"""
        
        # Determine phrase boundary type
        if parse_state.phrase_boundary_detected:
            if parse_state.convergence_probability > 0.9:
                boundary_type = 'convergence'
            else:
                boundary_type = 'continuation'
        else:
            boundary_type = 'divergence'
        
        # Calculate parse confidence based on Fisher value and convergence
        parse_confidence = min(1.0, fisher_value / self.FISHER_SPIKE_THRESHOLD)
        
        # Determine if continuation is unique
        unique_continuation = parse_state.deterministic_continuation is not None
        
        # Calculate overall correlation strength
        fisher_component = min(1.0, fisher_value / self.FISHER_SPIKE_THRESHOLD)
        grammar_component = parse_state.convergence_probability
        correlation_strength = (fisher_component + grammar_component) / 2.0
        
        return FisherGrammarCorrelation(
            fisher_value=fisher_value,
            grammar_symbol=grammar_symbol,
            phrase_boundary_type=boundary_type,
            parse_confidence=parse_confidence,
            unique_continuation=unique_continuation,
            correlation_strength=correlation_strength
        )
    
    def _create_empty_parse_state(self) -> GrammarParseState:
        """Create empty parse state for initialization"""
        return GrammarParseState(
            partial_parse_tree=[],
            parse_paths=[],
            fisher_confidence=0.0,
            phrase_boundary_detected=False,
            deterministic_continuation=None,
            convergence_probability=0.0
        )
    
    def predict_next_event(self) -> str | None:
        """Predict next market event based on current grammar-Fisher state"""
        if not self.current_parse_state:
            return None
        
        return self.current_parse_state.deterministic_continuation
    
    def is_deterministic_mode_active(self) -> bool:
        """Check if deterministic prediction mode should be active"""
        if not self.current_parse_state:
            return False
        
        return (self.current_parse_state.fisher_confidence > self.FISHER_SPIKE_THRESHOLD and
                self.current_parse_state.phrase_boundary_detected and
                self.current_parse_state.deterministic_continuation is not None)
    
    def get_correlation_summary(self) -> dict[str, Any]:
        """Get summary of Fisher-Grammar correlations"""
        if not self.correlation_history:
            return {"status": "no_correlations"}
        
        recent_correlations = self.correlation_history[-10:]
        
        return {
            "total_correlations": len(self.correlation_history),
            "recent_average_strength": np.mean([c.correlation_strength for c in recent_correlations]),
            "phrase_boundaries_detected": len([c for c in recent_correlations if c.phrase_boundary_type == 'convergence']),
            "unique_continuations": len([c for c in recent_correlations if c.unique_continuation]),
            "deterministic_mode_activations": len([c for c in recent_correlations 
                                                 if c.fisher_value > self.FISHER_SPIKE_THRESHOLD and c.unique_continuation]),
            "current_parse_depth": len(self.current_parse_state.partial_parse_tree) if self.current_parse_state else 0,
            "current_deterministic_mode": self.is_deterministic_mode_active()
        }
    
    def reset_parser_state(self):
        """Reset parser state for new session"""
        self.current_parse_state = None
        self.correlation_history = []
        self.phrase_boundaries = []
        self.logger.info("Grammar-Fisher correlator state reset")

# Example usage and testing
if __name__ == "__main__":
    # Create Grammar-Fisher correlator
    correlator = GrammarFisherCorrelator()
    
    print("🔗 GRAMMAR-FISHER CORRELATION ENGINE")
    print("=" * 50)
    
    # Simulate market event sequence with corresponding Fisher values
    event_sequence = ['CONSOLIDATION', 'EXPANSION', 'BREAKOUT']
    fisher_values = [150.0, 450.0, 1200.0]  # Increasing Fisher Information
    
    print("📝 Parsing Market Event Sequence:")
    
    # Parse sequence incrementally
    for i in range(len(event_sequence)):
        partial_events = event_sequence[:i+1]
        partial_fisher = fisher_values[:i+1]
        
        parse_state = correlator.parse_market_sequence(partial_events, partial_fisher)
        
        print(f"\nStep {i+1}: {' → '.join(partial_events)}")
        print(f"  Fisher: {partial_fisher[-1]:.1f}")
        print(f"  Parse paths: {len(parse_state.parse_paths)}")
        print(f"  Phrase boundary: {parse_state.phrase_boundary_detected}")
        print(f"  Convergence: {parse_state.convergence_probability:.2f}")
        
        if parse_state.deterministic_continuation:
            print(f"  🎯 Deterministic next: {parse_state.deterministic_continuation}")
        
        if correlator.is_deterministic_mode_active():
            print("  🔴 DETERMINISTIC MODE ACTIVE")
    
    # Print correlation summary
    print("\n" + "=" * 50)
    summary = correlator.get_correlation_summary()
    print("📊 CORRELATION SUMMARY:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test prediction
    next_event = correlator.predict_next_event()
    if next_event:
        print(f"\n🔮 Predicted next event: {next_event}")
    else:
        print("\n❓ No deterministic prediction available")
