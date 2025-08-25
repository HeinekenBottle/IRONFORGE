#!/usr/bin/env python3
"""
Enhanced Pattern Analyst for IRONFORGE TQE
Creative mathematician/theorist generating novel testable hypotheses

Integrates with TQE Orchestrator and Project Manager
Generates dynamic hypotheses based on research focus and data context
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class HypothesisSpec:
    """Specification for hypothesis generation"""
    domain: str
    mechanism: str
    prediction: str
    counterexamples: str
    ascii_diagram: str
    mathematical_framework: str
    effect_size: float
    confidence_level: float

class EnhancedPatternAnalyst:
    """
    Enhanced Pattern Analyst - Creative mathematician and theorist
    
    Generates novel but testable hypotheses grounded in mathematical frameworks
    Specializes in pattern recognition, cross-timeframe relationships, temporal clustering
    """
    
    def __init__(self):
        """Initialize Enhanced Pattern Analyst"""
        
        # Core expertise domains
        self.expertise_domains = {
            'pattern_recognition': ['fractal_structures', 'self_similarity', 'phase_transitions'],
            'cross_timeframe': ['entrainment', 'phase_coupling', 'hierarchical_dynamics'], 
            'temporal_clustering': ['poisson_deviations', 'cascade_mechanics', 'information_diffusion'],
            'phase_dynamics': ['kuramoto_models', 'coherence_indices', 'synchronization_windows'],
            'creative_hypothesis': ['novel_connections', 'emergent_behaviors', 'non_obvious_relationships']
        }
        
        # Mathematical frameworks available for hypothesis generation
        self.math_frameworks = {
            'temporal_clustering': {
                'models': ['exponential_decay', 'bi_exponential', 'poisson_process', 'hawkes_process'],
                'parameters': ['lambda_base', 'alpha_decay', 'tau_half_life', 'intensity_ratio']
            },
            'phase_dynamics': {
                'models': ['kuramoto_oscillator', 'phase_locking_index', 'coherence_measure'],
                'parameters': ['coupling_strength', 'phase_difference', 'synchronization_threshold']
            },
            'spatial_relationships': {
                'models': ['inverse_square_law', 'power_law_decay', 'gravitational_attractor'],
                'parameters': ['distance_decay', 'attraction_strength', 'power_exponent']
            },
            'information_diffusion': {
                'models': ['diffusion_equation', 'cascade_propagation', 'wave_equation'],
                'parameters': ['diffusion_coefficient', 'propagation_speed', 'damping_factor']
            },
            'fractal_scaling': {
                'models': ['fibonacci_ratios', 'golden_ratio_emergence', 'self_similar_scaling'],
                'parameters': ['scaling_exponent', 'fractal_dimension', 'ratio_precision']
            }
        }
        
        # Domain-specific hypothesis templates
        self.domain_templates = {
            'session_extremes': {
                'focus': 'First session high/low taken events as temporal anchors',
                'mechanisms': ['liquidity_cascade', 'information_asymmetry', 'algorithmic_response'],
                'math_framework': 'temporal_clustering'
            },
            'progress_anchors': {
                'focus': 'Session progress milestones as temporal resonance nodes',
                'mechanisms': ['fractal_geometry', 'harmonic_frequencies', 'temporal_anchoring'],
                'math_framework': 'fractal_scaling'
            },
            'range_milestones': {
                'focus': 'Range completion levels as spatial attractors',
                'mechanisms': ['gravitational_attraction', 'magnetic_fields', 'standing_waves'],
                'math_framework': 'spatial_relationships'
            },
            'formation_dynamics': {
                'focus': 'Gap formation as information shockwave catalyst',
                'mechanisms': ['shockwave_propagation', 'arbitrage_seeking', 'price_dislocation'],
                'math_framework': 'information_diffusion'
            },
            'cross_frame_resonance': {
                'focus': 'Multi-timeframe phase coupling and entrainment',
                'mechanisms': ['phase_entrainment', 'kuramoto_coupling', 'synchronization'],
                'math_framework': 'phase_dynamics'
            }
        }
        
        # Setup logging
        self.logger = logging.getLogger('ironforge.enhanced_pattern_analyst')
        self.logger.info("Enhanced Pattern Analyst initialized - creative mathematician mode")

    def generate_hypotheses(self, 
                          run_id: str, 
                          research_focus: str, 
                          data_context: str, 
                          n_hypotheses: int = 5,
                          domain_requirements: Optional[List[str]] = None,
                          save_output: bool = True) -> Dict[str, Any]:
        """
        Generate novel testable hypotheses for research run
        
        Args:
            run_id: Unique run identifier
            research_focus: Primary research focus area  
            data_context: Available data and experimental context
            n_hypotheses: Number of hypotheses to generate
            domain_requirements: Required domains to cover
            save_output: Whether to save output files
            
        Returns:
            Comprehensive hypothesis generation results
        """
        
        self.logger.info(f"Generating {n_hypotheses} hypotheses for run {run_id}")
        start_time = datetime.now()
        
        # Analyze research focus and determine domains
        domain_analysis = self._analyze_research_focus(research_focus, data_context)
        
        # Determine hypothesis domains
        hypothesis_domains = self._determine_hypothesis_domains(
            domain_analysis, n_hypotheses, domain_requirements
        )
        
        # Generate hypotheses for each domain
        generated_hypotheses = []
        for i, domain_spec in enumerate(hypothesis_domains, 1):
            hypothesis = self._generate_domain_hypothesis(
                f"H{i}", domain_spec, research_focus, data_context
            )
            generated_hypotheses.append(hypothesis)
        
        # Generate cross-hypothesis interactions
        interactions = self._generate_hypothesis_interactions(generated_hypotheses)
        
        # Create comprehensive results
        results = {
            'run_id': run_id,
            'research_focus': research_focus,
            'data_context': data_context,
            'generation_timestamp': datetime.now().isoformat(),
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'domain_analysis': domain_analysis,
            'hypotheses': generated_hypotheses,
            'interactions': interactions,
            'mathematical_frameworks': self._summarize_frameworks_used(generated_hypotheses),
            'testable_predictions': self._extract_testable_predictions(generated_hypotheses)
        }
        
        # Save output if requested
        if save_output:
            self._save_hypothesis_output(run_id, results)
        
        return results

    def _analyze_research_focus(self, research_focus: str, data_context: str) -> Dict[str, Any]:
        """Analyze research focus to determine optimal hypothesis domains"""
        
        focus_lower = research_focus.lower()
        context_lower = data_context.lower()
        
        # Keyword analysis for domain relevance
        domain_relevance = {}
        
        # Session extremes keywords
        extreme_keywords = ['liquidity', 'hunt', 'sweep', 'session', 'high', 'low', 'extreme']
        domain_relevance['session_extremes'] = sum(1 for kw in extreme_keywords if kw in focus_lower)
        
        # Progress anchors keywords  
        progress_keywords = ['progress', 'milestone', 'percentage', 'fibonacci', 'ratio', 'anchor']
        domain_relevance['progress_anchors'] = sum(1 for kw in progress_keywords if kw in focus_lower)
        
        # Range milestones keywords
        range_keywords = ['range', 'level', 'completion', 'milestone', 'attraction', 'magnet']
        domain_relevance['range_milestones'] = sum(1 for kw in range_keywords if kw in focus_lower)
        
        # Formation dynamics keywords
        formation_keywords = ['fpfvg', 'gap', 'formation', 'catalyst', 'shockwave', 'dislocation']
        domain_relevance['formation_dynamics'] = sum(1 for kw in formation_keywords if kw in focus_lower)
        
        # Cross-frame resonance keywords
        resonance_keywords = ['htf', 'ltf', 'resonance', 'entrainment', 'coupling', 'timeframe']
        domain_relevance['cross_frame_resonance'] = sum(1 for kw in resonance_keywords if kw in focus_lower)
        
        # Determine primary research themes
        primary_theme = max(domain_relevance, key=domain_relevance.get) if domain_relevance else 'session_extremes'
        complexity_level = 'high' if len(focus_lower.split()) > 10 else 'medium' if len(focus_lower.split()) > 5 else 'low'
        
        return {
            'primary_theme': primary_theme,
            'domain_relevance': domain_relevance,
            'complexity_level': complexity_level,
            'mathematical_sophistication': self._assess_mathematical_needs(focus_lower, context_lower),
            'novel_connections_potential': self._assess_novelty_potential(focus_lower)
        }

    def _determine_hypothesis_domains(self, 
                                    domain_analysis: Dict[str, Any], 
                                    n_hypotheses: int,
                                    domain_requirements: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Determine specific domains for each hypothesis"""
        
        domains = []
        
        if domain_requirements:
            # Use specified domain requirements
            available_domains = list(self.domain_templates.keys())
            for i, req_domain in enumerate(domain_requirements[:n_hypotheses]):
                if req_domain in available_domains:
                    domain_spec = self.domain_templates[req_domain].copy()
                    domain_spec['domain_name'] = req_domain
                    domain_spec['priority'] = 'required'
                    domains.append(domain_spec)
        else:
            # Auto-determine based on analysis
            relevance_scores = domain_analysis['domain_relevance']
            sorted_domains = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
            
            # Ensure cross-frame resonance is always included
            if 'cross_frame_resonance' not in sorted_domains[:n_hypotheses]:
                sorted_domains = sorted_domains[:n_hypotheses-1] + ['cross_frame_resonance']
            
            for i, domain_key in enumerate(sorted_domains[:n_hypotheses]):
                if domain_key in self.domain_templates:
                    domain_spec = self.domain_templates[domain_key].copy()
                    domain_spec['domain_name'] = domain_key
                    domain_spec['priority'] = 'high' if i == 0 else 'medium' if i < 3 else 'low'
                    domains.append(domain_spec)
        
        return domains

    def _generate_domain_hypothesis(self, 
                                  hypothesis_id: str, 
                                  domain_spec: Dict[str, Any], 
                                  research_focus: str, 
                                  data_context: str) -> HypothesisSpec:
        """Generate specific hypothesis for domain"""
        
        domain_name = domain_spec['domain_name']
        math_framework = domain_spec['math_framework']
        
        # Generate creative mechanism
        mechanism = self._generate_creative_mechanism(domain_spec, research_focus)
        
        # Generate quantifiable prediction
        prediction = self._generate_quantifiable_prediction(domain_spec, math_framework)
        
        # Generate counterexamples
        counterexamples = self._generate_counterexamples(domain_spec)
        
        # Generate ASCII diagram
        ascii_diagram = self._generate_ascii_diagram(domain_spec, mechanism)
        
        # Estimate effect size and confidence
        effect_size, confidence = self._estimate_effect_parameters(domain_spec, research_focus)
        
        return HypothesisSpec(
            domain=domain_name,
            mechanism=mechanism,
            prediction=prediction,
            counterexamples=counterexamples,
            ascii_diagram=ascii_diagram,
            mathematical_framework=math_framework,
            effect_size=effect_size,
            confidence_level=confidence
        )

    def _generate_creative_mechanism(self, domain_spec: Dict[str, Any], research_focus: str) -> str:
        """Generate creative mathematical mechanism description"""
        
        domain_name = domain_spec['domain_name']
        mechanisms = domain_spec.get('mechanisms', [])
        
        # Create novel mechanism based on domain
        if domain_name == 'session_extremes':
            return ("First session extreme breaches create information cascade vectors that propagate through "
                   "nested timeframe hierarchies via exponential decay mechanics. The initial breach establishes "
                   "a temporal singularity where subsequent events cluster according to λ(t) = λ₀e^(-αt) with "
                   "decay constant α representing market information absorption rate.")
        
        elif domain_name == 'progress_anchors':
            return ("Session temporal progress exhibits fractal resonance at Fibonacci ratio milestones where "
                   "market microstructure synchronizes with inherent geometric harmonics. These ratios emerge from "
                   "self-similar scaling laws governing participant behavior cycles, creating natural temporal "
                   "attractors with φⁿ-based clustering strength.")
        
        elif domain_name == 'range_milestones':
            return ("Range completion milestones generate standing wave patterns in price-time space through "
                   "quantum-like superposition effects where price discovery creates resonance nodes. The "
                   "completion levels act as gravitational attractors following inverse square decay: F(r) ∝ 1/r² "
                   "where r represents distance from milestone in normalized price space.")
        
        elif domain_name == 'formation_dynamics':
            return ("Gap formation events create temporal shockwaves that propagate through market microstructure "
                   "via information diffusion cascades. The instantaneous price dislocation generates follow-on "
                   "clustering through arbitrage-seeking dynamics with diffusion coefficient D = σ²/2τ where "
                   "σ is gap magnitude and τ is participant reaction time.")
        
        elif domain_name == 'cross_frame_resonance':
            return ("Multi-timeframe momentum oscillations exhibit Kuramoto-type phase coupling where higher "
                   "timeframe cycles entrain lower timeframe event timing. Phase coherence emerges when "
                   "|Δφ| < π/4, creating synchronization windows with coupling strength K proportional to "
                   "cross-timeframe volatility ratios and momentum alignment coefficients.")
        
        else:
            return f"Domain-specific mechanism for {domain_name} involving mathematical framework integration."

    def _generate_quantifiable_prediction(self, domain_spec: Dict[str, Any], math_framework: str) -> str:
        """Generate specific quantifiable predictions with effect sizes"""
        
        domain_name = domain_spec['domain_name']
        
        if domain_name == 'session_extremes':
            return ("Event clustering within ±5 minutes of first session extreme will exceed baseline by 420% "
                   "(effect size d=1.8), with clustering half-life of 3.2±0.4 minutes. Secondary clustering "
                   "peak expected at 14±2 minutes representing HTF confirmation with 2.6x baseline intensity.")
        
        elif domain_name == 'progress_anchors':
            return ("Event clustering coefficient increases by 340% (p<0.001) within ±90 seconds of Fibonacci "
                   "progress ratios, with strongest effect at 38.2% showing coefficient of 0.52 vs baseline 0.15. "
                   "Effect magnitude follows φⁿ decay with correlation coefficient R²=0.84.")
        
        elif domain_name == 'range_milestones':
            return ("Event probability increases by factor of 3.2x within ±0.12% of range milestones, with 75% "
                   "completion showing strongest attraction (3.8x baseline). Spatial clustering follows power law "
                   "γ=-2.1±0.3 for distances >0.25% from milestone levels.")
        
        elif domain_name == 'formation_dynamics':
            return ("Gap formation (>6 ticks) triggers 480% increase in event density within next 3.5 minutes, "
                   "with clustering strength proportional to gap_size^1.7 (R²=0.79). Secondary clustering at "
                   "retracement levels (50%, 62%, 79%) shows 2.4x±0.3x baseline intensity.")
        
        elif domain_name == 'cross_frame_resonance':
            return ("Phase-locking index (PLI) exceeds 0.72 during synchronization windows occurring every "
                   "19±2 minutes, with event clustering increasing 450% during coherence phases. HTF momentum "
                   "reversals predict LTF clustering with 1.8±0.2 minute lead time and 82% accuracy.")
        
        else:
            return f"Quantifiable prediction for {domain_name} with statistical significance p<0.01."

    def _generate_counterexamples(self, domain_spec: Dict[str, Any]) -> str:
        """Generate expected failure conditions for hypothesis"""
        
        domain_name = domain_spec['domain_name']
        
        counterexample_templates = {
            'session_extremes': "Low volatility consolidation sessions (VIX <11) and holiday-adjacent sessions where reduced participation suppresses cascade mechanics due to insufficient liquidity depth.",
            
            'progress_anchors': "Trend continuation sessions with momentum >2.8σ where directional bias overwhelms temporal anchoring effects, and sessions with major news releases creating temporal structure disruption.",
            
            'range_milestones': "Gap opening sessions where initial price dislocation invalidates range calculations, and sessions with range <12 points where measurement noise overwhelms percentage-based signals.",
            
            'formation_dynamics': "Gap formations during pre-market hours with <40% normal volume where insufficient participation prevents cascade propagation, and gaps near session boundaries with <3 minutes remaining.",
            
            'cross_frame_resonance': "Market regime transition periods where multi-scale relationships temporarily decohere, and sessions with conflicting HTF signals creating phase interference patterns that disrupt entrainment."
        }
        
        return counterexample_templates.get(domain_name, "Market conditions where underlying mechanisms are disrupted or overwhelmed by external factors.")

    def _generate_ascii_diagram(self, domain_spec: Dict[str, Any], mechanism: str) -> str:
        """Generate ASCII visualization of hypothesis relationship"""
        
        domain_name = domain_spec['domain_name']
        
        ascii_templates = {
            'session_extremes': """
Session Extreme Event (t0)
    |
    ↓ λ₀e^(-αt)
t+2m: ████████████████ (peak density)
t+5m: ████████         (decay phase) 
t+8m: ████             (residual)
t+14m: ████████         (HTF confirmation)
t+20m: ██               (baseline)
""",
            
            'progress_anchors': """
Session Timeline (0%→100%)
|--23.6%--38.2%--50%--61.8%--78.6%--|
    ↑     ↑      ↑     ↑      ↑
   ███   █████   ████  █████   ███   (clustering)
  weak STRONG  medium STRONG  weak
   φ³    φ²     φ¹     φ²     φ³    (ratio decay)
""",
            
            'range_milestones': """
Price Structure:
High ---|100%|--- █████ (strong attraction)
        |75%|---- █████████ (strongest)  
        |50%|---- ████ (medium)
        |25%|---- █████ (strong)
Low  ---|0%|---- ███ (moderate)
       F(r)∝1/r²  (inverse square law)
""",
            
            'formation_dynamics': """
FPFVG Formation Timeline:
t0: Gap Opens    |████████| (price dislocation)
    ↓ D=σ²/2τ diffusion
t+1m: ██████████████████████ (max clustering)
t+3m: ████████████          (decay)
t+5m: ██████                (baseline)
      ↑       ↑       ↑
    50%     62%     79%     (retracement)
    ███     ████    ███     (secondary)
""",
            
            'cross_frame_resonance': """
HTF (15m): ~~~∩~~~∪~~~∩~~~∪~~~∩~~~ (momentum)
           ↓     ↓     ↓     ↓     ↓  (coupling)
LTF (1m):  █████ ███ █████ ███ █████ (clusters)
           
Phase Sync: ◊-----◊-----◊-----◊    (windows)
PLI:       0.78  0.31  0.74  0.28   (coherence)
|Δφ|<π/4:   ✓     ✗     ✓     ✗     (coupling)
"""
        }
        
        return ascii_templates.get(domain_name, f"""
{domain_name.replace('_', ' ').title()} Relationship:
Input → Processing → Output
  ↓        ↓         ↓
Event   Analysis  Clustering
""")

    # TODO(human): Implement statistical parameter estimation
    # Context: The Enhanced Pattern Analyst needs to estimate effect sizes and confidence levels
    # for generated hypotheses based on the domain characteristics and research focus.
    # This function should analyze the mathematical sophistication and data context to provide
    # realistic statistical parameters.
    def _estimate_effect_parameters(self, domain_spec: Dict[str, Any], research_focus: str) -> Tuple[float, float]:
        """
        Estimate effect size and confidence level for hypothesis
        
        Args:
            domain_spec: Domain specification with mathematical framework info
            research_focus: Research focus to assess parameter realism
            
        Returns:
            Tuple of (effect_size, confidence_level) based on domain analysis
        """
        
        # Default values - you should implement sophisticated estimation
        effect_size = 1.2  # Cohen's d effect size
        confidence_level = 0.85  # Statistical confidence
        
        return effect_size, confidence_level

    def _generate_hypothesis_interactions(self, hypotheses: List[HypothesisSpec]) -> List[Dict[str, Any]]:
        """Generate cross-hypothesis interaction predictions"""
        
        interactions = []
        
        # Generate pairwise interactions
        for i, h1 in enumerate(hypotheses):
            for j, h2 in enumerate(hypotheses[i+1:], i+1):
                interaction = {
                    'hypothesis_pair': f"{h1.domain} × {h2.domain}",
                    'interaction_type': self._determine_interaction_type(h1, h2),
                    'predicted_effect': self._predict_interaction_effect(h1, h2),
                    'mechanism': self._describe_interaction_mechanism(h1, h2)
                }
                interactions.append(interaction)
        
        return interactions

    def _determine_interaction_type(self, h1: HypothesisSpec, h2: HypothesisSpec) -> str:
        """Determine type of interaction between two hypotheses"""
        
        # Interaction type logic based on mathematical frameworks
        framework_combinations = {
            ('temporal_clustering', 'fractal_scaling'): 'multiplicative_amplification',
            ('spatial_relationships', 'information_diffusion'): 'resonance_coupling',
            ('phase_dynamics', 'temporal_clustering'): 'entrainment_synchronization',
            ('fractal_scaling', 'phase_dynamics'): 'harmonic_interference'
        }
        
        fw1, fw2 = h1.mathematical_framework, h2.mathematical_framework
        return framework_combinations.get((fw1, fw2)) or framework_combinations.get((fw2, fw1)) or 'additive_combination'

    def _predict_interaction_effect(self, h1: HypothesisSpec, h2: HypothesisSpec) -> str:
        """Predict quantitative interaction effect"""
        
        # Estimate combined effect based on individual effect sizes
        combined_effect = (h1.effect_size * h2.effect_size) ** 0.5  # Geometric mean
        
        if combined_effect > 2.0:
            return f"Super-additive clustering effect with {combined_effect:.1f}x amplification"
        elif combined_effect > 1.5:
            return f"Synergistic enhancement with {combined_effect:.1f}x multiplicative factor"
        else:
            return f"Moderate interaction with {combined_effect:.1f}x combined effect"

    def _describe_interaction_mechanism(self, h1: HypothesisSpec, h2: HypothesisSpec) -> str:
        """Describe mechanism of interaction between hypotheses"""
        
        return (f"{h1.domain.replace('_', ' ')} events occurring near {h2.domain.replace('_', ' ')} "
                f"trigger cross-domain resonance through coupled mathematical frameworks")

    def _summarize_frameworks_used(self, hypotheses: List[HypothesisSpec]) -> Dict[str, List[str]]:
        """Summarize mathematical frameworks used in hypotheses"""
        
        frameworks = {}
        for hypothesis in hypotheses:
            framework = hypothesis.mathematical_framework
            if framework not in frameworks:
                frameworks[framework] = []
            frameworks[framework].append(hypothesis.domain)
        
        return frameworks

    def _extract_testable_predictions(self, hypotheses: List[HypothesisSpec]) -> List[Dict[str, Any]]:
        """Extract specific testable predictions from hypotheses"""
        
        predictions = []
        for i, hypothesis in enumerate(hypotheses, 1):
            prediction = {
                'hypothesis_id': f"H{i}",
                'domain': hypothesis.domain,
                'prediction_text': hypothesis.prediction,
                'expected_effect_size': hypothesis.effect_size,
                'confidence_level': hypothesis.confidence_level,
                'testable_metrics': self._extract_metrics_from_prediction(hypothesis.prediction),
                'statistical_tests': self._suggest_statistical_tests(hypothesis)
            }
            predictions.append(prediction)
        
        return predictions

    def _extract_metrics_from_prediction(self, prediction_text: str) -> List[str]:
        """Extract quantitative metrics from prediction text"""
        
        # Use regex to find quantitative metrics
        metrics = []
        
        # Look for percentage increases
        percent_matches = re.findall(r'(\d+(?:\.\d+)?%)', prediction_text)
        metrics.extend([f"percentage_increase: {match}" for match in percent_matches])
        
        # Look for effect sizes
        effect_matches = re.findall(r'd=(\d+(?:\.\d+)?)', prediction_text)
        metrics.extend([f"cohen_d: {match}" for match in effect_matches])
        
        # Look for correlation coefficients
        corr_matches = re.findall(r'R²=(\d+(?:\.\d+)?)', prediction_text)
        metrics.extend([f"r_squared: {match}" for match in corr_matches])
        
        # Look for time windows
        time_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:minutes?|seconds?)', prediction_text)
        metrics.extend([f"time_window: {match}" for match in time_matches])
        
        return metrics if metrics else ['clustering_coefficient', 'baseline_comparison', 'temporal_decay']

    def _suggest_statistical_tests(self, hypothesis: HypothesisSpec) -> List[str]:
        """Suggest appropriate statistical tests for hypothesis"""
        
        framework = hypothesis.mathematical_framework
        
        test_suggestions = {
            'temporal_clustering': ['poisson_test', 'kolmogorov_smirnov', 'chi_square_goodness_of_fit'],
            'phase_dynamics': ['circular_statistics', 'phase_locking_index', 'coherence_analysis'],
            'spatial_relationships': ['spatial_autocorrelation', 'distance_decay_regression', 'hotspot_analysis'],
            'information_diffusion': ['diffusion_equation_fit', 'cascade_detection', 'propagation_speed_test'],
            'fractal_scaling': ['hurst_exponent', 'fractal_dimension', 'self_similarity_test']
        }
        
        return test_suggestions.get(framework, ['t_test', 'mann_whitney_u', 'bootstrap_confidence_interval'])

    def _assess_mathematical_needs(self, focus_text: str, context_text: str) -> str:
        """Assess mathematical sophistication needed for research focus"""
        
        math_indicators = ['statistical', 'quantitative', 'mathematical', 'model', 'equation', 'coefficient']
        sophistication_score = sum(1 for indicator in math_indicators if indicator in focus_text + context_text)
        
        if sophistication_score >= 3:
            return 'high'
        elif sophistication_score >= 1:
            return 'medium'
        else:
            return 'basic'

    def _assess_novelty_potential(self, focus_text: str) -> str:
        """Assess potential for novel connection discovery"""
        
        novelty_indicators = ['novel', 'creative', 'unexplored', 'innovative', 'breakthrough', 'discovery']
        creativity_score = sum(1 for indicator in novelty_indicators if indicator in focus_text)
        
        if creativity_score >= 2:
            return 'high'
        elif creativity_score >= 1:
            return 'medium'
        else:
            return 'standard'

    def _save_hypothesis_output(self, run_id: str, results: Dict[str, Any]) -> None:
        """Save hypothesis output to files"""
        
        # Create run directory
        run_dir = Path(f"./runs/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hypotheses markdown
        hypotheses_md = self._format_hypotheses_markdown(results)
        with open(run_dir / "hypotheses.md", "w") as f:
            f.write(hypotheses_md)
        
        # Save JSON log
        json_log = self._format_json_log(results)
        with open(run_dir / "hypothesis_log.jsonl", "w") as f:
            for entry in json_log:
                f.write(json.dumps(entry) + "\n")
        
        self.logger.info(f"Hypothesis output saved to {run_dir}")

    def _format_hypotheses_markdown(self, results: Dict[str, Any]) -> str:
        """Format hypotheses as markdown report"""
        
        md_content = f"""# Pattern Analyst Hypotheses: {results['research_focus']}
**Run ID**: {results['run_id']}
**Data Context**: {results['data_context']}
**Generated**: {results['generation_timestamp']}
**Processing Time**: {results['processing_time']:.2f}s

## Research Focus Analysis
- **Primary Theme**: {results['domain_analysis']['primary_theme']}
- **Complexity Level**: {results['domain_analysis']['complexity_level']}
- **Mathematical Sophistication**: {results['domain_analysis']['mathematical_sophistication']}
- **Novelty Potential**: {results['domain_analysis']['novel_connections_potential']}

"""
        
        # Add individual hypotheses
        for i, hypothesis in enumerate(results['hypotheses'], 1):
            md_content += f"""## H{i}: {hypothesis.domain.replace('_', ' ').title()}
**Domain**: {hypothesis.domain}
**Mathematical Framework**: {hypothesis.mathematical_framework}

**Mechanism**: {hypothesis.mechanism}

**Prediction**: {hypothesis.prediction}

**Counterexamples**: {hypothesis.counterexamples}

**Effect Size**: d={hypothesis.effect_size:.1f}, Confidence: {hypothesis.confidence_level:.1%}

**ASCII Diagram**:
```{hypothesis.ascii_diagram}```

---

"""
        
        # Add interactions
        if results['interactions']:
            md_content += "## Cross-Hypothesis Interactions\n\n"
            for interaction in results['interactions']:
                md_content += f"- **{interaction['hypothesis_pair']}**: {interaction['predicted_effect']}\n"
                md_content += f"  - Mechanism: {interaction['mechanism']}\n\n"
        
        # Add mathematical framework summary
        md_content += "## Mathematical Frameworks Summary\n\n"
        for framework, domains in results['mathematical_frameworks'].items():
            md_content += f"- **{framework}**: {', '.join(domains)}\n"
        
        # Add testable predictions summary
        md_content += "\n## Testable Predictions Summary\n\n"
        for prediction in results['testable_predictions']:
            md_content += f"- **{prediction['hypothesis_id']}**: Effect size d={prediction['expected_effect_size']:.1f}\n"
            md_content += f"  - Metrics: {', '.join(prediction['testable_metrics'])}\n"
            md_content += f"  - Tests: {', '.join(prediction['statistical_tests'])}\n\n"
        
        md_content += f"\n*Generated by IRONFORGE Enhanced Pattern Analyst*\n"
        
        return md_content

    def _format_json_log(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format results as JSON-lines log entries"""
        
        log_entries = []
        
        # Main entry
        log_entries.append({
            'timestamp': results['generation_timestamp'],
            'run_id': results['run_id'],
            'event_type': 'hypothesis_generation',
            'research_focus': results['research_focus'],
            'hypotheses_count': len(results['hypotheses']),
            'processing_time': results['processing_time'],
            'mathematical_frameworks': list(results['mathematical_frameworks'].keys())
        })
        
        # Individual hypothesis entries
        for i, hypothesis in enumerate(results['hypotheses'], 1):
            log_entries.append({
                'timestamp': results['generation_timestamp'],
                'run_id': results['run_id'],
                'event_type': 'hypothesis_generated',
                'hypothesis_id': f"H{i}",
                'domain': hypothesis.domain,
                'mathematical_framework': hypothesis.mathematical_framework,
                'effect_size': hypothesis.effect_size,
                'confidence_level': hypothesis.confidence_level
            })
        
        return log_entries


def demo_enhanced_pattern_analyst():
    """Demonstrate Enhanced Pattern Analyst capabilities"""
    
    print("🔬 IRONFORGE Enhanced Pattern Analyst Demo")
    print("=" * 60)
    
    analyst = EnhancedPatternAnalyst()
    
    # Test hypothesis generation
    test_research_focus = "Microstructure clustering and resonance field relationships"
    test_data_context = "66 enhanced IRONFORGE sessions with FPFVG formations and temporal events"
    test_run_id = f"TEST_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n🎯 Generating hypotheses for research focus:")
    print(f"   Focus: {test_research_focus}")
    print(f"   Context: {test_data_context}")
    print(f"   Run ID: {test_run_id}")
    
    # Generate hypotheses
    results = analyst.generate_hypotheses(
        run_id=test_run_id,
        research_focus=test_research_focus,
        data_context=test_data_context,
        n_hypotheses=5,
        save_output=False  # Don't save in demo
    )
    
    print(f"\n📊 Results Summary:")
    print(f"   Hypotheses Generated: {len(results['hypotheses'])}")
    print(f"   Primary Theme: {results['domain_analysis']['primary_theme']}")
    print(f"   Processing Time: {results['processing_time']:.2f}s")
    print(f"   Mathematical Frameworks: {', '.join(results['mathematical_frameworks'].keys())}")
    
    # Show first hypothesis as example
    if results['hypotheses']:
        h1 = results['hypotheses'][0]
        print(f"\n🧪 Sample Hypothesis (H1):")
        print(f"   Domain: {h1.domain}")
        print(f"   Framework: {h1.mathematical_framework}")
        print(f"   Effect Size: d={h1.effect_size:.1f}")
        print(f"   Prediction: {h1.prediction[:100]}...")
    
    # Show interactions
    if results['interactions']:
        print(f"\n🔗 Cross-Hypothesis Interactions:")
        for interaction in results['interactions'][:2]:
            print(f"   {interaction['hypothesis_pair']}: {interaction['interaction_type']}")
    
    print(f"\n✅ Enhanced Pattern Analyst demo complete!")


if __name__ == "__main__":
    demo_enhanced_pattern_analyst()