# IRONFORGE Agent Development Template

This template provides the framework for building production-ready IRONFORGE agents that integrate seamlessly with the archaeological discovery pipeline.

## IRONFORGE Agent Requirements

### FEATURE:
[Describe the specific IRONFORGE enhancement this agent provides. Examples:]
- Enhanced TGAT discovery with archaeological intelligence
- Confluence scoring with temporal non-locality awareness
- Session boundary validation with pattern authenticity checking
- HTF cascade analysis with dimensional anchoring
- Archaeological zone detection with 40% range analysis

### IRONFORGE PIPELINE INTEGRATION:
[Specify which pipeline stage(s) this agent enhances:]
- **Discovery Stage**: TGAT discovery, enhanced graph building, pattern recognition
- **Confluence Stage**: Scoring algorithms, rule-based validation, weight configuration
- **Validation Stage**: Quality gates, authenticity thresholds, contract compliance
- **Reporting Stage**: Minidash generation, visualization, dashboard enhancement

### TOOLS:
[Define IRONFORGE-specific tools the agent needs:]
- **Enhanced Session Processing**: Tools for processing 45D/51D graph features
- **Pattern Discovery**: TGAT attention mechanism integration and pattern extraction
- **Confluence Analysis**: Rule-based scoring with configurable weights
- **Validation Tools**: Golden invariant checking and authenticity scoring
- **Performance Monitoring**: Real-time tracking of <3s session processing requirements
- **Container Integration**: IRONFORGE lazy loading and container system tools

### DEPENDENCIES:
[Specify IRONFORGE integration requirements:]
- **Container System**: Integration with IRONFORGE's lazy loading container
- **Data Contracts**: Compliance with golden invariants (6 events, 4 edge intents, 51D/20D features)
- **Performance Monitoring**: Tools for tracking <3s session, <180s full discovery requirements
- **Archaeological Intelligence**: Access to temporal non-locality and zone analysis frameworks
- **Pipeline APIs**: Integration with existing IRONFORGE discovery, confluence, validation APIs

### SYSTEM PROMPT(S):
[Define IRONFORGE-aware system prompts:]
- **Archaeological Intelligence Awareness**: Understanding of temporal non-locality, 40% zones, dimensional anchoring
- **Data Contract Compliance**: Strict adherence to golden invariants and session boundaries
- **Performance Requirements**: Awareness of <3s session processing, >87% authenticity thresholds
- **Pipeline Integration**: Knowledge of how to coordinate with other IRONFORGE components
- **Quality Standards**: Understanding of IRONFORGE's production-grade requirements

### PERFORMANCE REQUIREMENTS:
- **Session Processing**: <3 seconds per session
- **Full Discovery**: <180 seconds for complete pipeline (57 sessions)
- **Memory Usage**: <100MB footprint
- **Authenticity**: >87% threshold for production patterns
- **Initialization**: <2 seconds with lazy loading

### ARCHAEOLOGICAL INTELLIGENCE INTEGRATION:
[Specify archaeological discovery enhancements:]
- **Temporal Non-locality**: Integration with Theory B validation and precision scoring
- **Archaeological Zones**: 40% range analysis and dimensional anchor detection
- **Session Boundaries**: Respect for within-session learning and cross-session isolation
- **HTF Rules**: Last-closed only data processing, no intra-candle analysis
- **Pattern Authenticity**: Validation against 87% authenticity threshold

### DATA CONTRACT COMPLIANCE:
[Ensure adherence to IRONFORGE golden invariants:]
- **Events**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- **Feature Dimensions**: 51D nodes (f0-f50), 20D edges (e0-e19)
- **HTF Context**: Features f45-f50 for higher timeframe context
- **Session Isolation**: No cross-session edges or learning

### CONTAINER INTEGRATION:
[Specify IRONFORGE container system integration:]
- **Lazy Loading**: Integration with IRONFORGE's performance-optimized container
- **Dependency Injection**: Clean separation of concerns with container-managed dependencies
- **Performance Monitoring**: Built-in tracking and optimization capabilities
- **Component Coordination**: Proper integration with other IRONFORGE components

### VALIDATION & TESTING:
[Define IRONFORGE-specific validation requirements:]
- **Contract Testing**: Validate against golden invariants and data schemas
- **Performance Testing**: Verify <3s session and <180s full discovery requirements
- **Archaeological Testing**: Test temporal non-locality and zone analysis capabilities
- **Integration Testing**: Verify pipeline coordination and component interaction
- **Quality Testing**: Ensure >87% authenticity threshold maintenance

### EXAMPLES:
[Reference IRONFORGE-specific examples:]
- `examples/ironforge_integration/` - IRONFORGE container and API integration patterns
- `examples/archaeological_discovery/` - Temporal non-locality and zone analysis examples
- `examples/performance_optimization/` - Meeting IRONFORGE's strict performance requirements
- `examples/pipeline_coordination/` - Multi-stage pipeline integration patterns

### DOCUMENTATION REFERENCES:
- **IRONFORGE Architecture**: `/docs/04-ARCHITECTURE.md` - Core system architecture
- **Data Contracts**: `/ironforge/contracts/` - Golden invariants and validation
- **Performance Standards**: `/docs/specialized/PATTERN-DISCOVERY.md`
- **Archaeological Intelligence**: `/archaeological_analysis/` - Temporal non-locality frameworks
- **Container System**: `/ironforge/integration/ironforge_container.py`

### PRODUCTION CONSIDERATIONS:
- **Error Handling**: Fail fast on contract violations, graceful degradation on non-critical failures
- **Logging**: Comprehensive logging for archaeological discovery insights
- **Monitoring**: Real-time performance and quality metrics
- **Scalability**: Support for parallel processing of multiple sessions
- **Maintainability**: Clear separation of concerns and modular architecture

### QUALITY GATES:
Before delivery, the agent must pass:
- [ ] All data contract validation tests
- [ ] Performance benchmarks (<3s session, <180s full)
- [ ] Archaeological intelligence capability verification
- [ ] Pipeline integration tests
- [ ] Quality threshold compliance (>87% authenticity)
- [ ] Container system integration verification
- [ ] Production readiness assessment

## IRONFORGE Integration Notes:

1. **Always import IRONFORGE APIs properly**: Use the centralized API for stable interface
2. **Respect data contracts**: Never modify golden invariants or session boundaries
3. **Maintain performance**: All operations must meet strict timing requirements
4. **Coordinate with pipeline**: Ensure proper integration with discovery/confluence/validation/reporting stages
5. **Monitor quality**: Continuously validate archaeological discovery effectiveness

[Add any additional IRONFORGE-specific considerations for the agent development process.]