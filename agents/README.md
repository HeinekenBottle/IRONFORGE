# IRONFORGE Agent Ecosystem with Document-Mediated Communication

This document explains the IRONFORGE Agent Ecosystem that implements sophisticated document-mediated communication protocols based on the [context-engineering-intro](https://github.com/coleam00/context-engineering-intro) repository pattern.

## 🏗️ Architecture Overview

The IRONFORGE Agent Ecosystem uses a **Document-Mediated Communication Protocol** where agents don't communicate directly with each other. Instead, they communicate through structured markdown documents that serve as:

- **Context Carriers**: Each document contains accumulated knowledge from previous phases
- **Specification Contracts**: Clear interfaces between agent responsibilities  
- **Knowledge Evolution**: Documents build upon previous phases progressively
- **Audit Trails**: Complete history of how requirements evolved into implementation

## 📁 Planning Document Structure

Each agent follows a 4-phase planning document workflow:

```
agents/{agent_name}/
├── planning/                    # Context Communication Hub
│   ├── INITIAL.md              # Business Requirements Context
│   ├── prompts.md              # Behavioral Specification Context
│   ├── tools.md                # Functional Implementation Context
│   └── dependencies.md         # Infrastructure Architecture Context
├── agent.py                    # Final Implementation (reads all contexts)
├── tools.py                    # Implements tools.md specifications
├── prompts.py                  # Implements prompts.md specifications
├── dependencies.py             # Implements dependencies.md specifications
└── tests/                      # Validates against all contexts
```

## 🔄 Communication Flow

### Phase 1: Requirements Crystallization
**Document**: `INITIAL.md`  
**Context Type**: Business Requirements Context

Transforms vague user requests into precise, technical specifications:
- Executive Summary (business understanding)
- Functional Requirements (with acceptance criteria)
- Technical Requirements (models, APIs, integrations)
- Success Criteria (measurable outcomes)
- Risk Assessment and mitigation strategies

### Phase 2A: Behavioral Specification  
**Document**: `prompts.md`  
**Context Type**: Behavioral Specification Context

Translates requirements into agent personality and behavior:
- Primary System Prompt (agent personality)
- Dynamic Prompt Components (context-aware instructions)
- Behavioral Triggers (decision-making patterns)
- Communication Style guidelines

### Phase 2B: Functional Implementation
**Document**: `tools.md`  
**Context Type**: Functional Implementation Context

Defines concrete capabilities and actions:
- Tool Implementation Specifications (`@agent.tool` patterns)
- Parameter Definitions (with validation rules)
- Error Handling Strategies (retry, fallback, graceful degradation)
- Performance Considerations (caching, pooling, optimization)

### Phase 2C: Infrastructure Architecture
**Document**: `dependencies.md`  
**Context Type**: System Architecture Context

Configures runtime environment and dependencies:
- Environment Variables Configuration
- Settings Configuration (BaseSettings class structure)
- Model Provider Configuration
- Dependencies Dataclass (runtime context and state)

## 🤖 Available Agents

### Core Pipeline Coordination
1. **pipeline-orchestrator** - Central coordinator for all 4 IRONFORGE pipeline stages
2. **pipeline-performance-monitor** - Performance monitoring and optimization
3. **session-boundary-guardian** - Session isolation and boundary validation

### Quality & Validation
4. **contract-compliance-enforcer** - Golden invariant and data contract validation
5. **authenticity-validator** - Pattern authenticity and graduation validation
6. **tgat-attention-analyzer** - TGAT model analysis and optimization

### Intelligence & Analysis
7. **pattern-intelligence-analyst** - Deep pattern analysis with archaeological insights
8. **archaeological-zone-detector** - 40% zone detection and temporal non-locality
9. **motif-pattern-miner** - Recurring pattern mining and structural analysis
10. **htf-cascade-predictor** - Multi-timeframe cascade prediction

### Scoring & Reporting  
11. **confluence-intelligence** - Advanced confluence scoring optimization
12. **minidash-enhancer** - Dashboard visualization and reporting

## 🚀 Usage Examples

### Basic Agent Coordination

```python
from ironforge.agents import PipelineOrchestratorAgent, ContractComplianceEnforcer

# Initialize agents with planning document context
orchestrator = PipelineOrchestratorAgent.from_planning_documents()
contract_enforcer = ContractComplianceEnforcer.from_planning_documents()

# Execute pipeline with document-mediated coordination
session_data = {...}  # Your session data
result = await orchestrator.execute_pipeline_stage(
    "discovery", 
    session_data,
    quality_gates=True  # Uses context from dependencies.md
)
```

### Multi-Agent Coordination

```python
# Coordinate multiple agents using their specialized contexts
coordination_result = await orchestrator.coordinate_multiple_agents([
    ("contract-compliance-enforcer", "validate_contracts"),
    ("authenticity-validator", "validate_authenticity"),
    ("archaeological-zone-detector", "detect_zones")
], task_data)

# Each agent uses its planning document context automatically
for agent_name, result in coordination_result.items():
    print(f"{agent_name}: {result['status']}")
```

### Error Recovery with Context

```python
# Error recovery using behavioral context from prompts.md
error_details = {
    "error_type": "ValidationFailure",
    "stage": "confluence",
    "context": "Authenticity score below threshold"
}

recovery_result = await orchestrator.handle_pipeline_error(
    error_details,
    recovery_strategy="auto"  # Strategy defined in planning documents
)
```

## 🔧 Configuration

### Environment Variables (from dependencies.md context)
```bash
# IRONFORGE Pipeline Configuration
IRONFORGE_CONFIG_PATH=configs/dev.yml
PIPELINE_TIMEOUT_SECONDS=180
AUTHENTICITY_THRESHOLD=0.87

# Agent Coordination
AGENT_COMMUNICATION_TIMEOUT=30
MAX_AGENT_RETRIES=3
ENABLE_AGENT_FALLBACK=true

# Performance Monitoring  
ENABLE_PERFORMANCE_MONITORING=true
OPTIMIZATION_LEVEL=standard
```

### Agent Settings
```python
from ironforge.agents.settings import AgentEcosystemSettings

settings = AgentEcosystemSettings()
print(f"Quality threshold: {settings.authenticity_threshold}")
print(f"Pipeline timeout: {settings.pipeline_timeout_seconds}s")
```

## 🧪 Testing

### Integration Testing
```bash
# Run agent ecosystem integration tests
pytest tests/integration/test_agent_ecosystem.py -v

# Test specific agent coordination
pytest tests/integration/test_agent_ecosystem.py::TestAgentEcosystemIntegration::test_document_mediated_communication -v
```

### Planning Document Validation
```bash
# Validate planning document structure
pytest tests/integration/test_agent_ecosystem.py::TestAgentEcosystemIntegration::test_planning_document_structure_validation -v
```

## 📊 Context Communication Patterns

### 1. Reference-Based Communication
Agents reference specific sections of previous documents:

```markdown
# In tools.md:
"Based on the requirements from INITIAL.md, this agent needs 3 essential tools..."

# In prompts.md:  
"Following the functional requirements in INITIAL.md, the agent should exhibit..."

# In dependencies.md:
"From the technical requirements in INITIAL.md, we need OpenAI API..."
```

### 2. Assumption Inheritance
Later agents inherit and build upon earlier agents' assumptions:

```markdown
# INITIAL.md establishes:
"Agent will use OpenAI gpt-4o-mini for cost-effective processing"

# dependencies.md builds on this:
"LLM_MODEL=gpt-4o-mini (as specified in requirements)"

# prompts.md aligns:
"Optimized for gpt-4o-mini's capabilities and context window"
```

### 3. Constraint Propagation
Constraints flow through the document chain:

```markdown
# INITIAL.md constraint:
"Pipeline processing: <180 seconds total for 57 sessions"

# tools.md implements:
"Timeout handling: 180-second pipeline limit enforced"

# dependencies.md configures:
"PIPELINE_TIMEOUT_SECONDS=180"
```

## 🎯 Agent Specialization Through Context

Each agent has a specialized lens for interpreting requirements:

| Agent Role | Context Lens | Focus Area |
|------------|--------------|------------|
| **Pipeline Orchestrator** | "How do I coordinate?" | Workflow orchestration and stage management |
| **Contract Enforcer** | "What rules must be followed?" | Data integrity and invariant validation |
| **Authenticity Validator** | "Is this quality sufficient?" | Pattern quality and graduation criteria |
| **Pattern Analyst** | "What does this mean?" | Deep analysis and archaeological insights |
| **Zone Detector** | "Where are the significant areas?" | 40% zones and temporal non-locality |

## 🔍 Quality Assurance

### Golden Invariants (Enforced by contract-compliance-enforcer)
- **Event Types**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- **Node Features**: 51D (f0-f50) with proper dimensionality
- **Edge Features**: 20D (e0-e19) with proper dimensionality  
- **Session Boundaries**: No cross-session contamination

### Performance Requirements
- **Pipeline Processing**: <180 seconds total for 57 sessions
- **Single Session**: <5 seconds processing time
- **Memory Usage**: <100MB total footprint
- **Authenticity Threshold**: >87% for pattern graduation

## 🚨 Error Handling

### Failure Categories (defined in planning documents)
1. **Recoverable Errors**: Retry with exponential backoff
2. **Configuration Errors**: Clear remediation guidance provided
3. **Data Corruption**: Block processing with detailed violation reports
4. **Infrastructure Errors**: Fallback to alternative agents/processing

### Recovery Strategies
```python
# Automatic recovery using context from prompts.md
recovery_strategies = {
    "ValidationFailure": "pattern_revalidation",
    "PerformanceTimeout": "parallel_optimization", 
    "DataCorruption": "quality_gate_enforcement",
    "AgentCommunicationFailure": "circuit_breaker_activation"
}
```

## 📈 Performance Optimization

### Optimization Levels (from dependencies.md context)
- **Conservative**: Safe optimizations with minimal risk
- **Standard**: Balanced optimization with performance monitoring
- **Aggressive**: Maximum performance with quality safeguards

### Monitoring Integration
```python
# Real-time performance monitoring
performance_metrics = {
    "sessions_processed": 45,
    "avg_processing_time": 2.8,
    "memory_usage_mb": 78.5,
    "authenticity_rate": 0.923,
    "quality_gates_passed": 44,
    "bottlenecks_detected": 1
}
```

## 🔮 Advanced Features

### Dynamic Context Adaptation
```python
# Agents adapt behavior based on updated planning context
await orchestrator.update_planning_context("INITIAL.md", {
    "new_quality_threshold": 0.90,
    "performance_limit": 150,
    "additional_validation": "temporal_coherence"
})
```

### Planning Document Evolution Support
The system gracefully handles planning document updates while maintaining backward compatibility and operational continuity.

### Multi-Environment Configuration
- **Development**: Debug enabled, verbose logging, relaxed constraints
- **Staging**: Production-like settings with comprehensive monitoring  
- **Production**: Strict enforcement, optimized performance, minimal logging

## 📚 Further Reading

- [IRONFORGE Core Documentation](../docs/DATA_PIPELINE_ARCHITECTURE.md)
- [Agent Factory Pattern](https://github.com/coleam00/context-engineering-intro)
- [Planning Document Examples](./pipeline_orchestrator/planning/)
- [Integration Test Examples](../tests/integration/test_agent_ecosystem.py)

## 🤝 Contributing

When adding new agents to the ecosystem:

1. **Create Planning Documents**: Follow the 4-phase structure (INITIAL.md → prompts.md → tools.md → dependencies.md)
2. **Document Context Flow**: Show how your agent builds upon and communicates through planning documents
3. **Add Integration Tests**: Include tests that validate document-mediated communication
4. **Update Registry**: Register your agent in the ecosystem configuration
5. **Validate Quality**: Ensure compliance with IRONFORGE golden invariants and performance requirements

The document-mediated communication protocol ensures that agents can coordinate sophisticated workflows while maintaining clear separation of concerns, comprehensive documentation, and robust quality assurance.