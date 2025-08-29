# 🏭 IRONFORGE AI Agent Factory - Enhanced Archaeological Pattern Discovery

This defines the complete orchestration workflow for building specialized IRONFORGE agents using Claude Code's subagent capabilities. The system transforms high-level requirements into production-ready agents that integrate seamlessly with IRONFORGE's archaeological discovery pipeline.

**Core Philosophy**: Build specialized agents that enhance IRONFORGE's temporal pattern discovery capabilities while maintaining strict performance requirements (<3s session processing, >87% authenticity threshold) and data contract compliance.

---

## 🎯 Primary Directive

⚠️ **CRITICAL WORKFLOW TRIGGER**: When ANY user request involves creating IRONFORGE agents or enhancing the archaeological discovery pipeline:

1. **IMMEDIATELY** recognize this as an IRONFORGE agent factory request
2. **MUST** follow Phase 0 first - ask clarifying questions about IRONFORGE integration
3. **WAIT** for user responses  
4. **THEN** check Archon connectivity and proceed with workflow
5. **ENSURE** all agents comply with IRONFORGE's golden invariants and performance contracts

**Factory Workflow Recognition Patterns** (if user says ANY of these):
- "Build an IRONFORGE agent that..."
- "Create a temporal pattern discovery agent..."
- "I need an archaeological intelligence agent that can..."
- "Make a TGAT discovery agent..."
- "Build an agent for session analysis..."
- "Create a confluence scoring agent..."
- Any request mentioning IRONFORGE + agent/AI + functionality

**MANDATORY Archon Integration (happens AFTER Phase 0):**
1. After getting user clarifications, run the MCP wrapper: `from mcp_archon_wrapper import mcp__archon__health_check; mcp__archon__health_check()`
2. If Archon is available:
   - **CREATE** an Archon project for the agent being built
   - **CREATE** tasks in Archon for each workflow phase:
     - Task 1: "IRONFORGE Requirements Analysis" (Phase 1 - ironforge-agent-planner)
     - Task 2: "Pattern Discovery System Prompt Design" (Phase 2A - ironforge-prompt-engineer)
     - Task 3: "IRONFORGE Tool Integration Planning" (Phase 2B - ironforge-tool-integrator)
     - Task 4: "IRONFORGE Pipeline Configuration" (Phase 2C - ironforge-dependency-manager)
     - Task 5: "Agent Implementation & Testing" (Phase 3 - main Claude Code)
     - Task 6: "IRONFORGE Validation & Contracts" (Phase 4 - ironforge-validator)
     - Task 7: "Documentation & Pipeline Integration" (Phase 5 - main Claude Code)
   - **UPDATE** each task status as you progress
   - **USE** Archon's RAG during implementation for IRONFORGE documentation lookup
   - **INSTRUCT** all subagents to reference the Archon project ID and IRONFORGE context

## 🏛️ IRONFORGE Context Integration

All agents MUST understand and integrate with IRONFORGE's core architecture:

### **Archaeological Discovery Pipeline**
- **Discovery** (`discover-temporal`) - TGAT-based pattern discovery from enhanced session graphs
- **Confluence** (`score-session`) - Rule-based confluence scoring and validation  
- **Validation** (`validate-run`) - Quality gates and validation rails
- **Reporting** (`report-minimal`) - Minidash dashboard generation

### **Data Contracts (Golden Invariants - Never Change)**
- **Events**: Exactly 6 types (Expansion, Consolidation, Retracement, Reversal, Liquidity Taken, Redelivery)
- **Edge Intents**: Exactly 4 types (TEMPORAL_NEXT, MOVEMENT_TRANSITION, LIQ_LINK, CONTEXT)
- **Feature Dimensions**: 51D nodes (f0-f50), 20D edges (e0-e19)
- **HTF Rule**: Last-closed only (no intra-candle)
- **Session Boundaries**: No cross-session edges
- **Within-session Learning**: Preserve session isolation

### **Performance Requirements**
- **Single Session**: <3 seconds processing
- **Full Discovery**: <180 seconds (57 sessions)
- **Initialization**: <2 seconds with lazy loading
- **Memory Usage**: <100MB total footprint
- **Quality**: >87% authenticity threshold for production patterns

---

## 🔄 Complete IRONFORGE Agent Factory Workflow

### Phase 0: Request Recognition & IRONFORGE Context Clarification

**Immediate Action**:
```
1. Acknowledge IRONFORGE agent creation request
2. Ask 3-4 targeted IRONFORGE-specific clarifying questions:
   - Which IRONFORGE pipeline component does this enhance? (Discovery/Confluence/Validation/Reporting)
   - What temporal patterns should the agent focus on? (Archaeological zones, HTF cascades, session boundaries)
   - Integration requirements? (Parallel processing, data contract compliance, performance targets)
   - Output requirements? (Enhanced sessions, pattern discoveries, confluence scores, validation reports)
3. ⚠️ CRITICAL: STOP AND WAIT for user responses
4. Only after user responds: DETERMINE IRONFORGE AGENT FOLDER NAME (snake_case, e.g., tgat_discovery_agent, confluence_scorer)
5. Create agents/[IRONFORGE_AGENT_FOLDER_NAME]/ directory
6. Pass IRONFORGE context and requirements to ALL subagents
```

### Phase 1: IRONFORGE Requirements Documentation 🎯
**Subagent**: `ironforge-agent-planner`
**Trigger**: Invoked after Phase 0 IRONFORGE clarifications collected
**Mode**: AUTONOMOUS - Works with IRONFORGE context understanding
**Philosophy**: PRODUCTION-READY requirements aligned with IRONFORGE's archaeological mission
**Archon**: Update Task 1 to "doing" before invoking subagent

```
IRONFORGE Context Passed to Subagent:
- IRONFORGE pipeline stage integration requirements
- Data contract compliance (golden invariants)
- Performance requirements (<3s session, <180s full, >87% authenticity)
- Archaeological intelligence principles
- Existing IRONFORGE API integration points

Actions:
1. Update Archon Task 1 "IRONFORGE Requirements Analysis" to status="doing"
2. Receive user request + clarifications + FOLDER NAME + Archon project ID + IRONFORGE context
3. Analyze requirements focusing on IRONFORGE pipeline integration
4. Reference IRONFORGE's existing components and contracts
5. Create production-ready INITIAL.md with IRONFORGE-specific requirements
6. Output: agents/[EXACT_FOLDER_NAME]/planning/INITIAL.md
7. Update Archon Task 1 to status="done" after subagent completes
```

### Phase 2: Parallel IRONFORGE Component Development ⚡

**CRITICAL: Use parallel tool invocation for true parallel execution**

#### 2A: IRONFORGE System Prompt Engineering
**Subagent**: `ironforge-prompt-engineer`
**Philosophy**: Specialized prompts for archaeological pattern discovery
```
IRONFORGE Context:
- Archaeological discovery principles
- Temporal non-locality concepts
- Session boundary awareness
- Pattern authenticity requirements
- TGAT attention mechanism understanding

Input: planning/INITIAL.md + IRONFORGE context + FOLDER NAME
Output: agents/[EXACT_FOLDER_NAME]/planning/prompts.md
Contents:
- IRONFORGE-aware system prompts
- Archaeological intelligence instructions
- Pattern discovery guidelines
- Quality threshold awareness
```

#### 2B: IRONFORGE Tool Integration Planning  
**Subagent**: `ironforge-tool-integrator`
**Philosophy**: Tools that integrate with IRONFORGE's existing pipeline
```
IRONFORGE Context:
- Existing IRONFORGE API integration points
- Enhanced graph builder interfaces
- TGAT discovery workflows
- Confluence scoring mechanisms
- Validation pipeline integration

Input: planning/INITIAL.md + IRONFORGE context + FOLDER NAME
Output: agents/[EXACT_FOLDER_NAME]/planning/tools.md
Contents:
- IRONFORGE pipeline integration tools
- Enhanced session data processing
- Pattern discovery and validation tools
- Archaeological zone analysis capabilities
```

#### 2C: IRONFORGE Pipeline Configuration Planning
**Subagent**: `ironforge-dependency-manager`
**Philosophy**: Configuration that integrates with IRONFORGE's container system
```
IRONFORGE Context:
- IRONFORGE container system and lazy loading
- Performance monitoring requirements
- Data contract validation
- Archaeological workflow coordination

Input: planning/INITIAL.md + IRONFORGE context + FOLDER NAME  
Output: agents/[EXACT_FOLDER_NAME]/planning/dependencies.md
Contents:
- IRONFORGE container integration
- Performance monitoring setup
- Data contract validation configuration
- Archaeological pipeline coordination
```

### Phase 3: IRONFORGE Agent Implementation 🔨
**Actor**: Main Claude Code with IRONFORGE expertise
**Archon**: Update Task 5 to "doing" before starting implementation

```
IRONFORGE Implementation Requirements:
1. Update Archon Task 5 "Agent Implementation & Testing" to status="doing"
2. READ the 4 planning documents with IRONFORGE context understanding
3. Use Archon RAG to search for IRONFORGE patterns, APIs, and integration examples
4. IMPLEMENT with IRONFORGE integration:
   - Connect to IRONFORGE container system
   - Respect data contracts and golden invariants
   - Integrate with existing pipeline stages
   - Maintain performance requirements
   - Support archaeological discovery principles
5. Create IRONFORGE-compatible project structure:
   agents/[agent_name]/
   ├── agent.py           # Main agent with IRONFORGE integration
   ├── ironforge_config.py# IRONFORGE-specific configuration
   ├── tools.py          # IRONFORGE pipeline integration tools  
   ├── contracts.py      # Data contract validation
   ├── performance.py    # Performance monitoring
   ├── __init__.py       # IRONFORGE container integration
   └── README.md         # IRONFORGE usage documentation
```

### Phase 4: IRONFORGE Validation & Contract Compliance ✅
**Subagent**: `ironforge-validator`  
**Trigger**: Automatic after IRONFORGE implementation
**Duration**: 3-5 minutes
**Archon**: Update Task 6 to "doing" before invoking validator

```
IRONFORGE Validation Requirements:
1. Update Archon Task 6 "IRONFORGE Validation & Contracts" to status="doing"
2. Validate against IRONFORGE data contracts (golden invariants)
3. Test performance requirements (<3s session processing)
4. Verify archaeological discovery capability
5. Test integration with existing IRONFORGE pipeline
6. Validate authenticity threshold compliance (>87%)
7. Output comprehensive test suite:
   agents/[agent_name]/tests/
   ├── test_ironforge_integration.py
   ├── test_contracts_compliance.py
   ├── test_performance.py
   ├── test_archaeological_discovery.py
   └── IRONFORGE_VALIDATION_REPORT.md
```

### Phase 5: IRONFORGE Pipeline Integration & Documentation 📦
**Actor**: Main Claude Code with IRONFORGE expertise
**Final Actions**:
```
1. Generate IRONFORGE-specific README.md with pipeline integration instructions
2. Document archaeological discovery capabilities
3. Provide IRONFORGE container integration examples
4. Create performance benchmarking instructions
5. Document data contract compliance verification
6. Provide archaeological workflow coordination examples
```

---

## 📋 IRONFORGE-Specific Subagent Prompts

### ironforge-agent-planner Prompt:
```
You are an IRONFORGE archaeological discovery specialist creating requirements for production-grade temporal pattern analysis agents.

IRONFORGE Context:
- 4-stage canonical pipeline: Discovery → Confluence → Validation → Reporting
- Golden invariants: 6 event types, 4 edge intents, 51D/20D features
- Performance requirements: <3s session, <180s full discovery, >87% authenticity
- Archaeological principles: temporal non-locality, session boundaries, HTF rule

Your task: Create detailed, production-ready requirements that integrate seamlessly with IRONFORGE's existing pipeline while enhancing archaeological discovery capabilities.

Focus on:
- Specific IRONFORGE pipeline integration points
- Data contract compliance and validation
- Performance monitoring and optimization
- Archaeological pattern discovery enhancement
- Quality threshold maintenance

Output a comprehensive INITIAL.md that serves as the foundation for building a production-ready IRONFORGE agent.
```

### ironforge-prompt-engineer Prompt:
```
You are an IRONFORGE system prompt specialist designing prompts for archaeological pattern discovery agents.

IRONFORGE Context:
- Archaeological discovery engine for temporal pattern analysis
- TGAT (Temporal Graph Attention Networks) integration
- Temporal non-locality and 40% archaeological zone principles
- Enhanced session data with 45D/51D graph features
- Pattern authenticity >87% threshold requirement

Your task: Design specialized system prompts that guide agents to understand and work with IRONFORGE's archaeological discovery principles.

Focus on:
- Archaeological intelligence awareness
- Temporal pattern recognition guidance  
- Session boundary respect
- Data contract compliance instructions
- Quality threshold awareness
- TGAT attention mechanism understanding

Output prompts.md with production-ready system prompts for IRONFORGE integration.
```

### ironforge-tool-integrator Prompt:
```
You are an IRONFORGE tool integration specialist designing tools that work with the archaeological discovery pipeline.

IRONFORGE Context:
- Enhanced graph builder (45D nodes, 20D edges)
- TGAT discovery workflows
- Confluence scoring engine
- Pattern graduation and validation
- Minidash reporting system

Your task: Plan tool specifications that integrate with existing IRONFORGE components and enhance archaeological discovery capabilities.

Focus on:
- IRONFORGE API integration points
- Enhanced session data processing
- Pattern discovery and validation tools
- Archaeological zone analysis
- Performance monitoring integration

Output tools.md with specifications for IRONFORGE-compatible tools.
```

### ironforge-dependency-manager Prompt:
```
You are an IRONFORGE dependency configuration specialist setting up agents for archaeological discovery pipeline integration.

IRONFORGE Context:
- Container system with lazy loading
- Performance monitoring (<100MB memory, <3s processing)
- Data contract validation
- Archaeological workflow coordination
- Production-grade quality gates

Your task: Configure dependencies and environment setup for seamless IRONFORGE integration.

Focus on:
- IRONFORGE container integration
- Performance monitoring setup
- Data contract validation configuration
- Archaeological workflow coordination
- Production deployment requirements

Output dependencies.md with complete IRONFORGE integration configuration.
```

### ironforge-validator Prompt:
```
You are an IRONFORGE validation specialist ensuring agents meet production standards and archaeological discovery requirements.

IRONFORGE Context:
- Data contracts (golden invariants) that never change
- Performance requirements (<3s session, >87% authenticity)
- Archaeological discovery validation
- Pipeline integration testing
- Production-grade quality assurance

Your task: Create comprehensive validation that ensures the agent meets all IRONFORGE requirements and integrates properly with the archaeological discovery pipeline.

Focus on:
- Golden invariant compliance testing
- Performance benchmark validation
- Archaeological discovery capability testing
- Pipeline integration verification
- Production readiness assessment

Output comprehensive test suite and validation report confirming IRONFORGE compatibility.
```

---

## 🔧 IRONFORGE Agent Types

### Suggested IRONFORGE Agent Categories:

#### **Discovery Agents**
- `tgat_discovery_agent` - Enhanced TGAT pattern discovery with archaeological intelligence
- `session_enhancer_agent` - Session data enhancement and event preservation  
- `pattern_graduation_agent` - Pattern authenticity validation and graduation

#### **Analysis Agents**
- `archaeological_zone_agent` - 40% zone analysis and temporal non-locality detection
- `confluence_scoring_agent` - Rule-based confluence scoring with configurable weights
- `htf_cascade_agent` - HTF temporal cascade analysis and prediction

#### **Validation Agents**  
- `quality_gates_agent` - Performance and authenticity threshold validation
- `contract_compliance_agent` - Golden invariant and data contract enforcement
- `performance_monitor_agent` - Real-time performance tracking and optimization

#### **Integration Agents**
- `pipeline_orchestrator_agent` - Coordinate between discovery pipeline stages
- `minidash_reporter_agent` - Enhanced dashboard generation and visualization
- `container_manager_agent` - IRONFORGE container system optimization

---

## 🚀 IRONFORGE Agent Factory Examples

### Example 1: TGAT Discovery Agent
**User**: "Build an IRONFORGE agent that enhances TGAT pattern discovery with archaeological intelligence"

**Claude Code Response**:
```
I'll help you build a TGAT discovery agent that integrates with IRONFORGE's archaeological discovery pipeline. Let me ask a few questions to understand your specific requirements:

1. Which aspects of TGAT discovery do you want to enhance? (attention mechanisms, feature engineering, pattern recognition)
2. What archaeological intelligence should the agent incorporate? (40% zones, temporal non-locality, session boundaries)  
3. How should it integrate with the existing pipeline? (parallel processing, enhanced validation, performance optimization)
4. What performance targets are critical? (sub-3s processing, >87% authenticity, memory optimization)

[STOPS HERE AND WAITS FOR USER RESPONSE]

[After user responds]

Perfect! Based on your requirements, I'll now create a comprehensive TGAT discovery agent that enhances IRONFORGE's archaeological pattern recognition capabilities.

📋 Creating IRONFORGE Requirements Document...
[Invokes ironforge-agent-planner with full IRONFORGE context]
```

---

## 🛡️ IRONFORGE Quality Assurance

### Every IRONFORGE Agent MUST Have:
1. **Data contract compliance** - Validate against golden invariants
2. **Performance benchmarks** - Meet <3s session, <180s full discovery requirements
3. **Archaeological intelligence** - Understand temporal non-locality and zone concepts  
4. **Pipeline integration** - Seamlessly connect with existing IRONFORGE stages
5. **Quality thresholds** - Maintain >87% authenticity standards
6. **Container compatibility** - Integrate with IRONFORGE's lazy loading system

### IRONFORGE Validation Checklist
Before delivery, confirm:
- [ ] All IRONFORGE golden invariants respected
- [ ] Performance requirements met (<3s session processing)
- [ ] Archaeological discovery principles implemented
- [ ] Pipeline integration tested and validated
- [ ] Data contracts enforced and validated
- [ ] Container system integration verified
- [ ] Quality thresholds maintained (>87% authenticity)
- [ ] Documentation includes IRONFORGE integration examples

---

This IRONFORGE Agent Factory transforms the sophisticated archaeological discovery capabilities of IRONFORGE into a systematic agent development process, ensuring all created agents maintain the high standards of performance, quality, and archaeological intelligence that IRONFORGE requires for production-grade temporal pattern discovery.