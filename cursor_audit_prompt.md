# IRONFORGE Codebase Audit Prompt for Cursor CLI

## Overview
Perform a comprehensive, multi-phased audit of the IRONFORGE codebase - a sophisticated archaeological discovery engine for market pattern analysis. This is a large, complex system requiring systematic analysis across multiple dimensions.

## Audit Categories

### 1. Architecture & Design Patterns
- System architecture analysis (pipeline, components, data flow)
- Design pattern identification and evaluation
- Component coupling and cohesion assessment
- Dependency injection and container system evaluation
- Configuration management patterns
- Error handling and resilience patterns

### 2. Data Flow & Layering Analysis
- Data pipeline mapping (enhanced sessions → graphs → patterns → reports)
- Layer separation and boundary analysis
- Data contract validation (golden invariants)
- Session isolation and boundary enforcement
- Feature dimension consistency (45D/51D nodes, 20D edges)
- HTF rule compliance and temporal coherence

### 3. Code Quality & Standards
- Code style and formatting consistency
- Type safety and annotation coverage
- Documentation quality and completeness
- Naming conventions and code organization
- Import patterns and module structure
- Code duplication analysis

### 4. Security & Reliability
- Input validation and sanitization
- Authentication and authorization patterns
- Data protection and privacy considerations
- Error handling and logging security
- Dependency security analysis
- Configuration security

### 5. Performance & Optimization
- Algorithm complexity analysis
- Memory usage patterns
- I/O optimization opportunities
- Caching strategy evaluation
- Performance bottleneck identification
- Resource utilization analysis

### 6. Testing & Validation
- Test coverage analysis
- Test quality and effectiveness
- Contract testing implementation
- Integration test completeness
- Performance testing adequacy
- Quality gate effectiveness

### 7. Maintenance & Sustainability
- Dependency management and updates
- Technical debt assessment
- Code maintainability metrics
- Documentation maintenance
- Build and deployment processes
- Development workflow efficiency

## Multi-Phased Audit Approach

### Phase 1: Foundation Analysis (2-3 hours)
**Focus**: Core architecture and data flow understanding
- Map the 4-stage canonical pipeline (Discovery → Confluence → Validation → Reporting)
- Analyze enhanced graph builder and TGAT discovery components
- Document data contracts and golden invariants
- Identify key architectural patterns and design decisions
- Map primary data flows and transformation points

### Phase 2: Component Deep Dive (4-6 hours)
**Focus**: Individual component analysis
- Analyze each major module (learning, confluence, validation, reporting)
- Evaluate component interfaces and contracts
- Assess internal component quality and patterns
- Identify component-specific issues and improvements
- Analyze cross-component dependencies and interactions

### Phase 3: Quality & Security Assessment (2-3 hours)
**Focus**: Code quality, security, and maintainability
- Perform static code analysis for quality issues
- Security vulnerability assessment
- Performance analysis and optimization opportunities
- Testing coverage and effectiveness evaluation
- Documentation completeness and quality assessment

### Phase 4: Integration & System Analysis (2-3 hours)
**Focus**: System-wide integration and optimization
- End-to-end data flow validation
- Integration testing adequacy
- System performance characteristics
- Deployment and operational considerations
- Scalability and extensibility analysis

### Phase 5: Recommendations & Roadmap (1-2 hours)
**Focus**: Actionable improvements and prioritization
- Prioritize findings by impact and effort
- Create implementation roadmap
- Identify quick wins vs. major refactoring
- Document architectural improvement opportunities
- Establish monitoring and maintenance recommendations

## Audit Deliverables

### 1. Executive Summary
- Overall system health assessment
- Critical issues requiring immediate attention
- High-level architectural recommendations
- Risk assessment and mitigation strategies

### 2. Detailed Findings Report
- Component-by-component analysis results
- Data flow and layering assessment
- Quality metrics and measurements
- Security assessment results
- Performance analysis findings

### 3. Action Items & Roadmap
- Prioritized list of improvements
- Implementation phases and timelines
- Risk mitigation strategies
- Success metrics and monitoring approach

### 4. Technical Documentation Updates
- Architecture documentation improvements
- Data flow documentation enhancements
- Component interface documentation
- Operational runbook updates

## Success Criteria

### Quality Gates
- **Architecture**: Clear separation of concerns, well-defined interfaces
- **Data Flow**: Consistent data contracts, proper session isolation
- **Code Quality**: >80% test coverage, consistent patterns, good documentation
- **Security**: No critical vulnerabilities, proper input validation
- **Performance**: Meet established benchmarks (<3s session, <180s discovery)
- **Maintainability**: Clear code organization, manageable technical debt

### Audit Completion Standards
- All major components analyzed
- Data flows fully mapped and validated
- Critical issues identified and prioritized
- Actionable recommendations provided
- Implementation roadmap established

## Special Considerations

### IRONFORGE-Specific Requirements
- **Archaeological Integrity**: Preserve pattern authenticity and session isolation
- **Performance Constraints**: Maintain sub-3-second session processing
- **Data Contracts**: Never violate golden invariants (6 events, 4 edge intents, etc.)
- **Quality Thresholds**: Maintain >87% authenticity for production patterns
- **HTF Compliance**: Ensure last-closed only rule enforcement

### Technical Constraints
- Mixed language environment (Python, Lua)
- Complex ML components (TGAT, pattern discovery)
- Real-time performance requirements
- Large-scale data processing capabilities
- Distributed processing considerations

## Audit Methodology

### Analysis Techniques
- **Static Analysis**: Code quality tools, dependency analysis, security scanning
- **Dynamic Analysis**: Performance profiling, memory analysis, runtime behavior
- **Structural Analysis**: Architecture modeling, component interaction mapping
- **Data Flow Analysis**: Pipeline tracing, transformation validation, contract verification

### Tools & Frameworks
- Use appropriate static analysis tools for Python/Lua
- Leverage existing test frameworks and quality gates
- Utilize performance profiling tools
- Apply security scanning tools and methodologies

Begin with Phase 1 and progress systematically through each phase, ensuring thorough analysis while maintaining efficiency for this large codebase.