# TQE Send Command

Send coordinated messages to TQE specialists through the Project Manager.

## Syntax
```
tqe-send <specialist> '<message>' [priority]
```

## Parameters
- `<specialist>`: Target specialist ('pattern' or 'data')
- `'<message>'`: Message content (quoted)
- `[priority]`: Optional priority level ('low', 'normal', 'high', 'urgent') - defaults to 'normal'

## Available Specialists

### Pattern Specialist
- **Specializations**: FPFVG detection, Gauntlet sequences, Archaeological zones, HTF analysis
- **Capabilities**: ICT methodology integration, TGAT discovery (92.3/100 authenticity)
- **Use for**: Market structure analysis, pattern detection, temporal queries

### Data Specialist  
- **Specializations**: Session loading, data validation, preprocessing, enhanced features
- **Capabilities**: 66 sessions available, 57 enhanced with authentic features
- **Use for**: Data preparation, session management, preprocessing tasks

## Priority Levels
- `urgent`: Immediate processing, highest priority queue
- `high`: Prioritized processing, expedited handling
- `normal`: Standard processing (default)
- `low`: Background processing, when time permits

## Examples

### Pattern Analysis
```bash
tqe-send pattern 'Analyze FPFVG patterns in August 5th PM session' high
tqe-send pattern 'Execute comprehensive Gauntlet sequence detection'
tqe-send pattern 'Calculate archaeological zone confluences using Theory B' urgent
```

### Data Processing
```bash
tqe-send data 'Load and preprocess sessions for temporal analysis' normal
tqe-send data 'Validate session data integrity for recent enhancements'
tqe-send data 'Prepare enhanced feature extraction for 57 authentic sessions' high
```

## Response Format
```json
{
  "specialist": "pattern|data",
  "processing_mode": "immediate|prioritized|standard",
  "response_time": 2.3,
  "orchestrator_integration": true,
  "analysis_summary": "Pattern analysis executed successfully",
  "capabilities_available": ["FPFVG", "Gauntlet", "Archaeological Zones"],
  "coordination_context": "2025-08-22T...",
  "quality_standards_met": true
}
```

## Quality Standards
All specialists operate under Project Manager quality standards:
- **Pattern Accuracy**: Minimum 85% (IRONFORGE standard: 92.3%)
- **Data Completeness**: Minimum 90% 
- **Processing Time**: Maximum 30 seconds
- **Required Validations**: Data integrity, pattern authenticity, results coherence

## Integration Notes
- Messages are routed through TQE Orchestrator for actual processing
- Communication is logged for coordination tracking
- Quality metrics are automatically updated
- Supports both individual queries and project coordination
- Real-time coordination with other specialists when needed

## Team Coordination Context
The Project Manager provides coordination context including:
- Current project status and dependencies
- Quality standards and checkpoints
- Team communication patterns
- Resource availability and scheduling