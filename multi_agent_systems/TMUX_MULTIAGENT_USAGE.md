# Multi-Agent Tmux Setup - Usage Guide

## Quick Start

```bash
# Basic usage with Claude Code
./tmux_multiagent_setup.sh

# Custom investigation
RESEARCH_FOCUS="HTF archaeological zone interactions" ./tmux_multiagent_setup.sh

# Custom session name and workspace  
SESSION="my_research" RUN_ID="INVESTIGATION_001" ./tmux_multiagent_setup.sh
```

## Configuration Options

### Essential Variables
```bash
export CHAT_CMD="claude chat"              # Claude Code CLI command
export RESEARCH_FOCUS="Your investigation"  # Research topic
export INVESTIGATION_TYPE="Data Analysis"   # Investigation category
```

### Advanced Configuration
```bash
export RUN_ID="CUSTOM_$(date +%Y%m%d)"     # Custom run identifier
export SESSION="research_session"           # Tmux session name  
export ROOT="/custom/workspace/path"        # Custom workspace location
export DATA_CONTEXT="/path/to/data"         # Data source description
```

### Statistical Parameters
```bash
export CONFIDENCE_LEVEL="0.95"             # Statistical confidence level
export FDR_Q_VALUE="0.05"                  # False discovery rate
export PERMUTATION_ITERATIONS="5000"       # Permutation test iterations
```

### Pane Role Customization
```bash
export PANE_LEFT_ROLE="coordinator"        # Left pane role
export PANE_RT_ROLE="theorist"            # Top-right pane role
export PANE_RB_ROLE="validator"           # Bottom-right pane role
```

## In-Session Management

### Helper Functions (source in orchestrator pane)
```bash
source ./runs/$RUN_ID/scripts/multiagent_helpers.sh

# Delegation
delegate_to_analyst "Develop hypothesis for temporal patterns"
delegate_to_data "Analyze session range correlations"

# Progress tracking
capture_all_panes                          # Snapshot all panes
capture_pane $PANE_RT "analyst_progress"   # Capture specific pane
update_status orchestrator PLANNING task1 "Starting investigation"

# Status monitoring
wait_for_completion analyst 300            # Wait for analyst (5min timeout)
```

### Pane Communication
```bash
# Send prompt to specific pane
send_prompt $PANE_LEFT "Your instructions here"
send_line $PANE_RT "Multi-line message here"
send_file $PANE_RB "/path/to/prompt.md"

# Get pane by role
analyst_pane=$(get_pane_by_role analyst)
send_prompt $analyst_pane "Analyze this pattern..."
```

## Status Monitoring (Separate Terminal)

```bash
# Start status monitor
RUN_ID="your_run_id" ./runs/your_run_id/scripts/status_monitor.sh

# Manual status check
jq '.' ./runs/$RUN_ID/status/*.json
```

## File Structure Generated

```
runs/RUN_ID/
├── investigation_manifest.md    # Investigation overview
├── status/                     # Agent status tracking
│   ├── orchestrator.json
│   ├── analyst.json  
│   └── data.json
├── logs/                       # Timestamped pane outputs
│   ├── orchestrator.log
│   ├── analyst.log
│   └── data.log  
├── capture/                    # Pane snapshots
├── artifacts/                  # Analysis results
├── prompts/                    # Agent initialization prompts
└── scripts/                    # Helper utilities
    ├── multiagent_helpers.sh
    └── status_monitor.sh
```

## Example Workflows

### 1. Standard Research Investigation
```bash
# Setup
RESEARCH_FOCUS="News clustering amplification effects" ./tmux_multiagent_setup.sh

# In orchestrator pane:
source ./runs/$RUN_ID/scripts/multiagent_helpers.sh
delegate_to_analyst "Develop H1-H5 for news clustering mechanisms"
delegate_to_data "Validate 50.96x amplification with bootstrap CI"
```

### 2. Archaeological Zone Analysis  
```bash
# Setup
INVESTIGATION_TYPE="Temporal Non-locality Analysis" \
RESEARCH_FOCUS="Theory B validation across timeframes" \
./tmux_multiagent_setup.sh

# In orchestrator pane:
delegate_to_analyst "Explain 7.55-point precision mechanism"
delegate_to_data "Validate temporal non-locality with permutation tests"
```

### 3. Custom Multi-Agent Research
```bash
# Setup with custom roles
PANE_LEFT_ROLE="coordinator" \
PANE_RT_ROLE="theorist" \
PANE_RB_ROLE="validator" \
INVESTIGATION_TYPE="HTF Cascade Analysis" \
./tmux_multiagent_setup.sh
```

## Tips & Best Practices

### Effective Orchestration
1. **Clear Task Definition**: Provide specific, measurable objectives
2. **Progress Tracking**: Use `update_status` and `capture_pane` regularly  
3. **Quality Gates**: Review outputs before proceeding to next phase
4. **Documentation**: Save important findings to `./artifacts/`

### Troubleshooting
- **Chat Client Issues**: Script falls back to bash if Claude CLI unavailable
- **Pane Identification**: Use `get_pane_by_role <role>` for reliable targeting
- **Status Tracking**: Check `./status/*.json` files for agent state
- **Logs Review**: Timestamped logs in `./logs/` for debugging

### Performance Optimization  
- **Timeout Management**: Adjust `wait_for_completion` timeouts as needed
- **Capture Strategy**: Use targeted captures vs `capture_all_panes` for efficiency
- **Status Frequency**: Balance status updates vs performance overhead

## Integration with Existing IRONFORGE Workflows

The script integrates seamlessly with existing IRONFORGE analysis:

```bash
# Use existing data context
DATA_CONTEXT="/Users/jack/IRONFORGE/data - Enhanced sessions with TGAT discovery" \
RESEARCH_FOCUS="HTF amplification factor validation" \
./tmux_multiagent_setup.sh
```

## Sleep & Monitoring Patterns

The script uses:
- **0.1s polling** in `wait_for_repl()` for chat client detection
- **5s intervals** in `wait_for_completion()` for agent status polling  
- **2s refresh** in status monitor for real-time updates
- **No dedicated monitor.py** - uses shell + jq for JSON status parsing

This provides responsive monitoring without excessive resource usage.