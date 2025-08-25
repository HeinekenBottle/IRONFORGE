#!/usr/bin/env bash
# tmux_multiagent_setup.sh - Generalized 3-pane tmux orchestration for Claude Code multi-agent workflows
# Enhanced for flexible research, analysis, and data processing scenarios

set -euo pipefail

# === Configuration (Customizable) =============================================
# Session and workspace naming
RUN_ID="${RUN_ID:-RUN_$(date -u +%Y%m%d_%H%M%S)_MULTIAGENT}"
SESSION="${SESSION:-multiagent_$(date +%s)}"
ROOT="${ROOT:-$PWD/runs/$RUN_ID}"

# Command to start Claude Code in each pane 
CHAT_CMD="${CHAT_CMD:-claude chat}"   # Default to Claude Code CLI
BACKUP_CMD="${BACKUP_CMD:-bash}"      # Fallback if CHAT_CMD unavailable

# Research configuration (customize per investigation)
INVESTIGATION_TYPE="${INVESTIGATION_TYPE:-General Multi-Agent Analysis}"
RESEARCH_FOCUS="${RESEARCH_FOCUS:-Temporal pattern discovery and statistical validation}"
DATA_CONTEXT="${DATA_CONTEXT:-/Users/jack/IRONFORGE/data - Enhanced sessions with archaeological zones}"

# Analysis parameters (statistical)
CONFIDENCE_LEVEL="${CONFIDENCE_LEVEL:-0.95}"
FDR_Q_VALUE="${FDR_Q_VALUE:-0.10}"
PERMUTATION_ITERATIONS="${PERMUTATION_ITERATIONS:-2000}"

# Pane assignments (customizable roles)
PANE_LEFT_ROLE="${PANE_LEFT_ROLE:-orchestrator}"
PANE_RT_ROLE="${PANE_RT_ROLE:-analyst}"  
PANE_RB_ROLE="${PANE_RB_ROLE:-data}"

# === Dependency Checks ========================================================
command -v tmux >/dev/null 2>&1 || { 
    echo "❌ tmux not found. Install tmux: brew install tmux"; exit 1; 
}
command -v jq >/dev/null 2>&1 || { 
    echo "❌ jq not found. Install jq: brew install jq"; exit 1; 
}

# Check if Claude Code CLI is available
if ! command -v claude >/dev/null 2>&1; then
    echo "⚠️  Claude CLI not found, falling back to bash shells"
    CHAT_CMD="$BACKUP_CMD"
fi

# === Workspace Setup ==========================================================
echo "🏗️  Setting up workspace: $ROOT"
mkdir -p "$ROOT"/{status,logs,artifacts,prompts,scripts,capture}
touch "$ROOT/logs"/{${PANE_LEFT_ROLE}.log,${PANE_RT_ROLE}.log,${PANE_RB_ROLE}.log}

# Create investigation manifest
cat > "$ROOT/investigation_manifest.md" <<EOF
# Multi-Agent Investigation: $RUN_ID

**Investigation Type:** $INVESTIGATION_TYPE  
**Research Focus:** $RESEARCH_FOCUS  
**Data Context:** $DATA_CONTEXT  
**Started:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Agent Configuration
- **Left Pane ($PANE_LEFT_ROLE):** Orchestration and project management
- **Top Right ($PANE_RT_ROLE):** Analysis and hypothesis development  
- **Bottom Right ($PANE_RB_ROLE):** Data processing and validation

## Statistical Parameters  
- Confidence Level: $CONFIDENCE_LEVEL
- FDR q-value: $FDR_Q_VALUE  
- Permutation Iterations: $PERMUTATION_ITERATIONS

## Workspace Structure
- **Status:** ./status/*.json (agent state tracking)
- **Logs:** ./logs/*.log (timestamped pane outputs)  
- **Artifacts:** ./artifacts/ (results, reports, data)
- **Capture:** ./capture/ (pane capture snapshots)
- **Prompts:** ./prompts/ (agent initialization prompts)
EOF

# === Helper Scripts ===========================================================
cat > "$ROOT/scripts/multiagent_helpers.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Core tmux interaction functions
send_file() {  # send_file <target-pane> <file>
    tmux load-buffer -- "$(cat "$2")"
    tmux paste-buffer -t "$1"  
    tmux send-keys -t "$1" Enter
}

send_line() {  # send_line <target-pane> <text>
    tmux load-buffer -- "$2"
    tmux paste-buffer -t "$1"
    tmux send-keys -t "$1" Enter  
}

send_prompt() {  # send_prompt <target-pane> <prompt-text>
    echo "📤 Sending to $1: ${2:0:50}..."
    send_line "$1" "$2"
}

# Pane capture functions
capture_pane() {  # capture_pane <pane> [filename]
    local pane="$1"
    local filename="${2:-capture_$(date +%H%M%S).txt}"
    local capture_path="./runs/$RUN_ID/capture/$filename"
    
    tmux capture-pane -t "$pane" -p > "$capture_path"
    echo "📸 Captured $pane to $capture_path"
}

capture_all_panes() {  # capture_all_panes [suffix]
    local suffix="${1:-$(date +%H%M%S)}"
    local session="$SESSION"
    
    # Get all pane IDs
    readarray -t panes < <(tmux list-panes -t "$session:main" -F "#{pane_id}")
    
    for pane in "${panes[@]}"; do
        capture_pane "$pane" "all_panes_${suffix}_${pane}.txt"
    done
}

# Status management
update_status() {  # update_status <agent> <STATE> <TASK_ID> <NOTE>
    local agent="$1" state="$2" task="${3:-}" note="${4:-}"
    local ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local status_file="./runs/$RUN_ID/status/${agent}.json"
    
    printf '{"state":"%s","task_id":"%s","ts":"%s","note":"%s","agent":"%s"}\n' \
        "$state" "$task" "$ts" "$note" "$agent" > "$status_file"
    echo "📊 Status: $agent -> $state ($task) $note"
}

# Multi-agent communication
delegate_to_analyst() {  # delegate_to_analyst <message>
    local pane_id=$(get_pane_by_role "analyst")
    send_prompt "$pane_id" "$1"
    update_status "orchestrator" "DELEGATING" "$(date +%s)" "Sent to analyst: ${1:0:30}"
}

delegate_to_data() {  # delegate_to_data <message>  
    local pane_id=$(get_pane_by_role "data")
    send_prompt "$pane_id" "$1"
    update_status "orchestrator" "DELEGATING" "$(date +%s)" "Sent to data: ${1:0:30}"
}

# Utility functions
get_pane_by_role() {  # get_pane_by_role <role>
    case "$1" in
        orchestrator|left) echo "$PANE_LEFT" ;;
        analyst|right-top|rt) echo "$PANE_RT" ;;  
        data|right-bottom|rb) echo "$PANE_RB" ;;
        *) echo "❌ Unknown role: $1" >&2; return 1 ;;
    esac
}

# Wait for agent completion (poll status files)
wait_for_completion() {  # wait_for_completion <agent> <timeout_seconds>
    local agent="$1"
    local timeout="${2:-300}"  # 5 minute default
    local status_file="./runs/$RUN_ID/status/${agent}.json"
    local elapsed=0
    
    echo "⏳ Waiting for $agent completion (timeout: ${timeout}s)"
    
    while [ $elapsed -lt $timeout ]; do
        if [ -f "$status_file" ]; then
            local state=$(jq -r '.state // "UNKNOWN"' "$status_file")
            case "$state" in
                "DONE"|"COMPLETE") 
                    echo "✅ $agent completed"
                    return 0
                    ;;
                "ERROR"|"FAILED")
                    echo "❌ $agent failed" 
                    return 1
                    ;;
                "BUSY"|"WORKING")
                    echo "🔄 $agent working... ($elapsed/${timeout}s)"
                    ;;
            esac
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    echo "⏰ Timeout waiting for $agent"
    return 1  
}

# Export all functions for sourcing
export -f send_file send_line send_prompt capture_pane capture_all_panes
export -f update_status delegate_to_analyst delegate_to_data get_pane_by_role wait_for_completion
EOF
chmod +x "$ROOT/scripts/multiagent_helpers.sh"

# === Status Monitor ===========================================================
cat > "$ROOT/scripts/status_monitor.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ID:?RUN_ID must be set}"
WATCH_DIR="./runs/$RUN_ID/status"

echo "🎯 Multi-Agent Status Monitor"
echo "Run ID: $RUN_ID"
echo "Watching: $WATCH_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "Multi-Agent Status @ $(date '+%H:%M:%S')   (RUN: $RUN_ID)"
    echo "================================================================"
    
    if ls "$WATCH_DIR"/*.json >/dev/null 2>&1; then
        for f in "$WATCH_DIR"/*.json; do
            if [ -s "$f" ]; then  # Only process non-empty files
                agent="$(basename "$f" .json)"
                # Enhanced status display with agent role
                jq -r --arg a "$agent" '
                    "🤖 " + ($a | ascii_upcase) + ": " + 
                    (.state // "UNKNOWN") + 
                    (if .task_id and .task_id != "" then " | Task: " + .task_id else "" end) +
                    (if .note and .note != "" then " | " + .note else "" end) +
                    " | " + (.ts // "")
                ' "$f" 2>/dev/null || echo "🤖 $agent: [Invalid JSON]"
            fi
        done | sort
        
        # Show capture files available
        echo ""
        echo "📸 Recent Captures:"
        if ls "./runs/$RUN_ID/capture"/*.txt >/dev/null 2>&1; then
            ls -la "./runs/$RUN_ID/capture"/*.txt | tail -3 | awk '{print "   " $6 " " $7 " " $8 " " $9}'
        else
            echo "   (no captures yet)"
        fi
        
    else
        echo "   (no agent status yet)"
    fi
    
    echo ""
    echo "💡 Helper Commands:"
    echo "   source ./runs/$RUN_ID/scripts/multiagent_helpers.sh  # Load helper functions"
    echo "   capture_all_panes                                    # Snapshot all panes"
    echo "   update_status <agent> <state> <task> <note>          # Update agent status"
    
    sleep 2
done
EOF
chmod +x "$ROOT/scripts/status_monitor.sh"

# === Agent Prompt Templates ===================================================

# Orchestrator/Project Manager Prompt
cat > "$ROOT/prompts/orchestrator.md" <<EOF
# 🎯 MULTI-AGENT ORCHESTRATOR

**Role:** Project Manager & Orchestration Leader  
**Investigation:** $INVESTIGATION_TYPE  
**Focus:** $RESEARCH_FOCUS  
**Run ID:** $RUN_ID  
**Workspace:** $ROOT  

## 🚀 Your Mission
You coordinate a 3-agent research team to investigate: **$RESEARCH_FOCUS**

## 🤖 Team Roster  
- **ANALYST (Right-Top Pane):** Creative hypothesis development, theoretical frameworks
- **DATA (Right-Bottom Pane):** Statistical analysis, data processing, validation
- **YOU:** Strategic planning, delegation, progress tracking, quality assurance

## 📋 Communication Protocol
- **Delegate to Analyst:** Use clear task descriptions with expected deliverables
- **Delegate to Data:** Specify analysis requirements, statistical parameters  
- **Track Progress:** Monitor status files, use capture_pane for progress snapshots
- **Quality Control:** Review outputs before final integration

## 🔧 Available Tools
- \`capture_pane <pane>\` - Snapshot pane output
- \`update_status <agent> <state> <task> <note>\` - Track progress
- Status files: \`./status/*.json\` for inter-agent coordination
- Helper functions: \`source ./scripts/multiagent_helpers.sh\`

## 📊 Investigation Parameters
- **Data Context:** $DATA_CONTEXT
- **Statistical Confidence:** $CONFIDENCE_LEVEL  
- **FDR Correction:** q = $FDR_Q_VALUE
- **Permutation Testing:** $PERMUTATION_ITERATIONS iterations

## 🎯 First Steps
1. **Define Investigation Scope:** Break down the research focus into specific testable questions
2. **Delegate Hypothesis Development:** Send clear theoretical framework requests to ANALYST
3. **Delegate Data Analysis:** Send specific analysis requirements to DATA agent
4. **Coordinate Integration:** Synthesize findings into actionable insights

**Start by outlining your investigation strategy and delegating initial tasks!**
EOF

# Analyst Prompt
cat > "$ROOT/prompts/analyst.md" <<EOF
# 🧠 RESEARCH ANALYST

**Role:** Creative Theorist & Hypothesis Developer  
**Investigation:** $INVESTIGATION_TYPE  
**Focus:** $RESEARCH_FOCUS  
**Run ID:** $RUN_ID  

## 🎯 Your Mission  
Generate creative, testable hypotheses and theoretical frameworks for the investigation.

## 🔬 Analysis Framework
1. **Hypothesis Development:** Create H1-H5+ with clear mechanisms and predictions
2. **Theoretical Modeling:** Develop explanatory frameworks for observed phenomena
3. **Creative Connections:** Identify non-obvious relationships and patterns
4. **Counterexample Generation:** Anticipate alternative explanations and edge cases

## 📊 Deliverable Standards
- **Clear Hypotheses:** Specific, testable predictions with measurable outcomes
- **Mechanism Explanation:** How and why the proposed relationships work  
- **Visual Framework:** ASCII diagrams, flowcharts, or conceptual models
- **Statistical Expectations:** Expected effect sizes, confidence intervals

## 🤝 Collaboration Protocol
- **Status Updates:** Use \`update_status analyst <state> <task> <note>\` 
- **Artifact Creation:** Save hypotheses, frameworks to \`./artifacts/\`
- **Communication:** Respond to orchestrator requests with structured analysis
- **Documentation:** Maintain clear audit trail of theoretical development

## 📁 Data Context
$DATA_CONTEXT

## 🚀 Ready for Instructions
Await specific research questions from the Orchestrator, then develop comprehensive theoretical frameworks!
EOF

# Data Agent Prompt  
cat > "$ROOT/prompts/data.md" <<EOF
# 📊 DATA ANALYST & STATISTICAL VALIDATOR

**Role:** Rigorous Data Processor & Statistical Skeptic  
**Investigation:** $INVESTIGATION_TYPE  
**Focus:** $RESEARCH_FOCUS  
**Run ID:** $RUN_ID  

## 🎯 Your Mission
Execute rigorous statistical analysis AND red-team all results for robustness.

## 🔬 Analysis Standards
- **Statistical Rigor:** FDR correction (q=$FDR_Q_VALUE), permutation testing (N=$PERMUTATION_ITERATIONS)
- **Confidence Intervals:** $CONFIDENCE_LEVEL confidence level for all estimates
- **Robustness Testing:** Alternative windows, baselines, and sensitivity analysis  
- **Effect Size Reporting:** Practical significance alongside statistical significance

## 📋 Deliverable Requirements  
- **Reproducible Code:** Clear, commented analysis scripts
- **Statistical Tables:** Full results with confidence intervals, p-values, effect sizes
- **Visualization:** Publication-quality plots and charts
- **Skeptical Review:** Identify limitations, assumptions, potential confounds

## 🤝 Collaboration Protocol
- **Status Updates:** \`update_status data <state> <task> <note>\`
- **Results Storage:** Save all outputs to \`./artifacts/\`
- **Quality Assurance:** Cross-validate results with alternative methods
- **Documentation:** Detailed methodology and assumption documentation

## 📁 Data Context  
$DATA_CONTEXT

## 📊 Statistical Toolkit
- Permutation testing for null hypothesis generation
- FDR (Benjamini-Hochberg) multiple comparison correction
- Bootstrap confidence intervals for robust estimation
- Sensitivity analysis across parameter ranges

## 🚀 Ready for Analysis
Await specific data analysis requests from the Orchestrator. Execute with statistical rigor!
EOF

# === Start tmux session with 3 panes ==========================================
echo "🚀 Starting tmux session: $SESSION"

# Create session and split panes
tmux new-session -d -s "$SESSION" -n main
tmux split-window -h -t "${SESSION}:main"  # Split horizontally (left|right)
tmux split-window -v -t "${SESSION}:main.1"  # Split right pane vertically (top-right/bottom-right)

# Set layout to tiled for better space distribution
tmux select-layout -t "${SESSION}:main" tiled

# === Capture pane IDs by position =============================================
# Sort panes by position to get consistent assignment
readarray -t __PANES < <(tmux list-panes -t "${SESSION}:main" -F "#{pane_id} #{pane_left} #{pane_top}" | sort -k2,2n -k3,3n)

# Assign panes: left=orchestrator, top-right=analyst, bottom-right=data
PANE_LEFT="${__PANES[0]%% *}"    # Leftmost pane
PANE_RT="${__PANES[1]%% *}"      # Top-right pane  
PANE_RB="${__PANES[2]%% *}"      # Bottom-right pane

# Validate pane assignments
: "${PANE_LEFT:?Failed to get left pane}"
: "${PANE_RT:?Failed to get top-right pane}" 
: "${PANE_RB:?Failed to get bottom-right pane}"

echo "🎭 Pane assignments:"
echo "   Left (Orchestrator): $PANE_LEFT"
echo "   Top-Right (Analyst): $PANE_RT"  
echo "   Bottom-Right (Data): $PANE_RB"

# === Setup logging pipes ======================================================
echo "📝 Setting up logging..."

# Simple pipe-pane logging (avoiding long command line errors)
if command -v ts >/dev/null 2>&1; then
    tmux pipe-pane -o -t "$PANE_LEFT" "ts >> '$ROOT/logs/${PANE_LEFT_ROLE}.log'"
    tmux pipe-pane -o -t "$PANE_RT" "ts >> '$ROOT/logs/${PANE_RT_ROLE}.log'"  
    tmux pipe-pane -o -t "$PANE_RB" "ts >> '$ROOT/logs/${PANE_RB_ROLE}.log'"
else
    # Simple logging without timestamps to avoid command length issues
    tmux pipe-pane -o -t "$PANE_LEFT" "cat >> '$ROOT/logs/${PANE_LEFT_ROLE}.log'"
    tmux pipe-pane -o -t "$PANE_RT" "cat >> '$ROOT/logs/${PANE_RT_ROLE}.log'"
    tmux pipe-pane -o -t "$PANE_RB" "cat >> '$ROOT/logs/${PANE_RB_ROLE}.log'"
fi

# === Chat client initialization ===============================================
echo "💬 Initializing chat clients..."

# Helper functions (inline for setup)
send_file() { 
    tmux load-buffer -- "$(cat "$2")"
    tmux paste-buffer -t "$1"  
    tmux send-keys -t "$1" Enter
}

wait_for_repl() {
    local pane="$1" want="${2:-claude}" tries=0 cmd=""
    echo "⏳ Waiting for REPL in $pane..."
    
    while :; do
        cmd="$(tmux display -p -t "$pane" '#{pane_current_command}' 2>/dev/null || echo '')"
        
        # Accept claude, node (for Claude CLI), or target command
        if [[ "$cmd" == "$want" || "$cmd" == "node" || "$cmd" == "claude" ]]; then
            echo "✅ REPL ready in $pane ($cmd)"
            break
        fi
        
        ((tries++))
        if [[ $tries -ge 150 ]]; then  # 15 second timeout
            echo "⚠️  Timeout waiting for REPL in $pane (cmd: $cmd)"
            break
        fi
        sleep 0.1
    done
}

start_chat() {
    local pane="$1" role="$2"
    echo "🚀 Starting $CHAT_CMD in $pane ($role)"
    
    tmux send-keys -t "$pane" "$CHAT_CMD" C-m
    sleep 0.5  # Brief pause for command to register
    
    local base_cmd="${CHAT_CMD%% *}"  # Extract base command
    wait_for_repl "$pane" "$base_cmd"
}

# Start chat clients in each pane
start_chat "$PANE_LEFT" "$PANE_LEFT_ROLE"
start_chat "$PANE_RT" "$PANE_RT_ROLE"  
start_chat "$PANE_RB" "$PANE_RB_ROLE"

# Brief pause to ensure all clients are ready
sleep 1

# === Load agent prompts ======================================================
echo "📋 Loading agent prompts..."

send_file "$PANE_LEFT" "$ROOT/prompts/orchestrator.md"
send_file "$PANE_RT" "$ROOT/prompts/analyst.md"
send_file "$PANE_RB" "$ROOT/prompts/data.md"

# === Initialize status files =================================================
echo "📊 Initializing status tracking..."

current_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"state":"READY","task_id":"","ts":"%s","note":"initialized","agent":"orchestrator"}\n' "$current_ts" > "$ROOT/status/orchestrator.json"
printf '{"state":"READY","task_id":"","ts":"%s","note":"initialized","agent":"analyst"}\n' "$current_ts" > "$ROOT/status/analyst.json"  
printf '{"state":"READY","task_id":"","ts":"%s","note":"initialized","agent":"data"}\n' "$current_ts" > "$ROOT/status/data.json"

# Export key variables for helper scripts
export RUN_ID SESSION ROOT PANE_LEFT PANE_RT PANE_RB

# === Final setup and instructions ============================================
cat <<MSG

🎉 Multi-Agent Tmux Setup Complete!

📋 Session Information:
   Session:      $SESSION
   Run ID:       $RUN_ID  
   Workspace:    $ROOT
   Investigation: $INVESTIGATION_TYPE
   Focus:        $RESEARCH_FOCUS

🤖 Agent Configuration:
   Left Pane:    $PANE_LEFT_ROLE ($PANE_LEFT)
   Top-Right:    $PANE_RT_ROLE ($PANE_RT)  
   Bottom-Right: $PANE_RB_ROLE ($PANE_RB)

🔧 Management Tools:
   Status Monitor: RUN_ID="$RUN_ID" "$ROOT/scripts/status_monitor.sh"  
   Helper Functions: source "$ROOT/scripts/multiagent_helpers.sh"
   Logs: $ROOT/logs/*.log
   Status: $ROOT/status/*.json

💡 Quick Start Commands:
   # In orchestrator pane (left):
   source ./runs/$RUN_ID/scripts/multiagent_helpers.sh
   delegate_to_analyst "Develop H1-H5 hypotheses for: $RESEARCH_FOCUS"
   delegate_to_data "Validate data quality and provide statistical summary"
   
   # Capture progress:
   capture_all_panes
   update_status orchestrator PLANNING initial_setup "Starting investigation"

🚀 Ready to begin investigation!

Attaching to session...
MSG

# Select orchestrator pane and attach
tmux select-pane -t "$PANE_LEFT"
tmux attach -t "$SESSION"