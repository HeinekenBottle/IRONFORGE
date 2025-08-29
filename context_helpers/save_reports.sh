#!/bin/bash
# Save Context Reports for Documentation
# Usage: ./save_reports.sh [bug|arch] [component_name]

TIMESTAMP=$(date "+%Y%m%d_%H%M")
REPORT_DIR="/Users/jack/IRONFORGE/context_reports"

# Create reports directory if it doesn't exist
mkdir -p "$REPORT_DIR"

case "${1:-help}" in
    "bug")
        REPORT_FILE="$REPORT_DIR/bug_report_${TIMESTAMP}.md"
        echo "# Bug Hunting Report - $(date)" > "$REPORT_FILE"
        echo "Generated for: ${2:-entire_system}" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        ./context_helpers/bug_hunter.sh "$2" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "## Claude Code Analysis" >> "$REPORT_FILE"
        echo "<!-- Add Claude Code agent analysis here -->" >> "$REPORT_FILE"
        
        echo "📄 Bug report saved to: $REPORT_FILE"
        ;;
    "arch")
        REPORT_FILE="$REPORT_DIR/arch_report_${TIMESTAMP}.md"
        echo "# Architecture Analysis Report - $(date)" > "$REPORT_FILE"
        echo "Component: ${2:-entire_system}" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        ./context_helpers/arch_analyzer.sh "$2" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "## Architecture Decisions" >> "$REPORT_FILE"
        echo "<!-- Document decisions made based on this analysis -->" >> "$REPORT_FILE"
        
        echo "📄 Architecture report saved to: $REPORT_FILE"
        ;;
    *)
        echo "Usage: ./save_reports.sh [bug|arch] [component_name]"
        echo "Example: ./save_reports.sh bug authentication"
        echo "Example: ./save_reports.sh arch tgat_discovery"
        ;;
esac