#!/bin/bash
# Enhanced Context Workflow Helper
# Your go-to script for bug hunting and architecture decisions

echo "🚀 ENHANCED CONTEXT WORKFLOW HELPER"
echo "===================================="
echo ""

show_usage() {
    echo "Usage: ./workflow_helper.sh [command] [optional_args]"
    echo ""
    echo "Commands:"
    echo "  bug [search_term]     - Run bug hunting analysis"
    echo "  arch [component]      - Run architecture analysis" 
    echo "  quick                 - Quick overview of both"
    echo "  help                  - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./workflow_helper.sh bug authentication"
    echo "  ./workflow_helper.sh arch tgat_discovery"
    echo "  ./workflow_helper.sh quick"
    echo ""
}

run_bug_analysis() {
    echo "🐛 RUNNING BUG HUNTING ANALYSIS..."
    echo "================================="
    ./context_helpers/bug_hunter.sh "$1"
    echo ""
    echo "💡 NEXT STEPS WITH CLAUDE CODE:"
    echo "Copy the above report and use one of these approaches:"
    echo ""
    echo "For targeted bug hunting:"
    echo '  "Use the general-purpose agent to search for potential bugs related to the issues identified in this report"'
    echo ""
    echo "For comprehensive analysis:"
    echo '  "Use the data-scientist agent to analyze these error patterns and suggest fixes"'
    echo ""
}

run_arch_analysis() {
    echo "🏗️ RUNNING ARCHITECTURE ANALYSIS..."
    echo "==================================="
    ./context_helpers/arch_analyzer.sh "$1"
    echo ""
    echo "💡 NEXT STEPS WITH CLAUDE CODE:"
    echo "Copy the above report and use:"
    echo ""
    echo "For architecture decisions:"
    echo '  "Use the knowledge-architect agent to analyze this architecture report and document key decisions"'
    echo ""
    echo "For refactoring suggestions:"
    echo '  "Use the general-purpose agent to suggest refactoring opportunities based on this analysis"'
    echo ""
}

run_quick_overview() {
    echo "⚡ QUICK OVERVIEW MODE"
    echo "====================="
    echo ""
    
    # Quick bug indicators
    echo "🐛 Bug Indicators:"
    recent_changes=$(git log --since="3 days ago" --oneline | wc -l)
    todos=$(find . -name "*.py" -not -path "./.git/*" | xargs grep -i "TODO\|FIXME" | wc -l)
    prints=$(find . -name "*.py" -not -path "./.git/*" | xargs grep "print(" | wc -l)
    
    echo "  - Recent changes (3 days): $recent_changes"
    echo "  - TODO/FIXME items: $todos"
    echo "  - Debug prints: $prints"
    
    if [ "$todos" -gt 10 ]; then
        echo "  ⚠️  High number of TODO items - consider cleanup"
    fi
    
    if [ "$prints" -gt 5 ]; then
        echo "  ⚠️  Many print statements - check for debug code"
    fi
    echo ""
    
    # Quick architecture indicators
    echo "🏗️ Architecture Indicators:"
    python_files=$(find . -name "*.py" -not -path "./.git/*" | wc -l)
    large_files=$(find . -name "*.py" -not -path "./.git/*" | xargs wc -l | awk '$1 > 500' | wc -l)
    
    echo "  - Total Python files: $python_files"
    echo "  - Large files (>500 lines): $large_files"
    
    if [ -f "requirements.txt" ]; then
        deps=$(wc -l < requirements.txt)
        echo "  - Dependencies: $deps"
    fi
    
    if [ "$large_files" -gt 5 ]; then
        echo "  ⚠️  Many large files - consider breaking down"
    fi
    echo ""
    
    echo "💡 For detailed analysis, run:"
    echo "  ./workflow_helper.sh bug    (detailed bug hunting)"
    echo "  ./workflow_helper.sh arch   (detailed architecture)"
}

# Main execution
case "${1:-help}" in
    "bug")
        run_bug_analysis "$2"
        ;;
    "arch")
        run_arch_analysis "$2"
        ;;
    "quick")
        run_quick_overview
        ;;
    "help"|*)
        show_usage
        ;;
esac