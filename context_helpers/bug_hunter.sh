#!/bin/bash
# Bug Hunter Context Gatherer
# Usage: ./bug_hunter.sh [optional_search_term]

echo "🐛 BUG HUNTING CONTEXT REPORT"
echo "========================================="
echo "Generated: $(date)"
echo "Directory: $(pwd)"
echo "Search Term: ${1:-'all common issues'}"
echo ""

SEARCH_TERM="${1:-}"

# Recent changes that might introduce bugs
echo "📝 RECENT CHANGES (Last 7 days):"
echo "--------------------------------"
git log --since="7 days ago" --oneline --decorate | head -10
echo ""

# Error handling patterns
echo "❌ ERROR HANDLING PATTERNS:"
echo "----------------------------"
if [ -n "$SEARCH_TERM" ]; then
    echo "Searching for: $SEARCH_TERM"
    find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
    xargs grep -n -i "$SEARCH_TERM" | head -10
    echo ""
fi

# Common error patterns
echo "🚨 COMMON ERROR INDICATORS:"
echo "---------------------------"
echo "Exception handling:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -n "except.*:" | wc -l | xargs echo "  - Exception blocks found:"

echo "TODO/FIXME items:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -n -i "TODO\|FIXME\|BUG\|HACK" | wc -l | xargs echo "  - TODO/FIXME items:"

echo "Print/debug statements:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -n "print(" | wc -l | xargs echo "  - Print statements:"

echo ""

# Files with most recent modifications
echo "🔄 RECENTLY MODIFIED FILES:"
echo "---------------------------"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" -mtime -7 | \
head -10 | while read file; do
    echo "  - $file ($(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$file"))"
done
echo ""

# Test failures or missing tests
echo "🧪 TEST COVERAGE INDICATORS:"
echo "----------------------------"
if [ -d "tests" ]; then
    echo "Test directory exists: ✓"
    echo "Test files found: $(find tests -name "*.py" | wc -l)"
else
    echo "No tests directory found: ⚠️"
fi

if command -v pytest >/dev/null 2>&1; then
    echo "Pytest available: ✓"
else
    echo "Pytest not available: ⚠️"
fi
echo ""

# Performance indicators
echo "⚡ PERFORMANCE INDICATORS:"
echo "-------------------------"
echo "Large files (>1000 lines):"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs wc -l | sort -nr | head -5 | grep -v "total" | \
while read lines file; do
    if [ "$lines" -gt 1000 ]; then
        echo "  - $file: $lines lines"
    fi
done
echo ""

# Import complexity
echo "📦 IMPORT COMPLEXITY:"
echo "---------------------"
echo "Files with many imports (>10):"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
while read file; do
    import_count=$(grep -c "^import\|^from.*import" "$file" 2>/dev/null || echo 0)
    if [ "$import_count" -gt 10 ]; then
        echo "  - $file: $import_count imports"
    fi
done | head -5
echo ""

# Circular import risks
echo "🔄 POTENTIAL CIRCULAR IMPORT RISKS:"
echo "-----------------------------------"
echo "Files importing from parent directories:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "from \.\." | head -5 | \
while read file; do
    echo "  - $file"
done
echo ""

echo "========================================="
echo "💡 NEXT STEPS:"
echo "1. Review recent changes for potential issues"
echo "2. Check files with high complexity"
echo "3. Investigate TODO/FIXME items"
echo "4. Run tests if available"
echo "5. Use Claude Code general-purpose agent for deeper analysis"
echo ""
echo "Example Claude Code usage:"
echo '  "Use the general-purpose agent to analyze the files flagged in this bug hunting report"'