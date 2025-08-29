#!/bin/bash
# Architecture Decision Context Gatherer
# Usage: ./arch_analyzer.sh [component_name]

echo "🏗️ ARCHITECTURE ANALYSIS REPORT"
echo "======================================="
echo "Generated: $(date)"
echo "Directory: $(pwd)"
echo "Component Focus: ${1:-'entire system'}"
echo ""

COMPONENT="${1:-}"

# Project structure overview
echo "📁 PROJECT STRUCTURE OVERVIEW:"
echo "-------------------------------"
echo "Top-level directories:"
find . -maxdepth 2 -type d -not -path "./.git*" -not -path "./__pycache__*" | \
sort | head -15 | while read dir; do
    file_count=$(find "$dir" -name "*.py" 2>/dev/null | wc -l)
    echo "  - $dir ($file_count Python files)"
done
echo ""

# Core components identification
echo "🔧 CORE COMPONENTS:"
echo "-------------------"
if [ -n "$COMPONENT" ]; then
    echo "Focusing on component: $COMPONENT"
    find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
    xargs grep -l -i "$COMPONENT" | head -10 | \
    while read file; do
        echo "  - $file (references $COMPONENT)"
    done
else
    echo "Main modules (by file size and complexity):"
    find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
    xargs wc -l | sort -nr | head -10 | grep -v "total" | \
    while read lines file; do
        class_count=$(grep -c "^class " "$file" 2>/dev/null || echo 0)
        func_count=$(grep -c "^def " "$file" 2>/dev/null || echo 0)
        echo "  - $file: $lines lines, $class_count classes, $func_count functions"
    done
fi
echo ""

# Dependency analysis
echo "🔗 DEPENDENCY PATTERNS:"
echo "-----------------------"
echo "External dependencies:"
if [ -f "requirements.txt" ]; then
    echo "  From requirements.txt:"
    head -10 requirements.txt | sed 's/^/    /'
elif [ -f "pyproject.toml" ]; then
    echo "  From pyproject.toml:"
    grep -A 10 "dependencies" pyproject.toml | head -10 | sed 's/^/    /'
elif [ -f "setup.py" ]; then
    echo "  From setup.py:"
    grep -A 5 "install_requires" setup.py | head -10 | sed 's/^/    /'
else
    echo "  No dependency file found"
fi
echo ""

echo "Internal imports (most connected modules):"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
while read file; do
    internal_imports=$(grep "from \." "$file" 2>/dev/null | wc -l)
    if [ "$internal_imports" -gt 3 ]; then
        echo "  - $file: $internal_imports internal imports"
    fi
done | sort -k3 -nr | head -5
echo ""

# Design patterns identification
echo "🎨 DESIGN PATTERNS:"
echo "-------------------"
echo "Detected patterns:"

# Factory patterns
factory_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "def create_\|Factory" | wc -l)
echo "  - Factory pattern: $factory_files files"

# Singleton patterns  
singleton_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "__new__\|_instance" | wc -l)
echo "  - Singleton pattern: $singleton_files files"

# Observer patterns
observer_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "observer\|notify\|subscribe" | wc -l)
echo "  - Observer pattern: $observer_files files"

# Strategy patterns
strategy_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "strategy\|algorithm" | wc -l)
echo "  - Strategy pattern: $strategy_files files"

echo ""

# Configuration management
echo "⚙️ CONFIGURATION MANAGEMENT:"
echo "----------------------------"
config_files=$(find . -name "config*.py" -o -name "settings*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" | \
grep -v ".git" | wc -l)
echo "Configuration files found: $config_files"

echo "Configuration patterns:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "getenv\|config\|settings" | head -5 | \
while read file; do
    echo "  - $file (uses configuration)"
done
echo ""

# Error handling architecture
echo "🛡️ ERROR HANDLING ARCHITECTURE:"
echo "-------------------------------"
exception_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "class.*Exception\|raise.*Error" | wc -l)
echo "Custom exception classes: $exception_files files"

echo "Error handling patterns:"
find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
while read file; do
    try_blocks=$(grep -c "try:" "$file" 2>/dev/null || echo 0)
    if [ "$try_blocks" -gt 2 ]; then
        echo "  - $file: $try_blocks try/except blocks"
    fi
done | sort -k3 -nr | head -5
echo ""

# Performance architecture
echo "⚡ PERFORMANCE ARCHITECTURE:"
echo "---------------------------"
async_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "async def\|await" | wc -l)
echo "Async/await usage: $async_files files"

cache_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l "cache\|@lru_cache" | wc -l)
echo "Caching patterns: $cache_files files"

echo ""

# Testing architecture
echo "🧪 TESTING ARCHITECTURE:"
echo "------------------------"
if [ -d "tests" ]; then
    test_files=$(find tests -name "*.py" | wc -l)
    echo "Test files: $test_files"
    echo "Test structure:"
    find tests -type d | head -5 | while read dir; do
        echo "  - $dir"
    done
else
    echo "No dedicated tests directory found"
fi

# Look for test files elsewhere
other_tests=$(find . -name "*test*.py" -not -path "./tests/*" -not -path "./.git/*" | wc -l)
echo "Test files outside tests/: $other_tests"
echo ""

# Documentation architecture
echo "📚 DOCUMENTATION ARCHITECTURE:"
echo "------------------------------"
doc_files=$(find . -name "*.md" -o -name "*.rst" -o -name "*.txt" | grep -v ".git" | wc -l)
echo "Documentation files: $doc_files"

docstring_coverage=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | \
xargs grep -l '"""' | wc -l)
python_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./__pycache__/*" | wc -l)
if [ "$python_files" -gt 0 ]; then
    docstring_percent=$((docstring_coverage * 100 / python_files))
    echo "Docstring coverage: ~$docstring_percent% of Python files"
fi
echo ""

echo "======================================="
echo "💡 ARCHITECTURE RECOMMENDATIONS:"
echo "1. Review high-complexity modules for refactoring opportunities"
echo "2. Ensure consistent error handling patterns"
echo "3. Consider async patterns for I/O-heavy operations"
echo "4. Evaluate configuration management approach"
echo "5. Use Claude Code knowledge-architect agent for detailed analysis"
echo ""
echo "Example Claude Code usage:"
echo '  "Use the knowledge-architect agent to analyze this architecture report and suggest improvements"'