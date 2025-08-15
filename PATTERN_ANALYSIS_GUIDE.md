# IRONFORGE Pattern Analysis Guide
## Complete Session Analysis & Pattern Discovery

This guide shows you how to run IRONFORGE over all your logged sessions to extract patterns, data, timing analysis, and generate comprehensive visualizations.

## 🚀 Quick Start (Immediate Results)

### 1. Quick Pattern Discovery
Get immediate insights from your sessions:

```bash
# Analyze first 10 sessions quickly
python3 quick_pattern_discovery.py --limit 10

# Focus on specific session type
python3 quick_pattern_discovery.py --session-type NY_PM --limit 5
python3 quick_pattern_discovery.py --session-type LONDON

# Analyze all sessions (may take longer)
python3 quick_pattern_discovery.py
```

**Output**: Immediate console results showing:
- FVG redelivery events per session
- Expansion phase events per session  
- Session quality analysis
- Top performing sessions
- Session type comparisons
- Quick insights & recommendations

### 2. Real-Time Pattern Monitoring
Monitor for new sessions and get alerts:

```bash
# Start monitoring with default settings
python3 pattern_monitor.py

# Custom alert threshold
python3 pattern_monitor.py --alert-threshold 25

# Monitor specific directory
python3 pattern_monitor.py --watch-dir /path/to/sessions --alert-threshold 15
```

**Features**:
- Real-time session analysis as files are added
- Pattern-based alerts (high FVG activity, expansion phases, etc.)
- Continuous monitoring with summary reports
- Automatic logging of all discoveries

## 📊 Comprehensive Analysis (Full Reports & Graphs)

### 3. Full Session Analysis
Complete analysis with visualizations and detailed reports:

```bash
# Run comprehensive analysis on all sessions
python3 run_full_session_analysis.py
```

**Outputs**:
- `results/session_analysis_results.json` - Complete raw data
- `results/session_patterns_summary.csv` - Summary table for Excel/analysis
- `visualizations/session_dashboard.png` - Overview dashboard
- `visualizations/semantic_events_heatmap.png` - Pattern heatmap
- `visualizations/time_series_analysis.png` - Trends over time
- `reports/comprehensive_analysis_report.txt` - Detailed text report

## 📈 What You Get

### Semantic Pattern Discovery
- **FVG Redelivery Events**: Automatic detection of Fair Value Gap redeliveries
- **Expansion Phase Events**: Market expansion phase identification
- **Consolidation Events**: Consolidation pattern recognition
- **Session Phase Tracking**: Open/mid/close phase analysis
- **Liquidity Events**: Sweep and interaction detection

### Market Intelligence
- **Session Quality Assessment**: Excellent/good/adequate/poor ratings
- **Price Volatility Analysis**: Range and movement patterns
- **Energy State Tracking**: Market energy levels
- **Temporal Analysis**: Time-based pattern distribution
- **Cross-Session Comparisons**: Performance across different session types

### Performance Metrics
- **Processing Efficiency**: Time per session, nodes per second
- **Feature Optimization**: Constant feature filtering results
- **Discovery Rates**: Percentage of semantic events found
- **Archaeological Significance**: Pattern permanence and importance

## 🎯 Example Results

### Quick Discovery Output:
```
🎯 PATTERN DISCOVERY RESULTS
==================================================
📊 OVERALL STATISTICS:
   Total Sessions: 5
   Total Nodes: 253
   Total FVG Events: 41
   Total Expansion Events: 54
   Semantic Discovery Rate: 37.55%

🔥 TOP FVG REDELIVERY SESSIONS:
   1. LONDON (2025-07-30) - 22 events (42.3%)
   2. PREMARKET (2025-07-30) - 19 events (37.3%)

📈 TOP EXPANSION PHASE SESSIONS:
   1. NY_PM (2025-08-05) - 21 events (32.3%)
   2. ASIA (2025-08-05) - 13 events (28.3%)

💡 QUICK INSIGHTS & RECOMMENDATIONS:
   🏆 Most Active Session Type: LONDON
   🎯 Most Active Individual Session: LONDON on 2025-07-30
   📊 Best FVG Discovery Rate: LONDON (42.3%)
```

### Session Analysis Features:
- **Node-level semantic features**: Each market event has 45-dimensional rich features
- **Edge relationships**: 20-dimensional edge features connecting related events
- **Constant feature filtering**: Automatic removal of non-varying features for training
- **Archaeological context**: Session metadata with timing, quality, and characteristics

## 🔧 Advanced Usage

### Custom Analysis Scripts
You can extend the analysis by modifying the scripts:

1. **Add new pattern types**: Modify `_extract_semantic_patterns()` in the analyzer
2. **Custom visualizations**: Add new plots in `generate_visualizations()`
3. **Alert conditions**: Customize `_check_alerts()` in the pattern monitor
4. **Export formats**: Add CSV, Excel, or database exports

### Integration with Other Tools
- **Export to Excel**: Use the CSV output for spreadsheet analysis
- **Database integration**: Modify scripts to save to PostgreSQL/MongoDB
- **API endpoints**: Wrap analyzers in Flask/FastAPI for web access
- **Jupyter notebooks**: Import the analyzer classes for interactive analysis

## 📁 File Structure After Analysis

```
IRONFORGE/
├── results/
│   ├── session_analysis_results.json    # Complete raw data
│   └── session_patterns_summary.csv     # Summary table
├── visualizations/
│   ├── session_dashboard.png            # Overview charts
│   ├── semantic_events_heatmap.png      # Pattern heatmap
│   └── time_series_analysis.png         # Trends over time
├── reports/
│   └── comprehensive_analysis_report.txt # Detailed report
└── pattern_monitoring_log_*.json        # Monitor logs
```

## 🎪 Session Types Analyzed

Your current sessions include:
- **ASIA**: Asian market sessions
- **LONDON**: London market sessions  
- **LUNCH**: Lunch break sessions
- **MIDNIGHT**: Overnight sessions
- **NY_AM**: New York morning sessions
- **NY_PM**: New York afternoon sessions
- **PREMARKET**: Pre-market sessions

Each session type has different characteristics and pattern frequencies that IRONFORGE automatically discovers and analyzes.

## 🚨 Troubleshooting

### Common Issues:
1. **Missing dependencies**: Install with `pip install matplotlib seaborn pandas numpy`
2. **Memory issues**: Use `--limit` flag to process fewer sessions at once
3. **Processing time**: Large sessions may take 10-30 seconds each
4. **File permissions**: Ensure write access to results/, visualizations/, reports/ directories

### Performance Tips:
- Use quick discovery for immediate insights
- Run full analysis during off-hours for large datasets
- Monitor specific session types that interest you most
- Use the CSV output for custom analysis in Excel/R/Python

## ✅ Next Steps

1. **Start with quick discovery** to get immediate insights
2. **Run full analysis** for comprehensive reports and graphs
3. **Set up monitoring** for ongoing pattern detection
4. **Analyze the results** to identify your most profitable session types and patterns
5. **Customize the scripts** for your specific trading strategy needs

The semantic retrofit has transformed IRONFORGE from a generic pattern detector into a comprehensive **market archaeology system** - use these tools to discover the hidden patterns in your trading sessions!
