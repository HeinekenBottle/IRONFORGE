# IRONFORGE Getting Started Guide
**Quick Start for Archaeological Market Pattern Discovery**

---

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+ 
- PyTorch 1.9.0+
- 4GB+ RAM recommended
- Git for repository access

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd IRONFORGE

# 2. Create virtual environment (recommended)
python -m venv ironforge_env
source ironforge_env/bin/activate  # Linux/Mac
# ironforge_env\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading; print('✅ IRONFORGE installed successfully')"
```

---

## 🎯 First Discovery (5 Minutes)

### 1. Basic Pattern Discovery
```python
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
from ironforge.learning.tgat_discovery import IRONFORGEDiscovery

# Initialize the system (lazy loading for performance)
container = initialize_ironforge_lazy_loading()

# Get core components
graph_builder = container.get_enhanced_graph_builder()
discovery_engine = container.get_tgat_discovery()

print("🏛️ IRONFORGE Archaeological Discovery System initialized")
print("Ready for pattern discovery...")
```

### 2. Process Your First Session
```python
import json
from pathlib import Path

# Load a sample session (replace with your data path)
session_file = Path("data/raw/sample_session.json")

if session_file.exists():
    # Load session data
    with open(session_file) as f:
        session_data = json.load(f)
    
    # Build enhanced graph with semantic features
    enhanced_graph = graph_builder.enhance_session(session_data)
    print(f"📊 Graph created: {enhanced_graph.num_nodes} nodes, {enhanced_graph.num_edges} edges")
    
    # Discover archaeological patterns
    discoveries = discovery_engine.discover_patterns(enhanced_graph)
    print(f"🏛️ Discovered {len(discoveries)} archaeological patterns")
    
    # Display first few discoveries
    for i, pattern in enumerate(discoveries[:3]):
        print(f"  {i+1}. {pattern.get('description', 'Pattern discovered')}")
        print(f"     Confidence: {pattern.get('confidence', 0):.2f}")
        print(f"     Session: {pattern.get('session_name', 'Unknown')}")
else:
    print("❌ Sample session file not found. Please add session data to data/raw/")
```

---

## 📁 Project Structure Overview

```
IRONFORGE/
├── ironforge/                    # Main application package
│   ├── learning/                # 🧠 TGAT discovery engine
│   ├── analysis/                # 🔍 Pattern analysis & archaeology
│   ├── synthesis/               # ✅ Pattern validation & graduation
│   ├── integration/             # 🔗 System integration & containers
│   ├── utilities/               # 🛠️ Core utilities & monitoring
│   └── reporting/               # 📊 Analysis reporting
├── iron_core/                   # ⚡ Infrastructure & performance
├── data/                        # 📂 Organized data storage
│   ├── raw/                     # Level 1 raw market data
│   ├── enhanced/                # Enhanced/processed sessions
│   ├── adapted/                 # Adapted sessions with relativity
│   └── discoveries/             # Pattern discoveries
├── scripts/                     # 🔧 Utility scripts
├── tests/                       # 🧪 Comprehensive test suite
├── reports/                     # 📈 Generated reports & visualizations
└── docs/                        # 📚 Documentation
```

---

## 🎨 Daily Workflow Examples

### Morning Market Preparation
```python
from ironforge.analysis.daily_discovery_workflows import morning_prep

# Get comprehensive morning analysis
analysis = morning_prep(days_back=7)

# Results automatically printed with:
# - Dominant pattern types for the day
# - Cross-session continuation signals
# - Current market regime assessment
# - Session-specific focus areas
# - Actionable archaeological insights
```

### Session Pattern Hunting
```python
from ironforge.analysis.daily_discovery_workflows import hunt_patterns

# Focus on specific session type
ny_pm_patterns = hunt_patterns('NY_PM')
london_patterns = hunt_patterns('LONDON')

print(f"NY_PM patterns: {len(ny_pm_patterns)}")
print(f"LONDON patterns: {len(london_patterns)}")
```

### Cross-Session Analysis
```python
from ironforge.analysis.pattern_intelligence import find_similar_patterns

# Find patterns similar to those in NY_PM sessions
similar_patterns = find_similar_patterns('NY_PM', similarity_threshold=0.8)

for pattern in similar_patterns[:5]:
    print(f"- {pattern.description}")
    print(f"  Similarity: {pattern.similarity_score:.2f}")
    print(f"  Session: {pattern.session_name}")
```

---

## 🔧 Configuration

### Basic Configuration
Create `config.py` in your project root:

```python
# IRONFORGE Configuration
IRONFORGE_CONFIG = {
    # Data paths
    'raw_data_path': 'data/raw',
    'enhanced_data_path': 'data/enhanced', 
    'discoveries_path': 'data/discoveries',
    
    # Processing settings
    'max_sessions_per_batch': 10,
    'discovery_timeout_seconds': 300,
    'enable_caching': True,
    
    # Quality thresholds
    'pattern_confidence_threshold': 0.7,
    'authenticity_threshold': 87.0,
    'max_duplication_rate': 0.25,
    
    # Performance settings
    'lazy_loading': True,
    'max_memory_mb': 1000,
    'enable_monitoring': True
}
```

### Advanced Configuration
```python
# TGAT Discovery Settings
TGAT_CONFIG = {
    'node_features': 45,  # 37D base + 8D semantic
    'edge_features': 20,  # 17D base + 3D semantic
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'learning_rate': 0.001
}

# Semantic Feature Settings
SEMANTIC_CONFIG = {
    'preserve_fvg_events': True,
    'preserve_expansion_phases': True,
    'preserve_session_boundaries': True,
    'enable_htf_confluence': True,
    'semantic_relationship_detection': True
}
```

---

## 📊 Understanding Your Results

### Pattern Discovery Output
```python
# Example discovered pattern
pattern = {
    'pattern_id': 'NY_session_RPC_00',
    'session_name': 'NY_session',
    'session_start': '14:30:00',
    'anchor_timeframe': 'multi_timeframe',
    'archaeological_significance': {
        'archaeological_value': 'high_archaeological_value',
        'permanence_score': 0.933
    },
    'semantic_context': {
        'market_regime': 'transitional',
        'event_types': ['fvg_redelivery', 'expansion_phase'],
        'relationship_type': 'confluence_relationship'
    },
    'confidence': 0.87,
    'description': 'Multi-timeframe confluence with FVG redelivery in NY session'
}
```

### Quality Metrics
- **Confidence**: 0.0-1.0 (higher = more reliable pattern)
- **Authenticity**: 0-100 (>87 required for production)
- **Permanence Score**: 0.0-1.0 (pattern stability over time)
- **Archaeological Value**: low/medium/high significance rating

---

## 🚨 Common Issues & Solutions

### 1. Import Errors
```python
# ❌ Wrong (old structure)
from learning.enhanced_graph_builder import EnhancedGraphBuilder

# ✅ Correct (new structure)
from ironforge.learning.enhanced_graph_builder import EnhancedGraphBuilder
```

### 2. No Session Data Found
```bash
# Check data directory structure
ls -la data/raw/
# Should contain *.json session files

# If empty, add your session data:
cp your_sessions/*.json data/raw/
```

### 3. Slow Performance
```python
# Enable lazy loading (default)
container = initialize_ironforge_lazy_loading()

# Check memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
```

### 4. Low Pattern Discovery
```python
# Check session data quality
with open('data/raw/session.json') as f:
    data = json.load(f)
    
print(f"Price movements: {len(data.get('price_movements', []))}")
print(f"Enhanced features: {bool(data.get('enhanced_features'))}")

# Ensure sessions have sufficient data for pattern discovery
```

---

## 🎯 Next Steps

### 1. Explore Advanced Features
- Read the [User Guide](USER_GUIDE.md) for detailed workflows
- Check [Pattern Discovery Guide](PATTERN_DISCOVERY_GUIDE.md) for analysis techniques
- Review [API Reference](API_REFERENCE.md) for complete function documentation

### 2. Set Up Daily Workflows
- Configure morning preparation routines
- Set up session pattern hunting
- Enable cross-session analysis
- Implement pattern intelligence monitoring

### 3. Production Deployment
- Review [Deployment Guide](DEPLOYMENT_GUIDE.md)
- Set up automated discovery workflows
- Configure performance monitoring
- Implement result caching and archival

### 4. Contribute to Development
- Read [Developer Guide](DEVELOPER_GUIDE.md)
- Run the test suite: `python -m pytest tests/`
- Check [Architecture](ARCHITECTURE.md) for system design
- Review [Troubleshooting](TROUBLESHOOTING.md) for common issues

---

## ✅ Verification Checklist

- [ ] IRONFORGE imports working correctly
- [ ] Sample session processed successfully
- [ ] Patterns discovered with confidence scores
- [ ] Daily workflows accessible
- [ ] Configuration files created
- [ ] Data directory structure set up
- [ ] Performance within expected ranges (<5s initialization)

---

**🎉 Congratulations!** You're now ready to begin archaeological discovery of market patterns with IRONFORGE. The system is designed to reveal hidden market structures through advanced temporal graph attention networks while preserving complete semantic context.

*For detailed usage examples and advanced features, continue to the [User Guide](USER_GUIDE.md).*
