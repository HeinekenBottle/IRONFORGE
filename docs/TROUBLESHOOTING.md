# IRONFORGE Troubleshooting Guide
**Common Issues and Solutions for Archaeological Discovery System**

---

## 🚨 Quick Diagnostics

### System Health Check
Run this diagnostic script to quickly identify common issues:

```python
#!/usr/bin/env python3
"""IRONFORGE System Diagnostic"""

import sys
import json
import psutil
from pathlib import Path
from datetime import datetime

def run_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🏛️ IRONFORGE System Diagnostics")
    print("=" * 50)
    
    # 1. Import Test
    try:
        from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
        print("✅ IRONFORGE imports working")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 2. System Resources
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.available / 1024 / 1024:.0f}MB available")
    print(f"🖥️  CPU: {psutil.cpu_percent()}% usage")
    
    # 3. Data Directories
    data_paths = ['data/raw', 'data/enhanced', 'data/discoveries']
    for path in data_paths:
        path_obj = Path(path)
        if path_obj.exists():
            file_count = len(list(path_obj.glob('*.json')))
            print(f"📁 {path}: {file_count} files")
        else:
            print(f"❌ {path}: Directory not found")
    
    # 4. Container Initialization
    try:
        import time
        start_time = time.time()
        container = initialize_ironforge_lazy_loading()
        init_time = time.time() - start_time
        print(f"⚡ Container initialization: {init_time:.2f}s")
        
        if init_time > 5.0:
            print("⚠️  Slow initialization (>5s)")
        else:
            print("✅ Initialization within target (<5s)")
            
    except Exception as e:
        print(f"❌ Container initialization failed: {e}")
        return False
    
    print("\n🎯 Diagnostic complete")
    return True

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
```

---

## 🔧 Installation Issues

### Import Errors

#### Problem: `ModuleNotFoundError: No module named 'ironforge'`
```python
# Error message
ModuleNotFoundError: No module named 'ironforge'
```

**Solutions**:
1. **Install IRONFORGE package**:
   ```bash
   cd /path/to/IRONFORGE
   pip install -e .
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   # Ensure IRONFORGE directory is in path
   ```

3. **Verify virtual environment**:
   ```bash
   which python
   pip list | grep ironforge
   ```

#### Problem: `ImportError: cannot import name 'initialize_ironforge_lazy_loading'`
```python
# Error message
ImportError: cannot import name 'initialize_ironforge_lazy_loading'
```

**Solutions**:
1. **Update import path**:
   ```python
   # OLD (incorrect)
   from integration.ironforge_container import initialize_ironforge_lazy_loading
   
   # NEW (correct)
   from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
   ```

2. **Check package structure**:
   ```bash
   ls -la ironforge/integration/
   # Should contain ironforge_container.py
   ```

### Dependency Issues

#### Problem: PyTorch installation errors
```bash
# Error during pip install
ERROR: Could not find a version that satisfies the requirement torch>=1.9.0
```

**Solutions**:
1. **Install PyTorch separately**:
   ```bash
   # CPU version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # GPU version (if CUDA available)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Check Python version compatibility**:
   ```bash
   python --version
   # Ensure Python 3.8+
   ```

---

## 📊 Data Issues

### No Session Data Found

#### Problem: `FileNotFoundError: No session files found in data/raw/`
**Symptoms**:
- Empty discovery results
- "No enhanced sessions found" errors
- Zero patterns discovered

**Solutions**:
1. **Check data directory structure**:
   ```bash
   ls -la data/raw/
   # Should contain *.json session files
   ```

2. **Verify session file format**:
   ```python
   import json
   with open('data/raw/session.json') as f:
       data = json.load(f)
   
   # Check required fields
   required_fields = ['price_movements', 'timestamp', 'session_name']
   for field in required_fields:
       if field not in data:
           print(f"❌ Missing field: {field}")
   ```

3. **Create sample data directory**:
   ```bash
   mkdir -p data/{raw,enhanced,adapted,discoveries}
   ```

### Invalid Session Data Format

#### Problem: `KeyError: 'price_movements'` or similar data access errors
**Solutions**:
1. **Validate session data structure**:
   ```python
   def validate_session_data(session_file):
       """Validate Level 1 JSON session data"""
       with open(session_file) as f:
           data = json.load(f)
       
       # Required top-level fields
       required_fields = {
           'price_movements': list,
           'session_name': str,
           'timestamp': str,
           'enhanced_features': dict
       }
       
       for field, expected_type in required_fields.items():
           if field not in data:
               print(f"❌ Missing required field: {field}")
               return False
           if not isinstance(data[field], expected_type):
               print(f"❌ Invalid type for {field}: expected {expected_type}")
               return False
       
       print("✅ Session data format valid")
       return True
   ```

2. **Check price movements structure**:
   ```python
   # Each price movement should have:
   movement_fields = ['timestamp', 'price', 'event_type']
   for movement in data['price_movements']:
       for field in movement_fields:
           if field not in movement:
               print(f"❌ Missing movement field: {field}")
   ```

---

## 🧠 Discovery Issues

### No Patterns Discovered

#### Problem: Discovery returns empty results or very few patterns
**Symptoms**:
- `len(patterns) == 0` or very low pattern count
- All patterns have low confidence scores
- "No archaeological patterns found" messages

**Diagnostic Steps**:
1. **Check session data quality**:
   ```python
   def diagnose_session_quality(session_file):
       with open(session_file) as f:
           data = json.load(f)
       
       movements = data.get('price_movements', [])
       print(f"Price movements: {len(movements)}")
       
       if len(movements) < 10:
           print("⚠️  Too few price movements for pattern discovery")
           return False
       
       # Check for semantic events
       semantic_events = [m for m in movements if m.get('event_type')]
       print(f"Semantic events: {len(semantic_events)}")
       
       if len(semantic_events) == 0:
           print("⚠️  No semantic events found")
           return False
       
       return True
   ```

2. **Lower confidence threshold temporarily**:
   ```python
   # Temporary diagnostic - lower thresholds
   TGAT_CONFIG = {
       'attention_threshold': 0.7,  # Lower from 0.9
       'confidence_threshold': 0.5,  # Lower from 0.7
       'permanence_threshold': 0.5   # Lower from 0.7
   }
   ```

3. **Check TGAT model state**:
   ```python
   # Verify model is properly initialized
   discovery_engine = container.get_tgat_discovery()
   print(f"Model parameters: {sum(p.numel() for p in discovery_engine.model.parameters())}")
   
   # Check if model needs training
   if not hasattr(discovery_engine, '_trained'):
       print("⚠️  Model may need training on sample data")
   ```

### Low Quality Patterns

#### Problem: Patterns discovered but with low authenticity scores
**Solutions**:
1. **Analyze quality metrics**:
   ```python
   def analyze_pattern_quality(patterns):
       if not patterns:
           print("❌ No patterns to analyze")
           return
       
       confidences = [p.get('confidence', 0) for p in patterns]
       authenticity_scores = [p.get('authenticity_score', 0) for p in patterns]
       
       print(f"Average confidence: {sum(confidences)/len(confidences):.2f}")
       print(f"Average authenticity: {sum(authenticity_scores)/len(authenticity_scores):.1f}")
       
       low_quality = [p for p in patterns if p.get('confidence', 0) < 0.7]
       print(f"Low quality patterns: {len(low_quality)}/{len(patterns)}")
   ```

2. **Check for data contamination**:
   ```python
   # Check duplication rate
   descriptions = [p.get('description', '') for p in patterns]
   unique_descriptions = set(descriptions)
   duplication_rate = 1 - (len(unique_descriptions) / len(descriptions))
   
   if duplication_rate > 0.25:
       print(f"⚠️  High duplication rate: {duplication_rate:.1%}")
   ```

---

## ⚡ Performance Issues

### Slow Initialization

#### Problem: Container initialization takes >5 seconds
**Diagnostic**:
```python
import time
import cProfile

def profile_initialization():
    """Profile container initialization"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    container = initialize_ironforge_lazy_loading()
    init_time = time.time() - start_time
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    
    print(f"Total initialization time: {init_time:.2f}s")
    return init_time

# Run profiling
profile_initialization()
```

**Solutions**:
1. **Check system resources**:
   ```python
   import psutil
   
   # Memory check
   memory = psutil.virtual_memory()
   if memory.available < 1024 * 1024 * 1024:  # 1GB
       print("⚠️  Low memory available")
   
   # CPU check
   cpu_percent = psutil.cpu_percent(interval=1)
   if cpu_percent > 80:
       print("⚠️  High CPU usage")
   ```

2. **Disable unnecessary components**:
   ```python
   # Minimal initialization for testing
   IRONFORGE_CONFIG = {
       'lazy_loading': True,
       'enable_monitoring': False,  # Disable for faster init
       'enable_caching': False      # Disable for testing
   }
   ```

### High Memory Usage

#### Problem: Memory usage exceeds expected limits
**Diagnostic**:
```python
import psutil
import gc

def monitor_memory_usage():
    """Monitor memory usage during processing"""
    process = psutil.Process()
    
    print(f"Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Initialize container
    container = initialize_ironforge_lazy_loading()
    print(f"After container: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Process session
    builder = container.get_enhanced_graph_builder()
    print(f"After graph builder: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Force garbage collection
    gc.collect()
    print(f"After GC: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

**Solutions**:
1. **Enable memory optimization**:
   ```python
   # Optimize for memory usage
   IRONFORGE_CONFIG = {
       'max_memory_mb': 500,        # Lower limit
       'enable_caching': False,     # Reduce cache memory
       'batch_size': 5              # Smaller batches
   }
   ```

2. **Process sessions individually**:
   ```python
   # Instead of batch processing
   for session_file in session_files:
       # Process one at a time
       patterns = process_single_session(session_file)
       # Clear memory between sessions
       gc.collect()
   ```

---

## 🔗 Integration Issues

### Iron-Core Integration Problems

#### Problem: `ImportError: No module named 'iron_core'`
**Solutions**:
1. **Install iron-core separately**:
   ```bash
   cd iron_core
   pip install -e .
   ```

2. **Check iron-core installation**:
   ```python
   try:
       import iron_core
       print(f"✅ Iron-core version: {iron_core.__version__}")
   except ImportError:
       print("❌ Iron-core not installed")
   ```

### Container Dependency Issues

#### Problem: Components not loading through container
**Diagnostic**:
```python
def test_container_components():
    """Test all container components"""
    container = initialize_ironforge_lazy_loading()
    
    components = [
        'get_enhanced_graph_builder',
        'get_tgat_discovery',
        'get_pattern_graduation',
        'get_broad_spectrum_archaeology'
    ]
    
    for component_name in components:
        try:
            component = getattr(container, component_name)()
            print(f"✅ {component_name}: OK")
        except Exception as e:
            print(f"❌ {component_name}: {e}")
```

---

## 📝 Logging and Debugging

### Enable Debug Logging
```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ironforge_debug.log'),
        logging.StreamHandler()
    ]
)

# Set specific loggers
logging.getLogger('ironforge').setLevel(logging.DEBUG)
logging.getLogger('torch').setLevel(logging.WARNING)  # Reduce PyTorch noise
```

### Debug Session Processing
```python
def debug_session_processing(session_file):
    """Debug session processing step by step"""
    print(f"🔍 Debugging session: {session_file}")
    
    # 1. Load session data
    try:
        with open(session_file) as f:
            session_data = json.load(f)
        print(f"✅ Session data loaded: {len(session_data.get('price_movements', []))} movements")
    except Exception as e:
        print(f"❌ Failed to load session: {e}")
        return
    
    # 2. Build graph
    try:
        builder = container.get_enhanced_graph_builder()
        graph = builder.enhance_session(session_data)
        print(f"✅ Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
    except Exception as e:
        print(f"❌ Failed to build graph: {e}")
        return
    
    # 3. Discover patterns
    try:
        discovery = container.get_tgat_discovery()
        patterns = discovery.discover_patterns(graph)
        print(f"✅ Patterns discovered: {len(patterns)}")
    except Exception as e:
        print(f"❌ Failed to discover patterns: {e}")
        return
    
    return patterns
```

---

## 🆘 Getting Help

### Community Resources
- **Documentation**: Check [Architecture](ARCHITECTURE.md) and [API Reference](API_REFERENCE.md)
- **Examples**: Review [Getting Started](GETTING_STARTED.md) and [User Guide](USER_GUIDE.md)
- **Migration**: See [Migration Guide](MIGRATION_GUIDE.md) for version updates

### Reporting Issues
When reporting issues, include:
1. **System Information**: OS, Python version, memory, CPU
2. **IRONFORGE Version**: Check `pip list | grep ironforge`
3. **Error Messages**: Complete stack traces
4. **Data Sample**: Minimal example that reproduces the issue
5. **Configuration**: Any custom configuration settings

### Emergency Recovery
If IRONFORGE is completely broken:
1. **Clean reinstall**:
   ```bash
   pip uninstall ironforge iron-core
   rm -rf ironforge.egg-info iron_core.egg-info
   pip install -e .
   cd iron_core && pip install -e .
   ```

2. **Reset configuration**:
   ```bash
   rm -rf ~/.ironforge/config
   rm -rf discovery_cache/
   ```

3. **Verify with minimal test**:
   ```python
   from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
   container = initialize_ironforge_lazy_loading()
   print("✅ IRONFORGE recovered")
   ```

---

*For additional support, ensure you have the latest version and have followed the [Getting Started](GETTING_STARTED.md) guide completely.*
