# IRONFORGE Deployment Guide
**Production Deployment for Archaeological Discovery System**

---

## 🎯 Deployment Overview

This guide covers deploying IRONFORGE in production environments for systematic archaeological discovery of market patterns. The system is designed for high-performance, reliable operation with comprehensive monitoring and automated workflows.

### Production Requirements
- **Performance**: <5s initialization, <3s per session processing
- **Reliability**: 99.9% uptime, automatic error recovery
- **Scalability**: Handle 100+ sessions daily, batch processing
- **Quality**: >87% authenticity threshold, <25% duplication rate

---

## 🏗️ Infrastructure Requirements

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB SSD (100GB recommended)
- **Python**: 3.8+ with PyTorch 1.9.0+
- **Network**: Stable internet for data updates

#### Recommended Production Setup
- **CPU**: 8 cores, 3.0GHz+ (Intel Xeon or AMD EPYC)
- **RAM**: 32GB DDR4
- **Storage**: 200GB NVMe SSD
- **GPU**: Optional CUDA-compatible GPU for acceleration
- **OS**: Ubuntu 20.04 LTS or CentOS 8+

### Environment Setup

```bash
# 1. Create production user
sudo useradd -m -s /bin/bash ironforge
sudo usermod -aG sudo ironforge

# 2. Set up Python environment
sudo apt update && sudo apt install -y python3.8 python3.8-venv python3.8-dev
sudo apt install -y build-essential git curl

# 3. Create application directory
sudo mkdir -p /opt/ironforge
sudo chown ironforge:ironforge /opt/ironforge
```

---

## 📦 Production Installation

### 1. Application Deployment

```bash
# Switch to production user
sudo su - ironforge
cd /opt/ironforge

# Clone repository
git clone <repository-url> .
git checkout main  # or specific release tag

# Create production virtual environment
python3.8 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install IRONFORGE in production mode
pip install -e .
```

### 2. Configuration Setup

Create production configuration:

```bash
# Create configuration directory
mkdir -p /opt/ironforge/config

# Production configuration
cat > /opt/ironforge/config/production.py << 'EOF'
# IRONFORGE Production Configuration

import os
from pathlib import Path

# Base paths
BASE_DIR = Path("/opt/ironforge")
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
for dir_path in [DATA_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

IRONFORGE_CONFIG = {
    # Data paths
    'raw_data_path': str(DATA_DIR / 'raw'),
    'enhanced_data_path': str(DATA_DIR / 'enhanced'),
    'discoveries_path': str(DATA_DIR / 'discoveries'),
    'cache_path': str(CACHE_DIR),
    
    # Processing settings
    'max_sessions_per_batch': 20,  # Increased for production
    'discovery_timeout_seconds': 600,  # 10 minutes
    'enable_caching': True,
    'cache_ttl_hours': 24,
    
    # Quality thresholds
    'pattern_confidence_threshold': 0.75,  # Higher for production
    'authenticity_threshold': 87.0,
    'max_duplication_rate': 0.20,  # Stricter for production
    
    # Performance settings
    'lazy_loading': True,
    'max_memory_mb': 2000,  # Increased for production
    'enable_monitoring': True,
    'monitoring_interval_seconds': 300,
    
    # Logging
    'log_level': 'INFO',
    'log_file': str(LOGS_DIR / 'ironforge.log'),
    'max_log_size_mb': 100,
    'log_backup_count': 5,
    
    # Security
    'enable_api_auth': True,
    'api_key_required': True,
    'rate_limit_per_hour': 1000
}

# TGAT Production Configuration
TGAT_CONFIG = {
    'node_features': 45,
    'edge_features': 20,
    'hidden_dim': 128,
    'num_heads': 4,
    'dropout': 0.05,  # Reduced for production
    'learning_rate': 0.0005,  # Conservative for stability
    'max_epochs': 200,
    'early_stopping_patience': 20,
    'model_checkpoint_interval': 50
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'enable_performance_tracking': True,
    'enable_quality_monitoring': True,
    'enable_error_alerting': True,
    'metrics_retention_days': 30,
    'alert_thresholds': {
        'processing_time_seconds': 10.0,
        'memory_usage_percent': 80.0,
        'error_rate_percent': 5.0,
        'pattern_quality_score': 85.0
    }
}
EOF
```

### 3. Data Directory Structure

```bash
# Create production data structure
mkdir -p /opt/ironforge/data/{raw,enhanced,adapted,discoveries}
mkdir -p /opt/ironforge/logs
mkdir -p /opt/ironforge/cache/{discovery_results,pattern_intelligence,daily_workflows}
mkdir -p /opt/ironforge/backups

# Set proper permissions
chmod -R 755 /opt/ironforge/data
chmod -R 755 /opt/ironforge/logs
chmod -R 755 /opt/ironforge/cache
```

---

## 🔧 Service Configuration

### 1. Systemd Service Setup

Create systemd service for IRONFORGE:

```bash
sudo cat > /etc/systemd/system/ironforge.service << 'EOF'
[Unit]
Description=IRONFORGE Archaeological Discovery System
After=network.target
Wants=network.target

[Service]
Type=simple
User=ironforge
Group=ironforge
WorkingDirectory=/opt/ironforge
Environment=PYTHONPATH=/opt/ironforge
Environment=IRONFORGE_ENV=production
ExecStart=/opt/ironforge/venv/bin/python -m ironforge.services.discovery_service
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ironforge

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ironforge
sudo systemctl start ironforge
```

### 2. Discovery Service Implementation

Create the main discovery service:

```bash
mkdir -p /opt/ironforge/ironforge/services

cat > /opt/ironforge/ironforge/services/discovery_service.py << 'EOF'
#!/usr/bin/env python3
"""
IRONFORGE Production Discovery Service
Continuous archaeological pattern discovery service
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import IRONFORGE components
from ironforge.integration.ironforge_container import initialize_ironforge_lazy_loading
from ironforge.analysis.daily_discovery_workflows import DailyDiscoveryWorkflows
from ironforge.utilities.performance_monitor import PerformanceMonitor

# Load production configuration
sys.path.append('/opt/ironforge/config')
from production import IRONFORGE_CONFIG, MONITORING_CONFIG

class IRONFORGEDiscoveryService:
    """Production discovery service for continuous pattern discovery"""
    
    def __init__(self):
        self.setup_logging()
        self.running = True
        self.container = None
        self.workflows = None
        self.performance_monitor = None
        
    def setup_logging(self):
        """Configure production logging"""
        logging.basicConfig(
            level=getattr(logging, IRONFORGE_CONFIG['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(IRONFORGE_CONFIG['log_file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('ironforge.service')
        
    async def initialize(self):
        """Initialize IRONFORGE system"""
        try:
            self.logger.info("Initializing IRONFORGE Discovery Service...")
            
            # Initialize lazy loading container
            self.container = initialize_ironforge_lazy_loading()
            self.workflows = DailyDiscoveryWorkflows()
            self.performance_monitor = PerformanceMonitor()
            
            self.logger.info("IRONFORGE Discovery Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            return False
    
    async def run_discovery_cycle(self):
        """Run single discovery cycle"""
        try:
            cycle_start = datetime.now()
            self.logger.info(f"Starting discovery cycle at {cycle_start}")
            
            # Run morning preparation
            morning_analysis = self.workflows.morning_market_analysis()
            
            # Process focus sessions
            session_results = {}
            for session_type in ['NY_AM', 'NY_PM', 'LONDON', 'ASIA']:
                try:
                    result = self.workflows.hunt_session_patterns(session_type)
                    session_results[session_type] = result
                    self.logger.info(f"Processed {session_type}: {len(result.patterns_found)} patterns")
                except Exception as e:
                    self.logger.warning(f"Failed to process {session_type}: {e}")
            
            # Save results
            cycle_results = {
                'timestamp': cycle_start.isoformat(),
                'morning_analysis': morning_analysis,
                'session_results': session_results,
                'performance_metrics': self.performance_monitor.get_metrics()
            }
            
            # Save to cache
            cache_file = Path(IRONFORGE_CONFIG['cache_path']) / f"discovery_cycle_{cycle_start.strftime('%Y%m%d_%H%M%S')}.json"
            with open(cache_file, 'w') as f:
                json.dump(cycle_results, f, indent=2, default=str)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Discovery cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Discovery cycle failed: {e}")
    
    async def monitoring_loop(self):
        """Continuous monitoring and discovery loop"""
        while self.running:
            try:
                await self.run_discovery_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(MONITORING_CONFIG.get('monitoring_interval_seconds', 300))
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def run(self):
        """Main service run method"""
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Initialize system
        if not await self.initialize():
            sys.exit(1)
        
        # Start monitoring loop
        try:
            await self.monitoring_loop()
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        finally:
            self.logger.info("IRONFORGE Discovery Service stopped")

if __name__ == "__main__":
    service = IRONFORGEDiscoveryService()
    asyncio.run(service.run())
EOF

chmod +x /opt/ironforge/ironforge/services/discovery_service.py
```

---

## 📊 Monitoring & Alerting

### 1. Performance Monitoring

```bash
# Create monitoring script
cat > /opt/ironforge/scripts/monitor_performance.py << 'EOF'
#!/usr/bin/env python3
"""
IRONFORGE Performance Monitoring Script
"""

import psutil
import json
import time
from datetime import datetime
from pathlib import Path

def collect_metrics():
    """Collect system and IRONFORGE metrics"""
    return {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/opt/ironforge').percent,
            'load_average': psutil.getloadavg()
        },
        'ironforge': {
            'service_status': 'active' if is_service_running() else 'inactive',
            'cache_size_mb': get_cache_size(),
            'log_size_mb': get_log_size(),
            'last_discovery_time': get_last_discovery_time()
        }
    }

def is_service_running():
    """Check if IRONFORGE service is running"""
    try:
        import subprocess
        result = subprocess.run(['systemctl', 'is-active', 'ironforge'], 
                              capture_output=True, text=True)
        return result.stdout.strip() == 'active'
    except:
        return False

def get_cache_size():
    """Get cache directory size in MB"""
    cache_path = Path('/opt/ironforge/cache')
    if cache_path.exists():
        total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    return 0

def get_log_size():
    """Get log file size in MB"""
    log_file = Path('/opt/ironforge/logs/ironforge.log')
    if log_file.exists():
        return log_file.stat().st_size / (1024 * 1024)
    return 0

def get_last_discovery_time():
    """Get timestamp of last discovery cycle"""
    cache_path = Path('/opt/ironforge/cache')
    discovery_files = list(cache_path.glob('discovery_cycle_*.json'))
    if discovery_files:
        latest_file = max(discovery_files, key=lambda f: f.stat().st_mtime)
        return datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
    return None

if __name__ == "__main__":
    metrics = collect_metrics()
    print(json.dumps(metrics, indent=2))
EOF

chmod +x /opt/ironforge/scripts/monitor_performance.py
```

### 2. Health Check Endpoint

```bash
# Create health check script
cat > /opt/ironforge/scripts/health_check.py << 'EOF'
#!/usr/bin/env python3
"""
IRONFORGE Health Check Script
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

def check_service_health():
    """Comprehensive health check"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    # Check service status
    try:
        import subprocess
        result = subprocess.run(['systemctl', 'is-active', 'ironforge'], 
                              capture_output=True, text=True)
        service_active = result.stdout.strip() == 'active'
        health_status['checks']['service'] = {
            'status': 'pass' if service_active else 'fail',
            'message': 'Service is running' if service_active else 'Service is not running'
        }
    except Exception as e:
        health_status['checks']['service'] = {
            'status': 'fail',
            'message': f'Cannot check service status: {e}'
        }
    
    # Check recent discovery activity
    cache_path = Path('/opt/ironforge/cache')
    discovery_files = list(cache_path.glob('discovery_cycle_*.json'))
    if discovery_files:
        latest_file = max(discovery_files, key=lambda f: f.stat().st_mtime)
        last_discovery = datetime.fromtimestamp(latest_file.stat().st_mtime)
        time_since_last = datetime.now() - last_discovery
        
        if time_since_last < timedelta(hours=1):
            health_status['checks']['discovery_activity'] = {
                'status': 'pass',
                'message': f'Last discovery: {time_since_last.total_seconds():.0f}s ago'
            }
        else:
            health_status['checks']['discovery_activity'] = {
                'status': 'warn',
                'message': f'Last discovery: {time_since_last.total_seconds():.0f}s ago'
            }
            health_status['overall_status'] = 'degraded'
    else:
        health_status['checks']['discovery_activity'] = {
            'status': 'fail',
            'message': 'No recent discovery activity found'
        }
        health_status['overall_status'] = 'unhealthy'
    
    # Check disk space
    import psutil
    disk_usage = psutil.disk_usage('/opt/ironforge')
    disk_percent = (disk_usage.used / disk_usage.total) * 100
    
    if disk_percent < 80:
        health_status['checks']['disk_space'] = {
            'status': 'pass',
            'message': f'Disk usage: {disk_percent:.1f}%'
        }
    elif disk_percent < 90:
        health_status['checks']['disk_space'] = {
            'status': 'warn',
            'message': f'Disk usage: {disk_percent:.1f}%'
        }
        if health_status['overall_status'] == 'healthy':
            health_status['overall_status'] = 'degraded'
    else:
        health_status['checks']['disk_space'] = {
            'status': 'fail',
            'message': f'Disk usage: {disk_percent:.1f}%'
        }
        health_status['overall_status'] = 'unhealthy'
    
    return health_status

if __name__ == "__main__":
    health = check_service_health()
    print(json.dumps(health, indent=2))
    
    # Exit with appropriate code
    if health['overall_status'] == 'healthy':
        sys.exit(0)
    elif health['overall_status'] == 'degraded':
        sys.exit(1)
    else:
        sys.exit(2)
EOF

chmod +x /opt/ironforge/scripts/health_check.py
```

---

## 🔄 Backup & Recovery

### 1. Automated Backup Script

```bash
cat > /opt/ironforge/scripts/backup.sh << 'EOF'
#!/bin/bash
# IRONFORGE Backup Script

BACKUP_DIR="/opt/ironforge/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ironforge_backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup configuration
cp -r /opt/ironforge/config "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup data (excluding raw data - too large)
cp -r /opt/ironforge/data/discoveries "${BACKUP_DIR}/${BACKUP_NAME}/"
cp -r /opt/ironforge/data/enhanced "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup cache (recent results only)
find /opt/ironforge/cache -name "*.json" -mtime -7 -exec cp {} "${BACKUP_DIR}/${BACKUP_NAME}/" \;

# Backup logs (recent only)
find /opt/ironforge/logs -name "*.log" -mtime -7 -exec cp {} "${BACKUP_DIR}/${BACKUP_NAME}/" \;

# Create archive
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

# Keep only last 7 backups
ls -t ironforge_backup_*.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
EOF

chmod +x /opt/ironforge/scripts/backup.sh
```

### 2. Cron Jobs Setup

```bash
# Add cron jobs for automated operations
sudo crontab -u ironforge -e

# Add these lines:
# Backup every day at 2 AM
0 2 * * * /opt/ironforge/scripts/backup.sh >> /opt/ironforge/logs/backup.log 2>&1

# Health check every 5 minutes
*/5 * * * * /opt/ironforge/scripts/health_check.py >> /opt/ironforge/logs/health.log 2>&1

# Performance monitoring every hour
0 * * * * /opt/ironforge/scripts/monitor_performance.py >> /opt/ironforge/logs/performance.log 2>&1

# Log rotation weekly
0 0 * * 0 find /opt/ironforge/logs -name "*.log" -size +100M -exec gzip {} \;
```

---

## 🚀 Production Deployment Checklist

### Pre-Deployment
- [ ] Infrastructure requirements met
- [ ] Production configuration created
- [ ] Service user and directories set up
- [ ] Dependencies installed and tested
- [ ] Backup strategy implemented

### Deployment
- [ ] Application code deployed
- [ ] Configuration files in place
- [ ] Systemd service configured
- [ ] Monitoring scripts installed
- [ ] Cron jobs configured

### Post-Deployment
- [ ] Service starts successfully
- [ ] Health checks passing
- [ ] Discovery cycles running
- [ ] Performance metrics within targets
- [ ] Backup system operational
- [ ] Monitoring and alerting active

### Performance Validation
- [ ] Initialization time <5s
- [ ] Session processing time <3s
- [ ] Memory usage <2GB
- [ ] Pattern quality >87% authenticity
- [ ] Duplication rate <25%

---

*This deployment guide ensures IRONFORGE operates reliably in production environments with comprehensive monitoring, automated backups, and performance optimization.*
