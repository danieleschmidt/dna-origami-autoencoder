# 🚀 Deployment Guide

## DNA Origami AutoEncoder Deployment

### Prerequisites

#### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- Storage: 50GB free space
- OS: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- Python: 3.8+

**Recommended for Production:**
- CPU: 16+ cores, 3.0GHz+
- RAM: 64GB+
- Storage: 500GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (optional, for simulation acceleration)
- OS: Ubuntu 22.04 LTS or CentOS 8+

#### Dependencies

**Core Dependencies:**
```bash
# Python packages (automatically installed)
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pillow>=8.3.0
psutil>=5.8.0

# Optional dependencies
torch>=1.12.0  # For neural network features
numba>=0.56.0  # For JIT compilation
```

**System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    git \
    wget \
    curl

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install -y python3-devel git wget curl

# macOS (with Homebrew)
brew install git wget curl
```

### Installation Methods

#### Method 1: PyPI Installation (Recommended)

```bash
# Create virtual environment
python3 -m venv dna_origami_env
source dna_origami_env/bin/activate  # Linux/macOS
# dna_origami_env\Scripts\activate  # Windows

# Install package
pip install dna-origami-autoencoder

# Verify installation
python -c "import dna_origami_ae; print('Installation successful!')"
```

#### Method 2: Source Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

#### Method 3: Docker Installation

```bash
# Pull pre-built image
docker pull dnaorigamiae/dna-origami-autoencoder:latest

# Or build from source
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner
docker build -t dna-origami-ae .

# Run container
docker run -d \
  --name dna-origami-ae \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e DNA_ORIGAMI_MAX_WORKERS=4 \
  dna-origami-ae
```

### Configuration

#### Environment Variables

```bash
# Performance Configuration
export DNA_ORIGAMI_MAX_WORKERS=8
export DNA_ORIGAMI_CACHE_SIZE_MB=1000
export DNA_ORIGAMI_USE_GPU=true
export DNA_ORIGAMI_MEMORY_LIMIT_GB=16

# Data Storage
export DNA_ORIGAMI_DATA_DIR=/opt/dna_origami/data
export DNA_ORIGAMI_CACHE_DIR=/tmp/dna_origami_cache
export DNA_ORIGAMI_LOG_DIR=/var/log/dna_origami

# Compliance & Security
export DNA_ORIGAMI_REGULATIONS=gdpr,ccpa,pdpa
export DNA_ORIGAMI_DATA_RETENTION_DAYS=730
export DNA_ORIGAMI_ENCRYPTION_KEY=your-encryption-key
export DNA_ORIGAMI_AUDIT_ENABLED=true

# Internationalization
export DNA_ORIGAMI_LOCALE=en_US
export DNA_ORIGAMI_TIMEZONE=UTC

# External Services
export DNA_ORIGAMI_REDIS_URL=redis://localhost:6379
export DNA_ORIGAMI_DATABASE_URL=postgresql://user:pass@localhost/dnaorigami
```

#### Configuration File

Create `/etc/dna_origami_ae/config.yaml`:

```yaml
# Global Configuration
global:
  debug: false
  log_level: INFO
  locale: en_US
  timezone: UTC

# Performance Settings
performance:
  max_workers: 8
  cache_size_mb: 1000
  use_gpu: true
  memory_limit_gb: 16
  parallel_config:
    cpu_bound_workers: 8
    io_bound_workers: 32
    chunk_size: 100

# Storage Configuration
storage:
  data_dir: /opt/dna_origami/data
  cache_dir: /tmp/dna_origami_cache
  temp_dir: /tmp/dna_origami_temp
  backup_dir: /backup/dna_origami
  retention_days: 730

# Database Configuration
database:
  url: postgresql://dnaorigami:password@localhost:5432/dnaorigami
  pool_size: 10
  max_overflow: 20

# Cache Configuration
cache:
  redis:
    url: redis://localhost:6379
    db: 0
    max_connections: 100
  memory:
    size_mb: 500
    ttl_seconds: 3600

# Security & Compliance
security:
  encryption_key: ${DNA_ORIGAMI_ENCRYPTION_KEY}
  audit_enabled: true
  regulations: [gdpr, ccpa, pdpa]
  data_anonymization: true
  max_file_size_mb: 100

# External APIs
external_apis:
  github:
    token: ${GITHUB_TOKEN}
    timeout: 30
  
# Monitoring
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_port: 8080
  log_format: json
```

### Deployment Architectures

#### Single Server Deployment

```
┌─────────────────────────────────────┐
│           Load Balancer             │
│         (nginx/Apache)              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│        DNA Origami AE Server        │
│  ┌─────────────┐ ┌─────────────┐   │
│  │  Web API    │ │ Background  │   │
│  │  (FastAPI)  │ │ Workers     │   │
│  └─────────────┘ └─────────────┘   │
│  ┌─────────────┐ ┌─────────────┐   │
│  │   Cache     │ │ File Store  │   │
│  │  (Redis)    │ │ (Local FS)  │   │
│  └─────────────┘ └─────────────┘   │
└─────────────────────────────────────┘
```

#### High Availability Deployment

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │    (HAProxy)    │
                    └─────────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──┐  ┌─────────▼──┐  ┌─────────▼──┐
    │ App Server │  │ App Server │  │ App Server │
    │    #1      │  │    #2      │  │    #3      │
    └─────────┬──┘  └─────────┬──┘  └─────────┬──┘
              │               │               │
              └───────────────┼───────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌───▼────┐              ┌─────▼─────┐            ┌─────▼─────┐
│ Redis  │              │PostgreSQL │            │   File    │
│Cluster │              │ Primary/  │            │  Storage  │
│        │              │ Replica   │            │ (S3/NFS)  │
└────────┘              └───────────┘            └───────────┘
```

#### Microservices Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                             │
│                    (Kong/Istio)                            │
└─────────────┬───────────────┬───────────────┬───────────────┘
              │               │               │
    ┌─────────▼──┐  ┌─────────▼──┐  ┌─────────▼──┐
    │ Encoding   │  │  Design    │  │Simulation  │
    │ Service    │  │  Service   │  │ Service    │
    └─────────┬──┘  └─────────┬──┘  └─────────┬──┘
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │         Shared Services       │
              │ ┌─────────┐ ┌─────────┐      │
              │ │ Message │ │ Storage │      │
              │ │ Queue   │ │Service  │      │
              │ │(RabbitMQ│ │         │      │
              │ └─────────┘ └─────────┘      │
              └───────────────────────────────┘
```

### Container Orchestration

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dna-origami-ae:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DNA_ORIGAMI_MAX_WORKERS=4
      - DNA_ORIGAMI_CACHE_SIZE_MB=1000
      - DNA_ORIGAMI_DATABASE_URL=postgresql://postgres:password@db:5432/dnaorigami
      - DNA_ORIGAMI_REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=dnaorigami
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - dna-origami-ae
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dna-origami-ae
  labels:
    app: dna-origami-ae
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dna-origami-ae
  template:
    metadata:
      labels:
        app: dna-origami-ae
    spec:
      containers:
      - name: dna-origami-ae
        image: dnaorigamiae/dna-origami-autoencoder:latest
        ports:
        - containerPort: 8000
        env:
        - name: DNA_ORIGAMI_MAX_WORKERS
          value: "4"
        - name: DNA_ORIGAMI_CACHE_SIZE_MB
          value: "1000"
        - name: DNA_ORIGAMI_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: dna-origami-data
      - name: config-volume
        configMap:
          name: dna-origami-config

---
apiVersion: v1
kind: Service
metadata:
  name: dna-origami-ae-service
spec:
  selector:
    app: dna-origami-ae
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dna-origami-data
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dna-origami-config
data:
  config.yaml: |
    global:
      log_level: INFO
      locale: en_US
    performance:
      max_workers: 4
      cache_size_mb: 1000
    storage:
      data_dir: /app/data
```

Create `k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  url: cG9zdGdyZXNxbDovL3Bvc3RncmVzOnBhc3N3b3JkQGRiOjU0MzIvZG5hb3JpZ2FtaQ==
```

### Monitoring & Observability

#### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'dna-origami-ae'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
    metrics_path: /metrics
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "DNA Origami AE Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(dna_origami_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Encoding Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, dna_origami_encoding_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "dna_origami_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
```

#### Health Checks

```python
# health_check.py
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import psutil
import time

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    """Readiness check with dependency validation."""
    checks = {
        "database": check_database(),
        "cache": check_cache(),
        "memory": check_memory(),
        "disk": check_disk_space()
    }
    
    if all(checks.values()):
        return JSONResponse(
            content={"status": "ready", "checks": checks},
            status_code=status.HTTP_200_OK
        )
    else:
        return JSONResponse(
            content={"status": "not ready", "checks": checks},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

def check_memory():
    """Check if memory usage is within acceptable limits."""
    memory = psutil.virtual_memory()
    return memory.percent < 90

def check_disk_space():
    """Check if disk space is sufficient."""
    disk = psutil.disk_usage('/')
    return (disk.free / disk.total) > 0.1
```

### Security Configuration

#### SSL/TLS Setup

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    
    location / {
        proxy_pass http://dna-origami-ae:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# iptables
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -j DROP
```

### Backup & Recovery

#### Database Backup

```bash
#!/bin/bash
# backup_db.sh

DB_NAME="dnaorigami"
DB_USER="postgres"
BACKUP_DIR="/backup/database"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h localhost -U $DB_USER $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

#### Application Data Backup

```bash
#!/bin/bash
# backup_data.sh

DATA_DIR="/opt/dna_origami/data"
BACKUP_DIR="/backup/application"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz -C $DATA_DIR .

# Sync to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/dna-origami-ae/
```

#### Automated Backup with Cron

```bash
# Add to crontab (crontab -e)

# Database backup daily at 2 AM
0 2 * * * /opt/dna_origami/scripts/backup_db.sh

# Application data backup daily at 3 AM
0 3 * * * /opt/dna_origami/scripts/backup_data.sh

# Log rotation weekly
0 0 * * 0 /usr/sbin/logrotate /etc/logrotate.d/dna-origami-ae
```

### Performance Tuning

#### System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.somaxconn = 32768" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 32768" >> /etc/sysctl.conf
echo "net.core.netdev_max_backlog = 32768" >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

#### Application Tuning

```python
# performance_config.py

# Optimize for CPU-bound workloads
PERFORMANCE_CONFIG = {
    'max_workers': min(32, os.cpu_count() * 2),
    'worker_type': 'process',  # vs 'thread'
    'chunk_size': 100,
    'cache_size_mb': 2000,
    'use_jit': True,
    'optimize_memory': True
}

# Optimize for memory-intensive workloads
MEMORY_CONFIG = {
    'chunk_processing': True,
    'chunk_size_mb': 100,
    'memory_pool_enabled': True,
    'garbage_collection_threshold': 1000,
    'use_memory_mapping': True
}
```

### Troubleshooting

#### Common Issues

**Issue: High Memory Usage**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Solution: Enable memory optimization
export DNA_ORIGAMI_MEMORY_OPTIMIZATION=true
export DNA_ORIGAMI_CHUNK_PROCESSING=true
```

**Issue: Slow Performance**
```bash
# Check CPU usage
top
htop

# Solution: Increase worker count
export DNA_ORIGAMI_MAX_WORKERS=16
```

**Issue: Database Connection Errors**
```bash
# Check database connectivity
pg_isready -h localhost -p 5432

# Check connection pool
SELECT count(*) FROM pg_stat_activity;
```

#### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/dna-origami-ae/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    level: DEBUG
    formatter: json

loggers:
  dna_origami_ae:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

### Scaling Guidelines

#### Horizontal Scaling

```yaml
# Auto-scaling configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dna-origami-ae-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dna-origami-ae
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Scaling

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: dna-origami-ae-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dna-origami-ae
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: dna-origami-ae
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
```

---

For additional deployment assistance, please refer to our [support documentation](SUPPORT.md) or contact the development team.