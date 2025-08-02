# Health Check Configuration

Health checks ensure system components are functioning correctly and provide early warning of issues.

## Health Check Endpoints

### Application Health Check
```python
# dna_origami_ae/health.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import psutil
import torch

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "dna-origami-autoencoder"
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system metrics"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Check dependencies (database, cache, external APIs)
        dependencies = await check_dependencies()
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "dna-origami-autoencoder",
            "version": "0.1.0",
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "gpu": {
                "available": gpu_available,
                "count": gpu_count,
                "devices": get_gpu_info() if gpu_available else []
            },
            "dependencies": dependencies
        }
        
        # Determine overall health
        if any(dep["status"] != "healthy" for dep in dependencies.values()):
            health_status["status"] = "degraded"
            
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            health_status["status"] = "degraded"
            
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

async def check_dependencies() -> Dict[str, Dict[str, Any]]:
    """Check health of external dependencies"""
    dependencies = {}
    
    # Check database connection
    try:
        # Database health check logic
        dependencies["database"] = {
            "status": "healthy",
            "response_time_ms": 10,
            "last_check": time.time()
        }
    except Exception as e:
        dependencies["database"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }
    
    # Check Redis/cache
    try:
        # Cache health check logic
        dependencies["cache"] = {
            "status": "healthy",
            "response_time_ms": 5,
            "last_check": time.time()
        }
    except Exception as e:
        dependencies["cache"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": time.time()
        }
    
    # Check external APIs
    dependencies["external_apis"] = {
        "pdb_api": await check_external_api("https://data.rcsb.org/rest/v1/core/entry/1BNA"),
        "ncbi_api": await check_external_api("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi")
    }
    
    return dependencies

def get_gpu_info() -> list:
    """Get detailed GPU information"""
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        gpu_info.append({
            "device_id": i,
            "name": gpu.name,
            "memory_total_gb": gpu.total_memory / (1024**3),
            "memory_free_gb": torch.cuda.memory_reserved(i) / (1024**3),
            "memory_used_gb": torch.cuda.memory_allocated(i) / (1024**3)
        })
    return gpu_info

async def check_external_api(url: str) -> Dict[str, Any]:
    """Check external API health"""
    import aiohttp
    import asyncio
    
    try:
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=5)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "healthy" if response.status == 200 else "degraded",
                    "status_code": response.status,
                    "response_time_ms": response_time
                }
    except asyncio.TimeoutError:
        return {
            "status": "unhealthy",
            "error": "timeout",
            "response_time_ms": 5000
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Readiness and Liveness Probes

### Kubernetes Health Check Configuration
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dna-origami-autoencoder
spec:
  template:
    spec:
      containers:
      - name: api
        image: dna-origami-autoencoder:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
          timeoutSeconds: 10
          failureThreshold: 2
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
```

## Docker Health Check
```dockerfile
# Dockerfile
FROM python:3.11-slim

# ... other instructions ...

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

## Custom Health Checks

### ML Model Health Check
```python
@router.get("/health/models")
async def model_health_check() -> Dict[str, Any]:
    """Check health of ML models"""
    model_status = {}
    
    try:
        # Check encoder model
        from dna_origami_ae.encoding import DNAEncoder
        encoder = DNAEncoder()
        test_input = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        start_time = time.time()
        result = encoder.encode_image(test_input)
        encoding_time = (time.time() - start_time) * 1000
        
        model_status["encoder"] = {
            "status": "healthy",
            "inference_time_ms": encoding_time,
            "output_size": len(result)
        }
    except Exception as e:
        model_status["encoder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    try:
        # Check decoder model
        from dna_origami_ae.decoding import TransformerDecoder
        decoder = TransformerDecoder.from_pretrained('checkpoint')
        # ... decoder health check logic
        
        model_status["decoder"] = {
            "status": "healthy",
            "model_loaded": True
        }
    except Exception as e:
        model_status["decoder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return model_status
```

### Simulation Engine Health Check
```python
@router.get("/health/simulation")
async def simulation_health_check() -> Dict[str, Any]:
    """Check health of simulation components"""
    sim_status = {}
    
    try:
        # Check oxDNA availability
        import subprocess
        result = subprocess.run(['oxDNA', '--version'], 
                              capture_output=True, text=True, timeout=5)
        sim_status["oxdna"] = {
            "status": "healthy" if result.returncode == 0 else "unhealthy",
            "version": result.stdout.strip()
        }
    except Exception as e:
        sim_status["oxdna"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    try:
        # Check CUDA for GPU simulation
        import torch
        if torch.cuda.is_available():
            test_tensor = torch.randn(1000, 1000).cuda()
            start_time = time.time()
            result = torch.matmul(test_tensor, test_tensor.T)
            gpu_time = (time.time() - start_time) * 1000
            
            sim_status["gpu_compute"] = {
                "status": "healthy",
                "test_time_ms": gpu_time,
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2)
            }
        else:
            sim_status["gpu_compute"] = {
                "status": "unavailable",
                "reason": "CUDA not available"
            }
    except Exception as e:
        sim_status["gpu_compute"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return sim_status
```

## Monitoring Integration

### Prometheus Metrics for Health Checks
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Health check metrics
health_check_counter = Counter(
    'health_check_requests_total',
    'Total health check requests',
    ['endpoint', 'status']
)

health_check_duration = Histogram(
    'health_check_duration_seconds',
    'Health check duration',
    ['endpoint']
)

system_cpu_percent = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage'
)

system_memory_percent = Gauge(
    'system_memory_percent',
    'System memory usage percentage'
)

gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used',
    ['device_id']
)

@router.get("/health")
async def instrumented_health_check():
    start_time = time.time()
    status = "healthy"
    
    try:
        result = await health_check()
        return result
    except Exception as e:
        status = "unhealthy"
        raise
    finally:
        health_check_counter.labels(endpoint="health", status=status).inc()
        health_check_duration.labels(endpoint="health").observe(time.time() - start_time)
```

## Health Check Testing

### Unit Tests
```python
import pytest
from fastapi.testclient import TestClient
from dna_origami_ae.main import app

client = TestClient(app)

def test_basic_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_detailed_health_check():
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "gpu" in data
    assert "dependencies" in data

@pytest.mark.asyncio
async def test_dependency_health_checks():
    dependencies = await check_dependencies()
    assert "database" in dependencies
    assert "cache" in dependencies
    
    for dep_name, dep_status in dependencies.items():
        assert "status" in dep_status
        assert "last_check" in dep_status
```

## Alerting on Health Check Failures

### Prometheus Alert Rules
```yaml
# alerts/health-checks.yml
groups:
- name: health-checks
  rules:
  - alert: HealthCheckFailure
    expr: health_check_requests_total{status="unhealthy"} > 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Health check failing for {{ $labels.endpoint }}"
      description: "Health check for {{ $labels.endpoint }} has been failing for more than 2 minutes"

  - alert: HighSystemResourceUsage
    expr: system_cpu_percent > 90 or system_memory_percent > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High system resource usage"
      description: "System resources (CPU: {{ $value }}%, Memory: {{ $value }}%) are critically high"

  - alert: GPUMemoryHigh
    expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage is high"
      description: "GPU {{ $labels.device_id }} memory usage is above 90%"
```