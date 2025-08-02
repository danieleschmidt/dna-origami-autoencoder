# Deployment Documentation

This directory contains documentation and configuration for deploying the DNA-Origami-AutoEncoder system in various environments.

## Overview

The DNA-Origami-AutoEncoder system supports multiple deployment strategies:

- **Development**: Local development with hot-reloading
- **Production**: Optimized containers for production workloads
- **GPU-Enabled**: CUDA-accelerated deployment for intensive computations
- **Testing**: Automated testing in containerized environments
- **Monitoring**: Full observability stack with metrics and tracing

## Quick Start

### Development Deployment

```bash
# Start development environment
docker-compose --profile dev up -d

# Access Jupyter Lab
open http://localhost:8889

# View logs
docker-compose logs -f dna-origami-ae-dev
```

### Production Deployment

```bash
# Build production images
make docker-build

# Start production stack
docker-compose up -d

# Health check
make health-check
```

### GPU Deployment

```bash
# Build GPU-enabled image
make docker-build-gpu

# Start GPU services
docker-compose --profile gpu up -d

# Verify GPU access
docker exec dna-origami-ae-gpu nvidia-smi
```

## Container Architecture

### Multi-Stage Dockerfile

The project uses a multi-stage Dockerfile for optimized builds:

1. **Base Stage**: CUDA-enabled Ubuntu with Python
2. **Dependencies Stage**: Python packages installation
3. **Build Stage**: Compile native extensions (oxDNA)
4. **Production Stage**: Minimal runtime image
5. **Development Stage**: Full development environment
6. **Testing Stage**: Optimized for CI/CD
7. **GPU Stage**: GPU-optimized with RAPIDS

### Image Variants

| Image Tag | Purpose | Size | Use Case |
|-----------|---------|------|----------|
| `latest` | Production | ~2GB | Production deployment |
| `dev` | Development | ~4GB | Local development |
| `gpu` | GPU workloads | ~6GB | ML training/inference |
| `test` | Testing | ~3GB | CI/CD pipelines |

## Service Configuration

### Core Services

#### dna-origami-ae (Main Application)
- **Ports**: 8888 (Jupyter), 6006 (TensorBoard), 8080 (API)
- **Volumes**: Data, models, results, logs
- **Health Check**: Python import validation
- **Restart Policy**: unless-stopped

#### dna-origami-ae-dev (Development)
- **Profile**: dev
- **Mount**: Full source code for hot-reloading
- **Extensions**: Development tools and debuggers

#### dna-origami-ae-gpu (GPU)
- **Profile**: gpu
- **Runtime**: nvidia
- **GPU Access**: All available GPUs
- **Libraries**: CUDA, cuDNN, RAPIDS

### Supporting Services

#### Redis (Caching & Queues)
- **Purpose**: Task queues, caching, session storage
- **Port**: 6379
- **Persistence**: AOF enabled
- **Health Check**: Redis ping

#### PostgreSQL (Metadata Storage)
- **Purpose**: Experiment metadata, user data, results
- **Port**: 5432
- **Database**: dna_origami_ae
- **Backup**: Automated daily backups

#### MinIO (Object Storage)
- **Purpose**: Large file storage (models, datasets)
- **Ports**: 9000 (API), 9001 (Console)
- **S3 Compatible**: Full S3 API compatibility
- **Buckets**: Auto-created for different data types

### Monitoring Stack

#### Prometheus (Metrics)
- **Profile**: monitoring
- **Port**: 9090
- **Retention**: 30 days
- **Targets**: All application services

#### Grafana (Visualization)
- **Profile**: monitoring
- **Port**: 3000
- **Dashboards**: Pre-configured for DNA-OAE metrics
- **Alerting**: Email and Slack notifications

#### Jaeger (Tracing)
- **Profile**: monitoring
- **Port**: 16686
- **Sampling**: Configurable sampling rates
- **Storage**: In-memory (configurable to persistent)

## Environment Configuration

### Environment Variables

```bash
# Core configuration
DNA_ORIGAMI_AE_ENV=production
PYTHONPATH=/app
LOG_LEVEL=INFO

# GPU configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# Database configuration
DATABASE_URL=postgresql://user:pass@postgres:5432/dna_origami_ae
REDIS_URL=redis://redis:6379

# Object storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

### Secrets Management

Production deployments should use proper secrets management:

```bash
# Using Docker secrets
echo "secure_password" | docker secret create db_password -

# Using environment files
cp .env.example .env.production
# Edit .env.production with production values
```

## Build Configuration

### Build Arguments

```dockerfile
# Build with custom Python version
docker build --build-arg PYTHON_VERSION=3.11 .

# Build with specific CUDA version
docker build --build-arg CUDA_VERSION=12.1 .

# Enable build cache
docker build --build-arg BUILDKIT_INLINE_CACHE=1 .
```

### Build Optimization

The Dockerfile is optimized for:
- **Layer Caching**: Dependencies installed before code copy
- **Multi-arch**: ARM64 and AMD64 support
- **Security**: Non-root user, minimal attack surface
- **Size**: Multi-stage builds reduce final image size

## Volume Management

### Data Persistence

```yaml
volumes:
  # Application data
  - ./data:/app/data              # Datasets
  - ./models:/app/models          # Trained models
  - ./results:/app/results        # Experiment results
  - ./logs:/app/logs              # Application logs
  
  # Development
  - .:/app                        # Full source mount (dev only)
  
  # Database persistence
  - postgres-data:/var/lib/postgresql/data
  - redis-data:/data
  - minio-data:/data
```

### Backup Strategy

```bash
# Database backup
docker exec postgres pg_dump -U dna_origami dna_origami_ae > backup.sql

# Model backup
tar -czf models_backup.tar.gz models/

# Full data backup
docker run --rm -v dna-origami-ae_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Network Configuration

### Custom Network

```yaml
networks:
  dna-origami-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Service Discovery

Services communicate using Docker's built-in DNS:
- `postgres:5432` - Database
- `redis:6379` - Cache
- `minio:9000` - Object storage

### Port Mapping

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| App | 8888 | 8888 | Jupyter Lab |
| App | 6006 | 6006 | TensorBoard |
| App | 8080 | 8080 | REST API |
| Dev | 8888 | 8889 | Dev Jupyter |
| PostgreSQL | 5432 | 5432 | Database |
| Redis | 6379 | 6379 | Cache |
| MinIO | 9000/9001 | 9000/9001 | Storage |
| Grafana | 3000 | 3000 | Monitoring |
| Prometheus | 9090 | 9090 | Metrics |

## Health Checks

### Application Health

```bash
# Check main application
curl -f http://localhost:8080/health

# Check database connection
docker exec dna-origami-ae-app python -c "import psycopg2; print('DB OK')"

# Check GPU access
docker exec dna-origami-ae-gpu nvidia-smi
```

### Service Health

```bash
# All services status
docker-compose ps

# Specific service logs
docker-compose logs dna-origami-ae

# Health check details
docker inspect --format='{{.State.Health}}' dna-origami-ae-app
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# Scale application instances
docker-compose up -d --scale dna-origami-ae=3

# Load balancer configuration
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

### Resource Limits

```yaml
services:
  dna-origami-ae:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### GPU Resource Management

```yaml
services:
  dna-origami-ae-gpu:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Security Considerations

### Container Security

- **Non-root User**: Application runs as `dnaorigami` user
- **Read-only Filesystem**: Where possible, use read-only mounts
- **Security Scanning**: Regular vulnerability scans with Trivy
- **Secrets Management**: No secrets in environment variables

### Network Security

- **Internal Network**: Services communicate on private network
- **TLS Termination**: Use reverse proxy for HTTPS
- **Firewall Rules**: Restrict external access to necessary ports

### Data Security

- **Encryption at Rest**: Database and object storage encryption
- **Backup Encryption**: Encrypted backups
- **Access Control**: Role-based access to services

## Monitoring and Observability

### Metrics Collection

```yaml
# Prometheus configuration
prometheus:
  image: prom/prometheus
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### Log Aggregation

```yaml
# ELK Stack integration
elasticsearch:
  image: elasticsearch:7.17.0
  
logstash:
  image: logstash:7.17.0
  
kibana:
  image: kibana:7.17.0
```

### Alerting

```yaml
# Alert manager
alertmanager:
  image: prom/alertmanager
  volumes:
    - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs dna-origami-ae

# Check resource usage
docker stats

# Verify image
docker images | grep dna-origami-ae
```

#### GPU Not Accessible
```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Verify GPU in container
docker exec dna-origami-ae-gpu nvidia-smi
```

#### Database Connection Issues
```bash
# Check database status
docker exec postgres pg_isready

# Test connection
docker exec dna-origami-ae-app python -c "import psycopg2; conn = psycopg2.connect('postgresql://user:pass@postgres/db'); print('Connected')"
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check container limits
docker exec dna-origami-ae-app cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# Profile application
docker exec dna-origami-ae-app python -m cProfile -o profile.prof your_script.py
```

### Debug Mode

```bash
# Start in debug mode
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up -d

# Attach debugger
docker exec -it dna-origami-ae-dev python -m pdb your_script.py
```

## Production Deployment Checklist

### Pre-deployment
- [ ] Update all environment variables
- [ ] Configure proper secrets management
- [ ] Set up SSL certificates
- [ ] Configure monitoring and alerting
- [ ] Plan backup and recovery procedures

### Deployment
- [ ] Build and test images
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Deploy to production
- [ ] Verify all services are healthy

### Post-deployment
- [ ] Monitor application metrics
- [ ] Check log aggregation
- [ ] Verify backup procedures
- [ ] Update documentation
- [ ] Notify stakeholders

## Maintenance

### Regular Tasks

```bash
# Update base images
docker-compose pull

# Clean unused resources
docker system prune -f

# Update SSL certificates
certbot renew

# Database maintenance
docker exec postgres vacuumdb -U dna_origami dna_origami_ae
```

### Monitoring Tasks

- Monitor disk usage for data volumes
- Review application performance metrics
- Check security vulnerability reports
- Verify backup integrity

## Support and Documentation

For additional support:
- [Container Documentation](./containers.md)
- [Monitoring Setup](./monitoring.md)
- [Security Guidelines](./security.md)
- [Performance Tuning](./performance.md)

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)