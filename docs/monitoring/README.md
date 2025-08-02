# Monitoring & Observability

This directory contains configuration and documentation for monitoring the DNA-Origami-AutoEncoder system in production and development environments.

## Overview

The monitoring stack consists of:
- **Health Checks**: Application health endpoints
- **Metrics Collection**: Prometheus-compatible metrics
- **Logging**: Structured logging with correlation IDs
- **Alerting**: Critical system alerts
- **Observability**: Distributed tracing and performance monitoring

## Components

### Health Checks (`health-checks.md`)
- Application health endpoints
- Dependency health verification
- Custom health check implementation

### Metrics (`metrics-config.md`)
- Application metrics collection
- Infrastructure metrics
- Business metrics tracking

### Logging (`logging-config.md`)
- Structured logging configuration
- Log aggregation setup
- Log analysis and querying

### Alerting (`alerting-config.md`)
- Critical system alerts
- Performance threshold alerts
- Business metric alerts

### Observability (`observability-setup.md`)
- Distributed tracing configuration
- Performance monitoring
- User experience monitoring

## Quick Start

1. **Development Environment**:
   ```bash
   # Start monitoring stack
   docker-compose -f monitoring/docker-compose.monitoring.yml up -d
   
   # View metrics
   open http://localhost:9090  # Prometheus
   open http://localhost:3000  # Grafana
   ```

2. **Production Setup**: See `production-monitoring.md` for deployment guides

3. **Custom Metrics**: See `custom-metrics.md` for adding application-specific metrics

## Files

- `health-checks.md` - Health check configuration and endpoints
- `metrics-config.md` - Metrics collection and export
- `logging-config.md` - Structured logging setup
- `alerting-config.md` - Alert rules and notification setup
- `observability-setup.md` - Tracing and observability configuration
- `production-monitoring.md` - Production deployment guide
- `custom-metrics.md` - Guide for adding custom metrics
- `troubleshooting.md` - Common monitoring issues and solutions