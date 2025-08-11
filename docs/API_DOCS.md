# DNA Origami AutoEncoder - API Documentation

## Overview

The DNA Origami AutoEncoder provides a comprehensive REST API for encoding images into DNA sequences and decoding them back to images. The API is built with FastAPI and follows OpenAPI 3.0 specifications.

## Base URL

```
Production: https://api.dnaorigami.ai
Staging: https://staging-api.dnaorigami.ai
Local: http://localhost:8000
```

## Authentication

The API uses OAuth2 with JWT tokens for authentication.

### Obtain Token

```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Use Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Core Endpoints

### Health Check

Check API health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-08-11T04:25:41.113Z",
  "uptime": 3600,
  "checks": {
    "database": "healthy",
    "cache": "healthy",
    "storage": "healthy"
  }
}
```

### System Information

Get detailed system information.

```http
GET /info
```

**Response:**
```json
{
  "name": "DNA Origami AutoEncoder",
  "version": "1.0.0",
  "description": "Advanced DNA sequence encoding and decoding system",
  "build_info": {
    "commit": "abc123def",
    "build_date": "2024-08-11T00:00:00Z",
    "environment": "production"
  },
  "capabilities": {
    "max_image_size": "10MB",
    "supported_formats": ["PNG", "JPEG", "TIFF", "BMP"],
    "max_sequence_length": 100000,
    "gpu_acceleration": true,
    "distributed_processing": true
  }
}
```

## Image Encoding

### Single Image Encoding

Encode a single image to DNA sequence.

```http
POST /encode/image
Content-Type: multipart/form-data
Authorization: Bearer {token}

file: [image file]
optimization_level: "intermediate"
biological_constraints: true
```

**Parameters:**
- `file` (required): Image file (PNG, JPEG, TIFF, BMP)
- `optimization_level` (optional): "basic", "intermediate", "advanced", "maximum" (default: "intermediate")
- `biological_constraints` (optional): Apply biological constraints (default: true)
- `error_correction` (optional): Error correction method (default: "reed_solomon")
- `compression` (optional): Enable compression (default: false)

**Response:**
```json
{
  "task_id": "encode_20240811042541_abc123",
  "status": "completed",
  "image_info": {
    "filename": "sample.png",
    "size": [256, 256],
    "format": "PNG",
    "file_size": 65536
  },
  "dna_sequence": "ATGCATGCTAGC...GCTATGCAT",
  "encoding_stats": {
    "sequence_length": 131072,
    "encoding_time": 0.45,
    "constraint_satisfaction_rate": 0.98,
    "compression_ratio": 1.2
  },
  "validation": {
    "gc_content": 0.52,
    "palindromes": 3,
    "homopolymers": 1,
    "constraints_passed": true
  }
}
```

### Batch Image Encoding

Encode multiple images in a single request.

```http
POST /encode/batch
Content-Type: multipart/form-data
Authorization: Bearer {token}

files: [image files]
optimization_level: "advanced"
```

**Response:**
```json
{
  "batch_id": "batch_20240811042541_xyz789",
  "status": "in_progress",
  "total_images": 5,
  "completed": 3,
  "results": [
    {
      "image_id": "image_001",
      "status": "completed",
      "dna_sequence": "ATGC...",
      "encoding_stats": {...}
    },
    {
      "image_id": "image_002", 
      "status": "processing",
      "progress": 0.75
    }
  ],
  "estimated_completion": "2024-08-11T04:26:30Z"
}
```

## DNA Decoding

### Single Sequence Decoding

Decode a DNA sequence back to an image.

```http
POST /decode/sequence
Content-Type: application/json
Authorization: Bearer {token}

{
  "dna_sequence": "ATGCATGCTAGC...GCTATGCAT",
  "output_format": "PNG",
  "reconstruction_method": "transformer"
}
```

**Parameters:**
- `dna_sequence` (required): DNA sequence string
- `output_format` (optional): Output image format (default: "PNG")
- `reconstruction_method` (optional): "transformer", "cnn", "auto" (default: "transformer")
- `quality_enhancement` (optional): Apply quality enhancement (default: true)

**Response:**
```json
{
  "task_id": "decode_20240811042541_def456",
  "status": "completed",
  "sequence_info": {
    "length": 131072,
    "gc_content": 0.52,
    "validation_passed": true
  },
  "reconstructed_image": {
    "size": [256, 256],
    "format": "PNG",
    "data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUh...",
    "file_url": "/files/reconstructed/decode_20240811042541_def456.png"
  },
  "decoding_stats": {
    "decoding_time": 0.32,
    "reconstruction_accuracy": 0.999,
    "quality_score": 0.95
  }
}
```

### Batch Sequence Decoding

Decode multiple DNA sequences.

```http
POST /decode/batch
Content-Type: application/json
Authorization: Bearer {token}

{
  "sequences": [
    {
      "id": "seq_001",
      "sequence": "ATGC...",
      "metadata": {"original_filename": "image1.png"}
    },
    {
      "id": "seq_002", 
      "sequence": "TACG...",
      "metadata": {"original_filename": "image2.jpg"}
    }
  ],
  "output_format": "PNG"
}
```

## Task Management

### Get Task Status

Check the status of an encoding/decoding task.

```http
GET /tasks/{task_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "task_id": "encode_20240811042541_abc123",
  "status": "completed",
  "task_type": "image_encoding",
  "created_at": "2024-08-11T04:25:41Z",
  "updated_at": "2024-08-11T04:25:45Z",
  "progress": 1.0,
  "result": {
    "dna_sequence": "ATGC...",
    "encoding_stats": {...}
  },
  "error": null
}
```

### List Tasks

Get list of user tasks with filtering.

```http
GET /tasks?status=completed&limit=20&offset=0
Authorization: Bearer {token}
```

**Query Parameters:**
- `status`: Filter by status ("pending", "processing", "completed", "failed")
- `task_type`: Filter by type ("image_encoding", "sequence_decoding", "batch_processing")
- `limit`: Maximum results (default: 50, max: 100)
- `offset`: Pagination offset (default: 0)
- `sort`: Sort order ("created_at", "updated_at", "status")

### Cancel Task

Cancel a pending or processing task.

```http
DELETE /tasks/{task_id}
Authorization: Bearer {token}
```

## Performance & Optimization

### Performance Metrics

Get system performance metrics.

```http
GET /metrics
Authorization: Bearer {token}
```

**Response:**
```json
{
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "gpu_usage": 32.1,
    "active_tasks": 12,
    "queue_length": 5
  },
  "performance_stats": {
    "avg_encoding_time": 0.45,
    "avg_decoding_time": 0.32,
    "throughput": 150.5,
    "success_rate": 0.998
  },
  "optimization_stats": {
    "cache_hit_rate": 0.85,
    "gpu_acceleration_enabled": true,
    "distributed_processing_active": true
  }
}
```

### Benchmark Testing

Run performance benchmarks.

```http
POST /benchmark
Content-Type: application/json
Authorization: Bearer {token}

{
  "test_type": "throughput",
  "duration": 60,
  "concurrency": 10,
  "image_size": [256, 256]
}
```

## Algorithm Configuration

### Get Available Algorithms

List available encoding/decoding algorithms.

```http
GET /algorithms
```

**Response:**
```json
{
  "encoding_algorithms": [
    {
      "name": "standard_encoder",
      "description": "Standard base-4 encoding",
      "performance": "baseline",
      "biological_constraints": true
    },
    {
      "name": "adaptive_encoder", 
      "description": "ML-optimized adaptive encoding",
      "performance": "high",
      "biological_constraints": true,
      "features": ["genetic_optimization", "clustering", "caching"]
    }
  ],
  "decoding_algorithms": [
    {
      "name": "transformer_decoder",
      "description": "Transformer-based sequence decoder",
      "accuracy": "high",
      "speed": "medium"
    }
  ]
}
```

### Algorithm Configuration

Configure algorithm parameters.

```http
PUT /algorithms/{algorithm_name}/config
Content-Type: application/json
Authorization: Bearer {token}

{
  "parameters": {
    "optimization_level": "advanced",
    "genetic_algorithm_generations": 50,
    "clustering_enabled": true,
    "cache_size": "1GB"
  }
}
```

## Data Management

### Upload Data

Upload images or sequences for processing.

```http
POST /data/upload
Content-Type: multipart/form-data
Authorization: Bearer {token}

files: [files]
metadata: {"project": "research_001", "batch": "batch_001"}
```

### Download Results

Download processed results.

```http
GET /data/download/{file_id}
Authorization: Bearer {token}
```

### List Files

List uploaded/processed files.

```http
GET /data/files?project=research_001
Authorization: Bearer {token}
```

## Biological Validation

### Validate DNA Sequence

Validate a DNA sequence against biological constraints.

```http
POST /validate/sequence
Content-Type: application/json
Authorization: Bearer {token}

{
  "sequence": "ATGCATGC...",
  "constraints": {
    "max_homopolymer_length": 4,
    "gc_content_range": [0.4, 0.6],
    "forbidden_patterns": ["AAAA", "TTTT"]
  }
}
```

**Response:**
```json
{
  "valid": true,
  "validation_results": {
    "length": 1000,
    "gc_content": 0.52,
    "homopolymers": [
      {"pattern": "AAA", "positions": [45, 234], "length": 3}
    ],
    "palindromes": [
      {"sequence": "GAATTC", "position": 156}
    ],
    "forbidden_patterns": [],
    "secondary_structure_score": 0.75
  },
  "recommendations": [
    "Consider reducing GC content in region 200-300",
    "Break up homopolymer at position 234"
  ]
}
```

### Constraint Templates

Get predefined biological constraint templates.

```http
GET /validate/templates
```

**Response:**
```json
{
  "templates": {
    "strict": {
      "description": "Strict biological constraints for wet-lab synthesis",
      "max_homopolymer_length": 3,
      "gc_content_range": [0.45, 0.55],
      "max_palindrome_length": 6
    },
    "moderate": {
      "description": "Moderate constraints for computational work",
      "max_homopolymer_length": 5,
      "gc_content_range": [0.3, 0.7]
    }
  }
}
```

## Research & Analytics

### Experiment Tracking

Create and track research experiments.

```http
POST /research/experiments
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Optimization Comparison Study",
  "description": "Compare different optimization algorithms",
  "parameters": {
    "algorithms": ["standard", "adaptive", "genetic"],
    "dataset": "MNIST_subset",
    "metrics": ["throughput", "accuracy", "constraint_satisfaction"]
  }
}
```

### Statistical Analysis

Get statistical analysis of results.

```http
GET /research/analysis/{experiment_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "experiment_id": "exp_001",
  "statistical_summary": {
    "sample_size": 1000,
    "algorithms_compared": 3,
    "significance_tests": {
      "throughput_improvement": {
        "p_value": 0.001,
        "statistically_significant": true,
        "effect_size": 2.5
      }
    }
  },
  "performance_comparison": {
    "standard_encoder": {"mean_time": 0.5, "std_dev": 0.1},
    "adaptive_encoder": {"mean_time": 0.2, "std_dev": 0.05}
  }
}
```

## Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Invalid image format provided",
    "details": {
      "field": "file",
      "supported_formats": ["PNG", "JPEG", "TIFF", "BMP"]
    },
    "request_id": "req_20240811042541_abc123",
    "timestamp": "2024-08-11T04:25:41Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `SERVER_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limits

### Default Limits

- **Free Tier**: 100 requests/hour, 10 concurrent
- **Basic Tier**: 1,000 requests/hour, 25 concurrent
- **Pro Tier**: 10,000 requests/hour, 100 concurrent
- **Enterprise**: Custom limits

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1628707200
X-RateLimit-Window: 3600
```

## SDKs & Examples

### Python SDK

```python
from dna_origami_client import DNAOrigarmiClient

client = DNAOrigarmiClient(
    api_url="https://api.dnaorigami.ai",
    api_key="your_api_key"
)

# Encode image
with open("sample.png", "rb") as f:
    result = client.encode_image(
        file=f,
        optimization_level="advanced"
    )

print(f"DNA Sequence: {result.dna_sequence}")
print(f"Encoding Time: {result.encoding_stats.encoding_time}s")

# Decode sequence
decoded = client.decode_sequence(
    sequence=result.dna_sequence,
    output_format="PNG"
)

with open("reconstructed.png", "wb") as f:
    f.write(decoded.image_data)
```

### JavaScript SDK

```javascript
import { DNAOrigarmiClient } from '@dna-origami/client';

const client = new DNAOrigarmiClient({
  apiUrl: 'https://api.dnaorigami.ai',
  apiKey: 'your_api_key'
});

// Encode image
const formData = new FormData();
formData.append('file', imageFile);
formData.append('optimization_level', 'advanced');

const result = await client.encodeImage(formData);
console.log('DNA Sequence:', result.dna_sequence);

// Decode sequence
const decoded = await client.decodeSequence({
  dna_sequence: result.dna_sequence,
  output_format: 'PNG'
});
```

### cURL Examples

```bash
# Encode image
curl -X POST "https://api.dnaorigami.ai/encode/image" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample.png" \
  -F "optimization_level=advanced"

# Decode sequence
curl -X POST "https://api.dnaorigami.ai/decode/sequence" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dna_sequence": "ATGCATGC...",
    "output_format": "PNG"
  }'
```

## Webhooks

### Configure Webhooks

Set up webhooks for task completion notifications.

```http
POST /webhooks
Content-Type: application/json
Authorization: Bearer {token}

{
  "url": "https://your-app.com/webhook",
  "events": ["task.completed", "task.failed"],
  "secret": "webhook_secret_key"
}
```

### Webhook Payload

```json
{
  "event": "task.completed",
  "task_id": "encode_20240811042541_abc123",
  "timestamp": "2024-08-11T04:25:45Z",
  "data": {
    "status": "completed",
    "result": {
      "dna_sequence": "ATGC...",
      "encoding_stats": {...}
    }
  },
  "signature": "sha256=abc123def456..."
}
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

```
GET /openapi.json
GET /docs (Swagger UI)
GET /redoc (ReDoc)
```

## Support & Resources

- **Documentation**: https://docs.dnaorigami.ai
- **API Status**: https://status.dnaorigami.ai  
- **GitHub**: https://github.com/terragon-labs/dna-origami-autoencoder
- **Support**: support@dnaorigami.ai
- **Community**: https://discord.gg/dnaorigami

---

*API Documentation v1.0.0 - Last updated: August 11, 2024*