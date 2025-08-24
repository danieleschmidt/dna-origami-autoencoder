#!/usr/bin/env python3
"""
Simplified FastAPI Web Server for DNA Origami AutoEncoder
Working demo implementation with core functionality.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Any
import sys

sys.path.insert(0, '.')

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import io

# Core imports (working modules only)
from dna_origami_ae import DNASequence, ImageData, Base4Encoder, BiologicalConstraints
from dna_origami_ae.encoding.image_encoder import DNAEncoder, EncodingParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global app instance with lifespan
app = FastAPI(
    title="DNA Origami AutoEncoder API",
    description="Convert images to DNA sequences using origami principles",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task storage (for demo - use Redis in production)
tasks: Dict[str, Dict] = {}

# Pydantic models
class EncodeRequest(BaseModel):
    """Request model for basic encoding."""
    sequence: str = Field(description="DNA sequence to validate")

class EncodeResponse(BaseModel):
    """Response for basic encoding."""
    success: bool
    sequence_length: int
    gc_content: float
    is_valid: bool
    errors: List[str]

class ImageEncodeResponse(BaseModel):
    """Response for image encoding."""
    task_id: str
    status: str = "processing"
    message: str = "Task submitted"

class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str  # "processing", "completed", "failed"
    message: str
    dna_sequences: Optional[List[str]] = None
    total_bases: Optional[int] = None
    error: Optional[str] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DNA Origami AutoEncoder API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "encode_sequence": "/api/v1/encode/sequence",
            "encode_image": "/api/v1/encode/image", 
            "task_status": "/api/v1/task/{task_id}",
            "demo": "/api/v1/demo",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/v1/encode/sequence", response_model=EncodeResponse)
async def encode_sequence(request: EncodeRequest):
    """Encode and validate a DNA sequence."""
    try:
        # Create DNA sequence
        dna_seq = DNASequence(request.sequence, "api_sequence")
        
        # Validate with biological constraints
        constraints = BiologicalConstraints()
        is_valid, errors = constraints.validate_sequence(request.sequence)
        
        return EncodeResponse(
            success=True,
            sequence_length=len(dna_seq),
            gc_content=dna_seq.gc_content,
            is_valid=is_valid,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Sequence encoding failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/encode/image", response_model=ImageEncodeResponse)
async def encode_image(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    """Encode an uploaded image to DNA sequences."""
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task
    tasks[task_id] = {
        "status": "processing",
        "message": "Processing image...",
        "created_at": time.time()
    }
    
    # Schedule background processing
    background_tasks.add_task(process_image_encoding, task_id, image)
    
    return ImageEncodeResponse(
        task_id=task_id,
        status="processing", 
        message="Image encoding started"
    )

async def process_image_encoding(task_id: str, image: UploadFile):
    """Background task to process image encoding."""
    try:
        # Read image data
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale and resize for demo
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Resize to small size for demo (avoid constraint issues)
        pil_image = pil_image.resize((8, 8), Image.Resampling.LANCZOS)
        
        # Convert to ImageData
        img_array = np.array(pil_image)
        img_data = ImageData.from_array(img_array, "uploaded_image")
        
        # Create relaxed encoder for demo
        relaxed_constraints = BiologicalConstraints(
            gc_content_range=(0.2, 0.8),
            min_sequence_length=4,
            max_homopolymer_length=8,
            forbidden_sequences=[]
        )
        
        encoder = DNAEncoder(
            bits_per_base=2,
            error_correction=None,
            biological_constraints=relaxed_constraints
        )
        
        # Encoding parameters
        params = EncodingParameters(
            error_correction=None,
            compression_enabled=False,
            include_metadata=False,
            chunk_size=20,
            enforce_constraints=False
        )
        
        # Encode to DNA
        dna_sequences = encoder.encode_image(img_data, params)
        
        # Update task with success
        tasks[task_id].update({
            "status": "completed",
            "message": "Image successfully encoded to DNA",
            "dna_sequences": [seq.sequence for seq in dna_sequences],
            "total_bases": sum(len(seq) for seq in dna_sequences),
            "image_size": f"{img_array.shape[1]}x{img_array.shape[0]}",
            "completed_at": time.time()
        })
        
    except Exception as e:
        logger.error(f"Image encoding failed for task {task_id}: {e}")
        tasks[task_id].update({
            "status": "failed",
            "message": f"Encoding failed: {str(e)}",
            "error": str(e),
            "completed_at": time.time()
        })

@app.get("/api/v1/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of an encoding task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        dna_sequences=task.get("dna_sequences"),
        total_bases=task.get("total_bases"),
        error=task.get("error")
    )

@app.get("/api/v1/demo")
async def demo_endpoint():
    """Demo endpoint showing basic functionality."""
    try:
        # Demo sequence encoding
        test_seq = DNASequence("ATGCATGCATGCATGC", "demo_seq")
        
        # Demo image encoding
        demo_pattern = np.array([
            [255, 0, 255, 0],
            [0, 255, 0, 255],
            [255, 0, 255, 0], 
            [0, 255, 0, 255]
        ], dtype=np.uint8)
        
        demo_img = ImageData.from_array(demo_pattern, "demo_pattern")
        
        # Basic binary encoding demo
        encoder = Base4Encoder()
        binary = demo_img.to_binary(2)[:20]  # Use small sample
        dna_seq = encoder.encode_binary_to_dna(binary)
        
        # Validation demo
        constraints = BiologicalConstraints(
            gc_content_range=(0.2, 0.8),
            min_sequence_length=4
        )
        is_valid, errors = constraints.validate_sequence(dna_seq)
        
        return {
            "demo_sequence": {
                "sequence": test_seq.sequence,
                "length": len(test_seq),
                "gc_content": test_seq.gc_content,
                "melting_temp": test_seq.melting_temperature
            },
            "demo_encoding": {
                "original_bits": len(binary),
                "dna_sequence": dna_seq,
                "dna_length": len(dna_seq),
                "is_valid": is_valid,
                "validation_errors": errors
            },
            "demo_image": {
                "dimensions": f"{demo_img.metadata.width}x{demo_img.metadata.height}",
                "channels": demo_img.metadata.channels,
                "format": demo_img.metadata.format
            },
            "message": "Demo completed successfully!"
        }
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return {
            "error": str(e),
            "message": "Demo encountered an error"
        }

@app.get("/api/v1/stats")
async def get_stats():
    """Get server statistics."""
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks.values() if t["status"] == "completed"])
    failed_tasks = len([t for t in tasks.values() if t["status"] == "failed"])
    processing_tasks = len([t for t in tasks.values() if t["status"] == "processing"])
    
    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "processing_tasks": processing_tasks,
        "success_rate": completed_tasks / max(1, total_tasks),
        "uptime": time.time()
    }

# Cleanup old tasks periodically (demo implementation)
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("DNA Origami AutoEncoder API starting up...")
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_tasks())

async def cleanup_old_tasks():
    """Clean up old tasks periodically."""
    while True:
        await asyncio.sleep(300)  # Clean up every 5 minutes
        current_time = time.time()
        
        tasks_to_remove = [
            task_id for task_id, task in tasks.items()
            if current_time - task.get("created_at", 0) > 3600  # Remove tasks older than 1 hour
        ]
        
        for task_id in tasks_to_remove:
            del tasks[task_id]
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )