"""
FastAPI Web Server for DNA Origami AutoEncoder

High-performance async web API with auto-scaling, load balancing,
and distributed task processing capabilities.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

# Core imports
from ..encoding import DNAEncoder, Base4Encoder, BiologicalConstraints
from ..design import OrigamiDesigner
from ..simulation import OrigamiSimulator
from ..decoding import TransformerDecoder
from ..utils import gpu_acceleration, error_handling, monitoring
from ..utils.cache import get_cache_manager
from ..utils.distributed import DistributedTaskManager


# Pydantic models for API
class ImageEncodeRequest(BaseModel):
    """Request model for image encoding."""
    encoder_type: str = Field("base4", description="DNA encoding method")
    error_correction: str = Field("reed_solomon", description="Error correction method")
    compression: bool = Field(True, description="Enable compression")
    image_size: tuple[int, int] = Field((32, 32), description="Target image size")
    biological_constraints: Dict[str, Any] = Field(default_factory=dict)

class ImageEncodeResponse(BaseModel):
    """Response model for image encoding."""
    task_id: str = Field(description="Unique task identifier")
    dna_sequence: Optional[str] = Field(None, description="DNA sequence (if synchronous)")
    sequence_length: Optional[int] = Field(None, description="Length of DNA sequence")
    gc_content: Optional[float] = Field(None, description="GC content percentage")
    encoding_efficiency: Optional[float] = Field(None, description="Encoding efficiency")
    status: str = Field("processing", description="Task status")

class OrigamiDesignRequest(BaseModel):
    """Request model for origami design."""
    dna_sequence: str = Field(description="DNA sequence to design")
    shape: str = Field("square", description="Target origami shape")
    dimensions: tuple[float, float] = Field((100, 100), description="Dimensions in nm")
    scaffold_length: int = Field(7249, description="Scaffold length") 
    staple_length: int = Field(32, description="Staple length")
    optimize_sequences: bool = Field(True, description="Optimize staple sequences")

class OrigamiDesignResponse(BaseModel):
    """Response model for origami design."""
    task_id: str = Field(description="Unique task identifier")
    design_id: Optional[str] = Field(None, description="Design identifier")
    scaffold_length: Optional[int] = Field(None, description="Scaffold length")
    staple_count: Optional[int] = Field(None, description="Number of staples")
    estimated_yield: Optional[float] = Field(None, description="Estimated assembly yield")
    status: str = Field("processing", description="Task status")

class SimulationRequest(BaseModel):
    """Request model for simulation."""
    design_id: str = Field(description="Origami design identifier")
    force_field: str = Field("oxdna2", description="Force field")
    temperature: float = Field(300, description="Temperature in Kelvin")
    salt_concentration: float = Field(0.5, description="Salt concentration in M")
    time_steps: int = Field(1000000, description="Simulation time steps")
    gpu_acceleration: bool = Field(True, description="Use GPU acceleration")

class SimulationResponse(BaseModel):
    """Response model for simulation."""
    task_id: str = Field(description="Unique task identifier")
    simulation_id: Optional[str] = Field(None, description="Simulation identifier")
    progress: float = Field(0.0, description="Simulation progress (0-1)")
    estimated_completion: Optional[float] = Field(None, description="Estimated completion time")
    status: str = Field("processing", description="Task status")

class DecodingRequest(BaseModel):
    """Request model for decoding."""
    structure_id: str = Field(description="Structure identifier")
    model_type: str = Field("transformer", description="Decoder model type")
    confidence_threshold: float = Field(0.8, description="Minimum confidence threshold")

class DecodingResponse(BaseModel):
    """Response model for decoding."""
    task_id: str = Field(description="Unique task identifier") 
    decoded_image_id: Optional[str] = Field(None, description="Decoded image identifier")
    confidence: Optional[float] = Field(None, description="Decoding confidence")
    reconstruction_quality: Optional[float] = Field(None, description="Reconstruction quality score")
    status: str = Field("processing", description="Task status")

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(description="Task identifier")
    status: str = Field(description="Current status")
    progress: float = Field(description="Progress (0-1)")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: float = Field(description="Task creation timestamp")
    updated_at: float = Field(description="Last update timestamp")

class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    checks: Dict[str, Any] = Field(description="Individual health checks")
    metrics: Dict[str, Any] = Field(description="System metrics")


class DNAOrigarniAPI:
    """Main API class with business logic."""
    
    def __init__(self):
        self.task_manager = DistributedTaskManager()
        self.gpu_manager = gpu_acceleration.get_gpu_manager()
        self.error_handler = error_handling.get_error_handler()
        self.metrics_collector = monitoring.get_metrics_collector()
        self.profiler = monitoring.get_profiler()
        self.cache_manager = get_cache_manager()
        
        # Task storage (in production, use Redis/database)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
    async def encode_image_async(self, image_data: bytes, request: ImageEncodeRequest) -> ImageEncodeResponse:
        """Asynchronously encode image to DNA sequence."""
        
        task_id = str(uuid.uuid4())
        
        # Store task
        self.tasks[task_id] = {
            "type": "encode",
            "status": "processing",
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        try:
            # Process image
            with self.profiler.profile_operation("image_processing"):
                image = Image.open(io.BytesIO(image_data)).convert('L')
                if request.image_size != (32, 32):
                    image = image.resize(request.image_size)
                image_array = np.array(image)
            
            # Setup encoder
            with self.profiler.profile_operation("encoder_setup"):
                constraints = BiologicalConstraints(**request.biological_constraints) if request.biological_constraints else None
                
                if request.encoder_type == "base4":
                    encoder = Base4Encoder(constraints)
                else:
                    encoder = DNAEncoder(
                        encoding_method=request.encoder_type,
                        error_correction=request.error_correction,
                        compression=request.compression,
                        biological_constraints=constraints
                    )
            
            # Submit encoding task
            future = self.task_manager.submit_task(
                self._encode_image_task,
                encoder,
                image_array,
                task_id
            )
            
            # Check if we should return immediately or wait
            if hasattr(request, 'sync') and request.sync:
                result = await future
                return result
            else:
                # Return task ID for async processing
                return ImageEncodeResponse(
                    task_id=task_id,
                    status="processing"
                )
                
        except Exception as e:
            self.error_handler.handle_error(e, {"task_id": task_id, "operation": "encode_image"})
            self.tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": time.time()
            })
            raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")
    
    async def _encode_image_task(self, encoder, image_array: np.ndarray, task_id: str) -> ImageEncodeResponse:
        """Background task for image encoding."""
        
        try:
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.2,
                "updated_at": time.time()
            })
            
            # Perform encoding
            with self.profiler.profile_operation("dna_encoding"):
                dna_sequence = encoder.encode_image(image_array)
                
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.8,
                "updated_at": time.time()
            })
            
            # Calculate metrics
            gc_content = encoder.get_gc_content(dna_sequence)
            efficiency = encoder.get_efficiency()
            
            # Complete task
            result = ImageEncodeResponse(
                task_id=task_id,
                dna_sequence=dna_sequence,
                sequence_length=len(dna_sequence),
                gc_content=gc_content,
                encoding_efficiency=efficiency,
                status="completed"
            )
            
            self.tasks[task_id].update({
                "status": "completed",
                "progress": 1.0,
                "result": result.dict(),
                "updated_at": time.time()
            })
            
            # Record metrics
            self.metrics_collector.record_counter("api.encode.success")
            self.metrics_collector.record_gauge("api.encode.sequence_length", len(dna_sequence))
            
            return result
            
        except Exception as e:
            self.tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": time.time()
            })
            self.metrics_collector.record_counter("api.encode.failure")
            raise
    
    async def design_origami_async(self, request: OrigamiDesignRequest) -> OrigamiDesignResponse:
        """Asynchronously design origami structure."""
        
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "type": "design", 
            "status": "processing",
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        try:
            # Submit design task
            future = self.task_manager.submit_task(
                self._design_origami_task,
                request,
                task_id
            )
            
            return OrigamiDesignResponse(
                task_id=task_id,
                status="processing"
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, {"task_id": task_id, "operation": "design_origami"})
            raise HTTPException(status_code=500, detail=f"Design failed: {e}")
    
    async def _design_origami_task(self, request: OrigamiDesignRequest, task_id: str) -> OrigamiDesignResponse:
        """Background task for origami design."""
        
        try:
            # Initialize designer
            with self.profiler.profile_operation("designer_setup"):
                designer = OrigamiDesigner(
                    scaffold_length=request.scaffold_length,
                    staple_length=request.staple_length
                )
            
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.2,
                "updated_at": time.time()
            })
            
            # Design origami
            with self.profiler.profile_operation("origami_design"):
                origami = designer.design_origami(
                    request.dna_sequence,
                    target_shape=request.shape,
                    dimensions=request.dimensions,
                    optimize_sequences=request.optimize_sequences
                )
            
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.8,
                "updated_at": time.time()
            })
            
            # Generate design ID and cache result
            design_id = str(uuid.uuid4())
            await self.cache_manager.set(f"design:{design_id}", origami, ttl=3600)
            
            # Complete task
            result = OrigamiDesignResponse(
                task_id=task_id,
                design_id=design_id,
                scaffold_length=origami.scaffold_length,
                staple_count=len(origami.staples),
                estimated_yield=origami.get_estimated_yield(),
                status="completed"
            )
            
            self.tasks[task_id].update({
                "status": "completed",
                "progress": 1.0,
                "result": result.dict(),
                "updated_at": time.time()
            })
            
            # Record metrics
            self.metrics_collector.record_counter("api.design.success")
            self.metrics_collector.record_gauge("api.design.staple_count", len(origami.staples))
            
            return result
            
        except Exception as e:
            self.tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": time.time()
            })
            self.metrics_collector.record_counter("api.design.failure")
            raise
    
    async def simulate_folding_async(self, request: SimulationRequest) -> SimulationResponse:
        """Asynchronously simulate origami folding."""
        
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "type": "simulation",
            "status": "processing", 
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        try:
            # Submit simulation task
            future = self.task_manager.submit_task(
                self._simulate_folding_task,
                request,
                task_id
            )
            
            return SimulationResponse(
                task_id=task_id,
                status="processing"
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, {"task_id": task_id, "operation": "simulate_folding"})
            raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")
    
    async def _simulate_folding_task(self, request: SimulationRequest, task_id: str) -> SimulationResponse:
        """Background task for folding simulation."""
        
        try:
            # Load origami design
            origami = await self.cache_manager.get(f"design:{request.design_id}")
            if not origami:
                raise HTTPException(status_code=404, detail="Design not found")
            
            # Initialize simulator
            with self.profiler.profile_operation("simulator_setup"):
                if request.gpu_acceleration and self.gpu_manager.is_available:
                    simulator = gpu_acceleration.get_accelerated_simulator()
                else:
                    simulator = OrigamiSimulator(
                        force_field=request.force_field,
                        temperature=request.temperature,
                        salt_concentration=request.salt_concentration
                    )
            
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.1,
                "updated_at": time.time()
            })
            
            # Run simulation with progress updates
            with self.profiler.profile_operation("molecular_simulation"):
                trajectory = await self._run_simulation_with_progress(
                    simulator, origami, request, task_id
                )
            
            # Generate simulation ID and cache result
            simulation_id = str(uuid.uuid4())
            await self.cache_manager.set(f"simulation:{simulation_id}", trajectory, ttl=7200)
            
            # Complete task
            result = SimulationResponse(
                task_id=task_id,
                simulation_id=simulation_id,
                progress=1.0,
                status="completed"
            )
            
            self.tasks[task_id].update({
                "status": "completed",
                "progress": 1.0,
                "result": result.dict(),
                "updated_at": time.time()
            })
            
            # Record metrics
            self.metrics_collector.record_counter("api.simulation.success")
            
            return result
            
        except Exception as e:
            self.tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": time.time()
            })
            self.metrics_collector.record_counter("api.simulation.failure")
            raise
    
    async def _run_simulation_with_progress(self, simulator, origami, request, task_id):
        """Run simulation with periodic progress updates."""
        
        # Simplified simulation with progress tracking
        total_steps = request.time_steps
        update_interval = max(total_steps // 20, 1000)  # Update every 5%
        
        for step in range(0, total_steps, update_interval):
            # Simulate chunk of steps
            current_steps = min(update_interval, total_steps - step)
            
            # Update progress
            progress = 0.1 + (step / total_steps) * 0.8
            self.tasks[task_id].update({
                "progress": progress,
                "updated_at": time.time()
            })
            
            # Yield control to allow other tasks
            await asyncio.sleep(0.1)
        
        # Return mock trajectory
        return {"final_structure": "mock_trajectory_data", "steps": total_steps}
    
    async def decode_structure_async(self, request: DecodingRequest) -> DecodingResponse:
        """Asynchronously decode structure to image."""
        
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "type": "decoding",
            "status": "processing",
            "progress": 0.0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        try:
            # Submit decoding task
            future = self.task_manager.submit_task(
                self._decode_structure_task,
                request,
                task_id
            )
            
            return DecodingResponse(
                task_id=task_id,
                status="processing"
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, {"task_id": task_id, "operation": "decode_structure"})
            raise HTTPException(status_code=500, detail=f"Decoding failed: {e}")
    
    async def _decode_structure_task(self, request: DecodingRequest, task_id: str) -> DecodingResponse:
        """Background task for structure decoding."""
        
        try:
            # Load structure
            structure = await self.cache_manager.get(f"simulation:{request.structure_id}")
            if not structure:
                raise HTTPException(status_code=404, detail="Structure not found")
            
            # Initialize decoder
            with self.profiler.profile_operation("decoder_setup"):
                if self.gpu_manager.is_available:
                    decoder = gpu_acceleration.get_accelerated_decoder()
                else:
                    decoder = TransformerDecoder()
            
            # Update progress
            self.tasks[task_id].update({
                "progress": 0.2,
                "updated_at": time.time()
            })
            
            # Decode structure
            with self.profiler.profile_operation("structure_decoding"):
                decoded_image = await self._decode_with_progress(decoder, structure, task_id)
                confidence = 0.95  # Mock confidence score
            
            # Validate confidence
            if confidence < request.confidence_threshold:
                self.logger.warning(f"Low confidence: {confidence}")
            
            # Generate image ID and cache result
            image_id = str(uuid.uuid4())
            await self.cache_manager.set(f"image:{image_id}", decoded_image, ttl=1800)
            
            # Complete task
            result = DecodingResponse(
                task_id=task_id,
                decoded_image_id=image_id,
                confidence=confidence,
                reconstruction_quality=0.92,
                status="completed"
            )
            
            self.tasks[task_id].update({
                "status": "completed",
                "progress": 1.0,
                "result": result.dict(),
                "updated_at": time.time()
            })
            
            # Record metrics
            self.metrics_collector.record_counter("api.decoding.success")
            self.metrics_collector.record_gauge("api.decoding.confidence", confidence)
            
            return result
            
        except Exception as e:
            self.tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "updated_at": time.time()
            })
            self.metrics_collector.record_counter("api.decoding.failure")
            raise
    
    async def _decode_with_progress(self, decoder, structure, task_id):
        """Decode structure with progress updates."""
        
        # Update progress periodically
        for progress in [0.3, 0.5, 0.7, 0.9]:
            self.tasks[task_id].update({
                "progress": progress,
                "updated_at": time.time()
            })
            await asyncio.sleep(0.5)
        
        # Return mock decoded image
        return np.random.rand(32, 32).astype(np.uint8)
    
    def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """Get status of a task."""
        
        if task_id not in self.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.tasks[task_id]
        return TaskStatusResponse(
            task_id=task_id,
            status=task["status"],
            progress=task["progress"],
            result=task.get("result"),
            error=task.get("error"),
            created_at=task["created_at"],
            updated_at=task["updated_at"]
        )
    
    def get_health_status(self) -> HealthCheckResponse:
        """Get comprehensive health status."""
        
        health_monitor = monitoring.get_health_monitor()
        health_status = health_monitor.get_health_status()
        
        return HealthCheckResponse(
            status=health_status["overall_status"],
            version="0.1.0",
            uptime=time.time() - self.start_time,
            checks=health_status["checks"],
            metrics=self.metrics_collector.get_all_metrics()
        )


# Security dependency
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)."""
    if credentials is None:
        return None  # Allow anonymous access for demo
    
    # In production, verify JWT token here
    token = credentials.credentials
    if token != "demo-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"user": "demo_user"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    monitoring.start_monitoring()
    error_handling.setup_error_recovery()
    logging.info("DNA Origami API started")
    
    yield
    
    # Shutdown
    monitoring.stop_monitoring()
    logging.info("DNA Origami API stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="DNA Origami AutoEncoder API",
        description="RESTful API for encoding images into self-assembling DNA origami structures",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc", 
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Initialize API
    api = DNAOrigarniAPI()
    
    # Routes
    @app.post("/api/v1/encode", response_model=ImageEncodeResponse)
    async def encode_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        encoder_type: str = "base4",
        error_correction: str = "reed_solomon", 
        compression: bool = True,
        image_width: int = 32,
        image_height: int = 32,
        auth = Depends(verify_token)
    ):
        """Encode uploaded image to DNA sequence."""
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        
        request = ImageEncodeRequest(
            encoder_type=encoder_type,
            error_correction=error_correction,
            compression=compression,
            image_size=(image_width, image_height)
        )
        
        return await api.encode_image_async(image_data, request)
    
    @app.post("/api/v1/design", response_model=OrigamiDesignResponse)
    async def design_origami(
        request: OrigamiDesignRequest,
        auth = Depends(verify_token)
    ):
        """Design origami structure from DNA sequence."""
        return await api.design_origami_async(request)
    
    @app.post("/api/v1/simulate", response_model=SimulationResponse)
    async def simulate_folding(
        request: SimulationRequest,
        auth = Depends(verify_token)
    ):
        """Simulate origami folding dynamics."""
        return await api.simulate_folding_async(request)
    
    @app.post("/api/v1/decode", response_model=DecodingResponse)
    async def decode_structure(
        request: DecodingRequest,
        auth = Depends(verify_token)
    ):
        """Decode structure back to image.""" 
        return await api.decode_structure_async(request)
    
    @app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
    async def get_task_status(task_id: str, auth = Depends(verify_token)):
        """Get status of an async task."""
        return api.get_task_status(task_id)
    
    @app.get("/api/v1/health", response_model=HealthCheckResponse)
    async def health_check():
        """Get API health status."""
        return api.get_health_status()
    
    @app.get("/api/v1/metrics")
    async def get_metrics(auth = Depends(verify_token)):
        """Get performance metrics."""
        return api.metrics_collector.get_all_metrics()
    
    @app.get("/api/v1/status")
    async def get_status_report():
        """Get comprehensive status report."""
        return {"report": monitoring.get_status_report()}
    
    # File download endpoints
    @app.get("/api/v1/download/image/{image_id}")
    async def download_image(image_id: str, auth = Depends(verify_token)):
        """Download decoded image."""
        image_data = await api.cache_manager.get(f"image:{image_id}")
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Convert to PNG
        image = Image.fromarray(image_data.astype(np.uint8), mode='L')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=decoded_{image_id}.png"}
        )
    
    @app.get("/api/v1/download/design/{design_id}")
    async def download_design(design_id: str, format: str = "csv", auth = Depends(verify_token)):
        """Download origami design file."""
        design = await api.cache_manager.get(f"design:{design_id}")
        if not design:
            raise HTTPException(status_code=404, detail="Design not found")
        
        # Generate design file based on format
        if format == "csv":
            # Mock CSV export
            content = "staple_name,sequence,start,end\n"
            content += "staple_1,ATGCATGCATGCATGC,0,16\n"
            media_type = "text/csv"
            filename = f"design_{design_id}.csv"
        elif format == "json":
            content = '{"scaffold_length": 7249, "staples": []}'
            media_type = "application/json"
            filename = f"design_{design_id}.json"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    # WebSocket for real-time updates (future enhancement)
    # @app.websocket("/ws/tasks/{task_id}")
    # async def websocket_task_updates(websocket: WebSocket, task_id: str):
    #     """WebSocket endpoint for real-time task updates."""
    #     pass
    
    return app


# Create app instance for easy import
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the API server."""
    
    if workers > 1:
        # Multi-worker deployment
        uvicorn.run(
            "dna_origami_ae.api.server:create_app",
            host=host,
            port=port,
            workers=workers,
            factory=True,
            access_log=True
        )
    else:
        # Single worker development
        app = create_app()
        uvicorn.run(
            app,
            host=host,
            port=port,
            access_log=True,
            reload=True
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DNA Origami API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.workers)