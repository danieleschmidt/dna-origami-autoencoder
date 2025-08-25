"""API models for DNA Origami AutoEncoder REST API."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ImageUploadRequest(BaseModel):
    """Request model for image upload."""
    size: Optional[tuple] = Field(default=(32, 32), description="Target image size (width, height)")
    grayscale: bool = Field(default=True, description="Convert to grayscale")
    validate: bool = Field(default=True, description="Enable biological constraints validation")


class DNASequenceResponse(BaseModel):
    """Response model for DNA sequence."""
    sequence: str = Field(description="DNA sequence string")
    length: int = Field(description="Sequence length in bases")
    gc_content: float = Field(description="GC content percentage")
    name: Optional[str] = Field(default=None, description="Sequence name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EncodingResponse(BaseModel):
    """Response model for image encoding."""
    sequences: List[DNASequenceResponse] = Field(description="Generated DNA sequences")
    encoding_time_ms: float = Field(description="Encoding time in milliseconds")
    compression_ratio: float = Field(description="Compression ratio achieved")
    total_bases: int = Field(description="Total number of bases generated")
    status: str = Field(default="success", description="Encoding status")


class OrigamiDesignRequest(BaseModel):
    """Request model for origami design."""
    target_shape: str = Field(default="square", description="Target origami shape")
    dimensions: tuple = Field(default=(100, 100), description="Dimensions in nanometers")
    scaffold_length: int = Field(default=7249, description="Scaffold length (M13mp18=7249)")
    staple_length: int = Field(default=32, description="Staple strand length")
    optimize: bool = Field(default=True, description="Optimize staple sequences")


class StapleInfo(BaseModel):
    """Information about a staple strand."""
    sequence: str = Field(description="Staple sequence")
    start_position: int = Field(description="Start position on scaffold")
    end_position: int = Field(description="End position on scaffold")
    binding_regions: List[tuple] = Field(description="Binding regions as (start, end) tuples")


class OrigamiDesignResponse(BaseModel):
    """Response model for origami design."""
    staples: List[StapleInfo] = Field(description="Staple strand information")
    scaffold_length: int = Field(description="Scaffold sequence length")
    design_time_ms: float = Field(description="Design time in milliseconds")
    estimated_yield: float = Field(description="Estimated folding yield percentage")
    dimensions: tuple = Field(description="Design dimensions in nanometers")
    status: str = Field(default="success", description="Design status")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(default="healthy", description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(default="0.1.0", description="API version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    memory_usage_mb: float = Field(description="Memory usage in MB")
    active_requests: int = Field(description="Number of active requests")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class PipelineRequest(BaseModel):
    """Request model for full pipeline execution."""
    size: Optional[tuple] = Field(default=(32, 32), description="Image size")
    shape: str = Field(default="square", description="Origami shape")
    dimensions: tuple = Field(default=(100, 100), description="Origami dimensions in nm")
    simulate: bool = Field(default=False, description="Include folding simulation")
    decode: bool = Field(default=True, description="Include decoding step")


class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""
    encoding: EncodingResponse = Field(description="Encoding results")
    design: OrigamiDesignResponse = Field(description="Design results")
    total_time_ms: float = Field(description="Total pipeline execution time")
    pipeline_id: str = Field(description="Unique pipeline execution ID")
    status: str = Field(default="completed", description="Pipeline status")