"""API routes for DNA Origami AutoEncoder."""

from fastapi import APIRouter

# Create main router
router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "DNA Origami AutoEncoder API", "version": "0.1.0"}

@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is operational"}

# Export router for main app
__all__ = ["router"]