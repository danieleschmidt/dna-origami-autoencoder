#!/bin/bash
# DNA Origami AutoEncoder - Server Startup Script

set -e

echo "🧬 Starting DNA Origami AutoEncoder API Server"
echo "============================================"

# Configuration
HOST=${DNA_HOST:-"0.0.0.0"}
PORT=${DNA_PORT:-8000}
WORKERS=${DNA_WORKERS:-1}
LOG_LEVEL=${DNA_LOG_LEVEL:-"info"}

# Check if running in container
if [ -f /.dockerenv ]; then
    echo "📦 Running in Docker container"
    ENVIRONMENT="container"
else
    echo "🖥️  Running in host environment"
    ENVIRONMENT="host"
fi

# Validate Python dependencies
echo "🔍 Checking dependencies..."
python3 -c "import sys; sys.path.insert(0, '.'); from dna_origami_ae import DNASequence; print('✅ Core modules available')" || {
    echo "❌ Missing dependencies. Please install requirements."
    exit 1
}

# Run basic health check
echo "🧪 Running health check..."
python3 -c "
import sys
sys.path.insert(0, '.')
from api_server import app
print('✅ API server imports successfully')
" || {
    echo "❌ API server health check failed"
    exit 1
}

# Create logs directory
mkdir -p logs

# Start server
echo "🚀 Starting server on $HOST:$PORT"
echo "   Workers: $WORKERS"
echo "   Log Level: $LOG_LEVEL"
echo "   Environment: $ENVIRONMENT"
echo ""

if command -v uvicorn >/dev/null 2>&1; then
    # Use uvicorn if available
    exec uvicorn api_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --use-colors
else
    # Fallback to direct Python execution
    echo "⚠️  uvicorn not found, using fallback startup"
    exec python3 api_server.py
fi