#!/bin/bash
# DNA Origami AutoEncoder - Deployment Script

set -e

echo "🚀 DNA ORIGAMI AUTOENCODER - DEPLOYMENT SCRIPT"
echo "=============================================="

# Configuration
DEPLOYMENT_TYPE=${1:-"local"}
ENVIRONMENT=${2:-"production"}

echo "📋 Deployment Configuration:"
echo "   Type: $DEPLOYMENT_TYPE"
echo "   Environment: $ENVIRONMENT"
echo ""

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    if python3 run_tests.py; then
        echo "✅ All tests passed"
        return 0
    else
        echo "❌ Tests failed"
        return 1
    fi
}

# Function to build Docker image
build_docker() {
    echo "🐳 Building Docker image..."
    docker build -f Dockerfile.simple -t dna-origami-ae:latest .
    echo "✅ Docker image built successfully"
}

# Function to deploy locally
deploy_local() {
    echo "🏠 Deploying locally..."
    
    # Run tests first
    if ! run_tests; then
        echo "❌ Deployment aborted due to test failures"
        exit 1
    fi
    
    # Start server
    chmod +x start_server.sh
    echo "✅ Starting local server..."
    ./start_server.sh
}

# Function to deploy with Docker
deploy_docker() {
    echo "🐳 Deploying with Docker..."
    
    # Run tests first
    if ! run_tests; then
        echo "❌ Deployment aborted due to test failures"
        exit 1
    fi
    
    # Build and start with Docker Compose
    build_docker
    
    echo "🚢 Starting with Docker Compose..."
    docker-compose -f docker-compose.simple.yml up --build -d
    
    echo "✅ Docker deployment completed"
    echo "📡 API available at http://localhost:8000"
    echo "📊 Health check: curl http://localhost:8000/health"
}

# Function to deploy to cloud
deploy_cloud() {
    echo "☁️  Cloud deployment not implemented yet"
    echo "   Available options in future releases:"
    echo "   - AWS ECS/EKS"
    echo "   - Google Cloud Run/GKE" 
    echo "   - Azure Container Instances/AKS"
    echo "   - DigitalOcean App Platform"
    exit 1
}

# Function to show deployment status
show_status() {
    echo "📊 DEPLOYMENT STATUS"
    echo "==================="
    
    if command -v docker >/dev/null 2>&1; then
        echo "🐳 Docker containers:"
        docker-compose -f docker-compose.simple.yml ps 2>/dev/null || echo "   No containers running"
        echo ""
    fi
    
    echo "🌐 Testing API endpoints..."
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "   ✅ Health check: PASS"
    else
        echo "   ❌ Health check: FAIL"
    fi
    
    if curl -f http://localhost:8000/api/v1/demo >/dev/null 2>&1; then
        echo "   ✅ Demo endpoint: PASS"
    else
        echo "   ❌ Demo endpoint: FAIL"
    fi
    
    echo ""
    echo "📚 Available endpoints:"
    echo "   http://localhost:8000/          - API info"
    echo "   http://localhost:8000/docs      - API documentation"
    echo "   http://localhost:8000/health    - Health check"
    echo "   http://localhost:8000/api/v1/demo - Demo endpoint"
}

# Main deployment logic
case $DEPLOYMENT_TYPE in
    "local")
        deploy_local
        ;;
    "docker")
        deploy_docker
        ;;
    "cloud")
        deploy_cloud
        ;;
    "test")
        run_tests
        ;;
    "status")
        show_status
        ;;
    "stop")
        echo "🛑 Stopping services..."
        docker-compose -f docker-compose.simple.yml down 2>/dev/null || echo "No Docker services to stop"
        pkill -f "api_server.py" 2>/dev/null || echo "No local services to stop"
        echo "✅ Services stopped"
        ;;
    *)
        echo "❓ Usage: $0 {local|docker|cloud|test|status|stop} [environment]"
        echo ""
        echo "Examples:"
        echo "  $0 local                    # Deploy locally"
        echo "  $0 docker                   # Deploy with Docker"
        echo "  $0 test                     # Run tests only"
        echo "  $0 status                   # Check deployment status"
        echo "  $0 stop                     # Stop all services"
        exit 1
        ;;
esac