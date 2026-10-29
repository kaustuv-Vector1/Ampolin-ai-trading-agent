#!/bin/bash
# ==============================================================================
# Ampolin AI Trading Agent - Cloud Deployment Helper
# ==============================================================================

set -e

echo "=========================================================="
echo "🚀 Deploying Ampolin AI Trading Agent to the Cloud"
echo "=========================================================="

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "🐳 Building Docker container..."
    docker build -t ampolin-trader:latest .

    echo "▶️ Launching container with Docker Compose..."
    docker compose up -d

    echo "✅ Ampolin AI Trading Agent is running live in the cloud!"
    echo "📊 Web Dashboard: http://localhost:8000"
else
    echo "⚠️ Docker not detected. Running native Python background daemon..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    nohup python3 ui.py > ui.log 2>&1 &
    nohup python3 cloud_scheduler.py > cloud_scheduler.log 2>&1 &
    echo "✅ Daemon started in background!"
fi
