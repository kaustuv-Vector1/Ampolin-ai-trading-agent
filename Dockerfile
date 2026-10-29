# Ampolin AI Trading Agent - Dockerfile
# Production-ready container for Cloud Run, AWS ECS, GCP Compute, or Docker VPS

FROM python:3.12-slim

# Set timezone to Asia/Kolkata (IST) for Indian Equity Market Hours
ENV TZ=Asia/Kolkata
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Web Dashboard Port
EXPOSE 8000

# Default command starts the Web UI Dashboard and API server
CMD ["python", "ui.py"]
