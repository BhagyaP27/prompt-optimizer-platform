# ml-service/Dockerfile
# Multi-stage build for smaller production images and faster builds
# Stage 1 installs dependencies, Stage 2 copies only what's needed

# ============================================
# Stage 1: Builder
# ============================================
# Use official Python slim image as base (smaller than full Python image)
FROM python:3.10-slim as builder

# Set working directory for build stage
WORKDIR /build

# Install build dependencies needed for Python packages
# These are removed in final stage to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching)
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .

# Install Python dependencies to user directory
# --user installs to /root/.local (easier to copy to final stage)
# --no-cache-dir reduces image size by not caching downloads
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
# Start fresh with slim image (without build tools)
FROM python:3.10-slim

# Set working directory for application
WORKDIR /app

# Copy Python packages from builder stage
# This includes all installed dependencies from pip
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
# Done last so code changes don't invalidate previous layers
COPY . .

# Create directories for models and logs
RUN mkdir -p /models /app/logs

# Environment variables
# These can be overridden by docker-compose or deployment configs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/best_model \
    LOG_LEVEL=INFO \
    PORT=8000

# Expose port for API
EXPOSE 8000

# Health check - Docker will mark container as unhealthy if this fails
# Useful for orchestrators like Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user for security
# Running as root in containers is a security risk
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

# Start application with uvicorn
# --host 0.0.0.0 allows external connections
# --port 8000 matches exposed port
# --workers 1 for development (increase for production)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================
# Build instructions:
# docker build -t prompt-optimizer-ml:latest .
#
# Run instructions:
# docker run -p 8000:8000 \
#   -v $(pwd)/models:/models \
#   -e MODEL_PATH=/models/best_model \
#   prompt-optimizer-ml:latest
# ============================================