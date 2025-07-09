# Multi-stage Dockerfile for Orchestrator Agent
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY agents/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared dependencies
COPY shared/ shared/
COPY config/ config/

# Copy orchestrator-specific code
COPY agents/orchestrator/ agents/orchestrator/

# Install the application
RUN pip install -e .

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory and copy application
WORKDIR /app
COPY --from=builder /app .

# Create logs directory
RUN mkdir -p logs && chown -R app:app logs

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Expose port
EXPOSE ${PORT}

# Set entrypoint
ENTRYPOINT ["python", "-m", "agents.orchestrator.main"]

# =============================================================================
# Orchestrator Agent Docker Image
# =============================================================================
#
# This Dockerfile creates a production-ready image for the orchestrator agent.
#
# Build Arguments:
#   - PORT: Port to expose (default: 8000)
#
# Environment Variables:
#   - ORCHESTRATOR_HOST: Host to bind to (default: 0.0.0.0)
#   - ORCHESTRATOR_PORT: Port to bind to (default: 8000)
#   - CLASSIFIER_HOST: Classifier service host (default: classifier)
#   - CLASSIFIER_PORT: Classifier service port (default: 8001)
#   - LOG_LEVEL: Logging level (default: INFO)
#   - TRACE_ENABLED: Enable tracing (default: true)
#   - API_TITLE: API title (default: WhatsApp Sales Assistant)
#   - API_VERSION: API version (default: 1.0.0)
#
# Build Commands:
#   docker build -f agents/orchestrator/Dockerfile -t whatsapp-orchestrator .
#   docker run -p 8000:8000 whatsapp-orchestrator
#
# Features:
#   - Multi-stage build for smaller production image
#   - Non-root user for security
#   - Health check endpoint
#   - Proper logging directory
#   - Environment variable configuration
#   - Optimized for production deployment
#
# =============================================================================