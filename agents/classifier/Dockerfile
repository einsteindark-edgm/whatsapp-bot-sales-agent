# Multi-stage Dockerfile for Classifier Agent
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
COPY agents/classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared dependencies
COPY shared/ shared/
COPY config/ config/

# Copy classifier-specific code
COPY agents/classifier/ agents/classifier/

# Install the application
RUN pip install -e .

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8001

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
ENTRYPOINT ["python", "-m", "agents.classifier.main"]

# =============================================================================
# Classifier Agent Docker Image
# =============================================================================
#
# This Dockerfile creates a production-ready image for the classifier agent.
#
# Build Arguments:
#   - PORT: Port to expose (default: 8001)
#
# Environment Variables:
#   - GEMINI_API_KEY: Required - Gemini API key
#   - CLASSIFIER_HOST: Host to bind to (default: 0.0.0.0)
#   - CLASSIFIER_PORT: Port to bind to (default: 8001)
#   - LOG_LEVEL: Logging level (default: INFO)
#   - TRACE_ENABLED: Enable tracing (default: true)
#   - MODEL_NAME: Gemini model name (default: google-gla:gemini-2.0-flash)
#   - CONFIDENCE_THRESHOLD: Classification confidence threshold (default: 0.7)
#   - GEMINI_TEMPERATURE: Model temperature (default: 0.0)
#   - GEMINI_MAX_TOKENS: Max tokens (default: 100)
#
# Build Commands:
#   docker build -f agents/classifier/Dockerfile -t whatsapp-classifier .
#   docker run -p 8001:8001 -e GEMINI_API_KEY=your_key whatsapp-classifier
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