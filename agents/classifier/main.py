"""
Classifier Agent Main Entry Point.

This module serves as the entry point for the classifier agent service,
setting up the FastAPI application and starting the server.
"""

import os
import signal
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables explicitly before any other imports
# This ensures .env is loaded regardless of where the script is run from
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}")

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from agents.classifier.adapters.inbound.fastapi_router import router
from agents.classifier.agent import classifier_agent
from shared.observability import get_logger
from shared.observability_enhanced import enhanced_observability
from config.settings import settings


# Configure logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown logic for the classifier service.
    """
    # Startup
    logger.info(
        "Starting classifier service",
        service="classifier",
        version="1.0.0",
        model_name=settings.model_name,
        port=settings.classifier_port,
    )

    try:
        # Initialize classifier agent
        await classifier_agent.health_check()
        logger.info("Classifier agent initialized successfully")

        yield

    except Exception as e:
        logger.error("Failed to start classifier service", error=str(e))
        raise

    finally:
        # Shutdown
        logger.info("Shutting down classifier service")


# Create FastAPI application
app = FastAPI(
    title="WhatsApp Sales Assistant - Classifier Service",
    description="Message classification service using PydanticAI and Gemini Flash 2.5",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_docker_environment else None,
    redoc_url="/redoc" if not settings.is_docker_environment else None,
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware for response compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include router
app.include_router(router)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": "2024-01-01T00:00:00Z",  # Use proper timestamp
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "classifier",
        "status": "running",
        "version": "1.0.0",
        "description": "WhatsApp Sales Assistant - Classifier Service",
        "endpoints": {
            "health": "/api/v1/health",
            "classify": "/api/v1/classify",
            "classify_direct": "/api/v1/classify-direct",
            "metrics": "/api/v1/metrics",
            "observability_metrics": "/api/v1/observability-metrics",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "classifier"}


@app.get("/api/v1/observability-metrics")
async def get_observability_metrics():
    """Get enhanced observability metrics."""
    try:
        metrics_summary = enhanced_observability.get_metrics_summary()
        return metrics_summary
    except Exception as e:
        logger.error("Failed to get observability metrics", error=str(e))
        return JSONResponse(status_code=500, content={"error": f"Failed to get metrics: {str(e)}"})


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point for the classifier service."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Configure uvicorn
    uvicorn_config = {
        "app": "agents.classifier.main:app",
        "host": "0.0.0.0",
        "port": settings.classifier_port,
        "log_level": settings.log_level.lower(),
        "access_log": True,
        "reload": False,  # Disable reload in production
        "workers": 1,  # Single worker for simplicity
    }

    # Add uvloop for better performance if available
    try:
        import uvloop

        uvicorn_config["loop"] = "uvloop"
    except ImportError:
        pass

    # Log startup configuration
    logger.info(
        "Starting classifier service",
        host=uvicorn_config["host"],
        port=uvicorn_config["port"],
        log_level=uvicorn_config["log_level"],
        workers=uvicorn_config["workers"],
        model_name=settings.model_name,
    )

    # Start the server
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error("Failed to start server", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
