"""
Orchestrator Agent Main Entry Point.

This module serves as the entry point for the orchestrator agent service,
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

from agents.orchestrator.adapters.inbound.fastapi_router import router
from agents.orchestrator.adapters.inbound.whatsapp_webhook_router import router as whatsapp_router
from agents.orchestrator.agent import orchestrator_agent
from shared.observability import get_logger
from shared.observability_enhanced import enhanced_observability
from shared.observability_setup import initialize_app_observability, shutdown_observability
from config.settings import settings


# Configure logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown logic for the orchestrator service.
    """
    # Startup
    logger.info(
        "Starting orchestrator service",
        service="orchestrator",
        version="1.0.0",
        classifier_url=settings.classifier_url,
        port=settings.orchestrator_port,
    )

    try:
        # Initialize observability
        initialize_app_observability(app, settings)
        logger.info("Observability initialized successfully")

        # Initialize orchestrator agent
        await orchestrator_agent.health_check()
        logger.info("Orchestrator agent initialized successfully")

        yield

    except Exception as e:
        logger.error("Failed to start orchestrator service", error=str(e))
        raise

    finally:
        # Shutdown
        logger.info("Shutting down orchestrator service")
        
        # Shutdown observability
        shutdown_observability()
        logger.info("Observability shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="WhatsApp Sales Assistant - Orchestrator Service",
    description="Workflow orchestration service using Google ADK and A2A protocol",
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

# Include routers
app.include_router(router)
app.include_router(whatsapp_router)


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
        "service": "orchestrator",
        "status": "running",
        "version": "1.0.0",
        "description": "WhatsApp Sales Assistant - Orchestrator Service",
        "endpoints": {
            "health": "/api/v1/health",
            "orchestrate": "/api/v1/orchestrate",
            "orchestrate_direct": "/api/v1/orchestrate-direct",
            "metrics": "/api/v1/metrics",
            "observability_metrics": "/api/v1/observability-metrics",
            "status": "/api/v1/status",
            "whatsapp_webhook": "/webhook/whatsapp",
            "whatsapp_health": "/webhook/whatsapp/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "orchestrator"}


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
    """Main entry point for the orchestrator service."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Configure uvicorn
    uvicorn_config = {
        "app": "agents.orchestrator.main:app",
        "host": "0.0.0.0",
        "port": settings.orchestrator_port,
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
        "Starting orchestrator service",
        host=uvicorn_config["host"],
        port=uvicorn_config["port"],
        log_level=uvicorn_config["log_level"],
        workers=uvicorn_config["workers"],
        classifier_url=settings.classifier_url,
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
