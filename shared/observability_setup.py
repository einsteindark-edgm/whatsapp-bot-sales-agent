"""
Observability setup with auto-instrumentation for all supported packages.

This module provides centralized initialization of all observability platforms
including Logfire, Arize, and OpenTelemetry with proper auto-instrumentation.
"""

import os
from typing import Optional, Dict, Any
from shared.observability import get_logger
from config.settings import Settings

logger = get_logger(__name__)


def initialize_observability(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Initialize all observability integrations with auto-instrumentation.
    
    This function:
    1. Configures Logfire with auto-instrumentation for supported packages
    2. Sets up Arize with OpenTelemetry integration
    3. Configures OpenTelemetry trace exporters
    4. Returns status of each integration
    
    Args:
        settings: Application settings (will load from env if not provided)
        
    Returns:
        Dict with initialization status for each platform
    """
    if not settings:
        settings = Settings()
    
    status = {
        "logfire": False,
        "arize": False,
        "opentelemetry": False,
        "auto_instrumentation": {
            "fastapi": False,
            "httpx": False,
            "pydantic": False,
            "google_genai": False,
            "openai": False,
        }
    }
    
    # 1. Configure Logfire with auto-instrumentation
    if settings.logfire_token:
        try:
            import logfire
            
            # Configure Logfire
            logfire.configure(
                token=settings.logfire_token,
                project_name=settings.logfire_project_name,
                environment=settings.logfire_environment,
            )
            
            status["logfire"] = True
            logger.info(
                "Logfire configured successfully",
                extra={
                    "project": settings.logfire_project_name,
                    "environment": settings.logfire_environment,
                }
            )
            
            # Auto-instrument FastAPI
            try:
                logfire.instrument_fastapi()
                status["auto_instrumentation"]["fastapi"] = True
                logger.info("FastAPI auto-instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument FastAPI: {e}")
            
            # Auto-instrument HTTPX
            try:
                logfire.instrument_httpx()
                status["auto_instrumentation"]["httpx"] = True
                logger.info("HTTPX auto-instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument HTTPX: {e}")
            
            # Auto-instrument Pydantic
            try:
                logfire.instrument_pydantic()
                status["auto_instrumentation"]["pydantic"] = True
                logger.info("Pydantic auto-instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Pydantic: {e}")
            
            # Auto-instrument system metrics
            try:
                logfire.instrument_system_metrics()
                logger.info("System metrics instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument system metrics: {e}")
                
        except ImportError:
            logger.warning("Logfire not installed, skipping Logfire configuration")
        except Exception as e:
            logger.error(f"Failed to configure Logfire: {e}")
    else:
        logger.info("Logfire token not provided, skipping Logfire configuration")
    
    # 2. Configure Arize with OpenTelemetry
    if settings.arize_api_key and settings.arize_space_key:
        try:
            from arize.otel import register
            
            # Register Arize as OpenTelemetry exporter
            # Arize uses its own endpoint, don't pass custom endpoint
            tracer_provider = register(
                space_id=settings.arize_space_key,
                api_key=settings.arize_api_key,
                project_name=settings.arize_model_id,
            )
            
            status["arize"] = True
            logger.info(
                "Arize OpenTelemetry integration configured",
                extra={
                    "model_id": settings.arize_model_id,
                    "model_version": settings.arize_model_version,
                }
            )
            
            # Auto-instrument Google GenAI
            try:
                from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
                GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
                status["auto_instrumentation"]["google_genai"] = True
                logger.info("Google GenAI auto-instrumentation enabled")
            except ImportError:
                logger.warning("openinference-instrumentation-google-genai not installed")
            except Exception as e:
                logger.warning(f"Failed to instrument Google GenAI: {e}")
            
            # Auto-instrument OpenAI (in case we use it)
            try:
                from openinference.instrumentation.openai import OpenAIInstrumentor
                OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
                status["auto_instrumentation"]["openai"] = True
                logger.info("OpenAI auto-instrumentation enabled")
            except ImportError:
                logger.warning("openinference-instrumentation-openai not installed")
            except Exception as e:
                logger.warning(f"Failed to instrument OpenAI: {e}")
                
        except ImportError:
            logger.warning("arize-otel not installed, skipping Arize configuration")
        except Exception as e:
            logger.error(f"Failed to configure Arize: {e}")
    else:
        logger.info("Arize credentials not provided, skipping Arize configuration")
    
    # 3. Configure additional OpenTelemetry exporters if needed
    if settings.otel_exporter_endpoint and not status["arize"]:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource
            
            # Create resource with service information
            resource = Resource.create({
                "service.name": settings.otel_service_name,
                "service.version": settings.api_version,
                "deployment.environment": settings.logfire_environment,
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Parse headers if provided
            headers = None
            if settings.otel_headers:
                headers = dict(h.split("=", 1) for h in settings.otel_headers.split(","))
            
            # Create OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_endpoint,
                headers=headers,
            )
            
            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            status["opentelemetry"] = True
            logger.info(
                "OpenTelemetry configured with OTLP exporter",
                extra={"endpoint": settings.otel_exporter_endpoint}
            )
            
        except Exception as e:
            logger.error(f"Failed to configure OpenTelemetry: {e}")
    
    # Log final status
    logger.info(
        "Observability initialization complete",
        extra={"status": status}
    )
    
    return status


def initialize_app_observability(app: Any, settings: Optional[Settings] = None) -> None:
    """
    Initialize observability for a FastAPI application.
    
    This should be called during app startup to ensure all instrumentation
    is properly configured before handling requests.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    if not settings:
        settings = Settings()
    
    # Initialize general observability
    status = initialize_observability(settings)
    
    # Add middleware
    try:
        from shared.observability_middleware import TraceIDMiddleware
        app.add_middleware(TraceIDMiddleware)
        logger.info("TraceID middleware added to application")
    except Exception as e:
        logger.error(f"Failed to add TraceID middleware: {e}")
    
    # Store observability status in app state
    app.state.observability_status = status
    
    # Log startup with observability
    logger.info(
        "Application observability initialized",
        extra={
            "app_title": settings.api_title,
            "app_version": settings.api_version,
            "observability_status": status,
        }
    )


def get_tracer(name: str):
    """
    Get an OpenTelemetry tracer for manual instrumentation.
    
    Args:
        name: Name of the tracer (usually __name__)
        
    Returns:
        OpenTelemetry tracer instance
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        logger.warning("OpenTelemetry not available, returning None tracer")
        return None


def shutdown_observability():
    """
    Gracefully shutdown observability integrations.
    
    This should be called during application shutdown to ensure
    all pending spans and metrics are flushed.
    """
    try:
        # Flush OpenTelemetry
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
            logger.info("OpenTelemetry provider shutdown complete")
    except Exception as e:
        logger.error(f"Error during OpenTelemetry shutdown: {e}")
    
    try:
        # Flush Logfire
        import logfire
        if hasattr(logfire, "shutdown"):
            logfire.shutdown()
            logger.info("Logfire shutdown complete")
    except Exception as e:
        logger.error(f"Error during Logfire shutdown: {e}")
    
    logger.info("Observability shutdown complete")