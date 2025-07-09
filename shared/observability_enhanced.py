"""
Enhanced Observability Configuration with Logfire and Arize AX.

This module provides comprehensive observability features including:
- Pydantic Logfire for structured logging and tracing
- Arize AX for LLM monitoring and evaluation
- OpenTelemetry for distributed tracing
- Custom metrics and performance monitoring
"""

import logging
import logfire
import structlog
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import time
import uuid
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

# Arize imports
try:
    from arize.otel import register as arize_register
    from arize.pandas.logger import Client as ArizeClient
    from arize.utils.types import ModelTypes, Environments, Metrics, Schema
    import pandas as pd
    ARIZE_AVAILABLE = True
    ARIZE_OTEL_AVAILABLE = True
except ImportError:
    try:
        from arize.pandas.logger import Client as ArizeClient
        from arize.utils.types import ModelTypes, Environments, Metrics, Schema
        import pandas as pd
        ARIZE_AVAILABLE = True
        ARIZE_OTEL_AVAILABLE = False
        arize_register = None
    except ImportError:
        ARIZE_AVAILABLE = False
        ARIZE_OTEL_AVAILABLE = False
        ArizeClient = None
        ModelTypes = None
        Environments = None
        arize_register = None

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from config.settings import settings


class ObservabilityLevel:
    """Observability level constants."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMTrace(BaseModel):
    """Model for LLM interaction traces."""
    
    trace_id: str = Field(..., description="Unique trace identifier")
    model_name: str = Field(..., description="LLM model name")
    prompt: str = Field(..., description="Input prompt")
    response: str = Field(..., description="Model response")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    token_count_input: Optional[int] = Field(None, description="Input token count")
    token_count_output: Optional[int] = Field(None, description="Output token count")
    classification_label: Optional[str] = Field(None, description="Classification result")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    agent_name: str = Field(..., description="Agent that made the request")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ArizeIntegration:
    """Arize integration for LLM monitoring using OpenTelemetry."""
    
    def __init__(self):
        self.client = None
        self.tracer_provider = None
        self.enabled = False
        self.otel_enabled = False
        self.data_buffer = []
        self._initialize()
    
    def _initialize(self):
        """Initialize Arize client with OpenTelemetry integration."""
        if not ARIZE_AVAILABLE:
            logging.warning("Arize package not available. Install with: pip install arize arize-otel")
            return
        
        if not settings.arize_api_key or not settings.arize_space_key:
            logging.info("Arize credentials not configured. Skipping Arize integration.")
            return
        
        try:
            # First try OpenTelemetry integration (preferred method)
            if ARIZE_OTEL_AVAILABLE and arize_register:
                logging.info("Initializing Arize with OpenTelemetry integration...")
                
                # Convert space_key to space_id format for OTel
                space_id = settings.arize_space_key  # Assume this is already space_id format
                
                self.tracer_provider = arize_register(
                    space_id=space_id,
                    api_key=settings.arize_api_key,
                    project_name=settings.arize_model_id
                )
                
                self.otel_enabled = True
                self.enabled = True
                logging.info(f"✅ Arize OpenTelemetry integration initialized successfully for project: {settings.arize_model_id}")
                
            else:
                # Fallback to pandas client
                logging.info("Falling back to Arize pandas client...")
                self.client = ArizeClient(
                    api_key=settings.arize_api_key,
                    space_key=settings.arize_space_key
                )
                self.enabled = True
                logging.info(f"Arize pandas client initialized for model: {settings.arize_model_id}")
                
        except Exception as e:
            logging.error(f"Failed to initialize Arize: {e}")
            # Try fallback to pandas client if OTel fails
            try:
                if not self.enabled:
                    self.client = ArizeClient(
                        api_key=settings.arize_api_key,
                        space_key=settings.arize_space_key
                    )
                    self.enabled = True
                    logging.info(f"Arize fallback pandas client initialized for model: {settings.arize_model_id}")
            except Exception as fallback_error:
                logging.error(f"Arize fallback also failed: {fallback_error}")
                self.enabled = False
    
    def log_llm_trace(self, llm_trace: LLMTrace):
        """Log LLM interaction to Arize using OpenTelemetry when available."""
        if not self.enabled:
            logging.debug("Arize not enabled, skipping log")
            return
        
        try:
            # Use OpenTelemetry integration if available
            if self.otel_enabled and self.tracer_provider:
                logging.info(f"📡 Sending LLM trace to Arize via OpenTelemetry: {llm_trace.trace_id}")
                
                # Get tracer from the tracer provider
                tracer = self.tracer_provider.get_tracer(__name__)
                
                # Create span with LLM data
                with tracer.start_as_current_span("llm_classification") as span:
                    # Set span attributes for Arize
                    span.set_attributes({
                        "llm.request.model": llm_trace.model_name,
                        "llm.response.model": llm_trace.model_name,
                        "llm.model_name": llm_trace.model_name,
                        "llm.token_count.prompt": llm_trace.token_count_input or 0,
                        "llm.token_count.completion": llm_trace.token_count_output or 0,
                        "llm.latency": llm_trace.latency_ms,
                        "llm.agent_name": llm_trace.agent_name,
                        "llm.classification_label": llm_trace.classification_label or "unknown",
                        "llm.confidence_score": llm_trace.confidence_score or 0.0,
                        "input.value": llm_trace.prompt[:1000],  # Truncate long prompts
                        "output.value": llm_trace.response[:1000],  # Truncate long responses
                        "session.id": llm_trace.trace_id,
                        "user.id": llm_trace.agent_name,
                        "metadata.environment": settings.logfire_environment,
                    })
                    
                    logging.info(f"✅ Successfully sent LLM trace to Arize via OTel: {llm_trace.trace_id}")
                return
            
            # Fallback to pandas client if OTel not available
            if self.client:
                # Create DataFrame with LLM trace data - ensure all required fields
                df = pd.DataFrame([{
                    "prediction_id": llm_trace.trace_id,
                    "prediction_timestamp": llm_trace.timestamp,
                    "agent_name": llm_trace.agent_name,
                    "model_name": llm_trace.model_name,
                    "prompt_text": llm_trace.prompt[:1000],  # Truncate long prompts
                    "response_text": llm_trace.response[:1000],  # Truncate long responses
                    "latency_ms": float(llm_trace.latency_ms),
                    "token_count_input": int(llm_trace.token_count_input or 0),
                    "token_count_output": int(llm_trace.token_count_output or 0),
                    "classification_label": llm_trace.classification_label or "unknown",
                    "confidence_score": float(llm_trace.confidence_score or 0.0),
                    "environment": settings.logfire_environment,
                }])
                
                # Define schema - simplified to avoid complex nested features
                schema = Schema(
                    prediction_id_column_name="prediction_id",
                    timestamp_column_name="prediction_timestamp",
                    prediction_label_column_name="classification_label",
                    prediction_score_column_name="confidence_score",
                    feature_column_names=[
                        "agent_name", 
                        "model_name", 
                        "latency_ms", 
                        "token_count_input", 
                        "token_count_output"
                    ]
                )
                
                # Determine environment
                env = Environments.PRODUCTION if settings.logfire_environment == "production" else Environments.DEVELOPMENT
                
                # Log to Arize with explicit logging
                logging.info(f"📡 Sending LLM trace to Arize via pandas: {llm_trace.trace_id}")
                
                response = self.client.log(
                    dataframe=df,
                    model_id=settings.arize_model_id,
                    model_version=settings.arize_model_version,
                    model_type=ModelTypes.SCORE_CATEGORICAL,
                    environment=env,
                    schema=schema
                )
                
                if response.status_code == 200:
                    logging.info(f"✅ Successfully logged to Arize via pandas: {llm_trace.trace_id}")
                else:
                    logging.warning(f"❌ Arize pandas logging failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logging.error(f"❌ Failed to log to Arize: {e}")
            import traceback
            logging.debug(f"Arize error traceback: {traceback.format_exc()}")
    
    def log_evaluation_result(self, trace_id: str, metric_name: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        """Log evaluation results to Arize."""
        if not self.enabled:
            return
        
        try:
            # Create evaluation DataFrame
            df = pd.DataFrame([{
                "prediction_id": trace_id,
                "actual_label": metadata.get("actual_label", ""),
                "evaluation_score": score,
                "metric_name": metric_name,
                "timestamp": datetime.now(timezone.utc),
            }])
            
            # Define schema for evaluation
            schema = Schema(
                prediction_id_column_name="prediction_id",
                actual_label_column_name="actual_label",
                timestamp_column_name="timestamp",
            )
            
            # Log evaluation to Arize
            response = self.client.log(
                dataframe=df,
                model_id=settings.arize_model_id,
                model_version=settings.arize_model_version,
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.DEVELOPMENT if settings.logfire_environment == "development" else Environments.PRODUCTION,
                schema=schema
            )
            
            if response.status_code == 200:
                logging.debug(f"Successfully logged evaluation to Arize: {trace_id}")
            else:
                logging.warning(f"Arize evaluation logging failed with status {response.status_code}")
                
        except Exception as e:
            logging.error(f"Failed to log evaluation to Arize: {e}")


class OpenTelemetryIntegration:
    """OpenTelemetry integration for distributed tracing."""
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self.enabled = False
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenTelemetry."""
        if not OTEL_AVAILABLE:
            logging.warning("OpenTelemetry packages not available.")
            return
        
        try:
            # Create resource
            resource = Resource.create({
                "service.name": settings.otel_service_name,
                "service.version": settings.api_version,
                "deployment.environment": settings.logfire_environment,
            })
            
            # Setup tracing
            if settings.otel_exporter_endpoint:
                trace_exporter = OTLPSpanExporter(
                    endpoint=settings.otel_exporter_endpoint,
                    headers=self._parse_headers(settings.otel_headers)
                )
                span_processor = BatchSpanProcessor(trace_exporter)
                
                tracer_provider = TracerProvider(resource=resource)
                tracer_provider.add_span_processor(span_processor)
                trace.set_tracer_provider(tracer_provider)
            
            # Setup metrics
            if settings.otel_exporter_endpoint:
                metric_exporter = OTLPMetricExporter(
                    endpoint=settings.otel_exporter_endpoint.replace("traces", "metrics"),
                    headers=self._parse_headers(settings.otel_headers)
                )
                metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
                
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
            
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self.enabled = True
            
            # Auto-instrument FastAPI and HTTPX
            FastAPIInstrumentor.instrument()
            HTTPXClientInstrumentor.instrument()
            
            logging.info("OpenTelemetry integration initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize OpenTelemetry: {e}")
    
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse headers string into dictionary."""
        if not headers_str:
            return {}
        
        headers = {}
        for header in headers_str.split(","):
            if "=" in header:
                key, value = header.strip().split("=", 1)
                headers[key] = value
        return headers
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create OpenTelemetry span."""
        if not self.enabled or not self.tracer:
            return None
        
        span = self.tracer.start_span(name)
        if attributes:
            span.set_attributes(attributes)
        return span


class LogfireIntegration:
    """Enhanced Logfire integration."""
    
    def __init__(self):
        self.enabled = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Logfire with proper configuration."""
        try:
            if not settings.logfire_token:
                logging.info("Logfire token not configured. Skipping Logfire integration.")
                return
            
            # Configure Logfire with minimal configuration first
            logfire.configure(
                token=settings.logfire_token,
                service_name=settings.logfire_project_name,
                service_version=settings.api_version,
                environment=settings.logfire_environment,
                send_to_logfire=True,
                console=False,  # Avoid console conflicts
            )
            
            self.enabled = True
            logging.info(f"Logfire integration initialized successfully for project: {settings.logfire_project_name}")
            
            # Log successful initialization to Logfire
            logfire.info(
                "Logfire integration started",
                service_name=settings.logfire_project_name,
                environment=settings.logfire_environment
            )
            
        except Exception as e:
            logging.warning(f"Failed to configure Logfire: {e}")
            # Continue without Logfire
            self.enabled = False


class EnhancedObservability:
    """Enhanced observability manager."""
    
    def __init__(self):
        self.logfire = LogfireIntegration()
        self.arize = ArizeIntegration()
        self.otel = OpenTelemetryIntegration()
        self.llm_traces: List[LLMTrace] = []
        
        # Configure structured logging
        self._configure_structured_logging()
    
    def _configure_structured_logging(self):
        """Configure structured logging with structlog."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add JSON formatting for production
        if settings.logfire_environment != "development":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """Get structured logger."""
        return structlog.get_logger(name)
    
    def trace_llm_interaction(self,
                            agent_name: str,
                            model_name: str,
                            prompt: str,
                            response: str,
                            latency_ms: float,
                            classification_label: Optional[str] = None,
                            confidence_score: Optional[float] = None,
                            token_count_input: Optional[int] = None,
                            token_count_output: Optional[int] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Trace LLM interaction across all platforms."""
        
        trace_id = str(uuid.uuid4())
        
        llm_trace = LLMTrace(
            trace_id=trace_id,
            model_name=model_name,
            prompt=prompt,
            response=response,
            latency_ms=latency_ms,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            classification_label=classification_label,
            confidence_score=confidence_score,
            agent_name=agent_name,
            metadata=metadata or {}
        )
        
        # Store locally
        self.llm_traces.append(llm_trace)
        
        # Log to structured logger
        logger = self.get_logger(f"llm.{agent_name}")
        logger.info("LLM interaction",
                   trace_id=trace_id,
                   model_name=model_name,
                   agent_name=agent_name,
                   classification=classification_label,
                   confidence=confidence_score,
                   latency_ms=latency_ms,
                   prompt_length=len(prompt),
                   response_length=len(response))
        
        # Log to Logfire
        if self.logfire.enabled:
            try:
                logfire.info(
                    "LLM interaction completed",
                    trace_id=trace_id,
                    agent_type=agent_name,
                    model_type=model_name,
                    prompt_length=len(prompt),
                    response_length=len(response),
                    latency_ms=latency_ms,
                    classification=classification_label,
                    confidence_level=confidence_score
                )
            except Exception as e:
                logger.warning("Failed to log to Logfire", error=str(e))
        
        # Log to Arize AX
        self.arize.log_llm_trace(llm_trace)
        
        # Create OpenTelemetry span
        if self.otel.enabled and self.otel.tracer:
            try:
                with self.otel.tracer.start_as_current_span("llm_interaction") as span:
                    span.set_attributes({
                        "llm.agent_name": agent_name,
                        "llm.model_name": model_name,
                        "llm.prompt_length": len(prompt),
                        "llm.response_length": len(response),
                        "llm.latency_ms": latency_ms,
                        "llm.classification": classification_label or "",
                        "llm.confidence": confidence_score or 0.0,
                    })
            except Exception as e:
                logger.warning("Failed to create OTel span", error=str(e))
        
        return trace_id
    
    def trace_agent_operation(self,
                            agent_name: str,
                            operation_name: str,
                            trace_id: str,
                            status: str = "started",
                            duration: Optional[float] = None,
                            error: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None):
        """Trace agent operations."""
        
        logger = self.get_logger(f"agent.{agent_name}")
        
        log_data = {
            "agent_name": agent_name,
            "operation_name": operation_name,
            "trace_id": trace_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if duration is not None:
            log_data["duration"] = duration
        if error:
            log_data["error"] = error
        if metadata:
            log_data.update(metadata)
        
        # Log based on status
        if status == "failed" or error:
            logger.error("Agent operation failed", **log_data)
        elif status == "completed":
            logger.info("Agent operation completed", **log_data)
        else:
            logger.info("Agent operation", **log_data)
        
        # Log to Logfire
        if self.logfire.enabled:
            try:
                if status == "failed" or error:
                    logfire.error(
                        "Agent operation failed",
                        agent_type=agent_name,
                        operation_type=operation_name,
                        trace_id=trace_id,
                        error_message=error,
                        duration_ms=duration
                    )
                else:
                    logfire.info(
                        "Agent operation completed",
                        agent_type=agent_name,
                        operation_type=operation_name,
                        trace_id=trace_id,
                        operation_status=status,
                        duration_ms=duration
                    )
            except Exception as e:
                logger.warning("Failed to log to Logfire", error=str(e))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_llm_traces": len(self.llm_traces),
            "integrations": {
                "logfire_enabled": self.logfire.enabled,
                "arize_enabled": self.arize.enabled,
                "arize_otel_enabled": self.arize.otel_enabled,
                "otel_enabled": self.otel.enabled,
            },
            "recent_traces": [trace.model_dump() for trace in self.llm_traces[-5:]],
            "service_info": {
                "name": settings.otel_service_name,
                "version": settings.api_version,
                "environment": settings.logfire_environment,
            }
        }


# Global enhanced observability instance
enhanced_observability = EnhancedObservability()


# Convenience functions
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get structured logger."""
    return enhanced_observability.get_logger(name)


def trace_llm_interaction(**kwargs) -> str:
    """Trace LLM interaction."""
    return enhanced_observability.trace_llm_interaction(**kwargs)


def trace_agent_operation(**kwargs):
    """Trace agent operation."""
    return enhanced_observability.trace_agent_operation(**kwargs)


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return enhanced_observability.get_metrics_summary()


@asynccontextmanager
async def trace_context(operation_name: str,
                       trace_id: str,
                       agent_name: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
    """Async context manager for operation tracing."""
    start_time = time.time()
    
    # Start tracing
    if agent_name:
        trace_agent_operation(
            agent_name=agent_name,
            operation_name=operation_name,
            trace_id=trace_id,
            status="started",
            metadata=metadata,
        )
    
    try:
        yield {
            "trace_id": trace_id,
            "operation_name": operation_name,
            "start_time": start_time
        }
        
        duration = time.time() - start_time
        
        # Success tracing
        if agent_name:
            trace_agent_operation(
                agent_name=agent_name,
                operation_name=operation_name,
                trace_id=trace_id,
                status="completed",
                duration=duration,
                metadata=metadata,
            )
    
    except Exception as e:
        duration = time.time() - start_time
        
        # Error tracing
        if agent_name:
            trace_agent_operation(
                agent_name=agent_name,
                operation_name=operation_name,
                trace_id=trace_id,
                status="failed",
                duration=duration,
                error=str(e),
                metadata=metadata,
            )
        
        raise


# Export commonly used functions
__all__ = [
    "enhanced_observability",
    "get_logger",
    "trace_llm_interaction",
    "trace_agent_operation",
    "get_metrics_summary",
    "trace_context",
    "LLMTrace",
    "ArizeIntegration",
    "LogfireIntegration",
    "OpenTelemetryIntegration",
]