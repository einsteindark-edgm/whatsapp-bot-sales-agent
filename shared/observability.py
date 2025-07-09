"""
Observability and Logging Configuration.

This module provides structured logging and observability features using
Pydantic Logfire for the multi-agent WhatsApp sales assistant system.
"""

import logging
import logfire
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
from functools import wraps
import time
import asyncio
import json
import structlog
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from config.settings import settings


class LogLevel:
    """Log level constants."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TraceContext(BaseModel):
    """Trace context for distributed tracing."""

    trace_id: str = Field(..., description="Unique trace identifier")
    span_id: str = Field(..., description="Span identifier")
    parent_span_id: Optional[str] = Field(None, description="Parent span identifier")
    operation_name: str = Field(..., description="Operation name")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict, description="Trace tags")
    logs: list = Field(default_factory=list, description="Trace logs")


class MetricsCollector:
    """Metrics collection for performance monitoring."""

    def __init__(self):
        self.metrics = {}
        self.counters = {}
        self.timers = {}

    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = f"{metric_name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.counters[key] = self.counters.get(key, 0) + value

    def timing(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        key = f"{metric_name}:{json.dumps(tags or {}, sort_keys=True)}"
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(value)

    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        key = f"{metric_name}:{json.dumps(tags or {}, sort_keys=True)}"
        self.metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {"counters": self.counters, "timers": self.timers, "gauges": self.metrics}


# Global metrics collector instance
metrics = MetricsCollector()


class ObservabilityConfig:
    """Configuration for observability features."""

    def __init__(self):
        self.enabled = settings.trace_enabled
        self.log_level = settings.log_level
        self.service_name = "whatsapp-sales-assistant"
        self.environment = "development" if settings.is_docker_environment else "local"

        # Configure structured logging
        self._configure_structured_logging()

        # Configure Logfire if enabled
        if self.enabled:
            self._configure_logfire()

    def _configure_structured_logging(self):
        """Configure structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def _configure_logfire(self):
        """Configure Pydantic Logfire for observability."""
        try:
            logfire.configure(
                service_name=self.service_name,
                environment=self.environment,
                console=True,
                pydantic_plugin=logfire.PydanticPlugin(record="all"),
            )
            logfire.info("Logfire configured successfully")
        except Exception as e:
            # Fall back to standard logging if Logfire fails
            logging.warning(f"Failed to configure Logfire: {e}")


# Global observability configuration
observability_config = ObservabilityConfig()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def trace_a2a_message(
    message_type: str,
    sender_agent: str,
    receiver_agent: str,
    trace_id: str,
    direction: str = "outbound",
):
    """
    Trace A2A message flow.

    Args:
        message_type: Type of A2A message
        sender_agent: Agent sending the message
        receiver_agent: Agent receiving the message
        trace_id: Trace ID for correlation
        direction: Message direction (inbound/outbound)
    """
    logger = get_logger("a2a_protocol")

    # Log structured message
    logger.info(
        "A2A message",
        message_type=message_type,
        sender_agent=sender_agent,
        receiver_agent=receiver_agent,
        trace_id=trace_id,
        direction=direction,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Record metrics
    metrics.increment(
        "a2a_messages_total",
        tags={
            "message_type": message_type,
            "sender_agent": sender_agent,
            "receiver_agent": receiver_agent,
            "direction": direction,
        },
    )

    # Logfire tracing if enabled
    if observability_config.enabled:
        try:
            logfire.info(
                "A2A Message Flow",
                message_type=message_type,
                sender_agent=sender_agent,
                receiver_agent=receiver_agent,
                trace_id=trace_id,
                direction=direction,
            )
        except Exception as e:
            logger.warning("Failed to trace with Logfire", error=str(e))


def trace_agent_operation(
    agent_name: str,
    operation_name: str,
    trace_id: str,
    status: str = "started",
    duration: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Trace agent operations.

    Args:
        agent_name: Name of the agent
        operation_name: Operation being performed
        trace_id: Trace ID for correlation
        status: Operation status (started/completed/failed)
        duration: Operation duration in seconds
        error: Error message if operation failed
        metadata: Additional metadata
    """
    logger = get_logger(f"agent.{agent_name}")

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

    # Record metrics
    metrics.increment(
        "agent_operations_total",
        tags={"agent_name": agent_name, "operation_name": operation_name, "status": status},
    )

    if duration is not None:
        metrics.timing(
            "agent_operation_duration",
            duration,
            tags={"agent_name": agent_name, "operation_name": operation_name},
        )

    # Logfire tracing if enabled
    if observability_config.enabled:
        try:
            if status == "failed" or error:
                logfire.error(
                    "Agent Operation Failed",
                    agent_name=agent_name,
                    operation_name=operation_name,
                    trace_id=trace_id,
                    error=error,
                    **log_data,
                )
            else:
                logfire.info(
                    "Agent Operation",
                    agent_name=agent_name,
                    operation_name=operation_name,
                    trace_id=trace_id,
                    status=status,
                    duration=duration,
                    **log_data,
                )
        except Exception as e:
            logger.warning("Failed to trace with Logfire", error=str(e))


def trace_performance(
    operation_name: str,
    trace_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Trace performance metrics.

    Args:
        operation_name: Name of the operation
        trace_id: Trace ID for correlation
        start_time: Operation start time
        end_time: Operation end time
        duration: Pre-calculated duration
        metadata: Additional metadata
    """
    logger = get_logger("performance")

    # Calculate duration if not provided
    if duration is None and start_time and end_time:
        duration = (end_time - start_time).total_seconds()

    log_data = {
        "operation_name": operation_name,
        "trace_id": trace_id,
        "duration": duration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if metadata:
        log_data.update(metadata)

    logger.info("Performance metric", **log_data)

    # Record timing metric
    if duration is not None:
        metrics.timing("operation_duration", duration, tags={"operation": operation_name})

    # Logfire tracing if enabled
    if observability_config.enabled:
        try:
            logfire.info(
                "Performance Metric",
                operation_name=operation_name,
                trace_id=trace_id,
                duration=duration,
                **log_data,
            )
        except Exception as e:
            logger.warning("Failed to trace with Logfire", error=str(e))


def trace_decorator(operation_name: str, agent_name: Optional[str] = None):
    """
    Decorator for automatic function tracing.

    Args:
        operation_name: Name of the operation
        agent_name: Optional agent name for context

    Returns:
        Decorator function
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = kwargs.get("trace_id", f"trace_{int(time.time())}")
            start_time = time.time()

            # Start tracing
            if agent_name:
                trace_agent_operation(
                    agent_name=agent_name,
                    operation_name=operation_name,
                    trace_id=trace_id,
                    status="started",
                )

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Success tracing
                if agent_name:
                    trace_agent_operation(
                        agent_name=agent_name,
                        operation_name=operation_name,
                        trace_id=trace_id,
                        status="completed",
                        duration=duration,
                    )

                trace_performance(
                    operation_name=operation_name, trace_id=trace_id, duration=duration
                )

                return result

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
                    )

                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = kwargs.get("trace_id", f"trace_{int(time.time())}")
            start_time = time.time()

            # Start tracing
            if agent_name:
                trace_agent_operation(
                    agent_name=agent_name,
                    operation_name=operation_name,
                    trace_id=trace_id,
                    status="started",
                )

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Success tracing
                if agent_name:
                    trace_agent_operation(
                        agent_name=agent_name,
                        operation_name=operation_name,
                        trace_id=trace_id,
                        status="completed",
                        duration=duration,
                    )

                trace_performance(
                    operation_name=operation_name, trace_id=trace_id, duration=duration
                )

                return result

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
                    )

                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@asynccontextmanager
async def trace_context(
    operation_name: str,
    trace_id: str,
    agent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Async context manager for operation tracing.

    Args:
        operation_name: Name of the operation
        trace_id: Trace ID for correlation
        agent_name: Optional agent name
        metadata: Additional metadata

    Yields:
        Trace context
    """
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
        # Convert all metadata values to strings for TraceContext compatibility
        tags = {}
        if metadata:
            tags = {key: str(value) for key, value in metadata.items()}
        
        yield TraceContext(
            trace_id=trace_id,
            span_id=f"span_{int(time.time())}",
            operation_name=operation_name,
            tags=tags,
        )

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

        trace_performance(
            operation_name=operation_name, trace_id=trace_id, duration=duration, metadata=metadata
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


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of all collected metrics.

    Returns:
        Dictionary containing metrics summary
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics.get_metrics(),
        "service": observability_config.service_name,
        "environment": observability_config.environment,
    }


def reset_metrics():
    """Reset all collected metrics."""
    global metrics
    metrics = MetricsCollector()


# Export commonly used functions
__all__ = [
    "get_logger",
    "trace_a2a_message",
    "trace_agent_operation",
    "trace_performance",
    "trace_decorator",
    "trace_context",
    "get_metrics_summary",
    "reset_metrics",
    "metrics",
    "observability_config",
]
