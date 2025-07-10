"""
Middleware for trace ID propagation across services.

This module provides FastAPI middleware that ensures consistent trace ID propagation
through all service calls, supporting distributed tracing across the entire system.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable, Optional, Dict, Any
import uuid
import json
from shared.observability import get_logger
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode
import asyncio

logger = get_logger(__name__)


def extract_session_id_from_whatsapp(body: bytes) -> Optional[str]:
    """
    Extract session ID from WhatsApp webhook payload.
    
    Args:
        body: Raw request body
        
    Returns:
        Optional[str]: Session ID if found, None otherwise
    """
    try:
        data = json.loads(body)
        
        # WhatsApp webhook structure
        if "entry" in data and len(data["entry"]) > 0:
            entry = data["entry"][0]
            if "changes" in entry and len(entry["changes"]) > 0:
                change = entry["changes"][0]
                if "value" in change and "messages" in change["value"]:
                    messages = change["value"]["messages"]
                    if len(messages) > 0:
                        message = messages[0]
                        sender = message.get("from", "")
                        # Use consistent session ID format
                        if sender:
                            return f"whatsapp_{sender}"
        
    except Exception as e:
        logger.error(f"Failed to extract session ID from WhatsApp payload: {e}")
    
    return None


def generate_trace_id() -> str:
    """Generate a 32-character hex trace ID compatible with OpenTelemetry."""
    # OpenTelemetry expects 32 hex characters (16 bytes)
    return uuid.uuid4().hex


class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to propagate chat.session_id as global trace ID.
    
    This middleware:
    1. Extracts or generates trace IDs for all requests
    2. Propagates trace IDs through OpenTelemetry context
    3. Adds trace IDs to response headers
    4. Handles WhatsApp-specific session ID extraction
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.tracer = trace.get_tracer(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with trace ID propagation.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response with trace ID header
        """
        # Extract or generate trace ID
        trace_id = request.headers.get("X-Trace-ID")
        
        # Special handling for WhatsApp webhooks
        if not trace_id and request.url.path == "/webhook/whatsapp" and request.method == "POST":
            # Read body without consuming it
            body = await request.body()
            
            # Extract session ID from WhatsApp payload
            session_id = extract_session_id_from_whatsapp(body)
            if session_id:
                # Convert session ID to valid trace ID format
                # Use consistent hashing to generate same trace ID for same session
                import hashlib
                hash_object = hashlib.md5(session_id.encode())
                trace_id = hash_object.hexdigest()
            
            # Reconstruct request with body
            async def receive():
                return {"type": "http.request", "body": body}
            
            request = Request(request.scope, receive)
        
        # Generate trace ID if not found
        if not trace_id:
            trace_id = generate_trace_id()
        
        # Create span with trace ID
        with self.tracer.start_as_current_span(
            "http_request",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": request.url.path,
                "trace.id": trace_id,
            }
        ) as span:
            # Set trace ID in context
            ctx = trace.set_span_in_context(span)
            token = context.attach(ctx)
            
            # Add trace ID to request state for easy access
            request.state.trace_id = trace_id
            
            try:
                # Log request with trace ID
                logger.info(
                    "Processing request",
                    extra={
                        "trace_id": trace_id,
                        "method": request.method,
                        "path": request.url.path,
                        "client": request.client.host if request.client else None,
                    }
                )
                
                # Process request
                response = await call_next(request)
                
                # Add trace ID to response headers
                response.headers["X-Trace-ID"] = trace_id
                
                # Update span with response status
                span.set_attribute("http.status_code", response.status_code)
                
                if response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Log response
                logger.info(
                    "Request completed",
                    extra={
                        "trace_id": trace_id,
                        "status_code": response.status_code,
                        "method": request.method,
                        "path": request.url.path,
                    }
                )
                
                return response
                
            except Exception as e:
                # Log error with trace ID
                logger.error(
                    f"Request failed: {e}",
                    extra={
                        "trace_id": trace_id,
                        "method": request.method,
                        "path": request.url.path,
                        "error": str(e),
                    },
                    exc_info=True
                )
                
                # Update span with error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                raise
                
            finally:
                # Detach context
                context.detach(token)


class AsyncTraceContextManager:
    """
    Async context manager for maintaining trace context across async operations.
    
    This ensures trace IDs are properly propagated through async boundaries.
    """
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.tracer = trace.get_tracer(__name__)
        self.span = None
        self.token = None
    
    async def __aenter__(self):
        """Enter trace context."""
        self.span = self.tracer.start_span("async_operation")
        self.span.set_attribute("trace.id", self.trace_id)
        
        ctx = trace.set_span_in_context(self.span)
        self.token = context.attach(ctx)
        
        return self.trace_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context."""
        if exc_val:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(Status(StatusCode.OK))
        
        self.span.end()
        context.detach(self.token)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID from OpenTelemetry context.
    
    Returns:
        Optional[str]: Current trace ID if available
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span_context = span.get_span_context()
        if span_context.is_valid:
            # Return as 32-char hex string
            return format(span_context.trace_id, '032x')
    return None


def inject_trace_id_header(headers: Dict[str, str], trace_id: Optional[str] = None) -> Dict[str, str]:
    """
    Inject trace ID into HTTP headers.
    
    Args:
        headers: Existing headers dict
        trace_id: Trace ID to inject (uses current if not provided)
        
    Returns:
        Updated headers dict
    """
    if not trace_id:
        trace_id = get_current_trace_id()
    
    if trace_id:
        headers["X-Trace-ID"] = trace_id
    
    return headers