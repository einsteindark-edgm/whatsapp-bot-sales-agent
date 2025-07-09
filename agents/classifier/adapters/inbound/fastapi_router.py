"""
FastAPI Router for Classifier Agent.

This module provides the FastAPI router for the classifier service,
handling HTTP endpoints for message classification and health checks.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from datetime import datetime
import time

from agents.classifier.agent import (
    classifier_agent,
    classify_message_async,
    get_classifier_metrics,
    reset_classifier_metrics,
    classifier_health_check,
)
from agents.classifier.domain.models import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationError,
)
from shared.a2a_protocol import (
    ClassificationRequest as A2AClassificationRequest,
    parse_a2a_message,
    create_error_message,
    MessageType,
)
from shared.observability import get_logger, trace_a2a_message, trace_context
from shared.observability_enhanced import trace_agent_operation
from shared.utils import generate_trace_id
from config.settings import settings


# Configure logger
logger = get_logger(__name__)

# Create FastAPI router
router = APIRouter(
    prefix="/api/v1",
    tags=["classifier"],
    responses={404: {"description": "Not found"}, 500: {"description": "Internal server error"}},
)


def get_api_key(request: Request) -> str:
    """
    Extract API key from request headers or use default.

    Args:
        request: FastAPI request object

    Returns:
        API key string

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Try to get API key from headers
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")

    # Remove Bearer prefix if present
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Fall back to settings if no header provided
    if not api_key:
        api_key = settings.gemini_api_key

    # Validate API key
    if not api_key or len(api_key.strip()) < 10:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return api_key.strip()


def get_trace_id(request: Request) -> str:
    """
    Extract trace ID from request headers or generate new one.

    Args:
        request: FastAPI request object

    Returns:
        Trace ID string
    """
    trace_id = request.headers.get("X-Trace-Id")
    if not trace_id:
        trace_id = generate_trace_id()

    return trace_id


@router.post(
    "/classify",
    response_model=Dict[str, Any],
    summary="Classify message using A2A protocol",
    description="Classify a message received via A2A protocol",
)
async def classify_message(
    request: Request,
    message_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify a message using the A2A protocol.

    Args:
        request: FastAPI request object
        message_data: A2A message data
        api_key: API key for authentication
        trace_id: Trace ID for observability

    Returns:
        A2A classification response

    Raises:
        HTTPException: If classification fails
    """
    start_time = time.time()
    
    # Extract API key and trace_id with A2A-compatible fallbacks
    try:
        api_key = get_api_key(request)
    except HTTPException:
        # For A2A requests, fall back to settings
        api_key = settings.gemini_api_key
    
    trace_id = get_trace_id(request)

    try:
        # Parse A2A message
        try:
            a2a_message = parse_a2a_message(message_data)
        except Exception as e:
            logger.error("Failed to parse A2A message", error=str(e), trace_id=trace_id)
            error_message = create_error_message(
                sender_agent="classifier",
                receiver_agent=message_data.get("sender_agent", "unknown"),
                error_code="PARSE_ERROR",
                error_message=f"Failed to parse A2A message: {str(e)}",
                trace_id=trace_id,
                correlation_id=message_data.get("message_id"),
            )
            return JSONResponse(status_code=400, content=error_message.model_dump(mode='json'))

        # Validate message type
        if not isinstance(a2a_message, A2AClassificationRequest):
            logger.error("Invalid A2A message type", message_type=str(type(a2a_message)), trace_id=trace_id)
            error_message = create_error_message(
                sender_agent="classifier",
                receiver_agent=a2a_message.sender_agent if hasattr(a2a_message, 'sender_agent') else "unknown",
                error_code="INVALID_MESSAGE_TYPE",
                error_message=f"Expected ClassificationRequest, got {type(a2a_message)}",
                trace_id=trace_id,
                correlation_id=message_data.get("message_id"),
            )
            return JSONResponse(status_code=400, content=error_message.model_dump(mode='json'))

        # Log A2A message received
        trace_a2a_message(
            message_type=a2a_message.message_type,
            sender_agent=a2a_message.sender_agent,
            receiver_agent=a2a_message.receiver_agent,
            trace_id=a2a_message.trace_id,
            direction="inbound",
        )

        # Use A2A message trace ID if available
        effective_trace_id = a2a_message.trace_id or trace_id

        async with trace_context(
            operation_name="classify_a2a_message",
            trace_id=effective_trace_id,
            agent_name="classifier",
            metadata={
                "sender_agent": a2a_message.sender_agent,
                "text_length": len(a2a_message.text),
                "message_id": a2a_message.message_id,
            },
        ):
            # Perform classification
            classification = await classify_message_async(
                text=a2a_message.text,
                trace_id=effective_trace_id,
                api_key=api_key,
                include_reasoning=False,
                extract_keywords=True,
            )

            # Create A2A response
            response_payload = {
                "text": classification.text,
                "label": classification.label,
                "confidence": classification.confidence,
                "keywords": classification.keywords,
                "processing_time": classification.processing_time,
                "model_used": classification.model_used,
                "timestamp": classification.timestamp.isoformat(),
            }

            # Create A2A response message
            response_message = a2a_message.create_response(
                sender_agent="classifier",
                message_type=MessageType.CLASSIFICATION_RESPONSE,
                payload=response_payload,
            )

            # Log A2A response
            trace_a2a_message(
                message_type=response_message.message_type,
                sender_agent=response_message.sender_agent,
                receiver_agent=response_message.receiver_agent,
                trace_id=response_message.trace_id,
                direction="outbound",
            )

            # Log successful classification
            processing_time = time.time() - start_time
            logger.info(
                "A2A classification completed",
                sender_agent=a2a_message.sender_agent,
                label=classification.label,
                confidence=classification.confidence,
                processing_time=processing_time,
                trace_id=effective_trace_id,
            )

            # Enhanced observability - trace A2A operation success
            trace_agent_operation(
                agent_name="classifier",
                operation_name="a2a_classify",
                trace_id=effective_trace_id,
                status="completed",
                duration=processing_time * 1000,  # Convert to milliseconds
                metadata={
                    "sender_agent": a2a_message.sender_agent,
                    "classification_label": classification.label,
                    "confidence_score": classification.confidence,
                    "text_length": len(a2a_message.text),
                    "message_id": a2a_message.message_id,
                }
            )

            return response_message.model_dump(mode='json')

    except RuntimeError as e:
        # Handle classification-specific errors
        logger.error(
            "Classification error",
            error_message=str(e),
            trace_id=trace_id,
        )

        # Create error response for RuntimeError
        error_message = create_error_message(
            sender_agent="classifier",
            receiver_agent=message_data.get("sender_agent", "unknown"),
            error_code="CLASSIFICATION_FAILED",
            error_message=str(e),
            trace_id=trace_id,
            correlation_id=message_data.get("message_id"),
        )

        return JSONResponse(status_code=500, content=error_message.model_dump(mode='json'))

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Handle unexpected errors
        logger.error("Unexpected error in classification", error=str(e), trace_id=trace_id)

        # Create generic error response
        error_message = create_error_message(
            sender_agent="classifier",
            receiver_agent=message_data.get("sender_agent", "unknown"),
            error_code="INTERNAL_ERROR",
            error_message="Internal classification error",
            trace_id=trace_id,
            correlation_id=message_data.get("message_id"),
        )

        return JSONResponse(status_code=500, content=error_message.model_dump(mode='json'))


@router.post(
    "/classify-direct",
    response_model=ClassificationResponse,
    summary="Direct message classification",
    description="Classify a message directly without A2A protocol",
)
async def classify_message_direct(
    request: Request,
    classification_request: ClassificationRequest,
    api_key: str = Depends(get_api_key),
    trace_id: str = Depends(get_trace_id),
) -> ClassificationResponse:
    """
    Classify a message directly without A2A protocol.

    Args:
        request: FastAPI request object
        classification_request: Direct classification request
        api_key: API key for authentication
        trace_id: Trace ID for observability

    Returns:
        Classification response

    Raises:
        HTTPException: If classification fails
    """
    try:
        async with trace_context(
            operation_name="classify_direct_message",
            trace_id=trace_id,
            agent_name="classifier",
            metadata={
                "text_length": len(classification_request.text),
                "include_reasoning": classification_request.include_reasoning,
            },
        ):
            # Perform classification
            response = await classifier_agent.classify_request(
                request=classification_request, trace_id=trace_id, api_key=api_key
            )

            logger.info(
                "Direct classification completed",
                success=response.success,
                label=response.classification.label,
                confidence=response.classification.confidence,
                trace_id=trace_id,
            )

            return response

    except Exception as e:
        logger.error("Direct classification failed", error=str(e), trace_id=trace_id)

        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")



@router.post(
    "/test",
    summary="Simple test endpoint",
)
async def test_endpoint(data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple test endpoint."""
    return {"status": "ok", "received": data}


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check",
    description="Check the health of the classifier service",
)
async def health_check() -> Dict[str, Any]:
    """
    Perform health check of the classifier service.

    Returns:
        Health check results
    """
    try:
        # Perform health check
        health_result = await classifier_health_check()

        # Add service-specific information
        health_result.update(
            {
                "service": "classifier",
                "version": "1.0.0",
                "uptime": time.time(),  # In a real service, this would be actual uptime
                "endpoints": {
                    "classify": "/api/v1/classify",
                    "classify-direct": "/api/v1/classify-direct",
                    "health": "/api/v1/health",
                    "metrics": "/api/v1/metrics",
                },
            }
        )

        # Determine HTTP status code
        status_code = 200 if health_result.get("status") == "healthy" else 503

        return JSONResponse(status_code=status_code, content=health_result)

    except Exception as e:
        logger.error("Health check failed", error=str(e))

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "service": "classifier",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="Get metrics",
    description="Get classification metrics and statistics",
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get classification metrics and statistics.

    Returns:
        Metrics data
    """
    try:
        # Get classifier metrics
        metrics = get_classifier_metrics()

        # Convert to dictionary
        metrics_dict = metrics.model_dump(mode='json')

        # Add additional metadata
        metrics_dict.update(
            {
                "service": "classifier",
                "collected_at": datetime.utcnow().isoformat(),
                "metrics_version": "1.0.0",
            }
        )

        return metrics_dict

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))

        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post(
    "/metrics/reset",
    response_model=Dict[str, Any],
    summary="Reset metrics",
    description="Reset classification metrics",
)
async def reset_metrics() -> Dict[str, Any]:
    """
    Reset classification metrics.

    Returns:
        Reset confirmation
    """
    try:
        # Reset metrics
        reset_classifier_metrics()

        return {
            "status": "success",
            "message": "Metrics reset successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Failed to reset metrics", error=str(e))

        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")


@router.get(
    "/status",
    response_model=Dict[str, Any],
    summary="Get service status",
    description="Get detailed service status information",
)
async def get_status() -> Dict[str, Any]:
    """
    Get detailed service status information.

    Returns:
        Service status data
    """
    try:
        # Get health and metrics
        health_result = await classifier_health_check()
        metrics = get_classifier_metrics()

        # Combine status information
        status = {
            "service": "classifier",
            "status": health_result.get("status", "unknown"),
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "health": health_result,
            "metrics": metrics.model_dump(mode='json'),
            "configuration": {
                "model_name": settings.model_name,
                "confidence_threshold": 0.7,  # From config
                "max_tokens": 100,
                "temperature": 0.0,
            },
        }

        return status

    except Exception as e:
        logger.error("Failed to get status", error=str(e))

        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Note: Middleware should be added to the FastAPI app, not router
# This will be handled in main.py


# Export router
__all__ = ["router"]
