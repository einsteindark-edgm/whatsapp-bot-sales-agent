"""
FastAPI Router for Orchestrator Agent.

This module provides the FastAPI router for the orchestrator service,
handling HTTP endpoints for workflow orchestration and health checks.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import time

from agents.orchestrator.agent import (
    orchestrator_agent,
    process_workflow_request_async,
    get_orchestrator_metrics,
    reset_orchestrator_metrics,
    orchestrator_health_check,
)
from agents.orchestrator.domain.models import (
    WorkflowRequest,
    WorkflowResponse,
)
from shared.a2a_protocol import (
    OrchestrationRequest,
    parse_a2a_message,
    create_error_message,
    MessageType,
)
from shared.observability import get_logger, trace_a2a_message, trace_context
from shared.utils import generate_trace_id
from config.settings import settings


# Configure logger
logger = get_logger(__name__)

# Create FastAPI router
router = APIRouter(
    prefix="/api/v1",
    tags=["orchestrator"],
    responses={404: {"description": "Not found"}, 500: {"description": "Internal server error"}},
)


def get_api_key(request: Request) -> Optional[str]:
    """
    Extract API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        API key string or None
    """
    # Try to get API key from headers
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")

    # Remove Bearer prefix if present
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]

    return api_key.strip() if api_key else None


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
    "/orchestrate",
    response_model=Dict[str, Any],
    summary="Orchestrate workflow using A2A protocol",
    description="Orchestrate a workflow received via A2A protocol",
)
async def orchestrate_workflow(
    request: Request,
    message_data: Dict[str, Any],
    api_key: Optional[str] = Depends(get_api_key),
    trace_id: str = Depends(get_trace_id),
) -> Dict[str, Any]:
    """
    Orchestrate a workflow using the A2A protocol.

    Args:
        request: FastAPI request object
        message_data: A2A message data
        api_key: Optional API key for authentication
        trace_id: Trace ID for observability

    Returns:
        A2A orchestration response

    Raises:
        HTTPException: If orchestration fails
    """
    start_time = time.time()

    try:
        # Parse A2A message
        a2a_message = parse_a2a_message(message_data)

        # Validate message type
        if not isinstance(a2a_message, OrchestrationRequest):
            raise HTTPException(
                status_code=400, detail=f"Invalid message type: {a2a_message.message_type}"
            )

        # Trace incoming message
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
            operation_name="orchestrate_a2a_workflow",
            trace_id=effective_trace_id,
            agent_name="orchestrator",
            metadata={
                "sender_agent": a2a_message.sender_agent,
                "message_length": len(a2a_message.user_message),
                "message_id": a2a_message.message_id,
            },
        ):
            # Create workflow request
            workflow_request = WorkflowRequest(
                user_message=a2a_message.user_message,
                user_id=a2a_message.sender_agent,  # Use sender as user ID
                session_id=a2a_message.message_id,  # Use message ID as session
                include_classification=True,
            )

            # Process workflow
            workflow_response = await process_workflow_request_async(
                request=workflow_request, trace_id=effective_trace_id
            )

            # Create A2A response payload
            response_payload = {
                "response": workflow_response.response,
                "response_type": workflow_response.response_type,
                "workflow_status": workflow_response.workflow_status,
                "conversation_state": workflow_response.conversation_state,
                "processing_time": workflow_response.processing_time,
                "timestamp": workflow_response.timestamp.isoformat(),
            }

            # Include classification if present
            if workflow_response.classification:
                response_payload["classification"] = workflow_response.classification

            # Create A2A response message
            response_message = a2a_message.create_response(
                sender_agent="orchestrator",
                message_type=MessageType.ORCHESTRATION_RESPONSE,
                payload=response_payload,
            )

            # Trace outgoing message
            trace_a2a_message(
                message_type=response_message.message_type,
                sender_agent=response_message.sender_agent,
                receiver_agent=response_message.receiver_agent,
                trace_id=response_message.trace_id,
                direction="outbound",
            )

            # Log successful orchestration
            processing_time = time.time() - start_time
            logger.info(
                "A2A orchestration completed",
                sender_agent=a2a_message.sender_agent,
                response_type=workflow_response.response_type,
                workflow_status=workflow_response.workflow_status,
                processing_time=processing_time,
                trace_id=effective_trace_id,
            )

            return response_message.model_dump(mode='json')

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Handle unexpected errors
        logger.error("Unexpected error in orchestration", error=str(e), trace_id=trace_id)

        # Create generic error response
        error_message = create_error_message(
            sender_agent="orchestrator",
            receiver_agent=message_data.get("sender_agent", "unknown"),
            error_code="ORCHESTRATION_ERROR",
            error_message="Internal orchestration error",
            trace_id=trace_id,
            correlation_id=message_data.get("message_id"),
        )

        return JSONResponse(status_code=500, content=error_message.model_dump(mode='json'))


@router.post(
    "/orchestrate-direct",
    response_model=WorkflowResponse,
    summary="Direct workflow orchestration",
    description="Orchestrate a workflow directly without A2A protocol",
)
async def orchestrate_workflow_direct(
    request: Request,
    workflow_request: WorkflowRequest,
    api_key: Optional[str] = Depends(get_api_key),
    trace_id: str = Depends(get_trace_id),
) -> WorkflowResponse:
    """
    Orchestrate a workflow directly without A2A protocol.

    Args:
        request: FastAPI request object
        workflow_request: Direct workflow request
        api_key: Optional API key for authentication
        trace_id: Trace ID for observability

    Returns:
        Workflow response

    Raises:
        HTTPException: If orchestration fails
    """
    try:
        async with trace_context(
            operation_name="orchestrate_direct_workflow",
            trace_id=trace_id,
            agent_name="orchestrator",
            metadata={
                "user_id": workflow_request.user_id,
                "session_id": workflow_request.session_id,
                "message_length": len(workflow_request.user_message),
            },
        ):
            # Process workflow
            response = await process_workflow_request_async(
                request=workflow_request, trace_id=trace_id
            )

            logger.info(
                "Direct orchestration completed",
                success=response.success,
                response_type=response.response_type,
                workflow_status=response.workflow_status,
                trace_id=trace_id,
            )

            return response

    except Exception as e:
        logger.error("Direct orchestration failed", error=str(e), trace_id=trace_id)

        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check",
    description="Check the health of the orchestrator service",
)
async def health_check() -> Dict[str, Any]:
    """
    Perform health check of the orchestrator service.

    Returns:
        Health check results
    """
    try:
        # Perform health check
        health_result = await orchestrator_health_check()

        # Add service-specific information
        health_result.update(
            {
                "service": "orchestrator",
                "version": "1.0.0",
                "uptime": time.time(),  # In a real service, this would be actual uptime
                "endpoints": {
                    "orchestrate": "/api/v1/orchestrate",
                    "orchestrate-direct": "/api/v1/orchestrate-direct",
                    "health": "/api/v1/health",
                    "metrics": "/api/v1/metrics",
                    "status": "/api/v1/status",
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
                "service": "orchestrator",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get(
    "/metrics",
    response_model=Dict[str, Any],
    summary="Get metrics",
    description="Get orchestration metrics and statistics",
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get orchestration metrics and statistics.

    Returns:
        Metrics data
    """
    try:
        # Get orchestrator metrics
        metrics = get_orchestrator_metrics()

        # Convert to dictionary
        metrics_dict = metrics.model_dump()

        # Add additional metadata
        metrics_dict.update(
            {
                "service": "orchestrator",
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
    description="Reset orchestration metrics",
)
async def reset_metrics() -> Dict[str, Any]:
    """
    Reset orchestration metrics.

    Returns:
        Reset confirmation
    """
    try:
        # Reset metrics
        reset_orchestrator_metrics()

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
        health_result = await orchestrator_health_check()
        metrics = get_orchestrator_metrics()

        # Combine status information
        status = {
            "service": "orchestrator",
            "status": health_result.get("status", "unknown"),
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "health": health_result,
            "metrics": metrics.model_dump(),
            "configuration": {
                "classifier_url": settings.classifier_url,
                "orchestrator_port": settings.orchestrator_port,
                "api_title": settings.api_title,
                "api_version": settings.api_version,
            },
        }

        return status

    except Exception as e:
        logger.error("Failed to get status", error=str(e))

        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get(
    "/conversations/{user_id}/{session_id}",
    response_model=Dict[str, Any],
    summary="Get conversation",
    description="Get conversation context for a user and session",
)
async def get_conversation(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Get conversation context.

    Args:
        user_id: User identifier
        session_id: Session identifier

    Returns:
        Conversation context
    """
    try:
        # Get conversation
        conversation = orchestrator_agent.get_conversation(user_id, session_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return conversation.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get conversation", error=str(e), user_id=user_id, session_id=session_id
        )

        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@router.delete(
    "/conversations/{user_id}/{session_id}",
    response_model=Dict[str, Any],
    summary="Clear conversation",
    description="Clear conversation context for a user and session",
)
async def clear_conversation(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Clear conversation context.

    Args:
        user_id: User identifier
        session_id: Session identifier

    Returns:
        Clear confirmation
    """
    try:
        # Clear conversation
        orchestrator_agent.clear_conversation(user_id, session_id)

        return {
            "status": "success",
            "message": "Conversation cleared successfully",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(
            "Failed to clear conversation", error=str(e), user_id=user_id, session_id=session_id
        )

        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")


# Note: Middleware should be added to the FastAPI app, not router
# This will be handled in main.py


# Export router
__all__ = ["router"]
