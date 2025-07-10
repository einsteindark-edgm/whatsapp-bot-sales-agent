"""
WhatsApp webhook router for receiving and processing WhatsApp messages.

This module handles webhook verification and incoming message processing
from the WhatsApp Business API, integrating with the orchestrator workflow.
"""

from fastapi import APIRouter, Request, Response, Depends, HTTPException, Query
from typing import Optional, Dict, Any
import hmac
import hashlib
import os
import json
from shared.observability import get_logger
from shared.observability_enhanced import trace_context
from shared.observability_cost import CostCalculator, session_cost_aggregator
from agents.orchestrator.domain.whatsapp_models import WhatsAppWebhookPayload
from agents.orchestrator.domain.models import WorkflowRequest
from agents.orchestrator.adapters.inbound.fastapi_router import get_trace_id
from agents.orchestrator.adapters.outbound.whatsapp_api_client import WhatsAppAPIClient
from agents.orchestrator.agent import process_workflow_request_async
from config.settings import settings


router = APIRouter(prefix="/webhook", tags=["whatsapp"])
logger = get_logger(__name__)


def verify_webhook_signature(request: Request, payload: bytes) -> bool:
    """
    Verify webhook signature from Meta.

    Args:
        request: FastAPI request object
        payload: Raw request body

    Returns:
        bool: True if signature is valid, False otherwise
    """
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature or not signature.startswith("sha256="):
        return False

    app_secret = os.getenv("WHATSAPP_APP_SECRET", "")
    if not app_secret:
        logger.warning("WHATSAPP_APP_SECRET not configured, skipping signature verification")
        return True  # Allow in development, but log warning

    # Calculate expected signature
    expected_signature = hmac.new(
        app_secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    # Compare signatures (constant time comparison)
    provided_signature = signature.replace("sha256=", "")
    return hmac.compare_digest(provided_signature, expected_signature)


@router.get("/whatsapp")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
) -> Response:
    """
    Webhook verification endpoint for Meta.

    This endpoint is called by Meta when setting up the webhook.
    It must return the challenge value if the verify token matches.

    Args:
        hub_mode: Should be "subscribe"
        hub_verify_token: Token to verify against our configured token
        hub_challenge: Challenge string to return

    Returns:
        Response with challenge string or error
    """
    verify_token = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "")

    if not verify_token:
        logger.error("WHATSAPP_WEBHOOK_VERIFY_TOKEN not configured")
        raise HTTPException(status_code=500, detail="Webhook verify token not configured")

    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        logger.info("WhatsApp webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.error(
        "WhatsApp webhook verification failed",
        extra={
            "hub_mode": hub_mode,
            "token_match": hub_verify_token == verify_token,
        },
    )
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp")
async def process_webhook(
    request: Request,
    trace_id: str = Depends(get_trace_id),
) -> Dict[str, Any]:
    """
    Process incoming WhatsApp messages.

    This endpoint receives webhook events from WhatsApp, including:
    - Incoming messages from users
    - Status updates for sent messages
    - Other WhatsApp events

    Args:
        request: FastAPI request containing webhook payload
        trace_id: Trace ID for request tracking

    Returns:
        Dict with processing status
    """
    # Read raw body for signature verification
    body = await request.body()

    # Verify webhook signature
    if not verify_webhook_signature(request, body):
        logger.error("Invalid webhook signature", extra={"trace_id": trace_id})
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse webhook payload
    try:
        data = json.loads(body)
    except Exception as e:
        logger.error(
            f"Failed to parse webhook body: {e}",
            extra={"trace_id": trace_id, "error": str(e)},
        )
        return {"status": "error", "message": "Invalid JSON"}
    
    # Use the shared function to process the payload
    return await _process_webhook_payload(data, trace_id)


@router.post("/whatsapp/test")
async def process_webhook_test(
    request: Request,
    trace_id: str = Depends(get_trace_id),
) -> Dict[str, Any]:
    """
    Test endpoint for WhatsApp webhook without signature verification.
    FOR DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION.
    """
    logger.warning("Using test webhook endpoint without signature verification")
    
    # Parse body
    body = await request.body()
    data = json.loads(body)
    
    # Process as normal webhook but skip signature verification
    return await _process_webhook_payload(data, trace_id)


async def _process_webhook_payload(data: dict, trace_id: str) -> Dict[str, Any]:
    """
    Process webhook payload (extracted for reuse).
    """
    try:
        webhook_payload = WhatsAppWebhookPayload(**data)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}", extra={"trace_id": trace_id})
        return {"status": "error", "message": "Invalid payload"}

    async with trace_context(
        operation_name="whatsapp_message_processing",
        trace_id=trace_id,
        agent_name="whatsapp_adapter",
        metadata={"webhook_type": "message"}
    ) as trace_info:
        message_data = webhook_payload.extract_message()
        if not message_data:
            status_update = webhook_payload.extract_status_update()
            if status_update:
                logger.info(
                    "Received WhatsApp status update",
                    extra={
                        "trace_id": trace_id,
                        "status": status_update.get("status"),
                        "message_id": status_update.get("id"),
                    },
                )
            return {"status": "ok", "message": "No message to process"}

        text, sender, message_id = message_data

        logger.info(
            "Processing WhatsApp message",
            extra={
                "trace_id": trace_id,
                "sender": sender,
                "message_id": message_id,
                "message_length": len(text),
            },
        )

        try:
            whatsapp_client = WhatsAppAPIClient()
            await whatsapp_client.send_typing_indicator(sender, trace_id=trace_id)
            await whatsapp_client.mark_message_as_read(message_id, trace_id=trace_id)

            workflow_request = WorkflowRequest(
                user_message=text,
                user_id=sender,
                session_id=f"whatsapp_{sender}",
                conversation_id=f"whatsapp_{sender}_{message_id}",
                metadata={
                    "source": "whatsapp",
                    "message_id": message_id,
                    "phone_number": sender,
                    "channel": "whatsapp",
                },
            )

            workflow_response = await process_workflow_request_async(
                request=workflow_request,
                trace_id=trace_id
            )

            response_text = workflow_response.response
            classification = workflow_response.classification

            from shared.observability_enhanced import enhanced_observability
            
            if classification:
                # Extract token usage if available
                token_usage = workflow_response.token_usage or {}
                
                enhanced_observability.trace_llm_interaction(
                    agent_name="whatsapp_adapter",
                    model_name=settings.model_name,
                    prompt=text,
                    response=response_text,
                    latency_ms=workflow_response.processing_time * 1000 if workflow_response.processing_time else 0,
                    classification_label=classification.get("label") if classification else None,
                    confidence_score=classification.get("confidence") if classification else None,
                    token_count_input=token_usage.get("input"),
                    token_count_output=token_usage.get("output"),
                    metadata={
                        "channel": "whatsapp",
                        "sender": sender,
                        "message_id": message_id,
                        "trace_id": trace_id,
                        "user_id": sender,
                        "session_id": f"whatsapp_{sender}",
                        "conversation_id": f"whatsapp_{sender}_{message_id}",
                    }
                )

            await whatsapp_client.send_text_message(
                to=sender,
                message=response_text,
                trace_id=trace_id,
            )
            
            # Track WhatsApp message costs
            if classification:
                label = classification.get("label", "").lower()
                if label == "pqr":
                    message_type = "utility"
                else:
                    message_type = "service"
            else:
                message_type = "service"
            
            country = "US"
            if sender.startswith("52"):
                country = "MX"
            elif sender.startswith("55"):
                country = "BR"
            elif sender.startswith("91"):
                country = "IN"
            elif sender.startswith("44"):
                country = "GB"
            
            whatsapp_cost = CostCalculator.calculate_whatsapp_cost(
                message_type=message_type,
                country=country
            )
            
            CostCalculator.track_cost(
                cost_usd=whatsapp_cost,
                cost_type="whatsapp",
                metadata={
                    "message_type": message_type,
                    "country": country,
                    "channel": "whatsapp",
                    "sender": sender,
                    "message_id": message_id,
                    "trace_id": trace_id,
                    "classification": classification.get("label") if classification else None,
                    "session_id": f"whatsapp_{sender}",
                }
            )
            
            session_id = f"whatsapp_{sender}"
            session_cost_aggregator.add_cost(session_id, whatsapp_cost, "whatsapp")
            
            logger.info(
                "WhatsApp message cost tracked",
                cost_usd=whatsapp_cost,
                cost_formatted=CostCalculator.format_cost(whatsapp_cost),
                message_type=message_type,
                country=country,
                trace_id=trace_id,
                session_id=session_id,
            )

            return {
                "status": "ok",
                "processed": True,
                "message_id": message_id,
                "classification": classification.get("label") if classification else None,
                "cost": {
                    "whatsapp": CostCalculator.format_cost(whatsapp_cost),
                    "message_type": message_type,
                }
            }

        except Exception as e:
            logger.error(
                f"Failed to process WhatsApp message: {e}",
                extra={
                    "trace_id": trace_id,
                    "error": str(e),
                    "sender": sender,
                    "message_id": message_id,
                },
            )

            try:
                whatsapp_client = WhatsAppAPIClient()
                await whatsapp_client.send_text_message(
                    to=sender,
                    message="Lo siento, ocurrió un error al procesar tu mensaje. Por favor, intenta nuevamente más tarde.",
                    trace_id=trace_id,
                )
                
                country = "US"
                if sender.startswith("52"):
                    country = "MX"
                elif sender.startswith("55"):
                    country = "BR"
                
                error_message_cost = CostCalculator.calculate_whatsapp_cost(
                    message_type="utility",
                    country=country
                )
                
                CostCalculator.track_cost(
                    cost_usd=error_message_cost,
                    cost_type="whatsapp",
                    metadata={
                        "message_type": "utility",
                        "country": country,
                        "channel": "whatsapp",
                        "sender": sender,
                        "trace_id": trace_id,
                        "error_response": True,
                        "session_id": f"whatsapp_{sender}",
                    }
                )
                
                session_cost_aggregator.add_cost(f"whatsapp_{sender}", error_message_cost, "whatsapp")
                
            except Exception as send_error:
                logger.error(
                    f"Failed to send error message: {send_error}",
                    extra={"trace_id": trace_id},
                )

            return {"status": "error", "message": str(e)}


@router.get("/whatsapp/health")
async def whatsapp_health() -> Dict[str, Any]:
    """
    Health check endpoint for WhatsApp integration.

    Returns:
        Dict with health status and configuration info
    """
    whatsapp_client = WhatsAppAPIClient()
    health_status = await whatsapp_client.health_check()

    return {
        "status": "healthy" if health_status["status"] == "healthy" else "degraded",
        "whatsapp_api": health_status,
        "webhook_configured": bool(os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN")),
        "phone_number_configured": bool(os.getenv("WHATSAPP_PHONE_NUMBER_ID")),
    }