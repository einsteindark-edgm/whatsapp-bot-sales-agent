"""
WhatsApp Business API client for sending messages and managing conversations.

This module provides an async client for interacting with the WhatsApp Business API,
including sending messages, typing indicators, and marking messages as read.
"""

import os
import uuid
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import httpx
from shared.utils import AsyncHTTPClient, RetryConfig, TimeoutConfig
from shared.observability import get_logger
from shared.observability_enhanced import trace_context
from agents.orchestrator.domain.whatsapp_models import (
    WhatsAppSendMessageRequest,
    WhatsAppTextPayload,
    WhatsAppMarkReadRequest,
)
from config.settings import settings


logger = get_logger(__name__)


class WhatsAppConfig(BaseModel):
    """Configuration for WhatsApp API client."""

    access_token: str = Field(..., description="WhatsApp access token")
    phone_number_id: str = Field(..., description="WhatsApp phone number ID")
    api_version: str = Field(default="v18.0", description="WhatsApp API version")
    base_url: str = Field(default="https://graph.facebook.com", description="API base URL")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout")
    retries: int = Field(default=3, ge=0, le=10, description="Number of retries")

    @field_validator("access_token")
    def validate_access_token(cls, v):
        """Validate access token is provided."""
        if not v:
            raise ValueError("WhatsApp access token must be provided")
        return v

    @field_validator("phone_number_id")
    def validate_phone_number_id(cls, v):
        """Validate phone number ID is provided."""
        if not v:
            raise ValueError("WhatsApp phone number ID must be provided")
        return v


class WhatsAppAPIClient:
    """Async client for WhatsApp Business API."""

    def __init__(self, config: Optional[WhatsAppConfig] = None):
        """
        Initialize WhatsApp API client.

        Args:
            config: Optional WhatsApp configuration. If not provided, uses environment variables.
        """
        self.config = config or WhatsAppConfig(
            access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
            phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
            api_version=os.getenv("WHATSAPP_API_VERSION", "v18.0"),
            base_url=os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com"),
        )

        self.timeout_config = TimeoutConfig(
            connect=10.0,
            read=self.config.timeout,
            write=10.0,
            pool=5.0,
        )

        self.retry_config = RetryConfig(
            max_attempts=self.config.retries,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2,
            jitter=True,
        )

    def _get_message_url(self) -> str:
        """Get the messages endpoint URL."""
        return f"{self.config.base_url}/{self.config.api_version}/{self.config.phone_number_id}/messages"

    def _get_headers(self, trace_id: Optional[str] = None) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
        }
        if trace_id:
            headers["X-Trace-Id"] = trace_id
        return headers

    async def send_text_message(
        self,
        to: str,
        message: str,
        preview_url: bool = False,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send text message via WhatsApp API.

        Args:
            to: Recipient phone number (with country code, e.g., "+1234567890")
            message: Message text to send
            preview_url: Whether to show URL previews
            trace_id: Optional trace ID for request tracking

        Returns:
            Dict containing API response

        Raises:
            HTTPException: If the API request fails
        """
        async with trace_context(
            operation_name="send_whatsapp_message",
            trace_id=trace_id or str(uuid.uuid4()),
            agent_name="whatsapp_client",
            metadata={"recipient": to, "message_length": len(message)}
        ) as trace_info:
            url = self._get_message_url()
            headers = self._get_headers(trace_info["trace_id"])

            # Create request payload
            request = WhatsAppSendMessageRequest(
                to=to,
                text=WhatsAppTextPayload(body=message, preview_url=preview_url),
            )

            logger.info(
                f"Sending WhatsApp message to {to}",
                extra={"trace_id": trace_info["trace_id"], "phone_number": to},
            )

            async with AsyncHTTPClient(
                timeout_config=self.timeout_config,
                retry_config=self.retry_config,
                headers=headers,
            ) as client:
                try:
                    response = await client.post(url=url, json_data=request.model_dump())
                    
                    # Parse JSON response
                    response_data = response.json() if hasattr(response, 'json') else response
                    
                    # Log successful send
                    if "messages" in response_data:
                        message_id = response_data["messages"][0].get("id", "unknown")
                        logger.info(
                            "WhatsApp message sent successfully",
                            extra={
                                "trace_id": trace_info["trace_id"],
                                "message_id": message_id,
                                "recipient": to,
                            },
                        )
                    
                    return response_data
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"WhatsApp API error: {e.response.status_code}",
                        extra={
                            "trace_id": trace_info["trace_id"],
                            "status_code": e.response.status_code,
                            "response_body": e.response.text,
                        },
                    )
                    raise
                except Exception as e:
                    logger.error(
                        f"Failed to send WhatsApp message: {str(e)}",
                        extra={"trace_id": trace_info["trace_id"], "error": str(e)},
                    )
                    raise

    async def send_typing_indicator(
        self,
        to: str,
        typing: bool = True,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send typing indicator to user.

        Args:
            to: Recipient phone number
            typing: True to show typing, False to hide
            trace_id: Optional trace ID for request tracking

        Returns:
            Dict containing API response
        """
        url = self._get_message_url()
        headers = self._get_headers(trace_id)

        # Create request payload
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "typing": "on" if typing else "off",
        }

        logger.debug(
            f"Sending typing indicator to {to}",
            extra={"trace_id": trace_id, "typing": typing},
        )

        async with AsyncHTTPClient(
            timeout_config=self.timeout_config,
            retry_config=self.retry_config,
            headers=headers,
        ) as client:
            try:
                response = await client.post(url=url, json_data=payload)
                return response.json() if hasattr(response, 'json') else response
            except Exception as e:
                logger.error(
                    f"Failed to send typing indicator: {str(e)}",
                    extra={"trace_id": trace_id, "error": str(e)},
                )
                # Don't raise - typing indicator is not critical
                return {"error": str(e)}

    async def mark_message_as_read(
        self,
        message_id: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mark message as read.

        Args:
            message_id: WhatsApp message ID to mark as read
            trace_id: Optional trace ID for request tracking

        Returns:
            Dict containing API response
        """
        url = self._get_message_url()
        headers = self._get_headers(trace_id)

        # Create request payload
        request = WhatsAppMarkReadRequest(message_id=message_id)

        logger.debug(
            "Marking message as read",
            extra={"trace_id": trace_id, "message_id": message_id},
        )

        async with AsyncHTTPClient(
            timeout_config=self.timeout_config,
            retry_config=self.retry_config,
            headers=headers,
        ) as client:
            try:
                response = await client.post(url=url, json_data=request.model_dump())
                return response.json() if hasattr(response, 'json') else response
            except Exception as e:
                logger.error(
                    f"Failed to mark message as read: {str(e)}",
                    extra={"trace_id": trace_id, "error": str(e)},
                )
                # Don't raise - marking as read is not critical
                return {"error": str(e)}

    async def send_template_message(
        self,
        to: str,
        template_name: str,
        language_code: str = "en",
        components: Optional[List[Dict[str, Any]]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send template message via WhatsApp API.

        Args:
            to: Recipient phone number
            template_name: Name of the approved template
            language_code: Language code for the template
            components: Optional template components (header, body, button parameters)
            trace_id: Optional trace ID for request tracking

        Returns:
            Dict containing API response
        """
        url = self._get_message_url()
        headers = self._get_headers(trace_id)

        # Create template payload
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }

        if components:
            payload["template"]["components"] = components

        logger.info(
            f"Sending WhatsApp template message to {to}",
            extra={
                "trace_id": trace_id,
                "template_name": template_name,
                "language": language_code,
            },
        )

        async with AsyncHTTPClient(
            timeout_config=self.timeout_config,
            retry_config=self.retry_config,
            headers=headers,
        ) as client:
            try:
                response = await client.post(url=url, json_data=payload)
                return response.json() if hasattr(response, 'json') else response
            except Exception as e:
                logger.error(
                    f"Failed to send template message: {str(e)}",
                    extra={"trace_id": trace_id, "error": str(e)},
                )
                raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check WhatsApp API health by verifying phone number.

        Returns:
            Dict with health status
        """
        url = f"{self.config.base_url}/{self.config.api_version}/{self.config.phone_number_id}"
        headers = self._get_headers()

        try:
            async with AsyncHTTPClient(
                timeout_config=TimeoutConfig(connect=5.0, read=5.0),
                retry_config=RetryConfig(max_attempts=1),
                headers=headers,
            ) as client:
                response = await client.get(url=url)
                # Parse JSON response
                response_data = response.json() if hasattr(response, 'json') else response
                return {
                    "status": "healthy",
                    "phone_number_id": self.config.phone_number_id,
                    "verified": response_data.get("verified_name", {}).get("status") == "APPROVED" if isinstance(response_data, dict) else False,
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "phone_number_id": self.config.phone_number_id,
            }