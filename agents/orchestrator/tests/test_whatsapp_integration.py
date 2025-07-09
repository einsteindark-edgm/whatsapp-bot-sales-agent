"""
Unit tests for WhatsApp integration.

Tests webhook verification, message processing, and API client functionality.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import json
import hmac
import hashlib
import os

from agents.orchestrator.main import app
from agents.orchestrator.domain.whatsapp_models import (
    WhatsAppWebhookPayload,
    WhatsAppSendMessageRequest,
)
from agents.orchestrator.adapters.outbound.whatsapp_api_client import WhatsAppAPIClient, WhatsAppConfig


class TestWhatsAppWebhook:
    """Test cases for WhatsApp webhook endpoints."""

    @pytest.mark.asyncio
    async def test_webhook_verification_success(self):
        """Test successful webhook verification."""
        # Set up test environment
        os.environ["WHATSAPP_WEBHOOK_VERIFY_TOKEN"] = "test_verify_token"

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/webhook/whatsapp",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "test_verify_token",
                    "hub.challenge": "test_challenge_123",
                },
            )

        assert response.status_code == 200
        assert response.text == "test_challenge_123"

    @pytest.mark.asyncio
    async def test_webhook_verification_invalid_token(self):
        """Test webhook verification with invalid token."""
        os.environ["WHATSAPP_WEBHOOK_VERIFY_TOKEN"] = "test_verify_token"

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/webhook/whatsapp",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "wrong_token",
                    "hub.challenge": "test_challenge_123",
                },
            )

        assert response.status_code == 403
        assert "Verification failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_webhook_verification_missing_token(self):
        """Test webhook verification when verify token is not configured."""
        # Clear the environment variable
        os.environ.pop("WHATSAPP_WEBHOOK_VERIFY_TOKEN", None)

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/webhook/whatsapp",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "any_token",
                    "hub.challenge": "test_challenge_123",
                },
            )

        assert response.status_code == 500
        assert "not configured" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("agents.orchestrator.adapters.inbound.whatsapp_webhook_router.WhatsAppAPIClient")
    @patch("agents.orchestrator.adapters.inbound.whatsapp_webhook_router.HTTPAgentToAgentClient")
    async def test_process_webhook_message(self, mock_a2a_client, mock_whatsapp_client):
        """Test processing incoming WhatsApp message."""
        # Set up environment
        os.environ["WHATSAPP_APP_SECRET"] = "test_secret"

        # Mock WhatsApp client
        mock_whatsapp_instance = AsyncMock()
        mock_whatsapp_instance.send_typing_indicator = AsyncMock(return_value={"success": True})
        mock_whatsapp_instance.mark_message_as_read = AsyncMock(return_value={"success": True})
        mock_whatsapp_instance.send_text_message = AsyncMock(
            return_value={"messages": [{"id": "sent_msg_123"}]}
        )
        mock_whatsapp_client.return_value = mock_whatsapp_instance

        # Mock A2A client
        mock_a2a_instance = AsyncMock()
        mock_a2a_instance.orchestrate_workflow = AsyncMock(
            return_value={
                "response": "Hola, ¿en qué puedo ayudarte?",
                "classification": {"type": "greeting", "confidence": 0.95},
            }
        )
        mock_a2a_client.return_value = mock_a2a_instance

        # Create webhook payload
        webhook_data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "WABA_ID",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "15550123456",
                                    "phone_number_id": "123456789",
                                },
                                "messages": [
                                    {
                                        "from": "521234567890",
                                        "id": "wamid.123456",
                                        "timestamp": "1234567890",
                                        "text": {"body": "Hola"},
                                        "type": "text",
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        # Calculate signature
        payload_bytes = json.dumps(webhook_data).encode()
        signature = hmac.new(
            b"test_secret", payload_bytes, hashlib.sha256
        ).hexdigest()

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/webhook/whatsapp",
                json=webhook_data,
                headers={
                    "X-Hub-Signature-256": f"sha256={signature}",
                    "X-Trace-Id": "test-trace-123",
                },
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["processed"] is True
        assert response.json()["classification"] == "greeting"

        # Verify mocks were called
        mock_whatsapp_instance.send_typing_indicator.assert_called_once()
        mock_whatsapp_instance.mark_message_as_read.assert_called_once_with(
            "wamid.123456", trace_id="test-trace-123"
        )
        mock_whatsapp_instance.send_text_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_webhook_invalid_signature(self):
        """Test webhook with invalid signature."""
        os.environ["WHATSAPP_APP_SECRET"] = "test_secret"

        webhook_data = {"object": "whatsapp_business_account", "entry": []}

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/webhook/whatsapp",
                json=webhook_data,
                headers={"X-Hub-Signature-256": "sha256=invalid_signature"},
            )

        assert response.status_code == 401
        assert "Invalid signature" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_process_webhook_invalid_payload(self):
        """Test webhook with invalid payload structure."""
        os.environ["WHATSAPP_APP_SECRET"] = ""  # Disable signature verification

        webhook_data = {"invalid": "data"}

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/webhook/whatsapp", json=webhook_data)

        assert response.status_code == 200  # Should return 200 to prevent retries
        assert response.json()["status"] == "error"
        assert "Invalid payload" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_whatsapp_health_endpoint(self):
        """Test WhatsApp health check endpoint."""
        with patch(
            "agents.orchestrator.adapters.inbound.whatsapp_webhook_router.WhatsAppAPIClient"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(
                return_value={
                    "status": "healthy",
                    "phone_number_id": "123456789",
                    "verified": True,
                }
            )
            mock_client.return_value = mock_instance

            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/webhook/whatsapp/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["whatsapp_api"]["status"] == "healthy"


class TestWhatsAppAPIClient:
    """Test cases for WhatsApp API client."""

    @pytest.mark.asyncio
    async def test_send_text_message(self):
        """Test sending text message."""
        config = WhatsAppConfig(
            access_token="test_token",
            phone_number_id="123456789",
        )
        client = WhatsAppAPIClient(config)

        with patch("shared.utils.AsyncHTTPClient") as mock_http:
            mock_context = AsyncMock()
            mock_context.post = AsyncMock(
                return_value={
                    "messaging_product": "whatsapp",
                    "messages": [{"id": "msg_123"}],
                }
            )
            mock_http.return_value.__aenter__.return_value = mock_context

            result = await client.send_text_message(
                to="+1234567890", message="Test message", trace_id="trace_123"
            )

            assert result["messages"][0]["id"] == "msg_123"
            mock_context.post.assert_called_once()

            # Verify request payload
            call_args = mock_context.post.call_args
            assert call_args.kwargs["json_data"]["to"] == "+1234567890"
            assert call_args.kwargs["json_data"]["text"]["body"] == "Test message"

    @pytest.mark.asyncio
    async def test_send_typing_indicator(self):
        """Test sending typing indicator."""
        client = WhatsAppAPIClient(
            WhatsAppConfig(access_token="test_token", phone_number_id="123456789")
        )

        with patch("shared.utils.AsyncHTTPClient") as mock_http:
            mock_context = AsyncMock()
            mock_context.post = AsyncMock(return_value={"success": True})
            mock_http.return_value.__aenter__.return_value = mock_context

            result = await client.send_typing_indicator(to="+1234567890")

            assert result["success"] is True
            mock_context.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_message_as_read(self):
        """Test marking message as read."""
        client = WhatsAppAPIClient(
            WhatsAppConfig(access_token="test_token", phone_number_id="123456789")
        )

        with patch("shared.utils.AsyncHTTPClient") as mock_http:
            mock_context = AsyncMock()
            mock_context.post = AsyncMock(return_value={"success": True})
            mock_http.return_value.__aenter__.return_value = mock_context

            result = await client.mark_message_as_read(message_id="msg_123")

            assert result["success"] is True
            mock_context.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test WhatsApp API health check."""
        client = WhatsAppAPIClient(
            WhatsAppConfig(access_token="test_token", phone_number_id="123456789")
        )

        with patch("shared.utils.AsyncHTTPClient") as mock_http:
            mock_context = AsyncMock()
            mock_context.get = AsyncMock(
                return_value={
                    "id": "123456789",
                    "verified_name": {"status": "APPROVED"},
                }
            )
            mock_http.return_value.__aenter__.return_value = mock_context

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["verified"] is True


class TestWhatsAppModels:
    """Test cases for WhatsApp domain models."""

    def test_webhook_payload_extract_message(self):
        """Test extracting message from webhook payload."""
        payload_data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "WABA_ID",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "15550123456",
                                    "phone_number_id": "123456789",
                                },
                                "messages": [
                                    {
                                        "from": "521234567890",
                                        "id": "wamid.123456",
                                        "timestamp": "1234567890",
                                        "text": {"body": "Hello World"},
                                        "type": "text",
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        payload = WhatsAppWebhookPayload(**payload_data)
        text, sender, message_id = payload.extract_message()

        assert text == "Hello World"
        assert sender == "521234567890"
        assert message_id == "wamid.123456"

    def test_webhook_payload_no_message(self):
        """Test webhook payload with no messages."""
        payload_data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "WABA_ID",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "15550123456",
                                    "phone_number_id": "123456789",
                                },
                                "statuses": [{"status": "delivered", "id": "msg_123"}],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        payload = WhatsAppWebhookPayload(**payload_data)
        result = payload.extract_message()

        assert result is None

    def test_send_message_request_model(self):
        """Test WhatsApp send message request model."""
        request = WhatsAppSendMessageRequest(
            to="+1234567890",
            text={"body": "Test message", "preview_url": False},
        )

        data = request.model_dump()
        assert data["messaging_product"] == "whatsapp"
        assert data["recipient_type"] == "individual"
        assert data["to"] == "+1234567890"
        assert data["type"] == "text"
        assert data["text"]["body"] == "Test message"