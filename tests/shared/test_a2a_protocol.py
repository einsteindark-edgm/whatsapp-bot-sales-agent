"""
Tests for A2A Protocol integration and communication.

This module contains comprehensive tests for the Agent-to-Agent (A2A) protocol,
including message serialization, validation, and end-to-end communication.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from datetime import datetime, timezone

from shared.a2a_protocol import (
    A2AMessage,
    MessageType,
    A2AProtocolError,
    validate_message,
)


class TestA2AMessageValidation:
    """Test suite for A2A message validation."""

    def test_valid_classify_request_message(self, trace_id):
        """Test creation of valid classify request message."""
        message = A2AMessage(
            message_type=MessageType.CLASSIFICATION_REQUEST,
            trace_id=trace_id,
            sender_agent="orchestrator",
            receiver_agent="classifier",
            payload={
                "user_message": "What's the price of iPhone 15?",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Verify message structure
        assert message.message_type == MessageType.CLASSIFICATION_REQUEST
        assert message.trace_id == trace_id
        assert message.sender_agent == "orchestrator"
        assert message.receiver_agent == "classifier"
        assert message.payload["user_message"] == "What's the price of iPhone 15?"
        assert message.payload["user_id"] == "test_user"
        assert message.payload["session_id"] == "test_session"

        # Verify timestamp is set
        assert message.timestamp is not None
        assert isinstance(message.timestamp, datetime)

        # Verify validation passes
        assert validate_message(message) is True

    def test_valid_orchestrate_request_message(self, trace_id):
        """Test creation of valid orchestrate request message."""
        message = A2AMessage(
            message_type=MessageType.ORCHESTRATE_REQUEST,
            trace_id=trace_id,
            sender="cli",
            recipient="orchestrator",
            payload={
                "user_message": "My order is delayed",
                "user_id": "test_user",
                "session_id": "test_session",
                "include_classification": True,
            },
        )

        # Verify message structure
        assert message.message_type == MessageType.ORCHESTRATE_REQUEST
        assert message.trace_id == trace_id
        assert message.sender == "cli"
        assert message.recipient == "orchestrator"
        assert message.payload["include_classification"] is True

        # Verify validation passes
        assert validate_message(message) is True

    def test_invalid_message_type(self, trace_id):
        """Test validation fails for invalid message type."""
        with pytest.raises(ValueError):
            A2AMessage(
                message_type="invalid_type",
                trace_id=trace_id,
                sender="test",
                recipient="test",
                payload={},
            )

    def test_invalid_trace_id(self):
        """Test validation fails for invalid trace ID."""
        with pytest.raises(ValueError):
            A2AMessage(
                message_type=MessageType.CLASSIFY_REQUEST,
                trace_id="",  # Empty trace ID
                sender="test",
                recipient="test",
                payload={},
            )

    def test_invalid_sender_recipient(self, trace_id):
        """Test validation fails for invalid sender/recipient."""
        with pytest.raises(ValueError):
            A2AMessage(
                message_type=MessageType.CLASSIFY_REQUEST,
                trace_id=trace_id,
                sender="",  # Empty sender
                recipient="test",
                payload={},
            )

    def test_message_with_context(self, trace_id):
        """Test message with conversation context."""
        message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "What about the camera?",
                "user_id": "test_user",
                "session_id": "test_session",
                "context": {
                    "conversation_history": [{"role": "user", "content": "Tell me about iPhone 15"}]
                },
            },
        )

        # Verify context is preserved
        assert "context" in message.payload
        assert "conversation_history" in message.payload["context"]
        assert validate_message(message) is True

    def test_message_serialization(self, trace_id):
        """Test message serialization to/from JSON."""
        original_message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "Test message",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Serialize to JSON
        json_data = original_message.model_dump()
        json_str = json.dumps(json_data, default=str)

        # Deserialize from JSON
        parsed_data = json.loads(json_str)
        reconstructed_message = A2AMessage.model_validate(parsed_data)

        # Verify reconstruction
        assert reconstructed_message.message_type == original_message.message_type
        assert reconstructed_message.trace_id == original_message.trace_id
        assert reconstructed_message.sender == original_message.sender
        assert reconstructed_message.recipient == original_message.recipient
        assert reconstructed_message.payload == original_message.payload


class TestA2AResponseValidation:
    """Test suite for A2A response validation."""

    def test_valid_classify_response(self, trace_id):
        """Test creation of valid classify response."""
        response = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=True,
            payload={
                "classification": {
                    "label": "product_information",
                    "confidence": 0.92,
                    "reasoning": "User asking about product pricing",
                },
                "processing_time": 0.5,
            },
        )

        # Verify response structure
        assert response.response_type == ResponseType.CLASSIFY_RESPONSE
        assert response.trace_id == trace_id
        assert response.sender == "classifier"
        assert response.recipient == "orchestrator"
        assert response.success is True
        assert response.payload["classification"]["label"] == "product_information"
        assert response.payload["classification"]["confidence"] == 0.92
        assert response.payload["processing_time"] == 0.5

        # Verify validation passes
        assert validate_response(response) is True

    def test_valid_orchestrate_response(self, trace_id):
        """Test creation of valid orchestrate response."""
        response = A2AResponse(
            response_type=ResponseType.ORCHESTRATE_RESPONSE,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="cli",
            success=True,
            payload={
                "response": "The iPhone 15 is available starting at $799.",
                "response_type": "product_information",
                "classification": {"label": "product_information", "confidence": 0.92},
                "processing_time": 1.2,
            },
        )

        # Verify response structure
        assert response.response_type == ResponseType.ORCHESTRATE_RESPONSE
        assert response.payload["response"] == "The iPhone 15 is available starting at $799."
        assert response.payload["response_type"] == "product_information"
        assert response.payload["classification"]["label"] == "product_information"

        # Verify validation passes
        assert validate_response(response) is True

    def test_error_response(self, trace_id):
        """Test creation of error response."""
        response = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=False,
            error="Classification failed due to invalid input",
        )

        # Verify error response structure
        assert response.success is False
        assert response.error == "Classification failed due to invalid input"
        assert response.payload is None

        # Verify validation passes
        assert validate_response(response) is True

    def test_response_with_metadata(self, trace_id):
        """Test response with metadata."""
        response = A2AResponse(
            response_type=ResponseType.ORCHESTRATE_RESPONSE,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="cli",
            success=True,
            payload={"response": "Test response", "response_type": "general"},
            metadata={"model_version": "1.0.0", "processing_node": "node-1", "cache_hit": False},
        )

        # Verify metadata is preserved
        assert response.metadata is not None
        assert response.metadata["model_version"] == "1.0.0"
        assert response.metadata["processing_node"] == "node-1"
        assert response.metadata["cache_hit"] is False

        # Verify validation passes
        assert validate_response(response) is True

    def test_response_serialization(self, trace_id):
        """Test response serialization to/from JSON."""
        original_response = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=True,
            payload={"classification": {"label": "product_information", "confidence": 0.92}},
        )

        # Serialize to JSON
        json_data = original_response.model_dump()
        json_str = json.dumps(json_data, default=str)

        # Deserialize from JSON
        parsed_data = json.loads(json_str)
        reconstructed_response = A2AResponse.model_validate(parsed_data)

        # Verify reconstruction
        assert reconstructed_response.response_type == original_response.response_type
        assert reconstructed_response.trace_id == original_response.trace_id
        assert reconstructed_response.sender == original_response.sender
        assert reconstructed_response.recipient == original_response.recipient
        assert reconstructed_response.success == original_response.success
        assert reconstructed_response.payload == original_response.payload


class TestA2AProtocolIntegration:
    """Integration tests for A2A protocol communication."""

    @pytest.fixture
    def mock_http_client(self):
        """Create mock HTTP client for testing."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response_type": "classify_response",
            "trace_id": "test-trace",
            "sender": "classifier",
            "recipient": "orchestrator",
            "success": True,
            "payload": {"classification": {"label": "product_information", "confidence": 0.92}},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mock_client.request.return_value = mock_response
        return mock_client

    @pytest.mark.asyncio
    async def test_end_to_end_message_flow(self, mock_http_client, trace_id):
        """Test end-to-end message flow between agents."""
        # Create original message
        original_message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "What's the price of iPhone 15?",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Simulate sending message via HTTP
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_http_client):
            # Mock the HTTP client context manager
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)

            # Send message
            response_data = await self._send_a2a_message(original_message)

            # Verify response
            assert response_data is not None
            assert response_data["success"] is True
            assert response_data["trace_id"] == trace_id
            assert "classification" in response_data["payload"]
            assert response_data["payload"]["classification"]["label"] == "product_information"

    async def _send_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Helper method to simulate sending A2A message."""
        # This simulates the actual HTTP communication
        # In real implementation, this would be in the HTTP A2A client

        # Serialize message
        message_data = message.model_dump()

        # Make HTTP request (mocked)
        from shared.utils import AsyncHTTPClient, TimeoutConfig, RetryConfig

        timeout_config = TimeoutConfig(connect_timeout=5.0, read_timeout=30.0)
        retry_config = RetryConfig(max_attempts=3, base_delay=1.0)

        async with AsyncHTTPClient(
            timeout_config=timeout_config, retry_config=retry_config
        ) as client:
            response = await client.request(
                method="POST", url="http://localhost:8001/api/v1/classify", json_data=message_data
            )

            return response.json()

    @pytest.mark.asyncio
    async def test_message_retry_logic(self, mock_http_client, trace_id):
        """Test message retry logic on failure."""
        # Setup mock to fail first two attempts, succeed on third
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.json.return_value = {"error": "Internal server error"}

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "response_type": "classify_response",
            "trace_id": trace_id,
            "sender": "classifier",
            "recipient": "orchestrator",
            "success": True,
            "payload": {"classification": {"label": "product_information", "confidence": 0.92}},
        }

        mock_http_client.request.side_effect = [
            mock_response_fail,  # First attempt fails
            mock_response_fail,  # Second attempt fails
            mock_response_success,  # Third attempt succeeds
        ]

        # Create message
        message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "Test message",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Simulate sending with retry logic
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_http_client):
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)

            response_data = await self._send_a2a_message_with_retry(message)

            # Verify success after retries
            assert response_data is not None
            assert response_data["success"] is True
            assert response_data["trace_id"] == trace_id

            # Verify retry attempts
            assert mock_http_client.request.call_count == 3

    async def _send_a2a_message_with_retry(self, message: A2AMessage) -> Dict[str, Any]:
        """Helper method to simulate sending A2A message with retry logic."""
        from shared.utils import AsyncHTTPClient, TimeoutConfig, RetryConfig

        timeout_config = TimeoutConfig(connect_timeout=5.0, read_timeout=30.0)
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)  # Fast retry for testing

        async with AsyncHTTPClient(
            timeout_config=timeout_config, retry_config=retry_config
        ) as client:
            response = await client.request(
                method="POST",
                url="http://localhost:8001/api/v1/classify",
                json_data=message.model_dump(),
            )

            return response.json()

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, mock_http_client):
        """Test handling of concurrent A2A messages."""
        # Create multiple messages
        messages = []
        for i in range(10):
            message = A2AMessage(
                message_type=MessageType.CLASSIFY_REQUEST,
                trace_id=f"concurrent-trace-{i}",
                sender="orchestrator",
                recipient="classifier",
                payload={
                    "user_message": f"Test message {i}",
                    "user_id": "test_user",
                    "session_id": "test_session",
                },
            )
            messages.append(message)

        # Mock responses
        mock_responses = []
        for i in range(10):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response_type": "classify_response",
                "trace_id": f"concurrent-trace-{i}",
                "sender": "classifier",
                "recipient": "orchestrator",
                "success": True,
                "payload": {"classification": {"label": "product_information", "confidence": 0.92}},
            }
            mock_responses.append(mock_response)

        mock_http_client.request.side_effect = mock_responses

        # Send all messages concurrently
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_http_client):
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)

            tasks = [self._send_a2a_message(msg) for msg in messages]
            results = await asyncio.gather(*tasks)

            # Verify all messages were processed
            assert len(results) == 10
            for i, result in enumerate(results):
                assert result["success"] is True
                assert result["trace_id"] == f"concurrent-trace-{i}"

    @pytest.mark.asyncio
    async def test_message_timeout_handling(self, mock_http_client, trace_id):
        """Test handling of message timeouts."""
        # Setup mock to timeout
        mock_http_client.request.side_effect = asyncio.TimeoutError("Request timed out")

        # Create message
        message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "Test message",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Simulate sending with timeout
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_http_client):
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)

            # Should handle timeout gracefully
            with pytest.raises(asyncio.TimeoutError):
                await self._send_a2a_message(message)

    @pytest.mark.asyncio
    async def test_trace_id_propagation(self, mock_http_client, trace_id):
        """Test that trace IDs are properly propagated through the system."""
        # Create message with specific trace ID
        message = A2AMessage(
            message_type=MessageType.CLASSIFY_REQUEST,
            trace_id=trace_id,
            sender="orchestrator",
            recipient="classifier",
            payload={
                "user_message": "Test message",
                "user_id": "test_user",
                "session_id": "test_session",
            },
        )

        # Mock response with same trace ID
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response_type": "classify_response",
            "trace_id": trace_id,  # Same trace ID
            "sender": "classifier",
            "recipient": "orchestrator",
            "success": True,
            "payload": {"classification": {"label": "product_information", "confidence": 0.92}},
        }
        mock_http_client.request.return_value = mock_response

        # Send message
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_http_client):
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)

            response_data = await self._send_a2a_message(message)

            # Verify trace ID is preserved
            assert response_data["trace_id"] == trace_id

            # Verify trace ID was included in the request
            call_args = mock_http_client.request.call_args
            request_data = call_args[1]["json_data"]
            assert request_data["trace_id"] == trace_id


class TestA2AProtocolError:
    """Test suite for A2A protocol error handling."""

    def test_a2a_protocol_error_creation(self):
        """Test creation of A2A protocol errors."""
        error = A2AProtocolError(
            message="Invalid message format", error_code="INVALID_FORMAT", trace_id="test-trace"
        )

        assert str(error) == "Invalid message format"
        assert error.error_code == "INVALID_FORMAT"
        assert error.trace_id == "test-trace"

    def test_a2a_protocol_error_serialization(self):
        """Test serialization of A2A protocol errors."""
        error = A2AProtocolError(
            message="Classification failed",
            error_code="CLASSIFICATION_ERROR",
            trace_id="test-trace",
            details={"model": "gemini-2.0-flash", "confidence": 0.3},
        )

        error_dict = error.to_dict()

        assert error_dict["message"] == "Classification failed"
        assert error_dict["error_code"] == "CLASSIFICATION_ERROR"
        assert error_dict["trace_id"] == "test-trace"
        assert error_dict["details"]["model"] == "gemini-2.0-flash"
        assert error_dict["details"]["confidence"] == 0.3

    def test_a2a_protocol_error_in_response(self, trace_id):
        """Test including A2A protocol error in response."""
        error = A2AProtocolError(
            message="Classification failed", error_code="CLASSIFICATION_ERROR", trace_id=trace_id
        )

        response = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=False,
            error=str(error),
            metadata={"error_details": error.to_dict()},
        )

        assert response.success is False
        assert response.error == "Classification failed"
        assert response.metadata["error_details"]["error_code"] == "CLASSIFICATION_ERROR"
