"""
End-to-end integration tests for the WhatsApp Sales Assistant system.

This module contains comprehensive end-to-end tests that verify the entire
system works correctly from CLI input to final response.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from cli.client import OrchestratorClient, OrchestratorClientConfig, RequestStatus
from cli.main import async_main
from shared.a2a_protocol import A2AResponse, ResponseType
from shared.utils import generate_trace_id


class TestEndToEndFlow:
    """End-to-end test suite for the complete system flow."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for end-to-end testing."""
        # Mock Gemini client
        mock_gemini = AsyncMock()
        mock_gemini.classify_message.return_value = {
            "label": "product_information",
            "confidence": 0.92,
            "reasoning": "User asking about product pricing",
        }
        mock_gemini.health_check.return_value = {"status": "healthy"}

        # Mock classifier agent
        mock_classifier = AsyncMock()
        mock_classifier.classify_message.return_value = Mock(
            success=True,
            classification=Mock(
                label="product_information",
                confidence=0.92,
                reasoning="User asking about product pricing",
            ),
            processing_time=0.5,
            trace_id="test-trace",
        )

        # Mock A2A client
        mock_a2a_client = AsyncMock()
        mock_a2a_client.send_message.return_value = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id="test-trace",
            sender="classifier",
            recipient="orchestrator",
            success=True,
            payload={
                "classification": {
                    "label": "product_information",
                    "confidence": 0.92,
                    "reasoning": "User asking about product pricing",
                }
            },
        )

        # Mock orchestrator agent
        mock_orchestrator = AsyncMock()
        mock_orchestrator.orchestrate.return_value = Mock(
            success=True,
            response="The iPhone 15 is available starting at $799. Would you like more details?",
            response_type="product_information",
            classification={
                "label": "product_information",
                "confidence": 0.92,
                "reasoning": "User asking about product pricing",
            },
            processing_time=1.2,
            trace_id="test-trace",
        )

        # Mock HTTP client for CLI
        mock_http_client = AsyncMock()
        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = {
            "response": "The iPhone 15 is available starting at $799. Would you like more details?",
            "response_type": "product_information",
            "classification": {
                "label": "product_information",
                "confidence": 0.92,
                "reasoning": "User asking about product pricing",
            },
            "processing_time": 1.2,
            "trace_id": "test-trace",
        }
        mock_http_client.request.return_value = mock_http_response

        return {
            "gemini": mock_gemini,
            "classifier": mock_classifier,
            "a2a_client": mock_a2a_client,
            "orchestrator": mock_orchestrator,
            "http_client": mock_http_client,
        }

    @pytest.mark.asyncio
    async def test_product_information_flow(self, mock_services):
        """Test complete flow for product information request."""
        # Test input
        user_message = "What's the price of iPhone 15?"
        user_id = "test_user"
        session_id = "test_session"
        trace_id = generate_trace_id()

        # Mock the entire chain
        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient",
            return_value=mock_services["gemini"],
        ):
            with patch(
                "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient",
                return_value=mock_services["a2a_client"],
            ):
                with patch(
                    "shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]
                ):
                    # Setup HTTP client context manager
                    mock_services["http_client"].__aenter__ = AsyncMock(
                        return_value=mock_services["http_client"]
                    )
                    mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                    # Create orchestrator client
                    config = OrchestratorClientConfig(
                        base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                    )
                    client = OrchestratorClient(config)

                    # Execute the flow
                    result = await client.send_message(
                        user_message=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        trace_id=trace_id,
                    )

                    # Verify the result
                    assert result.status == RequestStatus.SUCCESS
                    assert result.data is not None
                    assert (
                        result.data["response"]
                        == "The iPhone 15 is available starting at $799. Would you like more details?"
                    )
                    assert result.data["response_type"] == "product_information"
                    assert result.data["classification"]["label"] == "product_information"
                    assert result.data["classification"]["confidence"] == 0.92
                    assert result.data["processing_time"] == 1.2
                    assert result.data["trace_id"] == "test-trace"

    @pytest.mark.asyncio
    async def test_pqr_flow(self, mock_services):
        """Test complete flow for PQR (Problems/Queries/Complaints) request."""
        # Update mocks for PQR scenario
        mock_services["gemini"].classify_message.return_value = {
            "label": "PQR",
            "confidence": 0.88,
            "reasoning": "User expressing complaint about delayed order",
        }

        mock_services["a2a_client"].send_message.return_value = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id="test-trace",
            sender="classifier",
            recipient="orchestrator",
            success=True,
            payload={
                "classification": {
                    "label": "PQR",
                    "confidence": 0.88,
                    "reasoning": "User expressing complaint about delayed order",
                }
            },
        )

        mock_services["http_client"].request.return_value.json.return_value = {
            "response": "I understand your concern about the delayed order. Let me help you with that. Can you please provide your order number?",
            "response_type": "PQR",
            "classification": {
                "label": "PQR",
                "confidence": 0.88,
                "reasoning": "User expressing complaint about delayed order",
            },
            "processing_time": 1.5,
            "trace_id": "test-trace",
        }

        # Test input
        user_message = "My order is delayed and I want to cancel it"
        user_id = "test_user"
        session_id = "test_session"
        trace_id = generate_trace_id()

        # Mock the entire chain
        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient",
            return_value=mock_services["gemini"],
        ):
            with patch(
                "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient",
                return_value=mock_services["a2a_client"],
            ):
                with patch(
                    "shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]
                ):
                    # Setup HTTP client context manager
                    mock_services["http_client"].__aenter__ = AsyncMock(
                        return_value=mock_services["http_client"]
                    )
                    mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                    # Create orchestrator client
                    config = OrchestratorClientConfig(
                        base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                    )
                    client = OrchestratorClient(config)

                    # Execute the flow
                    result = await client.send_message(
                        user_message=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        trace_id=trace_id,
                    )

                    # Verify the result
                    assert result.status == RequestStatus.SUCCESS
                    assert result.data is not None
                    assert "delayed order" in result.data["response"].lower()
                    assert result.data["response_type"] == "PQR"
                    assert result.data["classification"]["label"] == "PQR"
                    assert result.data["classification"]["confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_conversation_context_flow(self, mock_services):
        """Test complete flow with conversation context."""
        # Setup context-aware responses
        mock_services["gemini"].classify_message.return_value = {
            "label": "product_information",
            "confidence": 0.95,
            "reasoning": "User asking about product features with context",
        }

        mock_services["http_client"].request.return_value.json.return_value = {
            "response": "The iPhone 15 camera features a 48MP main sensor with advanced computational photography. It supports 4K video recording and has improved low-light performance.",
            "response_type": "product_information",
            "classification": {
                "label": "product_information",
                "confidence": 0.95,
                "reasoning": "User asking about product features with context",
            },
            "processing_time": 1.1,
            "trace_id": "test-trace",
        }

        # Test input with context
        user_message = "What about the camera quality?"
        user_id = "test_user"
        session_id = "test_session"
        trace_id = generate_trace_id()

        # Mock the entire chain
        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient",
            return_value=mock_services["gemini"],
        ):
            with patch(
                "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient",
                return_value=mock_services["a2a_client"],
            ):
                with patch(
                    "shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]
                ):
                    # Setup HTTP client context manager
                    mock_services["http_client"].__aenter__ = AsyncMock(
                        return_value=mock_services["http_client"]
                    )
                    mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                    # Create orchestrator client
                    config = OrchestratorClientConfig(
                        base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                    )
                    client = OrchestratorClient(config)

                    # Execute the flow
                    result = await client.send_message(
                        user_message=user_message,
                        user_id=user_id,
                        session_id=session_id,
                        trace_id=trace_id,
                    )

                    # Verify the result
                    assert result.status == RequestStatus.SUCCESS
                    assert result.data is not None
                    assert "camera" in result.data["response"].lower()
                    assert "48mp" in result.data["response"].lower()
                    assert result.data["classification"]["confidence"] == 0.95

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, mock_services):
        """Test handling multiple concurrent requests through the system."""
        # Create multiple different requests
        requests = [
            ("What's the price of iPhone 15?", "product_information"),
            ("My order is delayed", "PQR"),
            ("Do you have wireless headphones?", "product_information"),
            ("I want to return this item", "PQR"),
            ("What are your store hours?", "other"),
        ]

        # Setup different responses for each request
        responses = [
            {
                "response": "The iPhone 15 is available starting at $799.",
                "response_type": "product_information",
                "classification": {"label": "product_information", "confidence": 0.92},
            },
            {
                "response": "I understand your concern about the delayed order.",
                "response_type": "PQR",
                "classification": {"label": "PQR", "confidence": 0.88},
            },
            {
                "response": "Yes, we have a variety of wireless headphones available.",
                "response_type": "product_information",
                "classification": {"label": "product_information", "confidence": 0.90},
            },
            {
                "response": "I can help you with the return process.",
                "response_type": "PQR",
                "classification": {"label": "PQR", "confidence": 0.85},
            },
            {
                "response": "Our store hours are Monday-Friday 9AM-9PM, Saturday-Sunday 10AM-8PM.",
                "response_type": "other",
                "classification": {"label": "other", "confidence": 0.75},
            },
        ]

        # Mock HTTP responses
        mock_http_responses = []
        for i, (message, expected_type) in enumerate(requests):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                **responses[i],
                "processing_time": 1.0 + i * 0.1,
                "trace_id": f"concurrent-trace-{i}",
            }
            mock_http_responses.append(mock_response)

        mock_services["http_client"].request.side_effect = mock_http_responses

        # Mock the entire chain
        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient",
            return_value=mock_services["gemini"],
        ):
            with patch(
                "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient",
                return_value=mock_services["a2a_client"],
            ):
                with patch(
                    "shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]
                ):
                    # Setup HTTP client context manager
                    mock_services["http_client"].__aenter__ = AsyncMock(
                        return_value=mock_services["http_client"]
                    )
                    mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                    # Create orchestrator client
                    config = OrchestratorClientConfig(
                        base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                    )
                    client = OrchestratorClient(config)

                    # Execute all requests concurrently
                    tasks = []
                    for i, (message, expected_type) in enumerate(requests):
                        task = client.send_message(
                            user_message=message,
                            user_id="test_user",
                            session_id="test_session",
                            trace_id=f"concurrent-trace-{i}",
                        )
                        tasks.append(task)

                    results = await asyncio.gather(*tasks)

                    # Verify all requests succeeded
                    assert len(results) == 5
                    for i, result in enumerate(results):
                        assert result.status == RequestStatus.SUCCESS
                        assert result.data is not None
                        assert result.data["response_type"] == requests[i][1]
                        assert result.data["processing_time"] == 1.0 + i * 0.1

    @pytest.mark.asyncio
    async def test_error_handling_flow(self, mock_services):
        """Test error handling throughout the system."""
        # Setup classifier to fail
        mock_services["gemini"].classify_message.side_effect = Exception("Gemini API error")

        # Mock the entire chain
        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient",
            return_value=mock_services["gemini"],
        ):
            with patch(
                "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient",
                return_value=mock_services["a2a_client"],
            ):
                with patch(
                    "shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]
                ):
                    # Setup HTTP client context manager
                    mock_services["http_client"].__aenter__ = AsyncMock(
                        return_value=mock_services["http_client"]
                    )
                    mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                    # Mock HTTP response for error case
                    mock_error_response = Mock()
                    mock_error_response.status_code = 500
                    mock_error_response.json.return_value = {
                        "error": "Internal server error",
                        "message": "Classification failed",
                    }
                    mock_services["http_client"].request.return_value = mock_error_response

                    # Create orchestrator client
                    config = OrchestratorClientConfig(
                        base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                    )
                    client = OrchestratorClient(config)

                    # Execute the flow
                    result = await client.send_message(
                        user_message="Test message",
                        user_id="test_user",
                        session_id="test_session",
                        trace_id="error-trace",
                    )

                    # Verify error handling
                    assert result.status == RequestStatus.ERROR
                    assert result.error is not None
                    assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_health_check_flow(self, mock_services):
        """Test health check flow across all services."""
        # Mock health check responses
        mock_services["http_client"].request.return_value.json.return_value = {
            "status": "healthy",
            "service": "orchestrator",
            "version": "1.0.0",
            "health": {"classifier_connection": "healthy", "active_conversations": 5},
        }

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create orchestrator client
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
            )
            client = OrchestratorClient(config)

            # Execute health check
            result = await client.get_health()

            # Verify health check
            assert result.status == RequestStatus.SUCCESS
            assert result.data is not None
            assert result.data["status"] == "healthy"
            assert result.data["service"] == "orchestrator"
            assert result.data["health"]["classifier_connection"] == "healthy"

    @pytest.mark.asyncio
    async def test_metrics_collection_flow(self, mock_services):
        """Test metrics collection across the system."""
        # Mock metrics response
        mock_services["http_client"].request.return_value.json.return_value = {
            "total_requests": 150,
            "successful_requests": 142,
            "failed_requests": 8,
            "average_response_time": 1.2,
            "active_conversations": 12,
            "classification_distribution": {"product_information": 89, "PQR": 53, "other": 8},
            "response_time_percentiles": {"p50": 0.8, "p90": 1.5, "p95": 2.1, "p99": 3.2},
        }

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create orchestrator client
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
            )
            client = OrchestratorClient(config)

            # Execute metrics collection
            result = await client.get_metrics()

            # Verify metrics
            assert result.status == RequestStatus.SUCCESS
            assert result.data is not None
            assert result.data["total_requests"] == 150
            assert result.data["successful_requests"] == 142
            assert result.data["failed_requests"] == 8
            assert result.data["average_response_time"] == 1.2
            assert result.data["active_conversations"] == 12
            assert "classification_distribution" in result.data
            assert "response_time_percentiles" in result.data


class TestCLIIntegration:
    """Integration tests for CLI with the complete system."""

    @pytest.mark.asyncio
    async def test_cli_single_message_mode(self, mock_services):
        """Test CLI single message mode."""
        # Mock HTTP response
        mock_services["http_client"].request.return_value.json.return_value = {
            "response": "The iPhone 15 is available starting at $799.",
            "response_type": "product_information",
            "classification": {"label": "product_information", "confidence": 0.92},
            "processing_time": 1.2,
            "trace_id": "cli-trace",
        }

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            with patch("cli.client.OrchestratorClient.test_connection", return_value=True):
                # Setup HTTP client context manager
                mock_services["http_client"].__aenter__ = AsyncMock(
                    return_value=mock_services["http_client"]
                )
                mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

                # Create and configure client
                config = OrchestratorClientConfig(
                    base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
                )
                client = OrchestratorClient(config)

                # Execute CLI async main with single message
                await async_main(
                    client=client,
                    test_connection_flag=False,
                    single_message="What's the price of iPhone 15?",
                )

                # Verify HTTP request was made
                mock_services["http_client"].request.assert_called_once()
                call_args = mock_services["http_client"].request.call_args
                assert call_args[1]["method"] == "POST"
                assert "orchestrate-direct" in call_args[1]["url"]

    @pytest.mark.asyncio
    async def test_cli_connection_test_mode(self, mock_services):
        """Test CLI connection test mode."""
        # Mock health check response
        mock_services["http_client"].request.return_value.json.return_value = {
            "status": "healthy",
            "service": "orchestrator",
        }

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create and configure client
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
            )
            client = OrchestratorClient(config)

            # Execute CLI async main with connection test
            await async_main(client=client, test_connection_flag=True, single_message=None)

            # Verify health check request was made
            mock_services["http_client"].request.assert_called_once()
            call_args = mock_services["http_client"].request.call_args
            assert call_args[1]["method"] == "GET"
            assert "health" in call_args[1]["url"]


class TestSystemResilience:
    """Test system resilience and fault tolerance."""

    @pytest.mark.asyncio
    async def test_network_retry_resilience(self, mock_services):
        """Test system resilience with network retries."""
        # Setup mock to fail first two attempts, succeed on third
        mock_fail_response = Mock()
        mock_fail_response.status_code = 503
        mock_fail_response.json.return_value = {"error": "Service unavailable"}

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "response": "The iPhone 15 is available starting at $799.",
            "response_type": "product_information",
            "classification": {"label": "product_information", "confidence": 0.92},
            "processing_time": 1.2,
            "trace_id": "resilience-trace",
        }

        mock_services["http_client"].request.side_effect = [
            mock_fail_response,  # First attempt fails
            mock_fail_response,  # Second attempt fails
            mock_success_response,  # Third attempt succeeds
        ]

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create orchestrator client with retries
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
            )
            client = OrchestratorClient(config)

            # Execute request
            result = await client.send_message(
                user_message="What's the price of iPhone 15?",
                user_id="test_user",
                session_id="test_session",
                trace_id="resilience-trace",
            )

            # Verify success after retries
            assert result.status == RequestStatus.SUCCESS
            assert result.data is not None
            assert result.data["response"] == "The iPhone 15 is available starting at $799."

            # Verify retry attempts
            assert mock_services["http_client"].request.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_services):
        """Test system timeout handling."""
        # Setup mock to timeout
        mock_services["http_client"].request.side_effect = asyncio.TimeoutError("Request timed out")

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create orchestrator client
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080",
                request_timeout=1.0,  # Short timeout for testing
                max_retries=1,
            )
            client = OrchestratorClient(config)

            # Execute request
            result = await client.send_message(
                user_message="Test message",
                user_id="test_user",
                session_id="test_session",
                trace_id="timeout-trace",
            )

            # Verify timeout handling
            assert result.status == RequestStatus.TIMEOUT
            assert result.error is not None
            assert "timed out" in result.error.lower()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_system_load_handling(self, mock_services):
        """Test system behavior under load."""
        # Setup mock responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Load test response",
            "response_type": "product_information",
            "classification": {"label": "product_information", "confidence": 0.92},
            "processing_time": 0.5,
            "trace_id": "load-trace",
        }
        mock_services["http_client"].request.return_value = mock_response

        # Mock the entire chain
        with patch("shared.utils.AsyncHTTPClient", return_value=mock_services["http_client"]):
            # Setup HTTP client context manager
            mock_services["http_client"].__aenter__ = AsyncMock(
                return_value=mock_services["http_client"]
            )
            mock_services["http_client"].__aexit__ = AsyncMock(return_value=None)

            # Create orchestrator client
            config = OrchestratorClientConfig(
                base_url="http://localhost:8080", request_timeout=30.0, max_retries=3
            )
            client = OrchestratorClient(config)

            # Execute many concurrent requests
            tasks = []
            for i in range(100):
                task = client.send_message(
                    user_message=f"Load test message {i}",
                    user_id="test_user",
                    session_id="test_session",
                    trace_id=f"load-trace-{i}",
                )
                tasks.append(task)

            # Execute all requests
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()

            # Verify all requests succeeded
            assert len(results) == 100
            for result in results:
                assert result.status == RequestStatus.SUCCESS

            # Verify performance
            total_time = end_time - start_time
            avg_time_per_request = total_time / 100
            assert avg_time_per_request < 0.5  # Should be under 500ms per request

            # Verify all HTTP requests were made
            assert mock_services["http_client"].request.call_count == 100
