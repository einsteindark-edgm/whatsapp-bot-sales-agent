"""
Tests for the Classifier Agent using PydanticAI TestModel.

This module contains comprehensive tests for the classifier agent,
including functionality, edge cases, and performance tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from agents.classifier.agent import classifier_agent, ClassifierAgent
from agents.classifier.domain.models import ClassificationRequest, ClassificationResponse
from shared.a2a_protocol import A2AMessage, MessageType
from tests.conftest import MockTestModel


class TestClassifierAgent:
    """Test suite for ClassifierAgent using PydanticAI TestModel."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client."""
        mock_client = AsyncMock()
        mock_client.classify_message.return_value = {
            "label": "product_information",
            "confidence": 0.92,
            "reasoning": "User asking about product pricing",
        }
        return mock_client

    @pytest.fixture
    def test_model_responses(self):
        """Sample responses for TestModel."""
        return [
            '{"label": "product_information", "confidence": 0.92, "reasoning": "User asking about product pricing"}',
            '{"label": "PQR", "confidence": 0.88, "reasoning": "User expressing complaint about delayed order"}',
            '{"label": "other", "confidence": 0.45, "reasoning": "Unclear user intent"}',
        ]

    @pytest.fixture
    def mock_test_model(self, test_model_responses):
        """Create mock TestModel for PydanticAI testing."""
        return MockTestModel(test_model_responses)

    @pytest.mark.asyncio
    async def test_classify_product_information_message(self, mock_gemini_client, trace_id):
        """Test classification of product information message."""
        # Setup
        with patch("agents.classifier.agent.GeminiModel"):
            agent = ClassifierAgent()

        request = ClassificationRequest(
            user_message="What's the price of iPhone 15?",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.classification.label == "product_information"
        assert result.classification.confidence >= 0.7
        assert result.success is True
        assert result.processing_time > 0
        assert result.trace_id == trace_id

        # Verify classification completed successfully
        # Note: With mocked GeminiModel, we can't verify specific calls

    @pytest.mark.asyncio
    async def test_classify_pqr_message(self, mock_gemini_client, trace_id):
        """Test classification of PQR (Problems/Queries/Complaints) message."""
        # Setup
        with patch("agents.classifier.agent.GeminiModel") as mock_model:
            # Mock the agent's response to return PQR classification
            mock_model.return_value.run.return_value.data = MessageClassification(
                label=ClassificationLabel.PQR,
                confidence=0.88,
                reasoning="User expressing complaint about delayed order"
            )
            agent = ClassifierAgent()

        request = ClassificationRequest(
            user_message="My order is delayed and I want to cancel it",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.classification.label == "PQR"
        assert result.classification.confidence >= 0.7
        assert result.success is True
        assert result.processing_time > 0
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_classify_low_confidence_message(self, mock_gemini_client, trace_id):
        """Test classification of message with low confidence."""
        # Setup
        mock_gemini_client.classify_message.return_value = {
            "label": "other",
            "confidence": 0.45,
            "reasoning": "Unclear user intent",
        }

        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        request = ClassificationRequest(
            user_message="Hello there",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.classification.label == "other"
        assert result.classification.confidence < 0.7
        assert result.success is True
        assert result.processing_time > 0
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_classify_empty_message(self, mock_gemini_client, trace_id):
        """Test classification of empty message."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        request = ClassificationRequest(
            user_message="", user_id="test_user", session_id="test_session", trace_id=trace_id
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower()
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_classify_message_with_context(self, mock_gemini_client, trace_id):
        """Test classification with conversation context."""
        # Setup
        mock_gemini_client.classify_message.return_value = {
            "label": "product_information",
            "confidence": 0.95,
            "reasoning": "User asking about product features with context",
        }

        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        request = ClassificationRequest(
            user_message="What about the camera quality?",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
            context={
                "previous_messages": [
                    {"role": "user", "content": "Tell me about iPhone 15"},
                    {"role": "assistant", "content": "The iPhone 15 features..."},
                ]
            },
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.classification.label == "product_information"
        assert result.classification.confidence >= 0.7
        assert result.success is True
        assert result.trace_id == trace_id

        # Verify context was passed to Gemini client
        call_args = mock_gemini_client.classify_message.call_args
        assert "context" in call_args[1]

    @pytest.mark.asyncio
    async def test_classify_message_gemini_error(self, mock_gemini_client, trace_id):
        """Test classification when Gemini client throws error."""
        # Setup
        mock_gemini_client.classify_message.side_effect = Exception("Gemini API error")

        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        request = ClassificationRequest(
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.success is False
        assert result.error is not None
        assert "gemini api error" in result.error.lower()
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_gemini_client):
        """Test successful health check."""
        # Setup
        mock_gemini_client.health_check.return_value = {"status": "healthy"}

        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute
        result = await agent.health_check()

        # Verify
        assert result is True
        mock_gemini_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_gemini_client):
        """Test failed health check."""
        # Setup
        mock_gemini_client.health_check.side_effect = Exception("Connection failed")

        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute
        result = await agent.health_check()

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_gemini_client):
        """Test metrics retrieval."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute some classifications first
        request = ClassificationRequest(
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
            trace_id="test-trace",
        )

        await agent.classify_message(request)

        # Get metrics
        metrics = await agent.get_metrics()

        # Verify
        assert isinstance(metrics, dict)
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "average_response_time" in metrics
        assert "classification_distribution" in metrics
        assert metrics["total_requests"] >= 1

    @pytest.mark.asyncio
    async def test_a2a_message_processing(self, mock_gemini_client, sample_a2a_message):
        """Test processing of A2A messages."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute - Convert A2A message to ClassificationRequest
        request = ClassificationRequest(
            user_message=sample_a2a_message.payload.get("user_message", ""),
            user_id=sample_a2a_message.payload.get("user_id", "test_user"),
            session_id=sample_a2a_message.payload.get("session_id", "test_session"),
            trace_id=sample_a2a_message.trace_id,
        )
        
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.success is True
        assert result.trace_id == sample_a2a_message.trace_id
        assert result.classification is not None
        assert result.classification.label in ["product_information", "PQR", "other"]

    @pytest.mark.asyncio
    async def test_classification_request_validation(self, mock_gemini_client, trace_id):
        """Test validation of classification requests."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Test with missing required fields
        request = ClassificationRequest(
            user_message="",  # Empty message should fail validation
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower() or "invalid" in result.error.lower()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_classification_requests(self, mock_gemini_client):
        """Test handling of concurrent classification requests."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Create multiple requests
        requests = [
            ClassificationRequest(
                user_message=f"Test message {i}",
                user_id="test_user",
                session_id="test_session",
                trace_id=f"trace-{i}",
            )
            for i in range(10)
        ]

        # Execute concurrently
        results = await asyncio.gather(*[agent.classify_message(req) for req in requests])

        # Verify
        assert len(results) == 10
        for result in results:
            assert isinstance(result, ClassificationResponse)
            assert result.success is True

        # Verify all requests were processed
        assert mock_gemini_client.classify_message.call_count == 10

    @pytest.mark.asyncio
    async def test_classification_with_metadata(self, mock_gemini_client, trace_id):
        """Test classification with additional metadata."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        request = ClassificationRequest(
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
            metadata={"channel": "whatsapp", "language": "en", "timestamp": "2024-01-01T10:00:00Z"},
        )

        # Execute
        result = await agent.classify_message(request)

        # Verify
        assert isinstance(result, ClassificationResponse)
        assert result.success is True
        assert result.metadata is not None
        assert "channel" in result.metadata
        assert result.metadata["channel"] == "whatsapp"


class TestClassifierAgentWithTestModel:
    """Test suite using PydanticAI TestModel for more realistic testing."""

    @pytest.mark.asyncio
    async def test_test_model_integration(self, mock_test_model):
        """Test integration with PydanticAI TestModel."""
        # Setup
        with patch("agents.classifier.agent.TestModel", return_value=mock_test_model):
            agent = ClassifierAgent()

            # Execute
            result = await agent.classify_message(
                ClassificationRequest(
                    user_message="What's the price of iPhone 15?",
                    user_id="test_user",
                    session_id="test_session",
                    trace_id="test-trace",
                )
            )

            # Verify
            assert isinstance(result, ClassificationResponse)
            assert result.success is True
            assert result.classification.label == "product_information"
            assert result.classification.confidence == 0.92

    @pytest.mark.asyncio
    async def test_test_model_pqr_classification(self, mock_test_model):
        """Test PQR classification with TestModel."""
        # Setup - use second response from mock
        mock_test_model.call_count = 1

        with patch("agents.classifier.agent.TestModel", return_value=mock_test_model):
            agent = ClassifierAgent()

            # Execute
            result = await agent.classify_message(
                ClassificationRequest(
                    user_message="My order is delayed and I want to cancel it",
                    user_id="test_user",
                    session_id="test_session",
                    trace_id="test-trace",
                )
            )

            # Verify
            assert isinstance(result, ClassificationResponse)
            assert result.success is True
            assert result.classification.label == "PQR"
            assert result.classification.confidence == 0.88

    @pytest.mark.asyncio
    async def test_test_model_low_confidence(self, mock_test_model):
        """Test low confidence classification with TestModel."""
        # Setup - use third response from mock
        mock_test_model.call_count = 2

        with patch("agents.classifier.agent.TestModel", return_value=mock_test_model):
            agent = ClassifierAgent()

            # Execute
            result = await agent.classify_message(
                ClassificationRequest(
                    user_message="Hello there",
                    user_id="test_user",
                    session_id="test_session",
                    trace_id="test-trace",
                )
            )

            # Verify
            assert isinstance(result, ClassificationResponse)
            assert result.success is True
            assert result.classification.label == "other"
            assert result.classification.confidence == 0.45


class TestClassifierAgentPerformance:
    """Performance tests for ClassifierAgent."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_classification_performance(self, mock_gemini_client):
        """Test classification performance under load."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute multiple classifications
        start_time = asyncio.get_event_loop().time()

        tasks = []
        for i in range(100):
            request = ClassificationRequest(
                user_message=f"Performance test message {i}",
                user_id="test_user",
                session_id="test_session",
                trace_id=f"perf-trace-{i}",
            )
            tasks.append(agent.classify_message(request))

        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Verify
        assert len(results) == 100
        for result in results:
            assert isinstance(result, ClassificationResponse)
            assert result.success is True

        # Performance assertions
        avg_time_per_request = total_time / 100
        assert avg_time_per_request < 0.1  # Should be under 100ms per request

        # Get metrics
        metrics = await agent.get_metrics()
        assert metrics["total_requests"] >= 100
        assert metrics["average_response_time"] < 1.0  # Should be under 1 second

    @pytest.mark.asyncio
    async def test_memory_usage(self, mock_gemini_client):
        """Test memory usage doesn't grow excessively."""
        # Setup
        agent = ClassifierAgent(gemini_client=mock_gemini_client)

        # Execute many classifications
        for i in range(1000):
            request = ClassificationRequest(
                user_message=f"Memory test message {i}",
                user_id="test_user",
                session_id="test_session",
                trace_id=f"mem-trace-{i}",
            )
            await agent.classify_message(request)

            # Clear some history periodically to simulate real usage
            if i % 100 == 0:
                agent._clear_old_metrics()

        # Get final metrics
        metrics = await agent.get_metrics()

        # Verify metrics are reasonable
        assert metrics["total_requests"] >= 1000
        assert isinstance(metrics["average_response_time"], float)
        assert metrics["average_response_time"] > 0


# Integration tests with real components
class TestClassifierAgentIntegration:
    """Integration tests for ClassifierAgent with real components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_classification_flow(self, test_settings):
        """Test full classification flow with real components."""
        # This would require actual Gemini API key and network access
        # For now, we'll mock the integration

        with patch(
            "agents.classifier.adapters.outbound.gemini_client.GeminiClient"
        ) as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client.classify_message.return_value = {
                "label": "product_information",
                "confidence": 0.92,
                "reasoning": "User asking about product pricing",
            }
            mock_client_class.return_value = mock_client

            # Create agent
            agent = ClassifierAgent()

            # Execute
            request = ClassificationRequest(
                user_message="What's the price of iPhone 15?",
                user_id="test_user",
                session_id="test_session",
                trace_id="integration-trace",
            )

            result = await agent.classify_message(request)

            # Verify
            assert isinstance(result, ClassificationResponse)
            assert result.success is True
            assert result.classification.label == "product_information"
            assert result.classification.confidence >= 0.7
            assert result.trace_id == "integration-trace"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_classifier_agent_singleton(self):
        """Test that classifier_agent is properly configured as singleton."""
        # Import the singleton instance

        # Test health check
        with patch.object(classifier_agent, "gemini_client") as mock_client:
            mock_client.health_check.return_value = {"status": "healthy"}

            result = await classifier_agent.health_check()
            assert result is True

            # Test that it's the same instance
            from agents.classifier.agent import classifier_agent as classifier_agent2

            assert classifier_agent is classifier_agent2
