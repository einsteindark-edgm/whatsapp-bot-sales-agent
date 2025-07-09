"""
Tests for the Orchestrator Agent using Google ADK AgentEvaluator.

This module contains comprehensive tests for the orchestrator agent,
including workflow orchestration, agent evaluation, and performance tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from agents.orchestrator.agent import orchestrator_agent, WorkflowOrchestrator
from agents.orchestrator.domain.models import WorkflowRequest, WorkflowResponse
from shared.a2a_protocol import A2AMessage, MessageType
from tests.conftest import MockAgentEvaluator


class TestWorkflowOrchestrator:
    """Test suite for WorkflowOrchestrator."""

    @pytest.fixture
    def mock_a2a_client(self):
        """Create mock A2A client."""
        mock_client = AsyncMock()
        mock_client.send_classification_request.return_value = A2AMessage(
            message_type=MessageType.CLASSIFICATION_RESPONSE,
            trace_id="test-trace",
            sender_agent="classifier",
            receiver_agent="orchestrator",
            payload={
                "text": "Test message",
                "label": "product_information",
                "confidence": 0.92,
                "keywords": ["test", "message"],
                "processing_time": 0.5,
                "model_used": "google-gla:gemini-2.0-flash",
                "timestamp": "2024-01-01T10:00:00Z"
            },
        )
        return mock_client

    @pytest.fixture
    def mock_agent_evaluator(self):
        """Create mock AgentEvaluator for Google ADK testing."""
        evaluation_results = {
            "overall_score": 0.85,
            "response_quality": 0.9,
            "response_time": 0.8,
            "accuracy": 0.88,
            "completeness": 0.82,
            "conversation_flow": 0.87,
            "user_satisfaction": 0.85,
        }
        return MockAgentEvaluator(evaluation_results)

    @pytest.fixture
    def test_cases(self):
        """Sample test cases for agent evaluation."""
        return [
            {
                "input": "What's the price of iPhone 15?",
                "expected_classification": "product_information",
                "expected_response_type": "product_information",
                "user_id": "test_user_1",
                "session_id": "test_session_1",
            },
            {
                "input": "My order is delayed and I want to cancel it",
                "expected_classification": "PQR",
                "expected_response_type": "PQR",
                "user_id": "test_user_2",
                "session_id": "test_session_2",
            },
            {
                "input": "Hello, how are you?",
                "expected_classification": "other",
                "expected_response_type": "general",
                "user_id": "test_user_3",
                "session_id": "test_session_3",
            },
        ]

    @pytest.mark.asyncio
    async def test_process_workflow_product_information_request(self, mock_a2a_client, trace_id):
        """Test processing of product information workflow request."""
        # Setup
        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        request = WorkflowRequest(
            request_id="test-request-123",
            user_message="What's the price of iPhone 15?",
            user_id="test_user",
            session_id="test_session",
        )

        # Execute
        result = await agent.process_workflow_request(request, trace_id)

        # Verify
        assert isinstance(result, WorkflowResponse)
        assert result.success is True
        assert result.response is not None
        assert result.classification is not None
        assert result.classification["label"] == "product_information"
        assert result.processing_time > 0
        assert result.request_id == "test-request-123"

        # Verify A2A client was called
        mock_a2a_client.send_classification_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrate_pqr_request(self, mock_a2a_client, trace_id):
        """Test orchestration of PQR request."""
        # Setup
        mock_a2a_client.send_message.return_value = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=True,
            payload={
                "classification": {
                    "label": "PQR",
                    "confidence": 0.88,
                    "reasoning": "User expressing complaint",
                }
            },
        )

        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        request = OrchestrationRequest(
            user_message="My order is delayed and I want to cancel it",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.orchestrate(request)

        # Verify
        assert isinstance(result, OrchestrationResponse)
        assert result.success is True
        assert result.response is not None
        assert result.response_type == "PQR"
        assert result.classification is not None
        assert result.classification["label"] == "PQR"
        assert result.processing_time > 0
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_orchestrate_with_conversation_context(self, mock_a2a_client, trace_id):
        """Test orchestration with conversation context."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        request = OrchestrationRequest(
            user_message="What about the camera quality?",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
            context={
                "conversation_history": [
                    {"role": "user", "content": "Tell me about iPhone 15"},
                    {"role": "assistant", "content": "The iPhone 15 features..."},
                ]
            },
        )

        # Execute
        result = await agent.orchestrate(request)

        # Verify
        assert isinstance(result, OrchestrationResponse)
        assert result.success is True
        assert result.response is not None
        assert result.trace_id == trace_id

        # Verify context was included in the A2A message
        call_args = mock_a2a_client.send_message.call_args
        assert call_args is not None
        message = call_args[0][0]
        assert "context" in message.payload

    @pytest.mark.asyncio
    async def test_orchestrate_classification_failure(self, mock_a2a_client, trace_id):
        """Test orchestration when classification fails."""
        # Setup
        mock_a2a_client.send_message.return_value = A2AResponse(
            response_type=ResponseType.CLASSIFY_RESPONSE,
            trace_id=trace_id,
            sender="classifier",
            recipient="orchestrator",
            success=False,
            error="Classification failed",
        )

        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        request = OrchestrationRequest(
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
            trace_id=trace_id,
        )

        # Execute
        result = await agent.orchestrate(request)

        # Verify
        assert isinstance(result, OrchestrationResponse)
        assert result.success is False
        assert result.error is not None
        assert "classification failed" in result.error.lower()
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_orchestrate_empty_message(self, mock_a2a_client, trace_id):
        """Test orchestration with empty message."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        request = OrchestrationRequest(
            user_message="", user_id="test_user", session_id="test_session", trace_id=trace_id
        )

        # Execute
        result = await agent.orchestrate(request)

        # Verify
        assert isinstance(result, OrchestrationResponse)
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower()
        assert result.trace_id == trace_id

    @pytest.mark.asyncio
    async def test_orchestrate_a2a_client_error(self, mock_a2a_client, trace_id):
        """Test orchestration when A2A client throws error."""
        # Setup
        mock_a2a_client.send_classification_request.side_effect = Exception("A2A communication error")

        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        request = WorkflowRequest(
            request_id="test-request-a2a-error",
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
        )

        # Execute
        result = await agent.process_workflow_request(request, trace_id)

        # Verify
        assert isinstance(result, WorkflowResponse)
        assert result.success is False
        assert result.error is not None
        assert "a2a communication error" in result.error.lower()
        assert result.request_id == "test-request-a2a-error"

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_a2a_client):
        """Test successful health check."""
        # Setup
        mock_a2a_client.health_check.return_value = {"status": "healthy"}

        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        # Execute
        result = await agent.health_check()

        # Verify
        assert result is True
        mock_a2a_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_a2a_client):
        """Test failed health check."""
        # Setup
        mock_a2a_client.health_check.side_effect = Exception("Health check failed")

        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        # Execute
        result = await agent.health_check()

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_a2a_client):
        """Test metrics retrieval."""
        # Setup
        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        # Execute some workflow requests first
        request = WorkflowRequest(
            request_id="test-request-metrics",
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
        )

        await agent.process_workflow_request(request, "test-trace")

        # Get metrics
        metrics = await agent.get_metrics()

        # Verify
        assert isinstance(metrics, dict)
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "average_response_time" in metrics
        assert "active_conversations" in metrics
        assert "classification_distribution" in metrics
        assert metrics["total_requests"] >= 1

    @pytest.mark.asyncio
    async def test_a2a_message_processing(self, mock_a2a_client, trace_id):
        """Test processing of A2A messages."""
        # Setup
        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        # Create and process a workflow request directly
        request = WorkflowRequest(
            request_id="test-request-a2a",
            user_message="Test message",
            user_id="test_user",
            session_id="test_session",
        )

        # Execute
        result = await agent.process_workflow_request(request, trace_id)

        # Verify
        assert isinstance(result, WorkflowResponse)
        assert result.success is True
        assert result.response is not None
        assert result.request_id == "test-request-a2a"

    @pytest.mark.asyncio
    async def test_a2a_message_invalid_type(self, mock_a2a_client, trace_id):
        """Test processing of invalid A2A message type."""
        # Setup
        agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_a2a_client)

        # Test with invalid/empty request
        request = WorkflowRequest(
            request_id="test-request-invalid",
            user_message="",  # Empty message
            user_id="test_user",
            session_id="test_session",
        )

        # Execute
        result = await agent.process_workflow_request(request, trace_id)

        # Verify
        assert isinstance(result, WorkflowResponse)
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower() or "invalid" in result.error.lower()


class TestWorkflowOrchestratorAgentWithEvaluator:
    """Test suite using Google ADK AgentEvaluator for comprehensive testing."""

    @pytest.mark.asyncio
    async def test_agent_evaluator_integration(
        self, mock_a2a_client, mock_agent_evaluator, test_cases
    ):
        """Test integration with Google ADK AgentEvaluator."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Execute agent evaluation
        evaluation_results = await mock_agent_evaluator.evaluate_agent(agent, test_cases)

        # Verify
        assert isinstance(evaluation_results, dict)
        assert "overall_score" in evaluation_results
        assert "response_quality" in evaluation_results
        assert "response_time" in evaluation_results
        assert "accuracy" in evaluation_results
        assert "completeness" in evaluation_results
        assert "conversation_flow" in evaluation_results
        assert "user_satisfaction" in evaluation_results

        # Verify scores are within expected range
        assert 0.0 <= evaluation_results["overall_score"] <= 1.0
        assert 0.0 <= evaluation_results["response_quality"] <= 1.0
        assert 0.0 <= evaluation_results["response_time"] <= 1.0
        assert 0.0 <= evaluation_results["accuracy"] <= 1.0
        assert 0.0 <= evaluation_results["completeness"] <= 1.0
        assert evaluation_results["overall_score"] >= 0.8  # Should be high quality

    @pytest.mark.asyncio
    async def test_agent_test_suite_execution(self, mock_a2a_client, mock_agent_evaluator):
        """Test execution of agent test suite."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Execute test suite
        test_results = await mock_agent_evaluator.run_test_suite(agent, "product_information_suite")

        # Verify
        assert isinstance(test_results, dict)
        assert "test_suite" in test_results
        assert test_results["test_suite"] == "product_information_suite"
        assert "results" in test_results
        assert "passed" in test_results
        assert "total_tests" in test_results
        assert "passed_tests" in test_results
        assert "failed_tests" in test_results

        # Verify test execution results
        assert test_results["passed"] is True
        assert test_results["total_tests"] > 0
        assert test_results["passed_tests"] >= test_results["failed_tests"]

    @pytest.mark.asyncio
    async def test_agent_performance_evaluation(self, mock_a2a_client, mock_agent_evaluator):
        """Test agent performance evaluation."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Create performance test cases
        performance_test_cases = [
            {
                "input": f"Performance test message {i}",
                "expected_classification": "product_information",
                "user_id": f"perf_user_{i}",
                "session_id": f"perf_session_{i}",
            }
            for i in range(50)
        ]

        # Execute performance evaluation
        evaluation_results = await mock_agent_evaluator.evaluate_agent(
            agent, performance_test_cases
        )

        # Verify
        assert isinstance(evaluation_results, dict)
        assert evaluation_results["response_time"] >= 0.7  # Should be reasonably fast
        assert evaluation_results["overall_score"] >= 0.8  # Should maintain quality under load

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_orchestration_requests(self, mock_a2a_client):
        """Test handling of concurrent orchestration requests."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Create multiple requests
        requests = [
            WorkflowRequest(
                request_id=f"concurrent-request-{i}",
                user_message=f"Concurrent test message {i}",
                user_id="test_user",
                session_id="test_session",
            )
            for i in range(20)
        ]

        # Execute concurrently
        results = await asyncio.gather(*[
            agent.process_workflow_request(req, f"concurrent-trace-{i}") 
            for i, req in enumerate(requests)
        ])

        # Verify
        assert len(results) == 20
        for result in results:
            assert isinstance(result, WorkflowResponse)
            assert result.success is True

        # Verify all requests were processed
        assert mock_a2a_client.send_classification_request.call_count == 20

    @pytest.mark.asyncio
    async def test_conversation_state_management(self, mock_a2a_client):
        """Test conversation state management across multiple requests."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # First request
        request1 = WorkflowRequest(
            request_id="conv-request-1",
            user_message="Tell me about iPhone 15",
            user_id="test_user",
            session_id="test_session",
        )

        result1 = await agent.process_workflow_request(request1, "conv-trace-1")
        assert result1.success is True

        # Second request with context
        request2 = WorkflowRequest(
            request_id="conv-request-2",
            user_message="What about the camera quality?",
            user_id="test_user",
            session_id="test_session",
            context={
                "conversation_history": [
                    {"role": "user", "content": "Tell me about iPhone 15"},
                    {"role": "assistant", "content": result1.response},
                ]
            },
        )

        result2 = await agent.process_workflow_request(request2, "conv-trace-2")
        assert result2.success is True

        # Verify context was maintained
        assert result2.request_id == "conv-request-2"
        assert result2.response is not None

    @pytest.mark.asyncio
    async def test_multi_session_handling(self, mock_a2a_client):
        """Test handling of multiple concurrent sessions."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Create requests for different sessions
        session_requests = []
        for session_id in ["session_1", "session_2", "session_3"]:
            for i in range(5):
                request = WorkflowRequest(
                    request_id=f"{session_id}-request-{i}",
                    user_message=f"Message {i} for {session_id}",
                    user_id=f"user_{session_id}",
                    session_id=session_id,
                )
                session_requests.append(request)

        # Execute all requests concurrently with trace IDs
        trace_ids = [f"{session_id}-trace-{i}" for session_id in ["session_1", "session_2", "session_3"] for i in range(5)]
        results = await asyncio.gather(*[
            agent.process_workflow_request(req, trace_id) 
            for req, trace_id in zip(session_requests, trace_ids)
        ])

        # Verify
        assert len(results) == 15  # 3 sessions * 5 requests
        for result in results:
            assert isinstance(result, WorkflowResponse)
            assert result.success is True

        # Get metrics and verify session handling
        metrics = await agent.get_metrics()
        assert metrics["total_requests"] >= 15
        assert metrics["active_conversations"] >= 3


class TestWorkflowOrchestratorAgentPerformance:
    """Performance tests for WorkflowOrchestratorAgent."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_orchestration_performance(self, mock_a2a_client):
        """Test orchestration performance under load."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Execute multiple orchestrations
        start_time = asyncio.get_event_loop().time()

        tasks = []
        for i in range(100):
            request = WorkflowRequest(
                request_id=f"perf-request-{i}",
                user_message=f"Performance test message {i}",
                user_id="test_user",
                session_id="test_session",
            )
            tasks.append(agent.process_workflow_request(request, f"perf-trace-{i}"))

        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Verify
        assert len(results) == 100
        for result in results:
            assert isinstance(result, WorkflowResponse)
            assert result.success is True

        # Performance assertions
        avg_time_per_request = total_time / 100
        assert avg_time_per_request < 0.2  # Should be under 200ms per request

        # Get metrics
        metrics = await agent.get_metrics()
        assert metrics["total_requests"] >= 100
        assert metrics["average_response_time"] < 2.0  # Should be under 2 seconds

    @pytest.mark.asyncio
    async def test_memory_usage(self, mock_a2a_client):
        """Test memory usage doesn't grow excessively."""
        # Setup
        agent = WorkflowOrchestratorAgent(a2a_client=mock_a2a_client)

        # Execute many orchestrations
        for i in range(1000):
            request = WorkflowRequest(
                request_id=f"mem-request-{i}",
                user_message=f"Memory test message {i}",
                user_id="test_user",
                session_id="test_session",
            )
            await agent.process_workflow_request(request, f"mem-trace-{i}")

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
class TestWorkflowOrchestratorAgentIntegration:
    """Integration tests for WorkflowOrchestratorAgent with real components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_orchestration_flow(self, test_settings):
        """Test full orchestration flow with real components."""
        # This would require actual classifier service running
        # For now, we'll mock the integration

        with patch(
            "agents.orchestrator.adapters.outbound.http_a2a_client.HttpA2AClient"
        ) as mock_client_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_client.send_classification_request.return_value = A2AMessage(
                message_type=MessageType.CLASSIFICATION_RESPONSE,
                trace_id="integration-trace",
                sender_agent="classifier",
                receiver_agent="orchestrator",
                payload={
                    "text": "What's the price of iPhone 15?",
                    "label": "product_information",
                    "confidence": 0.92,
                    "keywords": ["price", "iPhone", "15"],
                    "processing_time": 0.5,
                    "model_used": "google-gla:gemini-2.0-flash",
                    "timestamp": "2024-01-01T10:00:00Z"
                },
            )
            mock_client_class.return_value = mock_client

            # Create agent
            agent = WorkflowOrchestrator(name="TestOrchestrator", a2a_client=mock_client)

            # Execute
            request = WorkflowRequest(
                request_id="integration-request",
                user_message="What's the price of iPhone 15?",
                user_id="test_user",
                session_id="test_session",
            )

            result = await agent.process_workflow_request(request, "integration-trace")

            # Verify
            assert isinstance(result, WorkflowResponse)
            assert result.success is True
            assert result.response is not None
            assert result.classification is not None
            assert result.classification["label"] == "product_information"
            assert result.request_id == "integration-request"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestrator_agent_singleton(self):
        """Test that orchestrator_agent is properly configured as singleton."""
        # Import the singleton instance

        # Test health check
        with patch.object(orchestrator_agent, "a2a_client") as mock_client:
            mock_client.health_check.return_value = {"status": "healthy"}

            result = await orchestrator_agent.health_check()
            assert result is True

            # Test that it's the same instance
            from agents.orchestrator.agent import orchestrator_agent as orchestrator_agent2

            assert orchestrator_agent is orchestrator_agent2
