"""
Pytest configuration and fixtures for the WhatsApp Sales Assistant tests.

This module provides shared fixtures and configuration for all test modules
in the project, ensuring consistent test environment setup.
"""

import os
import sys
import asyncio
import pytest
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from shared.a2a_protocol import A2AMessage, MessageType
from shared.observability import get_logger
from shared.observability_enhanced import get_logger as get_enhanced_logger
from shared.utils import generate_trace_id


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide test settings with overrides."""
    # Override environment variables for testing
    test_env = {
        "GEMINI_API_KEY": "AIzaSyTest-API-Key-123456789012345678901234",
        "CLASSIFIER_HOST": "localhost",
        "CLASSIFIER_PORT": "8001",
        "ORCHESTRATOR_HOST": "localhost",
        "ORCHESTRATOR_PORT": "8080",
        "LOG_LEVEL": "DEBUG",
        "TRACE_ENABLED": "true",
        "MODEL_NAME": "google-gla:gemini-2.0-flash",
        "CONFIDENCE_THRESHOLD": "0.7",
        "GEMINI_TEMPERATURE": "0.0",
        "GEMINI_MAX_TOKENS": "100",
        "API_TITLE": "WhatsApp Sales Assistant - Test",
        "API_VERSION": "1.0.0-test",
        "HTTP_TIMEOUT": "30.0",
        "HTTP_RETRIES": "3",
        "IS_DOCKER_ENVIRONMENT": "false",
        # Observability settings
        "LOGFIRE_TOKEN": "test-logfire-token",
        "LOGFIRE_PROJECT_NAME": "test-project",
        "LOGFIRE_ENVIRONMENT": "test",
        "ARIZE_API_KEY": "test-arize-key",
        "ARIZE_SPACE_KEY": "test-space-key",
        "ARIZE_MODEL_ID": "test-model",
        "ARIZE_MODEL_VERSION": "1.0.0-test",
    }

    with patch.dict(os.environ, test_env):
        return Settings()


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    return Mock(spec=get_logger("test"))


@pytest.fixture
def mock_observability():
    """Provide a mock observability system for testing."""
    with patch("shared.observability_enhanced.enhanced_observability") as mock_obs:
        mock_obs.trace_llm_interaction.return_value = "test-trace-id"
        mock_obs.trace_agent_operation.return_value = None
        mock_obs.get_metrics_summary.return_value = {
            "total_llm_traces": 0,
            "integrations": {
                "logfire_enabled": False,
                "arize_enabled": False,
                "arize_otel_enabled": False,
                "otel_enabled": False,
            },
            "recent_traces": [],
            "service_info": {
                "name": "test-service",
                "version": "1.0.0-test",
                "environment": "test",
            }
        }
        yield mock_obs


@pytest.fixture
def trace_id() -> str:
    """Provide a test trace ID."""
    return generate_trace_id()


@pytest.fixture
def sample_user_message() -> str:
    """Provide a sample user message for testing."""
    return "What's the price of iPhone 15?"


@pytest.fixture
def sample_user_message_pqr() -> str:
    """Provide a sample PQR user message for testing."""
    return "My order is delayed and I want to cancel it"


@pytest.fixture
def sample_a2a_message(trace_id: str) -> A2AMessage:
    """Provide a sample A2A message for testing."""
    return A2AMessage(
        message_type=MessageType.CLASSIFICATION_REQUEST,
        trace_id=trace_id,
        sender="test_sender",
        recipient="classifier",
        payload={
            "user_message": "What's the price of iPhone 15?",
            "user_id": "test_user",
            "session_id": "test_session",
        },
    )


@pytest.fixture
def sample_classification_response(trace_id: str) -> A2AMessage:
    """Provide a sample classification response for testing."""
    return A2AMessage(
        message_type=MessageType.CLASSIFICATION_RESPONSE,
        trace_id=trace_id,
        sender_agent="classifier",
        receiver_agent="test_sender",
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


@pytest.fixture
def sample_orchestration_response(trace_id: str) -> A2AMessage:
    """Provide a sample orchestration response for testing."""
    return A2AMessage(
        message_type=MessageType.ORCHESTRATE_RESPONSE,
        trace_id=trace_id,
        sender_agent="orchestrator",
        receiver_agent="test_sender",
        payload={
            "response": "The iPhone 15 is available in different storage capacities. The 128GB model starts at $799, 256GB at $899, and 512GB at $1,099. Would you like more details about any specific model?",
            "response_type": "product_information",
            "classification": {
                "label": "product_information",
                "confidence": 0.92,
                "reasoning": "User asking about product pricing",
            },
            "processing_time": 1.2,
        },
    )


@pytest.fixture
def mock_gemini_client():
    """Provide a mock Gemini client for testing."""
    mock_client = AsyncMock()
    mock_client.classify_message.return_value = {
        "label": "product_information",
        "confidence": 0.92,
        "reasoning": "User asking about product pricing",
    }
    return mock_client


@pytest.fixture
def mock_http_client():
    """Provide a mock HTTP client for testing."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_client.request.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_a2a_client():
    """Provide a mock A2A client for testing."""
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
def test_conversation_context() -> Dict[str, Any]:
    """Provide test conversation context."""
    return {
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "conversation_history": [
            {
                "role": "user",
                "content": "Hi, I'm looking for a smartphone",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "role": "assistant",
                "content": "Hello! I'd be happy to help you find a smartphone. What features are most important to you?",
                "timestamp": "2024-01-01T10:00:01Z",
            },
        ],
        "metadata": {"channel": "whatsapp", "language": "en", "timezone": "UTC"},
    }


@pytest.fixture
def test_metrics_data() -> Dict[str, Any]:
    """Provide test metrics data."""
    return {
        "total_requests": 150,
        "successful_requests": 142,
        "failed_requests": 8,
        "average_response_time": 0.85,
        "classifications": {"product_information": 89, "PQR": 53, "other": 8},
        "confidence_distribution": {
            "high": 120,  # > 0.8
            "medium": 22,  # 0.6 - 0.8
            "low": 8,  # < 0.6
        },
    }


@pytest.fixture
def cleanup_test_files():
    """Clean up test files after tests."""
    test_files = []

    def _add_file(file_path: str):
        test_files.append(file_path)

    yield _add_file

    # Cleanup
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add unit marker to unit tests
        if "unit" in item.nodeid or "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)


# Test utilities
class TestModelResponse:
    """Mock response for PydanticAI TestModel."""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def json(self):
        return self.content

    def __str__(self):
        return self.content


class MockTestModel:
    """Mock TestModel for PydanticAI testing."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0

    async def run(self, user_input: str) -> TestModelResponse:
        """Mock run method."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return TestModelResponse(response)

        # Default response
        return TestModelResponse("mock response")


class MockAgentEvaluator:
    """Mock AgentEvaluator for Google ADK testing."""

    def __init__(self, evaluation_results: Dict[str, Any] = None):
        self.evaluation_results = evaluation_results or {
            "overall_score": 0.85,
            "response_quality": 0.9,
            "response_time": 0.8,
            "accuracy": 0.88,
            "completeness": 0.82,
        }

    async def evaluate_agent(self, agent, test_cases: list) -> Dict[str, Any]:
        """Mock evaluate_agent method."""
        return self.evaluation_results

    async def run_test_suite(self, agent, test_suite_name: str) -> Dict[str, Any]:
        """Mock run_test_suite method."""
        return {
            "test_suite": test_suite_name,
            "results": self.evaluation_results,
            "passed": True,
            "total_tests": 10,
            "passed_tests": 9,
            "failed_tests": 1,
        }


# Export fixtures and utilities
__all__ = [
    "test_settings",
    "mock_logger",
    "trace_id",
    "sample_user_message",
    "sample_user_message_pqr",
    "sample_a2a_message",
    "sample_classification_response",
    "sample_orchestration_response",
    "mock_gemini_client",
    "mock_http_client",
    "mock_a2a_client",
    "test_conversation_context",
    "test_metrics_data",
    "cleanup_test_files",
    "TestModelResponse",
    "MockTestModel",
    "MockAgentEvaluator",
]
