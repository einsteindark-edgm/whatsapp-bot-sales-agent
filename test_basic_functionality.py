#!/usr/bin/env python3
"""
Basic functionality test to verify unit tests infrastructure works.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import AsyncMock, patch
from shared.observability_enhanced import enhanced_observability


def test_observability_import():
    """Test that observability module imports correctly."""
    assert enhanced_observability is not None
    assert hasattr(enhanced_observability, 'get_metrics_summary')
    assert hasattr(enhanced_observability, 'trace_llm_interaction')


@pytest.mark.asyncio 
async def test_observability_basic_functionality():
    """Test basic observability functionality."""
    with patch("shared.observability_enhanced.enhanced_observability.arize") as mock_arize:
        with patch("shared.observability_enhanced.enhanced_observability.logfire") as mock_logfire:
            mock_arize.enabled = False
            mock_logfire.enabled = False
            
            # Test metrics summary
            metrics = enhanced_observability.get_metrics_summary()
            assert isinstance(metrics, dict)
            assert "timestamp" in metrics
            assert "total_llm_traces" in metrics
            assert "integrations" in metrics
            
            # Test trace LLM interaction
            trace_id = enhanced_observability.trace_llm_interaction(
                agent_name="test_agent",
                model_name="test_model", 
                prompt="test prompt",
                response="test response",
                latency_ms=100.0,
                classification_label="test_label",
                confidence_score=0.85
            )
            
            assert isinstance(trace_id, str)
            assert len(trace_id) > 0


def test_settings_import():
    """Test that settings can be imported and loaded."""
    from config.settings import Settings
    
    # Create test settings with a valid API key format
    with patch.dict(os.environ, {
        "GEMINI_API_KEY": "AIzaSyTest-API-Key-123456789012345678901234",
        "LOG_LEVEL": "DEBUG"
    }):
        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.gemini_api_key == "AIzaSyTest-API-Key-123456789012345678901234"


def test_a2a_protocol_import():
    """Test that A2A protocol models work."""
    from shared.a2a_protocol import A2AMessage, MessageType
    
    message = A2AMessage(
        message_type=MessageType.CLASSIFICATION_REQUEST,
        trace_id="test-trace",
        sender_agent="test_sender",
        receiver_agent="test_receiver",
        payload={"text": "test message"}
    )
    
    assert message.message_type == MessageType.CLASSIFICATION_REQUEST
    assert message.trace_id == "test-trace"
    assert message.payload["text"] == "test message"


if __name__ == "__main__":
    # Run basic functionality test
    test_observability_import()
    test_settings_import()
    test_a2a_protocol_import()
    print("✅ All basic functionality tests passed!")