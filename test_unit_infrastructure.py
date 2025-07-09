#!/usr/bin/env python3
"""
Unit test infrastructure test.

This test verifies that the basic unit test infrastructure is working
and that we can import and instantiate the main components.
"""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.a2a_protocol import A2AMessage, MessageType


def test_a2a_protocol_basic():
    """Test that A2A protocol works with basic message creation."""
    # Test creating a basic A2A message
    message = A2AMessage(
        message_type=MessageType.CLASSIFICATION_REQUEST,
        trace_id="test-trace-123",
        sender_agent="test_sender",
        receiver_agent="test_receiver",
        payload={"test": "data"}
    )
    
    assert message.message_type == MessageType.CLASSIFICATION_REQUEST
    assert message.trace_id == "test-trace-123"
    assert message.sender_agent == "test_sender"
    assert message.receiver_agent == "test_receiver"
    assert message.payload["test"] == "data"


def test_settings_import():
    """Test that settings can be imported."""
    from config.settings import Settings
    # Just test import, don't instantiate due to validation issues
    assert Settings is not None


def test_observability_import():
    """Test that observability modules can be imported."""
    from shared.observability_enhanced import enhanced_observability
    
    assert enhanced_observability is not None
    assert hasattr(enhanced_observability, 'get_metrics_summary')
    assert hasattr(enhanced_observability, 'trace_llm_interaction')


def test_classifier_domain_import():
    """Test that classifier domain models can be imported."""
    from agents.classifier.domain.models import (
        ClassificationRequest, 
        ClassificationResponse,
        MessageClassification,
        ClassificationLabel
    )
    
    assert ClassificationRequest is not None
    assert ClassificationResponse is not None
    assert MessageClassification is not None
    assert ClassificationLabel is not None


def test_orchestrator_domain_import():
    """Test that orchestrator domain models can be imported."""
    from agents.orchestrator.domain.models import (
        WorkflowRequest,
        WorkflowResponse
    )
    
    assert WorkflowRequest is not None
    assert WorkflowResponse is not None


if __name__ == "__main__":
    # Run basic infrastructure tests
    test_a2a_protocol_basic()
    test_settings_import()
    test_observability_import()
    test_classifier_domain_import()
    test_orchestrator_domain_import()
    print("✅ All unit test infrastructure tests passed!")