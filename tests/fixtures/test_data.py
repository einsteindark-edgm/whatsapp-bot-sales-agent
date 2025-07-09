"""
Test fixtures and data for the WhatsApp Sales Assistant system.

This module provides comprehensive test data, fixtures, and utilities
for all test modules in the project.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class TestMessage:
    """Test message data structure."""

    text: str
    expected_classification: str
    expected_confidence_range: tuple
    description: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestConversation:
    """Test conversation data structure."""

    messages: List[TestMessage]
    context: Dict[str, Any]
    expected_flow: str
    description: str


class TestDataProvider:
    """Provider for comprehensive test data."""

    @staticmethod
    def get_product_information_messages() -> List[TestMessage]:
        """Get test messages for product information classification."""
        return [
            TestMessage(
                text="What's the price of iPhone 15?",
                expected_classification="product_information",
                expected_confidence_range=(0.85, 1.0),
                description="Direct product pricing inquiry",
                metadata={"category": "pricing", "product": "iPhone 15"},
            ),
            TestMessage(
                text="Do you have wireless headphones?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product availability inquiry",
                metadata={"category": "availability", "product": "wireless headphones"},
            ),
            TestMessage(
                text="Tell me about the iPhone 15 camera features",
                expected_classification="product_information",
                expected_confidence_range=(0.85, 1.0),
                description="Product feature inquiry",
                metadata={"category": "features", "product": "iPhone 15", "feature": "camera"},
            ),
            TestMessage(
                text="What colors are available for the MacBook Pro?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product variant inquiry",
                metadata={"category": "variants", "product": "MacBook Pro", "variant": "colors"},
            ),
            TestMessage(
                text="Is the iPhone 15 waterproof?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product specification inquiry",
                metadata={
                    "category": "specifications",
                    "product": "iPhone 15",
                    "feature": "waterproof",
                },
            ),
            TestMessage(
                text="What's the difference between iPhone 15 and iPhone 15 Pro?",
                expected_classification="product_information",
                expected_confidence_range=(0.85, 1.0),
                description="Product comparison inquiry",
                metadata={"category": "comparison", "products": ["iPhone 15", "iPhone 15 Pro"]},
            ),
            TestMessage(
                text="Do you have any deals on laptops?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product promotions inquiry",
                metadata={"category": "promotions", "product": "laptops"},
            ),
            TestMessage(
                text="What's included in the iPhone 15 box?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product contents inquiry",
                metadata={"category": "contents", "product": "iPhone 15"},
            ),
            TestMessage(
                text="Can you recommend a good smartphone under $500?",
                expected_classification="product_information",
                expected_confidence_range=(0.75, 1.0),
                description="Product recommendation inquiry",
                metadata={"category": "recommendation", "product": "smartphone", "budget": 500},
            ),
            TestMessage(
                text="What's the warranty on the iPad Pro?",
                expected_classification="product_information",
                expected_confidence_range=(0.8, 1.0),
                description="Product warranty inquiry",
                metadata={"category": "warranty", "product": "iPad Pro"},
            ),
        ]

    @staticmethod
    def get_pqr_messages() -> List[TestMessage]:
        """Get test messages for PQR (Problems/Queries/Complaints) classification."""
        return [
            TestMessage(
                text="My order is delayed and I want to cancel it",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Order cancellation complaint",
                metadata={"type": "complaint", "category": "order", "action": "cancel"},
            ),
            TestMessage(
                text="I received a damaged product, what should I do?",
                expected_classification="PQR",
                expected_confidence_range=(0.85, 1.0),
                description="Damaged product complaint",
                metadata={"type": "complaint", "category": "product", "issue": "damaged"},
            ),
            TestMessage(
                text="I want to return this item",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Return request",
                metadata={"type": "request", "category": "return"},
            ),
            TestMessage(
                text="My payment was charged twice",
                expected_classification="PQR",
                expected_confidence_range=(0.85, 1.0),
                description="Payment issue complaint",
                metadata={"type": "complaint", "category": "payment", "issue": "double_charge"},
            ),
            TestMessage(
                text="I haven't received my order yet",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Delivery delay complaint",
                metadata={"type": "complaint", "category": "delivery", "issue": "delay"},
            ),
            TestMessage(
                text="The product doesn't work as expected",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Product functionality complaint",
                metadata={"type": "complaint", "category": "product", "issue": "functionality"},
            ),
            TestMessage(
                text="I need help with my account",
                expected_classification="PQR",
                expected_confidence_range=(0.75, 1.0),
                description="Account support request",
                metadata={"type": "request", "category": "account"},
            ),
            TestMessage(
                text="Can you help me track my order?",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Order tracking request",
                metadata={"type": "request", "category": "order", "action": "track"},
            ),
            TestMessage(
                text="I'm not satisfied with the service",
                expected_classification="PQR",
                expected_confidence_range=(0.8, 1.0),
                description="Service complaint",
                metadata={"type": "complaint", "category": "service"},
            ),
            TestMessage(
                text="How do I change my shipping address?",
                expected_classification="PQR",
                expected_confidence_range=(0.75, 1.0),
                description="Shipping address change request",
                metadata={"type": "request", "category": "shipping", "action": "change_address"},
            ),
        ]

    @staticmethod
    def get_other_messages() -> List[TestMessage]:
        """Get test messages for other/general classification."""
        return [
            TestMessage(
                text="Hello, how are you?",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="General greeting",
                metadata={"type": "greeting"},
            ),
            TestMessage(
                text="What time is it?",
                expected_classification="other",
                expected_confidence_range=(0.7, 1.0),
                description="Time inquiry",
                metadata={"type": "time_inquiry"},
            ),
            TestMessage(
                text="What's the weather like?",
                expected_classification="other",
                expected_confidence_range=(0.7, 1.0),
                description="Weather inquiry",
                metadata={"type": "weather_inquiry"},
            ),
            TestMessage(
                text="Thank you for your help",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="Appreciation message",
                metadata={"type": "appreciation"},
            ),
            TestMessage(
                text="Goodbye",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="Farewell message",
                metadata={"type": "farewell"},
            ),
            TestMessage(
                text="What's your name?",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="Assistant name inquiry",
                metadata={"type": "name_inquiry"},
            ),
            TestMessage(
                text="Are you a robot?",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="Assistant nature inquiry",
                metadata={"type": "nature_inquiry"},
            ),
            TestMessage(
                text="What can you do?",
                expected_classification="other",
                expected_confidence_range=(0.6, 1.0),
                description="Capability inquiry",
                metadata={"type": "capability_inquiry"},
            ),
            TestMessage(
                text="Tell me a joke",
                expected_classification="other",
                expected_confidence_range=(0.7, 1.0),
                description="Entertainment request",
                metadata={"type": "entertainment"},
            ),
            TestMessage(
                text="How do I contact customer service?",
                expected_classification="other",
                expected_confidence_range=(0.5, 0.8),
                description="Contact information request",
                metadata={"type": "contact_info"},
            ),
        ]

    @staticmethod
    def get_edge_case_messages() -> List[TestMessage]:
        """Get test messages for edge cases."""
        return [
            TestMessage(
                text="",
                expected_classification="other",
                expected_confidence_range=(0.0, 0.5),
                description="Empty message",
                metadata={"type": "empty"},
            ),
            TestMessage(
                text="a",
                expected_classification="other",
                expected_confidence_range=(0.0, 0.5),
                description="Single character",
                metadata={"type": "single_char"},
            ),
            TestMessage(
                text="asdfghjkl qwertyuiop",
                expected_classification="other",
                expected_confidence_range=(0.0, 0.5),
                description="Random characters",
                metadata={"type": "random"},
            ),
            TestMessage(
                text="iPhone iPhone iPhone iPhone iPhone",
                expected_classification="product_information",
                expected_confidence_range=(0.3, 0.8),
                description="Repeated keywords",
                metadata={"type": "repeated"},
            ),
            TestMessage(
                text="🤔🤔🤔",
                expected_classification="other",
                expected_confidence_range=(0.0, 0.5),
                description="Only emojis",
                metadata={"type": "emoji_only"},
            ),
            TestMessage(
                text="Hello, I would like to know about the iPhone 15 price please thank you very much for your help and I hope you can provide me with detailed information about all the features and specifications and also the availability and shipping options and payment methods and warranty details and return policy and everything else I need to know about this product",
                expected_classification="product_information",
                expected_confidence_range=(0.7, 1.0),
                description="Very long message",
                metadata={"type": "long_message"},
            ),
            TestMessage(
                text="iPhone?",
                expected_classification="product_information",
                expected_confidence_range=(0.5, 0.8),
                description="Single word question",
                metadata={"type": "single_word"},
            ),
            TestMessage(
                text="HELP ME WITH MY ORDER NOW!!!",
                expected_classification="PQR",
                expected_confidence_range=(0.6, 1.0),
                description="All caps urgent message",
                metadata={"type": "urgent"},
            ),
            TestMessage(
                text="price iPhone 15 how much cost buy where",
                expected_classification="product_information",
                expected_confidence_range=(0.5, 0.9),
                description="Broken grammar",
                metadata={"type": "broken_grammar"},
            ),
            TestMessage(
                text="Can you help me with my iPhone 15 order that I placed yesterday but I haven't received confirmation and I'm worried it might be cancelled?",
                expected_classification="PQR",
                expected_confidence_range=(0.7, 1.0),
                description="Mixed intent message",
                metadata={"type": "mixed_intent"},
            ),
        ]

    @staticmethod
    def get_conversation_flows() -> List[TestConversation]:
        """Get test conversation flows."""
        return [
            TestConversation(
                messages=[
                    TestMessage(
                        text="Hi, I'm looking for a smartphone",
                        expected_classification="product_information",
                        expected_confidence_range=(0.8, 1.0),
                        description="Initial product inquiry",
                    ),
                    TestMessage(
                        text="What's the price range?",
                        expected_classification="product_information",
                        expected_confidence_range=(0.7, 1.0),
                        description="Follow-up pricing question",
                    ),
                    TestMessage(
                        text="What about the iPhone 15?",
                        expected_classification="product_information",
                        expected_confidence_range=(0.8, 1.0),
                        description="Specific product inquiry",
                    ),
                    TestMessage(
                        text="What colors are available?",
                        expected_classification="product_information",
                        expected_confidence_range=(0.8, 1.0),
                        description="Product variant inquiry",
                    ),
                ],
                context={
                    "user_id": "conversation_user_1",
                    "session_id": "conv_session_1",
                    "channel": "whatsapp",
                    "topic": "smartphone_shopping",
                },
                expected_flow="product_information_flow",
                description="Complete product inquiry conversation",
            ),
            TestConversation(
                messages=[
                    TestMessage(
                        text="I have a problem with my order",
                        expected_classification="PQR",
                        expected_confidence_range=(0.8, 1.0),
                        description="Initial complaint",
                    ),
                    TestMessage(
                        text="It hasn't arrived yet",
                        expected_classification="PQR",
                        expected_confidence_range=(0.8, 1.0),
                        description="Specific issue description",
                    ),
                    TestMessage(
                        text="Can you track it for me?",
                        expected_classification="PQR",
                        expected_confidence_range=(0.8, 1.0),
                        description="Specific action request",
                    ),
                    TestMessage(
                        text="I want to cancel if it's not shipped",
                        expected_classification="PQR",
                        expected_confidence_range=(0.8, 1.0),
                        description="Conditional cancellation request",
                    ),
                ],
                context={
                    "user_id": "conversation_user_2",
                    "session_id": "conv_session_2",
                    "channel": "whatsapp",
                    "topic": "order_issue",
                },
                expected_flow="pqr_flow",
                description="Complete complaint resolution conversation",
            ),
            TestConversation(
                messages=[
                    TestMessage(
                        text="Hello",
                        expected_classification="other",
                        expected_confidence_range=(0.6, 1.0),
                        description="General greeting",
                    ),
                    TestMessage(
                        text="What's the price of iPhone 15?",
                        expected_classification="product_information",
                        expected_confidence_range=(0.8, 1.0),
                        description="Product inquiry after greeting",
                    ),
                    TestMessage(
                        text="My order is delayed",
                        expected_classification="PQR",
                        expected_confidence_range=(0.8, 1.0),
                        description="Complaint after product inquiry",
                    ),
                    TestMessage(
                        text="Thank you",
                        expected_classification="other",
                        expected_confidence_range=(0.6, 1.0),
                        description="Appreciation message",
                    ),
                ],
                context={
                    "user_id": "conversation_user_3",
                    "session_id": "conv_session_3",
                    "channel": "whatsapp",
                    "topic": "mixed_conversation",
                },
                expected_flow="mixed_flow",
                description="Mixed conversation with multiple intents",
            ),
        ]

    @staticmethod
    def get_performance_test_messages(count: int = 100) -> List[TestMessage]:
        """Generate messages for performance testing."""
        messages = []

        # Product information messages
        for i in range(count // 3):
            messages.append(
                TestMessage(
                    text=f"What's the price of product {i}?",
                    expected_classification="product_information",
                    expected_confidence_range=(0.8, 1.0),
                    description=f"Performance test product message {i}",
                    metadata={"type": "performance_test", "category": "product_info"},
                )
            )

        # PQR messages
        for i in range(count // 3):
            messages.append(
                TestMessage(
                    text=f"I have a problem with order {i}",
                    expected_classification="PQR",
                    expected_confidence_range=(0.8, 1.0),
                    description=f"Performance test PQR message {i}",
                    metadata={"type": "performance_test", "category": "pqr"},
                )
            )

        # Other messages
        for i in range(count - 2 * (count // 3)):
            messages.append(
                TestMessage(
                    text=f"Hello, this is test message {i}",
                    expected_classification="other",
                    expected_confidence_range=(0.6, 1.0),
                    description=f"Performance test other message {i}",
                    metadata={"type": "performance_test", "category": "other"},
                )
            )

        return messages

    @staticmethod
    def get_a2a_message_templates() -> Dict[str, Dict[str, Any]]:
        """Get A2A message templates for testing."""
        return {
            "classify_request": {
                "message_type": "classify_request",
                "sender": "orchestrator",
                "recipient": "classifier",
                "payload": {
                    "user_message": "What's the price of iPhone 15?",
                    "user_id": "test_user",
                    "session_id": "test_session",
                    "context": {"conversation_history": []},
                },
            },
            "classify_response": {
                "response_type": "classify_response",
                "sender": "classifier",
                "recipient": "orchestrator",
                "success": True,
                "payload": {
                    "classification": {
                        "label": "product_information",
                        "confidence": 0.92,
                        "reasoning": "User asking about product pricing",
                    },
                    "processing_time": 0.5,
                },
            },
            "orchestrate_request": {
                "message_type": "orchestrate_request",
                "sender": "cli",
                "recipient": "orchestrator",
                "payload": {
                    "user_message": "What's the price of iPhone 15?",
                    "user_id": "test_user",
                    "session_id": "test_session",
                    "include_classification": True,
                },
            },
            "orchestrate_response": {
                "response_type": "orchestrate_response",
                "sender": "orchestrator",
                "recipient": "cli",
                "success": True,
                "payload": {
                    "response": "The iPhone 15 is available starting at $799. Would you like more details?",
                    "response_type": "product_information",
                    "classification": {
                        "label": "product_information",
                        "confidence": 0.92,
                        "reasoning": "User asking about product pricing",
                    },
                    "processing_time": 1.2,
                },
            },
            "error_response": {
                "response_type": "classify_response",
                "sender": "classifier",
                "recipient": "orchestrator",
                "success": False,
                "error": "Classification failed due to invalid input",
                "payload": None,
            },
        }

    @staticmethod
    def get_test_metrics() -> Dict[str, Any]:
        """Get test metrics data."""
        return {
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50,
            "average_response_time": 1.2,
            "median_response_time": 0.8,
            "p95_response_time": 2.5,
            "p99_response_time": 4.0,
            "active_conversations": 25,
            "classification_distribution": {"product_information": 600, "PQR": 300, "other": 100},
            "confidence_distribution": {
                "high": 800,  # > 0.8
                "medium": 150,  # 0.6 - 0.8
                "low": 50,  # < 0.6
            },
            "response_time_by_classification": {
                "product_information": 1.0,
                "PQR": 1.5,
                "other": 0.8,
            },
            "error_distribution": {
                "timeout": 20,
                "invalid_input": 15,
                "service_unavailable": 10,
                "other": 5,
            },
            "hourly_request_counts": [
                {"hour": 0, "count": 25},
                {"hour": 1, "count": 15},
                {"hour": 2, "count": 10},
                {"hour": 3, "count": 5},
                {"hour": 4, "count": 8},
                {"hour": 5, "count": 12},
                {"hour": 6, "count": 30},
                {"hour": 7, "count": 45},
                {"hour": 8, "count": 60},
                {"hour": 9, "count": 80},
                {"hour": 10, "count": 100},
                {"hour": 11, "count": 120},
                {"hour": 12, "count": 110},
                {"hour": 13, "count": 90},
                {"hour": 14, "count": 85},
                {"hour": 15, "count": 70},
                {"hour": 16, "count": 60},
                {"hour": 17, "count": 50},
                {"hour": 18, "count": 40},
                {"hour": 19, "count": 35},
                {"hour": 20, "count": 30},
                {"hour": 21, "count": 25},
                {"hour": 22, "count": 20},
                {"hour": 23, "count": 15},
            ],
        }

    @staticmethod
    def get_test_environment_config() -> Dict[str, Any]:
        """Get test environment configuration."""
        return {
            "services": {
                "classifier": {
                    "host": "localhost",
                    "port": 8001,
                    "health_endpoint": "/api/v1/health",
                    "classify_endpoint": "/api/v1/classify",
                },
                "orchestrator": {
                    "host": "localhost",
                    "port": 8080,
                    "health_endpoint": "/api/v1/health",
                    "orchestrate_endpoint": "/api/v1/orchestrate-direct",
                },
            },
            "timeouts": {"connection": 5.0, "read": 30.0, "total": 35.0},
            "retries": {"max_attempts": 3, "base_delay": 1.0, "max_delay": 10.0},
            "test_settings": {
                "default_user_id": "test_user",
                "default_session_id": "test_session",
                "trace_id_prefix": "test-trace",
                "concurrent_requests": 10,
                "performance_test_duration": 30.0,
                "performance_test_requests": 100,
            },
        }

    @staticmethod
    def save_test_data_to_file(filepath: str, data: Dict[str, Any]):
        """Save test data to JSON file."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_test_data_from_file(filepath: str) -> Dict[str, Any]:
        """Load test data from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def get_all_test_data() -> Dict[str, Any]:
        """Get all test data in a single dictionary."""
        return {
            "product_information_messages": [
                msg.__dict__ for msg in TestDataProvider.get_product_information_messages()
            ],
            "pqr_messages": [msg.__dict__ for msg in TestDataProvider.get_pqr_messages()],
            "other_messages": [msg.__dict__ for msg in TestDataProvider.get_other_messages()],
            "edge_case_messages": [
                msg.__dict__ for msg in TestDataProvider.get_edge_case_messages()
            ],
            "conversation_flows": [
                {
                    "messages": [msg.__dict__ for msg in conv.messages],
                    "context": conv.context,
                    "expected_flow": conv.expected_flow,
                    "description": conv.description,
                }
                for conv in TestDataProvider.get_conversation_flows()
            ],
            "a2a_message_templates": TestDataProvider.get_a2a_message_templates(),
            "test_metrics": TestDataProvider.get_test_metrics(),
            "test_environment_config": TestDataProvider.get_test_environment_config(),
        }


# Export commonly used classes and functions
__all__ = ["TestMessage", "TestConversation", "TestDataProvider"]
