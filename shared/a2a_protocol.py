"""
Agent-to-Agent (A2A) Protocol Implementation.

This module defines the A2A protocol for communication between agents in the
multi-agent WhatsApp sales assistant system. It provides type-safe message
structures with trace IDs for observability.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum
import uuid


class MessageType(str, Enum):
    """Supported A2A message types."""

    CLASSIFICATION_REQUEST = "classification_request"
    CLASSIFICATION_RESPONSE = "classification_response"
    ORCHESTRATION_REQUEST = "orchestration_request"
    ORCHESTRATION_RESPONSE = "orchestration_response"
    HEALTH_CHECK = "health_check"
    ERROR = "error"


class AgentStatus(str, Enum):
    """Agent status indicators."""

    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class A2AMessage(BaseModel):
    """
    Base A2A protocol message structure.

    This is the foundation for all agent-to-agent communication, ensuring
    consistent message structure and observability through trace IDs.
    """

    # Message Identification
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique message identifier"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Trace ID for observability across service calls",
    )

    # Agent Information
    sender_agent: str = Field(
        ..., description="Agent name/identifier sending the message", min_length=1, max_length=50
    )
    receiver_agent: str = Field(
        ..., description="Agent name/identifier receiving the message", min_length=1, max_length=50
    )

    # Message Metadata
    message_type: MessageType = Field(..., description="Type of A2A message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message creation timestamp"
    )

    # Message Content
    payload: Dict[str, Any] = Field(
        ..., description="Message payload with validation based on message type"
    )

    # Optional Metadata
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for request-response pairing"
    )
    retry_count: int = Field(default=0, ge=0, le=10, description="Number of retry attempts")
    priority: int = Field(
        default=5, ge=1, le=10, description="Message priority (1=highest, 10=lowest)"
    )

    @validator("payload")
    def validate_payload_not_empty(cls, v):
        """Ensure payload is not empty."""
        if not v:
            raise ValueError("Payload cannot be empty")
        return v

    @validator("sender_agent", "receiver_agent")
    def validate_agent_names(cls, v):
        """Validate agent names follow naming conventions."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Agent names must be alphanumeric with underscores or hyphens")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        arbitrary_types_allowed = True

    def create_response(
        self, sender_agent: str, message_type: MessageType, payload: Dict[str, Any]
    ) -> "A2AMessage":
        """Create a response message with proper correlation."""
        return A2AMessage(
            sender_agent=sender_agent,
            receiver_agent=self.sender_agent,
            message_type=message_type,
            payload=payload,
            trace_id=self.trace_id,  # Maintain trace ID for observability
            correlation_id=self.message_id,  # Link response to request
            priority=self.priority,
        )


class ClassificationRequest(A2AMessage):
    """
    Request for message classification.

    Sent from orchestrator to classifier to determine if a message
    is 'product_information' or 'PQR'.
    """

    message_type: Literal[MessageType.CLASSIFICATION_REQUEST] = Field(
        default=MessageType.CLASSIFICATION_REQUEST,
        description="Message type for classification requests",
    )

    @validator("payload")
    def validate_classification_request_payload(cls, v):
        """Validate classification request payload structure."""
        required_fields = ["text"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate text field
        if not isinstance(v["text"], str) or len(v["text"].strip()) == 0:
            raise ValueError("Text field must be a non-empty string")

        if len(v["text"]) > 10000:  # Reasonable limit
            raise ValueError("Text field too long (max 10000 characters)")

        return v

    @property
    def text(self) -> str:
        """Get the text to be classified."""
        return self.payload["text"]


class ClassificationResponse(A2AMessage):
    """
    Response with classification result.

    Sent from classifier to orchestrator with the classification
    result and confidence score.
    """

    message_type: Literal[MessageType.CLASSIFICATION_RESPONSE] = Field(
        default=MessageType.CLASSIFICATION_RESPONSE,
        description="Message type for classification responses",
    )

    @validator("payload")
    def validate_classification_response_payload(cls, v):
        """Validate classification response payload structure."""
        required_fields = ["text", "label", "confidence"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate label
        valid_labels = ["product_information", "PQR"]
        if v["label"] not in valid_labels:
            raise ValueError(f"Label must be one of: {valid_labels}")

        # Validate confidence
        if not isinstance(v["confidence"], (int, float)) or not (0.0 <= v["confidence"] <= 1.0):
            raise ValueError("Confidence must be a number between 0.0 and 1.0")

        return v

    @property
    def text(self) -> str:
        """Get the original text that was classified."""
        return self.payload["text"]

    @property
    def label(self) -> str:
        """Get the classification label."""
        return self.payload["label"]

    @property
    def confidence(self) -> float:
        """Get the classification confidence score."""
        return self.payload["confidence"]


class OrchestrationRequest(A2AMessage):
    """
    Request for workflow orchestration.

    Sent from CLI to orchestrator to initiate a conversation workflow.
    """

    message_type: Literal[MessageType.ORCHESTRATION_REQUEST] = Field(
        default=MessageType.ORCHESTRATION_REQUEST,
        description="Message type for orchestration requests",
    )

    @validator("payload")
    def validate_orchestration_request_payload(cls, v):
        """Validate orchestration request payload structure."""
        required_fields = ["user_message"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate user_message
        if not isinstance(v["user_message"], str) or len(v["user_message"].strip()) == 0:
            raise ValueError("User message must be a non-empty string")

        return v

    @property
    def user_message(self) -> str:
        """Get the user message to be processed."""
        return self.payload["user_message"]


class OrchestrationResponse(A2AMessage):
    """
    Response from workflow orchestration.

    Sent from orchestrator to CLI with the conversation response
    and classification context.
    """

    message_type: Literal[MessageType.ORCHESTRATION_RESPONSE] = Field(
        default=MessageType.ORCHESTRATION_RESPONSE,
        description="Message type for orchestration responses",
    )

    @validator("payload")
    def validate_orchestration_response_payload(cls, v):
        """Validate orchestration response payload structure."""
        required_fields = ["response", "classification"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate response
        if not isinstance(v["response"], str) or len(v["response"].strip()) == 0:
            raise ValueError("Response must be a non-empty string")

        # Validate classification structure
        classification = v["classification"]
        if not isinstance(classification, dict):
            raise ValueError("Classification must be a dictionary")

        required_classification_fields = ["label", "confidence"]
        missing_classification_fields = [
            field for field in required_classification_fields if field not in classification
        ]
        if missing_classification_fields:
            raise ValueError(f"Missing classification fields: {missing_classification_fields}")

        return v

    @property
    def response(self) -> str:
        """Get the orchestrated response."""
        return self.payload["response"]

    @property
    def classification(self) -> Dict[str, Any]:
        """Get the classification context."""
        return self.payload["classification"]


class HealthCheckMessage(A2AMessage):
    """
    Health check message for service monitoring.

    Used to verify agent availability and status.
    """

    message_type: Literal[MessageType.HEALTH_CHECK] = Field(
        default=MessageType.HEALTH_CHECK, description="Message type for health checks"
    )

    @validator("payload")
    def validate_health_check_payload(cls, v):
        """Validate health check payload structure."""
        required_fields = ["status"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate status
        if v["status"] not in [status.value for status in AgentStatus]:
            raise ValueError(f"Status must be one of: {list(AgentStatus)}")

        return v

    @property
    def status(self) -> AgentStatus:
        """Get the agent status."""
        return AgentStatus(self.payload["status"])


class ErrorMessage(A2AMessage):
    """
    Error message for exception handling.

    Used to communicate errors between agents.
    """

    message_type: Literal[MessageType.ERROR] = Field(
        default=MessageType.ERROR, description="Message type for errors"
    )

    @validator("payload")
    def validate_error_payload(cls, v):
        """Validate error payload structure."""
        required_fields = ["error_code", "error_message"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return v

    @property
    def error_code(self) -> str:
        """Get the error code."""
        return self.payload["error_code"]

    @property
    def error_message(self) -> str:
        """Get the error message."""
        return self.payload["error_message"]

    @property
    def error_details(self) -> Optional[Dict[str, Any]]:
        """Get additional error details."""
        return self.payload.get("error_details")


# Type union for all A2A message types
A2AMessageUnion = Union[
    ClassificationRequest,
    ClassificationResponse,
    OrchestrationRequest,
    OrchestrationResponse,
    HealthCheckMessage,
    ErrorMessage,
]


def parse_a2a_message(data: Dict[str, Any]) -> A2AMessageUnion:
    """
    Parse A2A message from dictionary data.

    Args:
        data: Dictionary containing message data

    Returns:
        Parsed A2A message of appropriate type

    Raises:
        ValueError: If message type is invalid or parsing fails
    """
    message_type = data.get("message_type")

    if message_type == MessageType.CLASSIFICATION_REQUEST:
        return ClassificationRequest(**data)
    elif message_type == MessageType.CLASSIFICATION_RESPONSE:
        return ClassificationResponse(**data)
    elif message_type == MessageType.ORCHESTRATION_REQUEST:
        return OrchestrationRequest(**data)
    elif message_type == MessageType.ORCHESTRATION_RESPONSE:
        return OrchestrationResponse(**data)
    elif message_type == MessageType.HEALTH_CHECK:
        return HealthCheckMessage(**data)
    elif message_type == MessageType.ERROR:
        return ErrorMessage(**data)
    else:
        raise ValueError(f"Unknown message type: {message_type}")


def create_error_message(
    sender_agent: str,
    receiver_agent: str,
    error_code: str,
    error_message: str,
    trace_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    error_details: Optional[Dict[str, Any]] = None,
) -> ErrorMessage:
    """
    Create an error message.

    Args:
        sender_agent: Agent sending the error
        receiver_agent: Agent receiving the error
        error_code: Error code
        error_message: Error message
        trace_id: Optional trace ID
        correlation_id: Optional correlation ID
        error_details: Optional additional error details

    Returns:
        Error message
    """
    payload = {"error_code": error_code, "error_message": error_message}

    if error_details:
        payload["error_details"] = error_details

    return ErrorMessage(
        sender_agent=sender_agent,
        receiver_agent=receiver_agent,
        payload=payload,
        trace_id=trace_id or str(uuid.uuid4()),
        correlation_id=correlation_id,
        priority=1,  # High priority for errors
    )
