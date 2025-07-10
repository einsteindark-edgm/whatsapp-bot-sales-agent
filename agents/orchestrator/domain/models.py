"""
Orchestrator Domain Models.

This module defines the domain models for the orchestrator agent, including
data structures for workflow orchestration, conversation management, and
agent coordination.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationState(str, Enum):
    """Conversation state management."""

    ACTIVE = "active"
    WAITING_FOR_CLASSIFICATION = "waiting_for_classification"
    PROCESSING_RESPONSE = "processing_response"
    COMPLETED = "completed"
    ERROR = "error"


class ResponseType(str, Enum):
    """Types of responses the orchestrator can generate."""

    PRODUCT_INFORMATION_RESPONSE = "product_information_response"
    PQR_RESPONSE = "pqr_response"
    CLARIFICATION_REQUEST = "clarification_request"
    ERROR_RESPONSE = "error_response"
    HANDOFF_RESPONSE = "handoff_response"


class OrchestratorDependencies(BaseModel):
    """Dependencies for the orchestrator agent."""

    # Service Configuration
    classifier_url: str = Field(
        ..., description="URL of the classifier service", pattern=r"^https?://.+"
    )

    # Session Configuration
    app_name: str = Field(
        default="whatsapp-orchestrator", description="Application name for ADK session management"
    )
    user_id: str = Field(..., description="User ID for session management")
    session_id: str = Field(..., description="Session ID for conversation tracking")

    # Tracing
    trace_id: str = Field(..., description="Trace ID for observability")

    # A2A Configuration
    a2a_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="A2A request timeout in seconds"
    )
    a2a_retries: int = Field(default=3, ge=0, le=10, description="A2A request retry attempts")

    # API Keys
    api_key: Optional[str] = Field(None, description="API key for external services")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class ConversationContext(BaseModel):
    """Context for conversation management."""

    # Conversation Identity
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique conversation identifier"
    )
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")

    # Conversation State
    state: ConversationState = Field(
        default=ConversationState.ACTIVE, description="Current conversation state"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Conversation creation timestamp"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Message History
    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conversation message history"
    )

    # Classification Context
    last_classification: Optional[Dict[str, Any]] = Field(
        None, description="Last classification result"
    )
    classification_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Classification history"
    )

    # Workflow Context
    current_workflow: Optional[str] = Field(None, description="Current workflow identifier")
    workflow_state: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow-specific state"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional conversation metadata"
    )

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        self.last_updated = datetime.utcnow()

    def update_classification(self, classification: Dict[str, Any]):
        """Update classification context."""
        self.last_classification = classification
        self.classification_history.append(
            {"classification": classification, "timestamp": datetime.utcnow().isoformat()}
        )
        self.last_updated = datetime.utcnow()

    def update_state(self, state: ConversationState):
        """Update conversation state."""
        self.state = state
        self.last_updated = datetime.utcnow()

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message."""
        return self.messages[-1] if self.messages else None

    @property
    def user_messages(self) -> List[Dict[str, Any]]:
        """Get all user messages."""
        return [msg for msg in self.messages if msg["role"] == "user"]

    @property
    def assistant_messages(self) -> List[Dict[str, Any]]:
        """Get all assistant messages."""
        return [msg for msg in self.messages if msg["role"] == "assistant"]

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class WorkflowRequest(BaseModel):
    """Request for workflow orchestration."""

    # Request Identity
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier"
    )

    # Message Content
    user_message: str = Field(
        ..., description="User message to process", min_length=1, max_length=10000
    )

    # Context
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")

    # Request Configuration
    include_classification: bool = Field(
        default=True, description="Whether to include classification in response"
    )
    require_handoff: bool = Field(default=False, description="Whether to require human handoff")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional request metadata"
    )

    @validator("user_message")
    def validate_user_message(cls, v):
        """Validate user message content."""
        if not v.strip():
            raise ValueError("User message cannot be empty or whitespace only")
        return v.strip()

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class WorkflowResponse(BaseModel):
    """Response from workflow orchestration."""

    # Response Identity
    request_id: str = Field(..., description="Request identifier this response is for")
    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique response identifier"
    )

    # Response Content
    response: str = Field(..., description="Orchestrated response message")
    response_type: ResponseType = Field(..., description="Type of response generated")

    # Classification Context
    classification: Optional[Dict[str, Any]] = Field(None, description="Classification result")

    # Workflow Context
    workflow_status: WorkflowStatus = Field(..., description="Workflow execution status")
    conversation_state: ConversationState = Field(
        ..., description="Conversation state after processing"
    )

    # Metadata
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    # Token usage (for LLM tracking)
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")

    # Success/Error Information
    success: bool = Field(default=True, description="Whether the workflow completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if workflow failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Next Actions
    requires_handoff: bool = Field(
        default=False, description="Whether this response requires human handoff"
    )
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested follow-up actions"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class WorkflowError(BaseModel):
    """Workflow execution error."""

    error_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique error identifier"
    )
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")

    # Context
    request_id: Optional[str] = Field(None, description="Request ID that caused the error")
    workflow_step: Optional[str] = Field(None, description="Workflow step where error occurred")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    # Error Details
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")

    # Recovery
    recoverable: bool = Field(default=True, description="Whether this error is recoverable")
    retry_count: int = Field(default=0, description="Number of retry attempts")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OrchestrationMetrics(BaseModel):
    """Metrics for orchestration performance."""

    # Request Metrics
    total_requests: int = Field(default=0, ge=0, description="Total number of requests processed")
    successful_requests: int = Field(default=0, ge=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Number of failed requests")

    # Response Type Metrics
    product_information_responses: int = Field(
        default=0, ge=0, description="Number of product information responses"
    )
    pqr_responses: int = Field(default=0, ge=0, description="Number of PQR responses")
    error_responses: int = Field(default=0, ge=0, description="Number of error responses")
    handoff_responses: int = Field(default=0, ge=0, description="Number of handoff responses")

    # Performance Metrics
    average_processing_time: float = Field(
        default=0.0, ge=0.0, description="Average processing time in seconds"
    )
    average_classification_time: float = Field(
        default=0.0, ge=0.0, description="Average classification time in seconds"
    )

    # Conversation Metrics
    active_conversations: int = Field(default=0, ge=0, description="Number of active conversations")
    total_conversations: int = Field(default=0, ge=0, description="Total number of conversations")

    # Error Metrics
    a2a_errors: int = Field(default=0, ge=0, description="Number of A2A communication errors")
    classification_errors: int = Field(
        default=0, ge=0, description="Number of classification errors"
    )

    # Time Window
    window_start: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics window start time"
    )
    window_end: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics window end time"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def handoff_rate(self) -> float:
        """Calculate handoff rate."""
        if self.total_requests == 0:
            return 0.0
        return self.handoff_responses / self.total_requests

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OrchestrationConfig(BaseModel):
    """Configuration for orchestration service."""

    # Service Configuration
    classifier_url: str = Field(
        default="http://localhost:8001", description="Classifier service URL"
    )

    # ADK Configuration
    app_name: str = Field(default="whatsapp-orchestrator", description="ADK application name")
    model_name: str = Field(default="gemini-2.0-flash", description="ADK model name")

    # Timeout Configuration
    classification_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Classification request timeout"
    )
    workflow_timeout: float = Field(
        default=60.0, ge=1.0, le=600.0, description="Workflow execution timeout"
    )

    # Retry Configuration
    classification_retries: int = Field(
        default=3, ge=0, le=10, description="Classification request retries"
    )

    # Response Configuration
    max_response_length: int = Field(
        default=1000, ge=50, le=10000, description="Maximum response length"
    )
    include_classification_in_response: bool = Field(
        default=True, description="Include classification in response"
    )

    # Handoff Configuration
    enable_handoff: bool = Field(default=True, description="Enable human handoff")
    handoff_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for handoff"
    )

    # Metrics Configuration
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


# Export commonly used models
__all__ = [
    "WorkflowStatus",
    "ConversationState",
    "ResponseType",
    "OrchestratorDependencies",
    "ConversationContext",
    "WorkflowRequest",
    "WorkflowResponse",
    "WorkflowError",
    "OrchestrationMetrics",
    "OrchestrationConfig",
]
