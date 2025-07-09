"""
Classification Domain Models.

This module defines the domain models for the classifier agent, including
data structures for classification requests, responses, and configuration.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ClassificationLabel(str, Enum):
    """Available classification labels."""

    PRODUCT_INFORMATION = "product_information"
    PQR = "PQR"  # Problems, Queries, Complaints


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""

    LOW = "low"  # 0.0 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.7
    HIGH = "high"  # 0.7 - 1.0


class ClassificationDependencies(BaseModel):
    """Dependencies for classification agent."""

    model_name: str = Field(
        default="google-gla:gemini-2.0-flash",
        description="Gemini model name to use for classification",
    )
    api_key: str = Field(..., description="Gemini API key for authentication", min_length=10)
    trace_id: str = Field(
        ..., description="Request trace ID for observability", min_length=1, max_length=100
    )

    # Optional configuration
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Model temperature for consistent classification"
    )
    max_tokens: int = Field(
        default=100, ge=1, le=1000, description="Maximum tokens for classification response"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold for classification"
    )

    @validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key format."""
        if not v or len(v.strip()) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v.strip()

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name format."""
        if not v.startswith("google-gla:"):
            raise ValueError('Model name must start with "google-gla:"')
        return v

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class MessageClassification(BaseModel):
    """Classification result for a message."""

    text: str = Field(
        ..., description="Original message text that was classified", min_length=1, max_length=10000
    )
    label: ClassificationLabel = Field(
        ..., description="Classification label assigned to the message"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")

    # Additional metadata
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Classification timestamp"
    )
    model_used: str = Field(..., description="Model used for classification")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")

    # Analysis metadata
    text_length: int = Field(description="Length of the input text")
    keywords: List[str] = Field(
        default_factory=list, description="Extracted keywords from the text"
    )
    reasoning: Optional[str] = Field(None, description="Reasoning behind the classification")


    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence < 0.4:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH

    @property
    def is_product_information(self) -> bool:
        """Check if classification is product information."""
        return self.label == ClassificationLabel.PRODUCT_INFORMATION

    @property
    def is_pqr(self) -> bool:
        """Check if classification is PQR."""
        return self.label == ClassificationLabel.PQR

    @property
    def is_high_confidence(self) -> bool:
        """Check if classification has high confidence."""
        return self.confidence_level == ConfidenceLevel.HIGH

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ClassificationRequest(BaseModel):
    """Request for message classification."""

    text: str = Field(..., description="Text to be classified", min_length=1, max_length=10000)
    context: Optional[str] = Field(None, description="Additional context for classification")
    user_id: Optional[str] = Field(None, description="User ID for tracking purposes")
    session_id: Optional[str] = Field(None, description="Session ID for tracking purposes")

    # Classification parameters
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold override"
    )
    include_reasoning: bool = Field(
        default=False, description="Whether to include reasoning in response"
    )
    extract_keywords: bool = Field(
        default=True, description="Whether to extract keywords from text"
    )

    @validator("text")
    def validate_text(cls, v):
        """Validate text content."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class ClassificationResponse(BaseModel):
    """Response from message classification."""

    request_id: str = Field(..., description="Request identifier")
    classification: MessageClassification = Field(..., description="Classification result")

    # Response metadata
    success: bool = Field(default=True, description="Whether classification was successful")
    error_message: Optional[str] = Field(None, description="Error message if classification failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class ClassificationError(BaseModel):
    """Classification error information."""

    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    trace_id: Optional[str] = Field(None, description="Trace ID for debugging")

    # Error details
    input_text: Optional[str] = Field(None, description="Input text that caused the error")
    model_response: Optional[str] = Field(None, description="Raw model response if available")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ClassificationMetrics(BaseModel):
    """Metrics for classification performance."""

    total_classifications: int = Field(
        default=0, ge=0, description="Total number of classifications performed"
    )
    product_information_count: int = Field(
        default=0, ge=0, description="Number of product information classifications"
    )
    pqr_count: int = Field(default=0, ge=0, description="Number of PQR classifications")

    # Performance metrics
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average confidence score"
    )
    average_processing_time: float = Field(
        default=0.0, ge=0.0, description="Average processing time in seconds"
    )

    # Error metrics
    error_count: int = Field(default=0, ge=0, description="Number of classification errors")
    low_confidence_count: int = Field(
        default=0, ge=0, description="Number of low confidence classifications"
    )

    # Time window
    window_start: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics window start time"
    )
    window_end: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics window end time"
    )

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_classifications == 0:
            return 0.0
        return self.error_count / self.total_classifications

    @property
    def product_information_rate(self) -> float:
        """Calculate product information rate."""
        if self.total_classifications == 0:
            return 0.0
        return self.product_information_count / self.total_classifications

    @property
    def pqr_rate(self) -> float:
        """Calculate PQR rate."""
        if self.total_classifications == 0:
            return 0.0
        return self.pqr_count / self.total_classifications

    @property
    def low_confidence_rate(self) -> float:
        """Calculate low confidence rate."""
        if self.total_classifications == 0:
            return 0.0
        return self.low_confidence_count / self.total_classifications

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ClassificationConfig(BaseModel):
    """Configuration for classification service."""

    # Model configuration
    model_name: str = Field(default="google-gla:gemini-2.0-flash", description="Gemini model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=100, ge=1, le=1000, description="Maximum tokens")

    # Classification thresholds
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    # Feature flags
    enable_keyword_extraction: bool = Field(default=True, description="Enable keyword extraction")
    enable_reasoning: bool = Field(default=False, description="Enable reasoning in responses")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

    # Performance settings
    request_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


# Export commonly used models
__all__ = [
    "ClassificationLabel",
    "ConfidenceLevel",
    "ClassificationDependencies",
    "MessageClassification",
    "ClassificationRequest",
    "ClassificationResponse",
    "ClassificationError",
    "ClassificationMetrics",
    "ClassificationConfig",
]
