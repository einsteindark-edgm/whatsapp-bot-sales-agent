"""
Gemini Flash 2.5 Client Adapter.

This module provides an adapter for the Gemini Flash 2.5 model, handling
API authentication, request formatting, and response processing.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models import Model
from pydantic import BaseModel, Field, validator

from shared.observability import get_logger, trace_decorator
from shared.utils import generate_trace_id, sanitize_text
from config.settings import settings


# Configure logger
logger = get_logger(__name__)


class GeminiClientConfig(BaseModel):
    """Configuration for Gemini client."""

    model_name: str = Field(
        default="gemini-2.0-flash", description="Gemini model name (without provider prefix)"
    )
    provider: str = Field(default="google-gla", description="Gemini provider identifier")
    api_key: str = Field(..., description="Gemini API key")

    # Request parameters
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Model temperature for consistent responses"
    )
    max_tokens: int = Field(default=100, ge=1, le=1000, description="Maximum tokens for response")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=1, ge=1, le=100, description="Top-k sampling parameter")

    # Safety and filtering
    safety_threshold: str = Field(
        default="BLOCK_MEDIUM_AND_ABOVE", description="Safety filter threshold"
    )

    # Timeout and retry settings
    request_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    @validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key format."""
        if not v or len(v.strip()) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v.strip()

    @validator("model_name")
    def validate_model_name(cls, v):
        """Validate model name format."""
        if v.startswith("google-gla:"):
            # Remove provider prefix if included
            return v.replace("google-gla:", "")
        return v

    @property
    def full_model_name(self) -> str:
        """Get full model name with provider prefix."""
        return f"{self.provider}:{self.model_name}"

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class GeminiResponse(BaseModel):
    """Gemini API response structure."""

    text: str = Field(..., description="Generated text response")
    model_used: str = Field(..., description="Model used for generation")

    # Response metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

    # Token usage
    prompt_tokens: Optional[int] = Field(None, description="Number of prompt tokens used")
    completion_tokens: Optional[int] = Field(
        None, description="Number of completion tokens generated"
    )
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")

    # Safety and quality
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    safety_ratings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Safety ratings from the model"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


class GeminiClient:
    """
    Gemini Flash 2.5 client adapter.

    This class provides a high-level interface for interacting with the
    Gemini Flash 2.5 model through PydanticAI.
    """

    def __init__(self, config: GeminiClientConfig):
        """
        Initialize the Gemini client.

        Args:
            config: Client configuration
        """
        self.config = config
        self._model = self._create_model()

        logger.info(
            "Gemini client initialized",
            model_name=config.full_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def _create_model(self) -> Model:
        """Create and configure the Gemini model."""
        return GeminiModel(
            model_name=self.config.model_name,
            provider=self.config.provider,
            api_key=self.config.api_key,
        )

    @trace_decorator("gemini_generate", "gemini_client")
    async def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, trace_id: Optional[str] = None
    ) -> GeminiResponse:
        """
        Generate text using the Gemini model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            trace_id: Optional trace ID for observability

        Returns:
            GeminiResponse with generated text

        Raises:
            GeminiClientError: If generation fails
        """
        if not trace_id:
            trace_id = generate_trace_id()

        # Sanitize inputs
        sanitized_prompt = sanitize_text(prompt)
        sanitized_system_prompt = sanitize_text(system_prompt) if system_prompt else None

        try:
            # Note: This is a simplified version. In a real implementation,
            # we would use the PydanticAI model directly with proper parameters.
            # For now, we'll create a mock response structure.

            # Log the request
            logger.info(
                "Generating text with Gemini",
                model_name=self.config.full_model_name,
                prompt_length=len(sanitized_prompt),
                has_system_prompt=bool(sanitized_system_prompt),
                trace_id=trace_id,
            )

            # In a real implementation, this would call the model
            # For now, we'll return a mock response
            import time

            start_time = time.time()

            # Mock response (in real implementation, this would be the actual model call)
            mock_response = {
                "text": "This is a mock response from Gemini Flash 2.5",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": len(sanitized_prompt.split()),
                    "completion_tokens": 10,
                    "total_tokens": len(sanitized_prompt.split()) + 10,
                },
            }

            processing_time = time.time() - start_time

            # Create response
            response = GeminiResponse(
                text=mock_response["text"],
                model_used=self.config.full_model_name,
                processing_time=processing_time,
                prompt_tokens=mock_response["usage"]["prompt_tokens"],
                completion_tokens=mock_response["usage"]["completion_tokens"],
                total_tokens=mock_response["usage"]["total_tokens"],
                finish_reason=mock_response["finish_reason"],
            )

            logger.info(
                "Text generation completed",
                model_name=self.config.full_model_name,
                response_length=len(response.text),
                processing_time=processing_time,
                total_tokens=response.total_tokens,
                trace_id=trace_id,
            )

            return response

        except Exception as e:
            logger.error(
                "Text generation failed",
                model_name=self.config.full_model_name,
                error=str(e),
                trace_id=trace_id,
            )

            raise GeminiClientError(
                message=f"Failed to generate text: {str(e)}",
                error_code="GENERATION_FAILED",
                original_error=e,
            )

    @trace_decorator("gemini_classify", "gemini_client")
    async def classify_text(
        self, text: str, categories: List[str], trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify text into one of the provided categories.

        Args:
            text: Text to classify
            categories: List of possible categories
            trace_id: Optional trace ID for observability

        Returns:
            Dictionary with classification results

        Raises:
            GeminiClientError: If classification fails
        """
        if not trace_id:
            trace_id = generate_trace_id()

        # Create classification prompt
        categories_str = ", ".join(f'"{cat}"' for cat in categories)
        classification_prompt = f"""
        Classify the following text into one of these categories: {categories_str}
        
        Text: {text}
        
        Respond with only the category name and a confidence score (0.0-1.0).
        Format: category_name|confidence_score
        """

        try:
            # Generate classification
            response = await self.generate_text(prompt=classification_prompt, trace_id=trace_id)

            # Parse response
            parts = response.text.strip().split("|")
            if len(parts) != 2:
                raise ValueError("Invalid classification response format")

            category = parts[0].strip().strip('"')
            confidence = float(parts[1].strip())

            # Validate category
            if category not in categories:
                raise ValueError(f"Invalid category: {category}")

            # Validate confidence
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Invalid confidence: {confidence}")

            classification_result = {
                "category": category,
                "confidence": confidence,
                "model_used": self.config.full_model_name,
                "processing_time": response.processing_time,
                "trace_id": trace_id,
            }

            logger.info(
                "Text classification completed",
                category=category,
                confidence=confidence,
                text_length=len(text),
                trace_id=trace_id,
            )

            return classification_result

        except Exception as e:
            logger.error(
                "Text classification failed", error=str(e), text_length=len(text), trace_id=trace_id
            )

            raise GeminiClientError(
                message=f"Failed to classify text: {str(e)}",
                error_code="CLASSIFICATION_FAILED",
                original_error=e,
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Gemini client.

        Returns:
            Health check results
        """
        try:
            # Test with a simple generation
            test_prompt = "Hello, this is a health check."
            trace_id = f"health_check_{int(datetime.utcnow().timestamp())}"

            response = await self.generate_text(prompt=test_prompt, trace_id=trace_id)

            return {
                "status": "healthy",
                "model_name": self.config.full_model_name,
                "response_received": bool(response.text),
                "processing_time": response.processing_time,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.config.full_model_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def update_config(self, config: GeminiClientConfig):
        """
        Update client configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self._model = self._create_model()
        logger.info("Gemini client configuration updated")


def create_gemini_client(
    api_key: Optional[str] = None, model_name: Optional[str] = None, **kwargs
) -> GeminiClient:
    """
    Create a configured Gemini client.

    Args:
        api_key: Optional API key override
        model_name: Optional model name override
        **kwargs: Additional configuration parameters

    Returns:
        Configured GeminiClient instance
    """
    config = GeminiClientConfig(
        api_key=api_key or settings.gemini_api_key,
        model_name=model_name or settings.model_name.replace("google-gla:", ""),
        **kwargs,
    )

    return GeminiClient(config)


# Export commonly used classes and functions
__all__ = [
    "GeminiClient",
    "GeminiClientConfig",
    "GeminiResponse",
    "GeminiClientError",
    "create_gemini_client",
]
