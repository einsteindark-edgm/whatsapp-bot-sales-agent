"""
Classifier Agent Implementation.

This module implements the classifier agent using PydanticAI with Gemini Flash 2.5
for classifying WhatsApp messages as either 'product_information' or 'PQR'.
"""

import time
from typing import Optional, Dict, Any
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel

from .domain.models import (
    ClassificationDependencies,
    MessageClassification,
    ClassificationLabel,
    ClassificationRequest,
    ClassificationResponse,
    ClassificationError,
    ClassificationMetrics,
    ClassificationConfig,
)
from shared.observability import get_logger, trace_decorator
from shared.observability_enhanced import trace_llm_interaction, trace_agent_operation
from shared.utils import sanitize_text, generate_message_id
from config.settings import settings


# Configure logger
logger = get_logger(__name__)


class ClassifierAgent:
    """
    PydanticAI-based classifier agent for WhatsApp message classification.

    This agent uses Gemini Flash 2.5 to classify messages as either
    'product_information' or 'PQR' (Problems, Queries, Complaints).
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize the classifier agent.

        Args:
            config: Optional configuration for the classifier
        """
        self.config = config or ClassificationConfig()
        self.metrics = ClassificationMetrics()
        self._agent = self._create_agent()

        logger.info(
            "Classifier agent initialized",
            model_name=self.config.model_name,
            confidence_threshold=self.config.confidence_threshold,
        )

    def _create_agent(self) -> Agent:
        """Create and configure the PydanticAI agent."""
        
        # Set API key in environment (required by PydanticAI)
        import os
        os.environ['GEMINI_API_KEY'] = settings.gemini_api_key

        # Create Gemini model
        model = GeminiModel(
            model_name=self.config.model_name.replace("google-gla:", ""), 
            provider="google-gla"
        )

        # Create agent with configuration
        agent = Agent(
            model=model,
            deps_type=ClassificationDependencies,
            output_type=MessageClassification,
            system_prompt=self._get_system_prompt(),
        )

        # Add system prompt function for trace context
        @agent.system_prompt
        async def add_trace_context(ctx: RunContext[ClassificationDependencies]) -> str:
            """Add trace ID to system context for observability."""
            return f"Trace ID: {ctx.deps.trace_id}"

        return agent

    def _get_system_prompt(self) -> str:
        """Get the system prompt for classification."""
        return """
You are a specialized message classifier for WhatsApp commerce conversations.

Your task is to classify messages into exactly one of these categories:
1. "product_information" - Questions about products, pricing, availability, features, specifications, comparisons, or any product-related inquiries
2. "PQR" - Problems, Queries, or Complaints about orders, services, support, delivery, returns, or any issues

Classification Guidelines:
- Product Information: "What's the price of iPhone 15?", "Do you have wireless headphones?", "What colors are available?", "What's the warranty?"
- PQR: "My order is delayed", "I want to return this item", "The product is defective", "I need customer support"

Instructions:
1. Analyze the message content carefully
2. Focus on the primary intent of the message
3. Classify based on the main subject matter
4. Be decisive and confident in your classification
5. Provide a confidence score between 0.0 and 1.0
6. Extract relevant keywords if requested

Always respond with high confidence (>0.7) unless the message is genuinely ambiguous.
""".strip()

    @trace_decorator("classify_message", "classifier")
    async def classify_message(
        self,
        text: str,
        trace_id: str,
        api_key: str,
        include_reasoning: bool = False,
        extract_keywords: bool = True,
        confidence_threshold: Optional[float] = None,
    ) -> MessageClassification:
        """
        Classify a message using the PydanticAI agent.

        Args:
            text: Text to classify
            trace_id: Trace ID for observability
            api_key: Gemini API key
            include_reasoning: Whether to include reasoning
            extract_keywords: Whether to extract keywords
            confidence_threshold: Override confidence threshold

        Returns:
            MessageClassification result

        Raises:
            ClassificationError: If classification fails
        """
        start_time = time.time()

        # Sanitize input text
        sanitized_text = sanitize_text(text, max_length=10000)

        # Create dependencies
        deps = ClassificationDependencies(
            api_key=api_key,
            trace_id=trace_id,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            confidence_threshold=confidence_threshold or self.config.confidence_threshold,
        )

        try:
            # Run classification
            result = await self._agent.run(
                user_prompt=f"Classify this message: {sanitized_text}", deps=deps
            )

            # Extract keywords if requested
            keywords = []
            if extract_keywords:
                from shared.utils import extract_keywords as extract_keywords_func
                keywords = extract_keywords_func(sanitized_text)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create classification result
            classification = MessageClassification(
                text=sanitized_text,
                label=result.data.label,
                confidence=result.data.confidence,
                model_used=self.config.model_name,
                processing_time=processing_time,
                text_length=len(sanitized_text),
                keywords=keywords,
                reasoning=result.data.reasoning if include_reasoning else None,
            )

            # Update metrics
            self._update_metrics(classification)

            # Log successful classification
            logger.info(
                "Message classified successfully",
                text_length=len(sanitized_text),
                label=classification.label,
                confidence=classification.confidence,
                processing_time=processing_time,
                trace_id=trace_id,
            )

            # Enhanced observability - trace LLM interaction
            trace_llm_interaction(
                agent_name="classifier",
                model_name=self.config.model_name,
                prompt=f"Classify this message: {sanitized_text}",
                response=f"Label: {classification.label}, Confidence: {classification.confidence}",
                latency_ms=processing_time * 1000,  # Convert to milliseconds
                classification_label=classification.label,
                confidence_score=classification.confidence,
                token_count_input=len(sanitized_text.split()),  # Approximate token count
                token_count_output=len(str(result.data).split()),  # Approximate token count
                metadata={
                    "text_length": len(sanitized_text),
                    "keywords": keywords,
                    "trace_id": trace_id,
                    "include_reasoning": include_reasoning,
                }
            )

            return classification

        except Exception as e:
            # Handle classification error
            processing_time = time.time() - start_time

            logger.error(
                "Classification failed",
                error=str(e),
                text_length=len(sanitized_text),
                processing_time=processing_time,
                trace_id=trace_id,
            )

            # Update error metrics
            self.metrics.error_count += 1

            # Create classification error for logging
            error_info = ClassificationError(
                error_code="CLASSIFICATION_FAILED",
                error_message=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
                input_text=sanitized_text[:1000],  # Truncate for logging
            )
            
            # Log the error details
            logger.error("Classification error details", error_details=error_info.model_dump())

            # Enhanced observability - trace failed operation
            trace_agent_operation(
                agent_name="classifier",
                operation_name="classify_message",
                trace_id=trace_id,
                status="failed",
                duration=processing_time * 1000,  # Convert to milliseconds
                error=str(e),
                metadata={
                    "text_length": len(sanitized_text),
                    "model_name": self.config.model_name,
                    "error_type": type(e).__name__,
                }
            )

            # Raise simple RuntimeError
            raise RuntimeError(f"Classification failed: {str(e)}")

    async def classify_request(
        self, request: ClassificationRequest, trace_id: str, api_key: str
    ) -> ClassificationResponse:
        """
        Classify a message from a classification request.

        Args:
            request: Classification request
            trace_id: Trace ID for observability
            api_key: Gemini API key

        Returns:
            ClassificationResponse with result
        """
        request_id = generate_message_id()

        try:
            # Perform classification
            classification = await self.classify_message(
                text=request.text,
                trace_id=trace_id,
                api_key=api_key,
                include_reasoning=request.include_reasoning,
                extract_keywords=request.extract_keywords,
                confidence_threshold=request.confidence_threshold,
            )

            # Create successful response
            response = ClassificationResponse(
                request_id=request_id, classification=classification, success=True
            )

            # Check for warnings
            warnings = []
            if classification.confidence < self.config.confidence_threshold:
                warnings.append(f"Low confidence classification: {classification.confidence:.2f}")

            response.warnings = warnings

            return response

        except RuntimeError as e:
            # Create error response
            return ClassificationResponse(
                request_id=request_id,
                classification=MessageClassification(
                    text=request.text,
                    label=ClassificationLabel.PQR,  # Default fallback
                    confidence=0.0,
                    model_used=self.config.model_name,
                    text_length=len(request.text),
                ),
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                "Unexpected error in classification request",
                error=str(e),
                request_id=request_id,
                trace_id=trace_id,
            )

            return ClassificationResponse(
                request_id=request_id,
                classification=MessageClassification(
                    text=request.text,
                    label=ClassificationLabel.PQR,  # Default fallback
                    confidence=0.0,
                    model_used=self.config.model_name,
                    text_length=len(request.text),
                ),
                success=False,
                error_message="Internal classification error",
            )

    def _update_metrics(self, classification: MessageClassification):
        """Update classification metrics."""
        if not self.config.enable_metrics:
            return

        self.metrics.total_classifications += 1

        if classification.label == ClassificationLabel.PRODUCT_INFORMATION:
            self.metrics.product_information_count += 1
        elif classification.label == ClassificationLabel.PQR:
            self.metrics.pqr_count += 1

        # Update average confidence
        current_avg = self.metrics.average_confidence
        total = self.metrics.total_classifications
        self.metrics.average_confidence = (
            current_avg * (total - 1) + classification.confidence
        ) / total

        # Update average processing time
        if classification.processing_time:
            current_avg_time = self.metrics.average_processing_time
            self.metrics.average_processing_time = (
                current_avg_time * (total - 1) + classification.processing_time
            ) / total

        # Update low confidence count
        if classification.confidence < self.config.confidence_threshold:
            self.metrics.low_confidence_count += 1

    def get_metrics(self) -> ClassificationMetrics:
        """Get current classification metrics."""
        return self.metrics

    def reset_metrics(self):
        """Reset classification metrics."""
        self.metrics = ClassificationMetrics()
        logger.info("Classification metrics reset")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the classifier agent.

        Returns:
            Health check results
        """
        try:
            # Test classification with a simple message
            test_message = "What's the price of this product?"
            test_trace_id = "health_check_" + str(int(time.time()))

            # This would normally use a real API key, but for health check
            # we'll use a dummy key and expect it to fail gracefully
            test_deps = ClassificationDependencies(
                api_key="dummy_key_for_health_check",
                trace_id=test_trace_id,
                model_name=self.config.model_name,
            )

            # The health check doesn't actually call the model
            # but verifies the agent configuration is valid
            agent_configured = self._agent is not None

            return {
                "status": "healthy" if agent_configured else "unhealthy",
                "agent_configured": agent_configured,
                "model_name": self.config.model_name,
                "confidence_threshold": self.config.confidence_threshold,
                "total_classifications": self.metrics.total_classifications,
                "error_count": self.metrics.error_count,
                "error_rate": self.metrics.error_rate,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def update_config(self, config: ClassificationConfig):
        """
        Update classifier configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self._agent = self._create_agent()
        logger.info("Classifier configuration updated")


# Global classifier agent instance
classifier_agent = ClassifierAgent()


async def classify_message_async(
    text: str,
    trace_id: str,
    api_key: str,
    include_reasoning: bool = False,
    extract_keywords: bool = True,
    confidence_threshold: Optional[float] = None,
) -> MessageClassification:
    """
    Async function for message classification.

    Args:
        text: Text to classify
        trace_id: Trace ID for observability
        api_key: Gemini API key
        include_reasoning: Whether to include reasoning
        extract_keywords: Whether to extract keywords
        confidence_threshold: Override confidence threshold

    Returns:
        MessageClassification result
    """
    return await classifier_agent.classify_message(
        text=text,
        trace_id=trace_id,
        api_key=api_key,
        include_reasoning=include_reasoning,
        extract_keywords=extract_keywords,
        confidence_threshold=confidence_threshold,
    )


async def classify_request_async(
    request: ClassificationRequest, trace_id: str, api_key: str
) -> ClassificationResponse:
    """
    Async function for classification request processing.

    Args:
        request: Classification request
        trace_id: Trace ID for observability
        api_key: Gemini API key

    Returns:
        ClassificationResponse with result
    """
    return await classifier_agent.classify_request(
        request=request, trace_id=trace_id, api_key=api_key
    )


def get_classifier_metrics() -> ClassificationMetrics:
    """Get current classifier metrics."""
    return classifier_agent.get_metrics()


def reset_classifier_metrics():
    """Reset classifier metrics."""
    classifier_agent.reset_metrics()


async def classifier_health_check() -> Dict[str, Any]:
    """Perform classifier health check."""
    return await classifier_agent.health_check()


# Export commonly used functions
__all__ = [
    "ClassifierAgent",
    "classifier_agent",
    "classify_message_async",
    "classify_request_async",
    "get_classifier_metrics",
    "reset_classifier_metrics",
    "classifier_health_check",
]
