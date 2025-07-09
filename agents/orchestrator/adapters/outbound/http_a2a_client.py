"""
HTTP A2A Client for Orchestrator.

This module provides an HTTP client for A2A protocol communication
between the orchestrator and classifier agents.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from shared.a2a_protocol import (
    ClassificationRequest,
    ClassificationResponse,
    parse_a2a_message,
)
from shared.observability import get_logger, trace_decorator, trace_a2a_message
from shared.utils import AsyncHTTPClient, RetryConfig, TimeoutConfig
from config.settings import settings


# Configure logger
logger = get_logger(__name__)


class A2AClientConfig(BaseModel):
    """Configuration for A2A HTTP client."""

    classifier_url: str = Field(..., description="URL of the classifier service")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    retries: int = Field(default=3, ge=0, le=10, description="Number of retry attempts")

    # HTTP Configuration
    connect_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Connection timeout in seconds"
    )
    read_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Read timeout in seconds"
    )

    # Headers
    user_agent: str = Field(default="WhatsApp-Orchestrator/1.0.0", description="User agent string")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class A2AHttpClient:
    """
    HTTP client for A2A protocol communication.

    This client handles communication between the orchestrator and classifier
    agents using the A2A protocol over HTTP.
    """

    def __init__(
        self,
        classifier_url: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ):
        """
        Initialize the A2A HTTP client.

        Args:
            classifier_url: Optional classifier service URL
            timeout: Optional request timeout
            retries: Optional retry attempts
        """
        self.config = A2AClientConfig(
            classifier_url=classifier_url or settings.classifier_url,
            timeout=timeout or 30.0,
            retries=retries or 3,
        )

        # Configure HTTP client
        self.timeout_config = TimeoutConfig(
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
            total_timeout=self.config.timeout,
        )

        self.retry_config = RetryConfig(
            max_attempts=self.config.retries, base_delay=1.0, max_delay=10.0
        )

        logger.info(
            "A2A HTTP client initialized",
            classifier_url=self.config.classifier_url,
            timeout=self.config.timeout,
            retries=self.config.retries,
        )

    @trace_decorator("send_classification_request", "orchestrator")
    async def send_classification_request(
        self, request: ClassificationRequest, api_key: Optional[str] = None
    ) -> ClassificationResponse:
        """
        Send classification request to classifier agent.

        Args:
            request: Classification request
            api_key: Optional API key for authentication

        Returns:
            Classification response

        Raises:
            Exception: If request fails after retries
        """
        start_time = time.time()

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
            "X-Trace-Id": request.trace_id,
        }

        if api_key:
            headers["X-API-Key"] = api_key

        # Prepare URL
        url = f"{self.config.classifier_url}/api/v1/classify"

        # Trace outbound message
        trace_a2a_message(
            message_type=request.message_type,
            sender_agent=request.sender_agent,
            receiver_agent=request.receiver_agent,
            trace_id=request.trace_id,
            direction="outbound",
        )

        logger.info(
            "Sending classification request",
            url=url,
            receiver_agent=request.receiver_agent,
            text_length=len(request.text),
            trace_id=request.trace_id,
        )

        try:
            # Send request using async HTTP client
            async with AsyncHTTPClient(
                timeout_config=self.timeout_config, retry_config=self.retry_config, headers=headers
            ) as client:
                response = await client.post(url=url, json_data=request.model_dump(mode='json'))

                # Check response status
                if response.status_code != 200:
                    error_msg = f"Classification request failed with status {response.status_code}"
                    logger.error(
                        error_msg,
                        status_code=response.status_code,
                        response_text=response.text[:1000],  # Truncate for logging
                        trace_id=request.trace_id,
                    )
                    raise Exception(error_msg)

                # Parse response
                response_data = response.json()
                classification_response = parse_a2a_message(response_data)

                # Validate response type
                if not isinstance(classification_response, ClassificationResponse):
                    error_msg = f"Invalid response type: {type(classification_response)}"
                    logger.error(error_msg, trace_id=request.trace_id)
                    raise Exception(error_msg)

                # Trace inbound message
                trace_a2a_message(
                    message_type=classification_response.message_type,
                    sender_agent=classification_response.sender_agent,
                    receiver_agent=classification_response.receiver_agent,
                    trace_id=classification_response.trace_id,
                    direction="inbound",
                )

                # Log successful response
                processing_time = time.time() - start_time
                logger.info(
                    "Classification response received",
                    label=classification_response.payload.get("label"),
                    confidence=classification_response.payload.get("confidence"),
                    processing_time=processing_time,
                    trace_id=request.trace_id,
                )

                return classification_response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Classification request failed",
                error=str(e),
                url=url,
                processing_time=processing_time,
                trace_id=request.trace_id,
            )
            raise

    @trace_decorator("health_check", "orchestrator")
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the classifier service.

        Returns:
            Health check results
        """
        url = f"{self.config.classifier_url}/api/v1/health"

        try:
            async with AsyncHTTPClient(
                timeout_config=TimeoutConfig(
                    connect_timeout=5.0, read_timeout=10.0, total_timeout=15.0
                ),
                headers={"User-Agent": self.config.user_agent},
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    health_data = response.json()
                    return {
                        "status": "healthy",
                        "classifier_status": health_data.get("status", "unknown"),
                        "classifier_url": self.config.classifier_url,
                        "response_time": (
                            response.elapsed.total_seconds()
                            if hasattr(response, "elapsed")
                            else None
                        ),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "classifier_url": self.config.classifier_url,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        except Exception as e:
            logger.error("Health check failed", error=str(e), url=url)
            return {
                "status": "unhealthy",
                "error": str(e),
                "classifier_url": self.config.classifier_url,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_classifier_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from classifier service.

        Returns:
            Classifier metrics
        """
        url = f"{self.config.classifier_url}/api/v1/metrics"

        try:
            async with AsyncHTTPClient(
                timeout_config=self.timeout_config, headers={"User-Agent": self.config.user_agent}
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        "Failed to get classifier metrics",
                        status_code=response.status_code,
                        url=url,
                    )
                    return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error("Failed to get classifier metrics", error=str(e), url=url)
            return {"error": str(e)}

    async def get_classifier_status(self) -> Dict[str, Any]:
        """
        Get detailed status from classifier service.

        Returns:
            Classifier status
        """
        url = f"{self.config.classifier_url}/api/v1/status"

        try:
            async with AsyncHTTPClient(
                timeout_config=self.timeout_config, headers={"User-Agent": self.config.user_agent}
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        "Failed to get classifier status", status_code=response.status_code, url=url
                    )
                    return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error("Failed to get classifier status", error=str(e), url=url)
            return {"error": str(e)}

    def update_config(self, config: A2AClientConfig):
        """
        Update client configuration.

        Args:
            config: New configuration
        """
        self.config = config

        # Update timeout and retry configs
        self.timeout_config = TimeoutConfig(
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            total_timeout=config.timeout,
        )

        self.retry_config = RetryConfig(max_attempts=config.retries, base_delay=1.0, max_delay=10.0)

        logger.info("A2A client configuration updated")


# Create global A2A client instance
a2a_client = A2AHttpClient()


async def send_classification_request_async(
    request: ClassificationRequest, api_key: Optional[str] = None
) -> ClassificationResponse:
    """
    Async function for sending classification requests.

    Args:
        request: Classification request
        api_key: Optional API key

    Returns:
        Classification response
    """
    return await a2a_client.send_classification_request(request, api_key)


async def a2a_health_check() -> Dict[str, Any]:
    """Perform A2A client health check."""
    return await a2a_client.health_check()


async def get_classifier_metrics_async() -> Dict[str, Any]:
    """Get classifier metrics via A2A client."""
    return await a2a_client.get_classifier_metrics()


async def get_classifier_status_async() -> Dict[str, Any]:
    """Get classifier status via A2A client."""
    return await a2a_client.get_classifier_status()


# Export commonly used classes and functions
__all__ = [
    "A2AHttpClient",
    "A2AClientConfig",
    "a2a_client",
    "send_classification_request_async",
    "a2a_health_check",
    "get_classifier_metrics_async",
    "get_classifier_status_async",
]
