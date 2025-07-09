"""
CLI HTTP Client for Orchestrator Communication.

This module provides an HTTP client for the CLI to communicate with
the orchestrator service.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from shared.utils import AsyncHTTPClient, RetryConfig, TimeoutConfig, generate_trace_id
from shared.observability import get_logger
from config.settings import cli_settings


# Configure logger
logger = get_logger(__name__)


class RequestStatus(Enum):
    """HTTP request status."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class RequestResult:
    """Result of an HTTP request."""

    status: RequestStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    trace_id: Optional[str] = None


class OrchestratorClientConfig(BaseModel):
    """Configuration for orchestrator client."""

    base_url: str = Field(
        default="http://localhost:8080", description="Base URL of the orchestrator service"
    )

    # Timeout settings
    connect_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Connection timeout in seconds"
    )
    request_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds"
    )

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Base delay between retries"
    )

    # Client settings
    user_agent: str = Field(default="WhatsApp-CLI/1.0.0", description="User agent string")

    # Authentication
    api_key: Optional[str] = Field(None, description="Optional API key for authentication")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class OrchestratorClient:
    """
    HTTP client for communicating with the orchestrator service.

    This client handles all HTTP communication between the CLI and
    the orchestrator service, including error handling and retries.
    """

    def __init__(self, config: Optional[OrchestratorClientConfig] = None):
        """
        Initialize the orchestrator client.

        Args:
            config: Optional client configuration
        """
        self.config = config or OrchestratorClientConfig()

        # Configure HTTP client components
        self.timeout_config = TimeoutConfig(
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.request_timeout,
            total_timeout=self.config.request_timeout,
        )

        self.retry_config = RetryConfig(
            max_attempts=self.config.max_retries, base_delay=self.config.retry_delay, max_delay=10.0
        )

        # Default headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
        }

        if self.config.api_key:
            self.headers["X-API-Key"] = self.config.api_key

        logger.info(
            "Orchestrator client initialized",
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            retries=self.config.max_retries,
        )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> RequestResult:
        """
        Make an HTTP request to the orchestrator service.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Optional request data
            params: Optional query parameters
            trace_id: Optional trace ID

        Returns:
            Request result
        """
        if not trace_id:
            trace_id = generate_trace_id()

        # Prepare URL
        url = f"{self.config.base_url}{endpoint}"

        # Add trace ID to headers
        headers = {**self.headers, "X-Trace-Id": trace_id}

        start_time = time.time()

        try:
            async with AsyncHTTPClient(
                timeout_config=self.timeout_config, retry_config=self.retry_config, headers=headers
            ) as client:
                # Make request
                response = await client.request(
                    method=method, url=url, json_data=data, params=params
                )

                response_time = time.time() - start_time

                # Parse response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = {"content": response.text}

                # Check status
                if response.status_code == 200:
                    logger.debug(
                        "Request successful",
                        method=method,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        response_time=response_time,
                        trace_id=trace_id,
                    )

                    return RequestResult(
                        status=RequestStatus.SUCCESS,
                        data=response_data,
                        status_code=response.status_code,
                        response_time=response_time,
                        trace_id=trace_id,
                    )
                else:
                    logger.warning(
                        "Request failed",
                        method=method,
                        endpoint=endpoint,
                        status_code=response.status_code,
                        response_time=response_time,
                        trace_id=trace_id,
                    )

                    return RequestResult(
                        status=RequestStatus.ERROR,
                        error=f"HTTP {response.status_code}: {response_data.get('detail', 'Unknown error')}",
                        status_code=response.status_code,
                        response_time=response_time,
                        trace_id=trace_id,
                    )

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            logger.error(
                "Request timed out",
                method=method,
                endpoint=endpoint,
                timeout=self.config.request_timeout,
                response_time=response_time,
                trace_id=trace_id,
            )

            return RequestResult(
                status=RequestStatus.TIMEOUT,
                error=f"Request timed out after {self.config.request_timeout}s",
                response_time=response_time,
                trace_id=trace_id,
            )

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(
                "Request failed with exception",
                method=method,
                endpoint=endpoint,
                error=str(e),
                response_time=response_time,
                trace_id=trace_id,
            )

            return RequestResult(
                status=RequestStatus.ERROR,
                error=str(e),
                response_time=response_time,
                trace_id=trace_id,
            )

    async def send_message(
        self,
        user_message: str,
        user_id: str = "cli_user",
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> RequestResult:
        """
        Send a message to the orchestrator for processing.

        Args:
            user_message: Message from the user
            user_id: User identifier
            session_id: Optional session identifier
            trace_id: Optional trace ID

        Returns:
            Request result with orchestrator response
        """
        if not session_id:
            session_id = f"cli_session_{int(time.time())}"

        # Prepare request data
        request_data = {
            "user_message": user_message,
            "user_id": user_id,
            "session_id": session_id,
            "include_classification": True,
        }

        logger.info(
            "Sending message to orchestrator",
            message_length=len(user_message),
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
        )

        # Make request
        result = await self._make_request(
            method="POST",
            endpoint="/api/v1/orchestrate-direct",
            data=request_data,
            trace_id=trace_id,
        )

        if result.status == RequestStatus.SUCCESS:
            logger.info(
                "Message processed successfully",
                response_time=result.response_time,
                trace_id=trace_id,
            )
        else:
            logger.error(
                "Message processing failed",
                error=result.error,
                status=result.status,
                trace_id=trace_id,
            )

        return result

    async def get_health(self, trace_id: Optional[str] = None) -> RequestResult:
        """
        Get health status of the orchestrator service.

        Args:
            trace_id: Optional trace ID

        Returns:
            Request result with health status
        """
        return await self._make_request(method="GET", endpoint="/api/v1/health", trace_id=trace_id)

    async def get_status(self, trace_id: Optional[str] = None) -> RequestResult:
        """
        Get detailed status of the orchestrator service.

        Args:
            trace_id: Optional trace ID

        Returns:
            Request result with detailed status
        """
        return await self._make_request(method="GET", endpoint="/api/v1/status", trace_id=trace_id)

    async def get_metrics(self, trace_id: Optional[str] = None) -> RequestResult:
        """
        Get metrics from the orchestrator service.

        Args:
            trace_id: Optional trace ID

        Returns:
            Request result with metrics
        """
        return await self._make_request(method="GET", endpoint="/api/v1/metrics", trace_id=trace_id)

    async def get_conversation(
        self, user_id: str, session_id: str, trace_id: Optional[str] = None
    ) -> RequestResult:
        """
        Get conversation history.

        Args:
            user_id: User identifier
            session_id: Session identifier
            trace_id: Optional trace ID

        Returns:
            Request result with conversation history
        """
        return await self._make_request(
            method="GET",
            endpoint=f"/api/v1/conversations/{user_id}/{session_id}",
            trace_id=trace_id,
        )

    async def clear_conversation(
        self, user_id: str, session_id: str, trace_id: Optional[str] = None
    ) -> RequestResult:
        """
        Clear conversation history.

        Args:
            user_id: User identifier
            session_id: Session identifier
            trace_id: Optional trace ID

        Returns:
            Request result with clear confirmation
        """
        return await self._make_request(
            method="DELETE",
            endpoint=f"/api/v1/conversations/{user_id}/{session_id}",
            trace_id=trace_id,
        )

    async def test_connection(self, trace_id: Optional[str] = None) -> bool:
        """
        Test connection to the orchestrator service.

        Args:
            trace_id: Optional trace ID

        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = await self.get_health(trace_id)
            return result.status == RequestStatus.SUCCESS
        except Exception:
            return False

    def update_config(self, config: OrchestratorClientConfig):
        """
        Update client configuration.

        Args:
            config: New configuration
        """
        self.config = config

        # Update timeout and retry configs
        self.timeout_config = TimeoutConfig(
            connect_timeout=config.connect_timeout,
            read_timeout=config.request_timeout,
            total_timeout=config.request_timeout,
        )

        self.retry_config = RetryConfig(
            max_attempts=config.max_retries, base_delay=config.retry_delay, max_delay=10.0
        )

        # Update headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": config.user_agent,
            "Accept": "application/json",
        }

        if config.api_key:
            self.headers["X-API-Key"] = config.api_key

        logger.info("Orchestrator client configuration updated")


def create_orchestrator_client(
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    retries: Optional[int] = None,
    api_key: Optional[str] = None,
) -> OrchestratorClient:
    """
    Create an orchestrator client with the specified configuration.

    Args:
        base_url: Optional base URL override
        timeout: Optional timeout override
        retries: Optional retries override
        api_key: Optional API key

    Returns:
        Configured orchestrator client
    """
    config = OrchestratorClientConfig(
        base_url=base_url or "http://localhost:8080",
        request_timeout=timeout or cli_settings.http_timeout,
        max_retries=retries or cli_settings.http_retries,
        api_key=api_key,
    )

    return OrchestratorClient(config)


# Global client instance
default_client = create_orchestrator_client()


async def send_message_async(
    user_message: str,
    user_id: str = "cli_user",
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> RequestResult:
    """
    Send a message using the default client.

    Args:
        user_message: Message from the user
        user_id: User identifier
        session_id: Optional session identifier
        trace_id: Optional trace ID

    Returns:
        Request result
    """
    return await default_client.send_message(
        user_message=user_message, user_id=user_id, session_id=session_id, trace_id=trace_id
    )


async def get_health_async(trace_id: Optional[str] = None) -> RequestResult:
    """Get health status using the default client."""
    return await default_client.get_health(trace_id)


async def get_status_async(trace_id: Optional[str] = None) -> RequestResult:
    """Get detailed status using the default client."""
    return await default_client.get_status(trace_id)


async def test_connection_async(trace_id: Optional[str] = None) -> bool:
    """Test connection using the default client."""
    return await default_client.test_connection(trace_id)


# Export commonly used classes and functions
__all__ = [
    "OrchestratorClient",
    "OrchestratorClientConfig",
    "RequestResult",
    "RequestStatus",
    "create_orchestrator_client",
    "default_client",
    "send_message_async",
    "get_health_async",
    "get_status_async",
    "test_connection_async",
]
