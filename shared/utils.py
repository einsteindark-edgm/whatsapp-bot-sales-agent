"""
Common Utilities for the Multi-Agent WhatsApp Sales Assistant.

This module provides shared utility functions and classes used across
the entire system.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
import time
import hashlib
import re
from pathlib import Path
import os
from dataclasses import dataclass
from enum import Enum
import httpx
from pydantic import BaseModel, Field

T = TypeVar("T")
R = TypeVar("R")


class RetryStrategy(str, Enum):
    """Retry strategy options."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"


class TimeoutConfig(BaseModel):
    """Configuration for timeout settings."""

    connect_timeout: float = Field(default=10.0, ge=0.1, le=60.0)
    read_timeout: float = Field(default=30.0, ge=0.1, le=300.0)
    write_timeout: float = Field(default=30.0, ge=0.1, le=300.0)
    total_timeout: float = Field(default=60.0, ge=0.1, le=600.0)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL_BACKOFF)
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0)
    backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    jitter: bool = Field(default=True, description="Add randomness to delay")


@dataclass
class TimestampedResult:
    """Result with timestamp information."""

    data: Any
    timestamp: datetime
    duration: float
    success: bool
    error: Optional[str] = None


class AsyncHTTPClient:
    """
    Async HTTP client with retry and timeout support.

    Provides a robust HTTP client for A2A communication with automatic
    retries, timeouts, and error handling.
    """

    def __init__(
        self,
        timeout_config: Optional[TimeoutConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.timeout_config = timeout_config or TimeoutConfig()
        self.retry_config = retry_config or RetryConfig()
        self.default_headers = headers or {}
        self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = httpx.Timeout(
            connect=self.timeout_config.connect_timeout,
            read=self.timeout_config.read_timeout,
            write=self.timeout_config.write_timeout,
            pool=self.timeout_config.total_timeout,
        )

        self._client = httpx.AsyncClient(timeout=timeout, headers=self.default_headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.retry_config.base_delay
        elif self.retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.retry_config.base_delay * attempt
        else:  # EXPONENTIAL_BACKOFF
            delay = self.retry_config.base_delay * (
                self.retry_config.backoff_factor ** (attempt - 1)
            )

        # Apply max delay limit
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter if enabled
        if self.retry_config.jitter:
            import random

            delay += random.uniform(0, delay * 0.1)

        return delay

    async def _should_retry(
        self, response: Optional[httpx.Response], exception: Optional[Exception]
    ) -> bool:
        """Determine if request should be retried."""
        if exception:
            # Retry on connection errors, timeouts, etc.
            return isinstance(
                exception, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)
            )

        if response:
            # Retry on server errors (5xx) and some client errors
            return response.status_code >= 500 or response.status_code in [408, 429]

        return False

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional headers
            json_data: Optional JSON data
            data: Optional request data
            params: Optional query parameters

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: On request failure after retries
        """
        if not self._client:
            raise RuntimeError("HTTP client not initialized. Use async context manager.")

        merged_headers = {**self.default_headers, **(headers or {})}
        last_exception = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=merged_headers,
                    json=json_data,
                    data=data,
                    params=params,
                )

                # Check if we should retry based on response
                if not await self._should_retry(response, None):
                    return response

                last_exception = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )

            except Exception as e:
                last_exception = e

                # Check if we should retry based on exception
                if not await self._should_retry(None, e):
                    raise

            # Don't delay after the last attempt
            if attempt < self.retry_config.max_attempts:
                delay = await self._calculate_delay(attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_exception or httpx.HTTPError("All retry attempts failed")

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.

    Prevents cascading failures by monitoring error rates and temporarily
    disabling failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def _is_recovery_timeout_elapsed(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout

    async def call(self, func: Callable, *args, **kwargs):
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._is_recovery_timeout_elapsed():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Success - reset failure count
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return str(uuid.uuid4())


def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp as ISO string."""
    return timestamp.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string."""
    return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))


def calculate_duration(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    """Calculate duration in seconds between timestamps."""
    if end_time is None:
        end_time = get_current_timestamp()
    return (end_time - start_time).total_seconds()


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_phone_number(phone: str) -> bool:
    """Validate phone number format (basic validation)."""
    # Remove common formatting characters
    cleaned = re.sub(r"[\s\-\(\)\+]", "", phone)
    # Check if it's all digits and reasonable length
    return cleaned.isdigit() and 10 <= len(cleaned) <= 15


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """
    Sanitize text input for safe processing.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Limit length
    if len(text) > max_length:
        text = text[:max_length]

    # Strip leading/trailing whitespace
    return text.strip()


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for logging/storage.

    Args:
        data: Sensitive data to hash

    Returns:
        SHA256 hash of the data
    """
    return hashlib.sha256(data.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text.

    Args:
        text: Input text
        min_length: Minimum keyword length

    Returns:
        List of keywords
    """
    # Simple keyword extraction - can be enhanced
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if len(word) >= min_length]


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with default fallback.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: Any = None) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Object to serialize
        default: Default serializer for non-serializable objects

    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, default=default, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def get_env_var(name: str, default: str = None, required: bool = False) -> str:
    """
    Get environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not found
        required: Whether the variable is required

    Returns:
        Environment variable value

    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{name}' not found")

    return value


def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """
    Create directory with parent directories.

    Args:
        path: Directory path
        exist_ok: Whether to ignore if directory exists

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=exist_ok)
    return path_obj


def read_file_safely(file_path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
    """
    Safely read file content.

    Args:
        file_path: File path
        encoding: File encoding

    Returns:
        File content or None if failed
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except (IOError, OSError, UnicodeDecodeError):
        return None


def write_file_safely(
    file_path: Union[str, Path], content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> bool:
    """
    Safely write content to file.

    Args:
        file_path: File path
        content: Content to write
        encoding: File encoding
        create_dirs: Whether to create parent directories

    Returns:
        True if successful, False otherwise
    """
    try:
        path_obj = Path(file_path)

        if create_dirs:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding=encoding) as f:
            f.write(content)

        return True
    except (IOError, OSError, UnicodeEncodeError):
        return False


async def run_with_timeout(coro: Callable, timeout: float, default: Any = None) -> Any:
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout

    Returns:
        Coroutine result or default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


# Common constants
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_FACTOR = 2.0
MAX_MESSAGE_LENGTH = 10000
MAX_TRACE_ID_LENGTH = 36
MAX_AGENT_NAME_LENGTH = 50

# Export commonly used functions
__all__ = [
    "AsyncHTTPClient",
    "CircuitBreaker",
    "TimeoutConfig",
    "RetryConfig",
    "TimestampedResult",
    "generate_trace_id",
    "generate_message_id",
    "generate_correlation_id",
    "get_current_timestamp",
    "format_timestamp",
    "parse_timestamp",
    "calculate_duration",
    "validate_email",
    "validate_phone_number",
    "sanitize_text",
    "hash_sensitive_data",
    "truncate_text",
    "extract_keywords",
    "format_file_size",
    "safe_json_loads",
    "safe_json_dumps",
    "deep_merge_dicts",
    "get_env_var",
    "create_directory",
    "read_file_safely",
    "write_file_safely",
    "run_with_timeout",
]
