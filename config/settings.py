"""
Configuration settings for the multi-agent WhatsApp sales assistant.

Uses pydantic-settings for environment variable management with validation.
"""

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    # LLM Configuration
    gemini_api_key: str = Field(..., description="Gemini API key for classification")

    # Service Configuration
    orchestrator_host: str = Field(default="localhost", description="Orchestrator service host")
    orchestrator_port: int = Field(default=8080, description="Orchestrator service port")
    classifier_host: str = Field(default="localhost", description="Classifier service host")
    classifier_port: int = Field(default=8001, description="Classifier service port")

    # Docker Configuration
    docker_network: str = Field(
        default="whatsapp-assistant-network", description="Docker network name"
    )
    
    # Phoenix Evaluations Configuration
    phoenix_enabled: bool = Field(default=False, env="PHOENIX_ENABLED", description="Enable Phoenix evaluations")
    phoenix_eval_model: str = Field(default="gpt-4", env="PHOENIX_EVAL_MODEL", description="Model for evaluations")
    phoenix_eval_concurrency: int = Field(default=4, env="PHOENIX_EVAL_CONCURRENCY", description="Evaluation concurrency")
    
    # Cost Threshold Configuration
    cost_alert_per_call_usd: float = Field(default=0.05, env="COST_ALERT_PER_CALL_USD", description="Cost alert threshold per LLM call")
    cost_alert_per_session_usd: float = Field(default=1.0, env="COST_ALERT_PER_SESSION_USD", description="Cost alert threshold per session")
    
    # Trace Configuration
    trace_id_header: str = Field(default="X-Trace-ID", env="TRACE_ID_HEADER", description="HTTP header for trace ID")

    # Observability Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    trace_enabled: bool = Field(default=True, description="Enable tracing")
    
    # Logfire Configuration
    logfire_token: str = Field(default="", description="Logfire authentication token")
    logfire_project_name: str = Field(default="whatsapp-sales-assistant", description="Logfire project name")
    logfire_environment: str = Field(default="development", description="Logfire environment")
    
    # Arize AX Configuration
    arize_api_key: str = Field(default="", description="Arize AI API key")
    arize_space_key: str = Field(default="", description="Arize AI space key")
    arize_model_id: str = Field(default="whatsapp-chatcommerce-bot", description="Arize model identifier")
    arize_model_version: str = Field(default="1.0.0", description="Arize model version")
    
    # OpenTelemetry Configuration
    otel_service_name: str = Field(default="whatsapp-sales-assistant", description="OpenTelemetry service name")
    otel_exporter_endpoint: str = Field(default="", description="OTLP exporter endpoint")
    otel_headers: str = Field(default="", description="OTLP exporter headers")

    # Model Configuration
    model_name: str = Field(default="google-gla:gemini-2.0-flash", description="Gemini model name")

    # WhatsApp Business API Configuration
    whatsapp_access_token: str = Field(default="", description="WhatsApp Business API access token")
    whatsapp_phone_number_id: str = Field(default="", description="WhatsApp phone number ID")
    whatsapp_business_account_id: str = Field(default="", description="WhatsApp Business Account ID")
    whatsapp_webhook_verify_token: str = Field(default="", description="Webhook verification token")
    whatsapp_app_secret: str = Field(default="", description="WhatsApp app secret for webhook signature")
    whatsapp_api_version: str = Field(default="v18.0", description="WhatsApp API version")
    whatsapp_api_base_url: str = Field(default="https://graph.facebook.com", description="WhatsApp API base URL")

    # FastAPI Configuration
    api_title: str = Field(default="WhatsApp Sales Assistant", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")

    # CLI Configuration
    cli_host: str = Field(
        default="localhost", description="CLI host for connecting to orchestrator"
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @validator("orchestrator_port", "classifier_port")
    def validate_ports(cls, v):
        """Validate port numbers are in valid range."""
        if not (1024 <= v <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        return v

    @validator("gemini_api_key")
    def validate_gemini_api_key(cls, v):
        """Validate Gemini API key format."""
        if not v or len(v) < 10:
            raise ValueError("GEMINI_API_KEY must be provided and valid")
        return v

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""

        # Environment variable mapping
        fields = {
            "gemini_api_key": {"env": "GEMINI_API_KEY"},
            "orchestrator_host": {"env": "ORCHESTRATOR_HOST"},
            "orchestrator_port": {"env": "ORCHESTRATOR_PORT"},
            "classifier_host": {"env": "CLASSIFIER_HOST"},
            "classifier_port": {"env": "CLASSIFIER_PORT"},
            "docker_network": {"env": "DOCKER_NETWORK"},
            "log_level": {"env": "LOG_LEVEL"},
            "trace_enabled": {"env": "TRACE_ENABLED"},
            "logfire_token": {"env": "LOGFIRE_TOKEN"},
            "logfire_project_name": {"env": "LOGFIRE_PROJECT_NAME"},
            "logfire_environment": {"env": "LOGFIRE_ENVIRONMENT"},
            "arize_api_key": {"env": "ARIZE_API_KEY"},
            "arize_space_key": {"env": "ARIZE_SPACE_KEY"},
            "arize_model_id": {"env": "ARIZE_MODEL_ID"},
            "arize_model_version": {"env": "ARIZE_MODEL_VERSION"},
            "otel_service_name": {"env": "OTEL_SERVICE_NAME"},
            "otel_exporter_endpoint": {"env": "OTEL_EXPORTER_ENDPOINT"},
            "otel_headers": {"env": "OTEL_HEADERS"},
            "model_name": {"env": "MODEL_NAME"},
            "whatsapp_access_token": {"env": "WHATSAPP_ACCESS_TOKEN"},
            "whatsapp_phone_number_id": {"env": "WHATSAPP_PHONE_NUMBER_ID"},
            "whatsapp_business_account_id": {"env": "WHATSAPP_BUSINESS_ACCOUNT_ID"},
            "whatsapp_webhook_verify_token": {"env": "WHATSAPP_WEBHOOK_VERIFY_TOKEN"},
            "whatsapp_app_secret": {"env": "WHATSAPP_APP_SECRET"},
            "whatsapp_api_version": {"env": "WHATSAPP_API_VERSION"},
            "whatsapp_api_base_url": {"env": "WHATSAPP_API_BASE_URL"},
            "api_title": {"env": "API_TITLE"},
            "api_version": {"env": "API_VERSION"},
            "cli_host": {"env": "CLI_HOST"},
        }

    @property
    def orchestrator_url(self) -> str:
        """Get full orchestrator URL."""
        return f"http://{self.orchestrator_host}:{self.orchestrator_port}"

    @property
    def classifier_url(self) -> str:
        """Get full classifier URL."""
        return f"http://{self.classifier_host}:{self.classifier_port}"

    @property
    def is_docker_environment(self) -> bool:
        """Check if running in Docker environment."""
        return os.getenv("DOCKER_CONTAINER") == "true"


class OrchestratorSettings(BaseModel):
    """Orchestrator-specific settings."""

    app_name: str = Field(default="orchestrator", description="Application name")
    user_id: str = Field(default="user-001", description="Default user ID")
    session_id: str = Field(default="session-001", description="Default session ID")

    # ADK Configuration
    adk_model: str = Field(default="gemini-2.0-flash", description="ADK model")

    # A2A Configuration
    a2a_timeout: int = Field(default=30, description="A2A request timeout in seconds")
    a2a_retries: int = Field(default=3, description="A2A request retries")


class ClassifierSettings(BaseModel):
    """Classifier-specific settings."""

    app_name: str = Field(default="classifier", description="Application name")

    # Classification Configuration
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")

    # Gemini Configuration
    gemini_temperature: float = Field(
        default=0.0, description="Gemini temperature for consistent classification"
    )
    gemini_max_tokens: int = Field(
        default=100, description="Maximum tokens for classification response"
    )


class CLISettings(BaseModel):
    """CLI-specific settings."""

    app_name: str = Field(
        default="WhatsApp Sales Assistant CLI", description="CLI application name"
    )

    # Spinner Configuration
    spinner_delay: float = Field(default=0.1, description="Spinner animation delay")
    spinner_chars: str = Field(default="|/-\\", description="Spinner characters")

    # HTTP Configuration
    http_timeout: int = Field(default=30, description="HTTP request timeout")
    http_retries: int = Field(default=3, description="HTTP request retries")


# Global settings instance
settings = Settings()
orchestrator_settings = OrchestratorSettings()
classifier_settings = ClassifierSettings()
cli_settings = CLISettings()
