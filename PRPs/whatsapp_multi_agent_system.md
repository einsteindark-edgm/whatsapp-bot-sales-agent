name: "Multi-Agent WhatsApp Sales Assistant Proof-of-Concept"
description: |

## Purpose
Build a local proof-of-concept for a multi-agent WhatsApp sales assistant that demonstrates advanced orchestration patterns, A2A protocol communication, and microservice architecture using Google ADK and PydanticAI.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready multi-agent system that demonstrates:
- CLI front-end with async messaging and loading spinner
- Workflow Orchestrator Agent using Google ADK in Docker container
- Classifier Agent using PydanticAI + Gemini Flash 2.5 in separate container
- A2A protocol for agent communication with trace IDs
- FastAPI for HTTP communication between containers
- >90% unit test coverage using pytest, ADK's AgentEvaluator, and PydanticAI's TestModel

## Why
- **Business value**: Demonstrates scalable multi-agent architecture for WhatsApp commerce
- **Integration**: Showcases Google ADK orchestration with PydanticAI classification
- **Problems solved**: Provides foundation for human-like sales assistant with proper handoff protocols

## What
A complete containerized system where:
- Users interact via CLI with async messaging and spinner feedback
- Workflow Orchestrator (ADK) coordinates the conversation flow
- Classifier Agent (PydanticAI) labels messages as `product_information` or `PQR`
- Agents communicate via A2A protocol with structured payloads
- All components run in separate Docker containers with FastAPI interfaces

### Success Criteria
- [ ] CLI provides async messaging with loading spinner during agent communication
- [ ] Workflow Orchestrator Agent successfully orchestrates conversation flow
- [ ] Classifier Agent accurately labels messages using Gemini Flash 2.5
- [ ] A2A protocol enables seamless agent-to-agent communication with trace IDs
- [ ] FastAPI interfaces enable HTTP communication between containers
- [ ] >90% unit test coverage achieved with comprehensive test suite
- [ ] All agents run in separate Docker containers with proper networking
- [ ] System handles error cases gracefully with proper logging

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://google.github.io/adk-docs/agents/custom-agents/
  why: ADK custom agent patterns for workflow orchestration
  critical: StoryFlowAgent pattern for complex orchestration logic
  
- url: https://ai.pydantic.dev/
  why: PydanticAI agent creation, dependency injection, and testing
  critical: Agent with deps_type pattern and TestModel for testing
  
- url: https://fastapi.tiangolo.com/
  why: FastAPI request/response patterns and Docker deployment
  critical: Async handlers and Pydantic integration
  
- url: https://google.github.io/adk-docs/evaluate/
  why: AgentEvaluator usage and pytest integration
  critical: Programmatic testing with pytest.mark.anyio
  
- url: https://stackoverflow.com/q/48854567
  why: Async CLI spinner pattern for user feedback
  critical: Threading approach for non-blocking UI

- file: examples/adk_orchestrator_snippet.py
  why: Complete ADK custom agent implementation pattern
  critical: BaseAgent inheritance and _run_async_impl method
  
- file: examples/pydanticai_bank_support_snippet.py
  why: PydanticAI agent with deps_type and output_type patterns
  critical: Agent creation with structured outputs
  
- file: examples/pydanticai_gemini_model.py
  why: Gemini Flash 2.5 integration pattern
  critical: GeminiModel initialization with google-gla provider
  
- file: examples/fastapi_request_body.py
  why: FastAPI async handler patterns with Pydantic schemas
  critical: Request/response validation
  
- file: examples/cli_spinner.py
  why: Threading-based async spinner implementation
  critical: Non-blocking UI during async operations
  
- file: examples/adk_pytest_evaluate.py
  why: ADK evaluation integration with pytest
  critical: AgentEvaluator.evaluate async pattern
  
- file: examples/pydanticai_test_fixture.py
  why: PydanticAI testing with TestModel and capture_run_messages
  critical: Mock LLM responses for deterministic testing
```

### Current Codebase tree
```bash
.
├── CLAUDE.md
├── INITIAL.md
├── INITIAL_template.md
├── LICENSE
├── PRPs/
│   ├── EXAMPLE_multi_agent_prp.md
│   └── templates/
│       └── prp_base.md
├── README.md
└── examples/
    ├── adk_orchestrator_snippet.py
    ├── adk_pytest_evaluate.py
    ├── cli_spinner.py
    ├── fastapi_request_body.py
    ├── pydanticai_bank_support_snippet.py
    ├── pydanticai_gemini_model.py
    └── pydanticai_test_fixture.py
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
.
├── agents/
│   ├── orchestrator/
│   │   ├── __init__.py              # Package init
│   │   ├── agent.py                 # Workflow Orchestrator Agent (ADK)
│   │   ├── domain/
│   │   │   ├── __init__.py          # Domain models
│   │   │   └── models.py            # Orchestration domain models
│   │   ├── application/
│   │   │   ├── __init__.py          # Application layer
│   │   │   └── workflow_service.py  # Workflow orchestration logic
│   │   ├── ports/
│   │   │   ├── __init__.py          # Port definitions
│   │   │   └── outbound/
│   │   │       ├── __init__.py      # Outbound port definitions
│   │   │       └── a2a_client.py    # A2A protocol client interface
│   │   ├── adapters/
│   │   │   ├── __init__.py          # Adapter implementations
│   │   │   ├── inbound/
│   │   │   │   ├── __init__.py      # Inbound adapters
│   │   │   │   └── fastapi_router.py # FastAPI router for orchestrator
│   │   │   └── outbound/
│   │   │       ├── __init__.py      # Outbound adapters
│   │   │       └── http_a2a_client.py # HTTP A2A client implementation
│   │   ├── tests/
│   │   │   ├── __init__.py          # Test package
│   │   │   ├── test_agent.py        # Orchestrator agent tests
│   │   │   └── test_workflow_service.py # Workflow service tests
│   │   ├── Dockerfile               # Orchestrator container
│   │   └── requirements.txt         # Orchestrator dependencies
│   └── classifier/
│       ├── __init__.py              # Package init
│       ├── agent.py                 # Classifier Agent (PydanticAI)
│       ├── domain/
│       │   ├── __init__.py          # Domain models
│       │   └── models.py            # Classification domain models
│       ├── application/
│       │   ├── __init__.py          # Application layer
│       │   └── classification_service.py # Classification logic
│       ├── ports/
│       │   ├── __init__.py          # Port definitions
│       │   └── outbound/
│       │       ├── __init__.py      # Outbound port definitions
│       │       └── llm_client.py    # LLM client interface
│       ├── adapters/
│       │   ├── __init__.py          # Adapter implementations
│       │   ├── inbound/
│       │   │   ├── __init__.py      # Inbound adapters
│       │   │   └── fastapi_router.py # FastAPI router for classifier
│       │   └── outbound/
│       │       ├── __init__.py      # Outbound adapters
│       │       └── gemini_client.py  # Gemini Flash 2.5 client
│       ├── tests/
│       │   ├── __init__.py          # Test package
│       │   ├── test_agent.py        # Classifier agent tests
│       │   └── test_classification_service.py # Classification service tests
│       ├── Dockerfile               # Classifier container
│       └── requirements.txt         # Classifier dependencies
├── shared/
│   ├── __init__.py                  # Shared package init
│   ├── a2a_protocol.py              # A2A protocol models and utilities
│   ├── observability.py             # Pydantic Logfire integration
│   └── utils.py                     # Common utilities
├── cli/
│   ├── __init__.py                  # CLI package init
│   ├── main.py                      # CLI entry point
│   ├── spinner.py                   # Async spinner implementation
│   └── client.py                    # HTTP client for agent communication
├── config/
│   ├── __init__.py                  # Config package init
│   ├── settings.py                  # Environment and config management
│   └── logging.yaml                 # Logging configuration
├── tests/
│   ├── __init__.py                  # Test package init
│   ├── integration/
│   │   ├── __init__.py              # Integration test package
│   │   ├── test_a2a_communication.py # A2A protocol integration tests
│   │   └── test_end_to_end.py       # End-to-end system tests
│   └── fixtures/
│       ├── __init__.py              # Test fixtures
│       └── test_data.py             # Test data and fixtures
├── docker-compose.yml               # Multi-container orchestration
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore patterns
├── Makefile                         # Build automation
├── requirements.txt                 # Root dependencies
└── README.md                        # Comprehensive documentation
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Google ADK requires async throughout - BaseAgent._run_async_impl must be async
# CRITICAL: PydanticAI Agent requires deps_type for dependency injection
# CRITICAL: Gemini Flash 2.5 uses 'google-gla:gemini-2.0-flash' model identifier
# CRITICAL: FastAPI async handlers require proper async/await for all operations
# CRITICAL: A2A protocol payloads must include trace-ID for observability
# CRITICAL: Docker containers need proper networking for inter-container communication
# CRITICAL: ADK AgentEvaluator requires specific JSON test format
# CRITICAL: PydanticAI TestModel requires models.ALLOW_MODEL_REQUESTS = False
# CRITICAL: CLI spinner must use threading to avoid blocking async operations
# CRITICAL: All tests must use pytest.mark.anyio for async test compatibility
# CRITICAL: Docker images should use python:3.12-slim for size optimization
# CRITICAL: Pydantic Logfire requires proper configuration for observability
```

## Implementation Blueprint

### Data models and structure

```python
# shared/a2a_protocol.py - A2A protocol models
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class A2AMessage(BaseModel):
    """Base A2A protocol message structure."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str = Field(..., description="Agent name sending the message")
    receiver_agent: str = Field(..., description="Agent name receiving the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(..., description="Message payload")
    message_type: str = Field(..., description="Type of message")

class ClassificationRequest(A2AMessage):
    """Request for message classification."""
    message_type: str = Field(default="classification_request")
    payload: Dict[str, Any] = Field(..., description="Must contain 'text' key")

class ClassificationResponse(A2AMessage):
    """Response with classification result."""
    message_type: str = Field(default="classification_response")
    payload: Dict[str, Any] = Field(..., description="Must contain 'text' and 'label' keys")

# agents/classifier/domain/models.py - Classification domain models
from pydantic import BaseModel, Field
from typing import Literal

class MessageClassification(BaseModel):
    """Classification result for a message."""
    text: str = Field(..., description="Original message text")
    label: Literal["product_information", "PQR"] = Field(..., description="Classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")

class ClassificationDependencies(BaseModel):
    """Dependencies for classification agent."""
    model_name: str = Field(default="google-gla:gemini-2.0-flash")
    api_key: str = Field(..., description="Gemini API key")
    trace_id: str = Field(..., description="Request trace ID")
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Setup Project Structure and Configuration
CREATE config/settings.py:
  - PATTERN: Use pydantic-settings for environment management
  - Load all required API keys and configuration
  - Validate required environment variables
  - Include Docker networking configuration

CREATE .env.example:
  - Include all required environment variables with descriptions
  - GEMINI_API_KEY, ORCHESTRATOR_PORT, CLASSIFIER_PORT, CLI_HOST
  - Follow CLAUDE.md environment variable patterns

CREATE docker-compose.yml:
  - PATTERN: Multi-container setup with proper networking
  - Define services for orchestrator, classifier, and shared network
  - Include environment variable injection from .env

Task 2: Implement Shared A2A Protocol
CREATE shared/a2a_protocol.py:
  - PATTERN: Pydantic models for type-safe communication
  - Define A2AMessage base class with trace_id and observability
  - Create ClassificationRequest and ClassificationResponse models
  - Include validation for required payload fields

CREATE shared/observability.py:
  - PATTERN: Pydantic Logfire integration for tracing
  - Configure structured logging with trace IDs
  - Setup observability for A2A message flow

Task 3: Implement Classifier Agent (PydanticAI)
CREATE agents/classifier/agent.py:
  - PATTERN: Follow pydanticai_bank_support_snippet.py structure
  - Use Agent with deps_type=ClassificationDependencies
  - Configure output_type=MessageClassification
  - System prompt for product_information vs PQR classification

CREATE agents/classifier/adapters/outbound/gemini_client.py:
  - PATTERN: Follow pydanticai_gemini_model.py pattern
  - Use GeminiModel with google-gla provider
  - Configure gemini-2.0-flash model
  - Handle API key authentication

CREATE agents/classifier/adapters/inbound/fastapi_router.py:
  - PATTERN: Follow fastapi_request_body.py structure
  - Create /classify endpoint accepting A2AMessage
  - Return ClassificationResponse with trace ID
  - Handle async agent execution

Task 4: Implement Workflow Orchestrator Agent (ADK)
CREATE agents/orchestrator/agent.py:
  - PATTERN: Follow adk_orchestrator_snippet.py StoryFlowAgent pattern
  - Inherit from BaseAgent with custom _run_async_impl
  - Orchestrate conversation flow with classifier agent
  - Manage session state and conversation context

CREATE agents/orchestrator/adapters/outbound/http_a2a_client.py:
  - PATTERN: Use httpx for async HTTP client
  - Implement A2A protocol communication
  - Send ClassificationRequest to classifier agent
  - Handle response parsing and error handling

CREATE agents/orchestrator/adapters/inbound/fastapi_router.py:
  - PATTERN: FastAPI router for message handling
  - Accept user messages and initiate workflow
  - Return conversation responses with classification context
  - Include proper error handling and logging

Task 5: Implement CLI Interface
CREATE cli/main.py:
  - PATTERN: Async CLI with proper session management
  - Handle user input and display conversation flow
  - Integrate with orchestrator agent via HTTP
  - Include error handling and user feedback

CREATE cli/spinner.py:
  - PATTERN: Follow cli_spinner.py threading approach
  - Implement non-blocking spinner during agent communication
  - Thread-safe start/stop operations
  - Visual feedback for async operations

CREATE cli/client.py:
  - PATTERN: HTTP client for orchestrator communication
  - Use httpx for async HTTP requests
  - Handle connection errors and timeouts
  - Include proper JSON serialization

Task 6: Create Docker Containers
CREATE agents/orchestrator/Dockerfile:
  - PATTERN: Multi-stage build with python:3.12-slim
  - Install ADK dependencies and application code
  - Configure proper entrypoint and health checks
  - Optimize for production deployment

CREATE agents/classifier/Dockerfile:
  - PATTERN: Multi-stage build with python:3.12-slim
  - Install PydanticAI and Gemini dependencies
  - Configure proper entrypoint and health checks
  - Optimize for production deployment

Task 7: Implement Comprehensive Testing
CREATE agents/orchestrator/tests/test_agent.py:
  - PATTERN: Follow adk_pytest_evaluate.py pattern
  - Use AgentEvaluator for orchestrator testing
  - Test conversation flow and classifier integration
  - Include async test markers

CREATE agents/classifier/tests/test_agent.py:
  - PATTERN: Follow pydanticai_test_fixture.py pattern
  - Use TestModel for deterministic testing
  - Test classification accuracy and edge cases
  - Include pytest.mark.anyio for async tests

CREATE tests/integration/test_a2a_communication.py:
  - PATTERN: Integration tests for A2A protocol
  - Test end-to-end message flow between agents
  - Validate trace ID propagation and observability
  - Include container networking tests

Task 8: Create Build Automation
CREATE Makefile:
  - PATTERN: Standard make targets for Docker operations
  - Include build, run, test, and cleanup targets
  - Support for development and production environments
  - Integrate with CI/CD pipeline requirements

CREATE README.md:
  - PATTERN: Comprehensive documentation
  - Include setup instructions and architecture overview
  - API key configuration and Docker setup
  - Testing and deployment guidelines
```

### Per task pseudocode as needed

```python
# Task 3: Classifier Agent Implementation
# agents/classifier/agent.py
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from .domain.models import ClassificationDependencies, MessageClassification

classifier_agent = Agent(
    model=GeminiModel('gemini-2.0-flash', provider='google-gla'),
    deps_type=ClassificationDependencies,
    output_type=MessageClassification,
    system_prompt=(
        'You are a message classifier for WhatsApp commerce. '
        'Classify messages as either "product_information" (questions about products, '
        'pricing, availability) or "PQR" (problems, queries, complaints). '
        'Be decisive and confident in your classification.'
    ),
)

@classifier_agent.system_prompt
async def add_trace_context(ctx: RunContext[ClassificationDependencies]) -> str:
    """Add trace ID to system context for observability."""
    return f"Trace ID: {ctx.deps.trace_id}"

# Task 4: Workflow Orchestrator Implementation  
# agents/orchestrator/agent.py
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
from google.adk.events import Event

class WorkflowOrchestrator(BaseAgent):
    """Custom orchestrator for WhatsApp sales workflow."""
    
    def __init__(self, name: str, classifier_client: A2AClient):
        self.classifier_client = classifier_client
        super().__init__(name=name)
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Orchestrate conversation flow with classification."""
        # Get user message from context
        user_message = ctx.session.state.get("user_message", "")
        
        # Send to classifier via A2A protocol
        classification_request = ClassificationRequest(
            sender_agent="orchestrator",
            receiver_agent="classifier",
            payload={"text": user_message}
        )
        
        # CRITICAL: Await classifier response
        classification_response = await self.classifier_client.send_message(
            classification_request
        )
        
        # Store classification in session
        ctx.session.state["classification"] = classification_response.payload
        
        # Generate response based on classification
        if classification_response.payload["label"] == "product_information":
            response = f"I can help with product information about: {user_message}"
        else:
            response = f"I understand this is a query/complaint: {user_message}"
        
        # Yield final response event
        yield Event(
            content=response,
            author=self.name,
            is_final=True
        )

# Task 5: CLI Implementation with Spinner
# cli/main.py
import asyncio
from .spinner import SpinnerThread
from .client import OrchestratorClient

async def main():
    """Main CLI loop with async spinner."""
    client = OrchestratorClient()
    
    print("WhatsApp Sales Assistant CLI")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() == 'quit':
            break
        
        # Start spinner for visual feedback
        spinner = SpinnerThread()
        spinner.start()
        
        try:
            # CRITICAL: Await async operation
            response = await client.send_message(user_input)
            spinner.stop()
            
            print(f"Assistant: {response['content']}")
            if response.get('classification'):
                print(f"[Classification: {response['classification']['label']}]")
        
        except Exception as e:
            spinner.stop()
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration Points
```yaml
DOCKER_NETWORKING:
  - Network: whatsapp-assistant-network
  - Orchestrator: Container port 8000, exposed as 8080
  - Classifier: Container port 8001, exposed as 8081
  - Inter-container communication via container names

ENVIRONMENT_VARIABLES:
  - GEMINI_API_KEY: Required for classifier agent
  - ORCHESTRATOR_HOST: Default localhost:8080
  - CLASSIFIER_HOST: Default localhost:8081
  - LOG_LEVEL: Default INFO
  - TRACE_ENABLED: Default true

API_ENDPOINTS:
  - Orchestrator: POST /orchestrate (accepts user messages)
  - Classifier: POST /classify (accepts A2A messages)
  - Health checks: GET /health for both services

OBSERVABILITY:
  - Pydantic Logfire for structured logging
  - Trace ID propagation through A2A protocol
  - Performance metrics for agent response times
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check . --fix                    # Auto-fix style issues
mypy .                               # Type checking
black .                              # Code formatting

# Expected: No errors. If errors, READ and fix systematically.
```

### Level 2: Unit Tests
```python
# agents/classifier/tests/test_agent.py
import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import models
from ..agent import classifier_agent
from ..domain.models import ClassificationDependencies

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False  # CRITICAL: Disable real LLM calls

async def test_product_information_classification():
    """Test classification of product information messages."""
    deps = ClassificationDependencies(
        api_key="test-key",
        trace_id="test-trace-123"
    )
    
    with classifier_agent.override(model=TestModel()):
        result = await classifier_agent.run(
            "What's the price of iPhone 15?",
            deps=deps
        )
    
    assert result.data.label == "product_information"
    assert result.data.confidence > 0.7

async def test_pqr_classification():
    """Test classification of PQR messages."""
    deps = ClassificationDependencies(
        api_key="test-key", 
        trace_id="test-trace-456"
    )
    
    with classifier_agent.override(model=TestModel()):
        result = await classifier_agent.run(
            "My order is delayed and I'm not happy",
            deps=deps
        )
    
    assert result.data.label == "PQR"
    assert result.data.confidence > 0.7

# agents/orchestrator/tests/test_agent.py
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

@pytest.mark.asyncio
async def test_orchestrator_workflow():
    """Test orchestrator agent workflow."""
    await AgentEvaluator.evaluate(
        agent_module="agents.orchestrator.agent",
        eval_dataset_file_path_or_dir="tests/fixtures/orchestrator_test.json",
    )

# tests/integration/test_a2a_communication.py
import pytest
import httpx
from shared.a2a_protocol import ClassificationRequest, ClassificationResponse

@pytest.mark.asyncio
async def test_a2a_message_flow():
    """Test A2A protocol message flow between agents."""
    # Send classification request to classifier
    request = ClassificationRequest(
        sender_agent="orchestrator",
        receiver_agent="classifier",
        payload={"text": "What's the warranty on laptops?"}
    )
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8081/classify",
            json=request.model_dump()
        )
    
    assert response.status_code == 200
    result = ClassificationResponse(**response.json())
    assert result.payload["label"] in ["product_information", "PQR"]
    assert result.trace_id == request.trace_id
```

```bash
# Run tests iteratively until passing:
pytest tests/ -v --cov=agents --cov=shared --cov=cli --cov-report=term-missing

# Target: >90% coverage as specified in requirements
# If failing: Debug specific test, fix code, re-run
```

### Level 3: Integration Test
```bash
# Start the system
make build && make run

# Expected: Both containers start successfully
# Check container health
docker ps

# Test CLI interaction
python -m cli.main

# Expected interaction:
# User: What's the price of iPhone 15?
# [Processing with spinner...]
# Assistant: I can help with product information about: What's the price of iPhone 15?
# [Classification: product_information]
#
# User: My order is delayed
# [Processing with spinner...]  
# Assistant: I understand this is a query/complaint: My order is delayed
# [Classification: PQR]

# Test A2A communication directly
curl -X POST http://localhost:8080/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"message": "Do you have wireless headphones?"}'

# Expected: JSON response with classification and orchestrator response
```

## Final validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] >90% test coverage achieved: `pytest --cov=. --cov-report=term`
- [ ] No linting errors: `ruff check .`
- [ ] No type errors: `mypy .`
- [ ] Code formatted: `black .`
- [ ] Docker containers build successfully: `make build`
- [ ] Multi-container system runs: `make run`
- [ ] CLI spinner works during async operations
- [ ] A2A protocol communication successful
- [ ] Trace IDs propagate through system
- [ ] Gemini Flash 2.5 classification works
- [ ] ADK orchestrator manages workflow correctly
- [ ] Error cases handled gracefully
- [ ] Observability logging functional
- [ ] Documentation complete and accurate

---

## Anti-Patterns to Avoid
- ❌ Don't use sync functions in async agent context
- ❌ Don't skip trace ID propagation in A2A messages
- ❌ Don't hardcode API keys - use environment variables
- ❌ Don't ignore Docker networking requirements
- ❌ Don't use real LLM calls in unit tests
- ❌ Don't skip async test markers (pytest.mark.anyio)
- ❌ Don't create files >500 lines (split into modules)
- ❌ Don't ignore error handling in HTTP communication
- ❌ Don't skip health checks in Docker containers
- ❌ Don't commit sensitive credentials or keys

## Confidence Score: 9/10

High confidence due to:
- Comprehensive examples provided in /examples folder
- Clear ADK and PydanticAI documentation patterns
- Well-defined A2A protocol structure
- Proven FastAPI and Docker patterns
- Extensive testing framework with >90% coverage target
- Clear validation gates for iterative development

Minor uncertainty around:
- Docker networking configuration specifics
- Exact AgentEvaluator test file format requirements
- Pydantic Logfire integration complexity

The comprehensive context, examples, and validation loops provide strong foundation for successful one-pass implementation.