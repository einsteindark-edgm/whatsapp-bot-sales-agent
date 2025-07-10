# PRP: End-to-End Observability Implementation for ChatCommerce Bot

## Context
This PRP implements comprehensive observability for the WhatsApp ChatCommerce Bot, including global trace propagation, structured logging, cost telemetry, and LLM quality evaluations.

## Objective
Implement end-to-end observability features:
1. Global trace-ID propagation using `chat.session_id`
2. Structured Logfire spans with auto-instrumentation
3. Cost telemetry for LLM tokens and WhatsApp messages
4. Online LLM evaluations using Phoenix (Arize)
5. Dashboards, monitors, and alert rules
6. Comprehensive testing and validation

## Current State Analysis

### Existing Observability Infrastructure
- **Basic observability** (`shared/observability.py`): Structured logging, Logfire integration, metrics collection, trace decorators
- **Enhanced observability** (`shared/observability_enhanced.py`): Arize AX integration, OpenTelemetry support, LLM tracing
- **Configuration** (`config/settings.py`): Environment-based settings for Logfire, Arize, and OpenTelemetry
- **Logging** (`config/logging.yaml`): Comprehensive logging configuration with rotation and environment-specific settings

### Key Patterns in Use
- Graceful degradation when external services unavailable
- Trace decorators and context managers
- A2A message tracing
- LLM interaction tracking with metadata
- Metrics endpoints for services

## External Research & Documentation

### Logfire Auto-instrumentation
- **Docs**: https://logfire.pydantic.dev/docs/integrations/
- **Pattern**: Call `logfire.instrument_<package>()` after `logfire.configure()`
- **Supported**: FastAPI, HTTPX, Pydantic, OpenAI, Anthropic, System Metrics
- **Key Feature**: Request spans show parsing/validation time separately

### Arize OpenTelemetry Integration
- **Docs**: https://docs.arize.com/arize/observe/tracing-integrations-auto/opentelemetry-arize-otel
- **Package**: `arize-otel` (NOT the pandas client)
- **Pattern**: `arize.otel.register()` returns TracerProvider
- **Auto-instrumentation**: Use OpenInference instrumentors

### Phoenix LLM Evaluations
- **Docs**: https://docs.arize.com/phoenix/evaluation/concepts-evals/evaluation
- **Evaluators**: QA Correctness, Hallucination, Toxicity
- **Pattern**: LLM-as-a-Judge approach with explanations
- **Performance**: Run with `concurrency=4` for speed

### Cost Calculations
- **Gemini 1.5 Flash**: $0.075/1M input tokens, $0.30/1M output tokens
- **WhatsApp**: Service conversations FREE (as of July 2025), other types per-message
- **Token Cost Library**: https://github.com/AgentOps-AI/tokencost

## Implementation Blueprint

### 1. Trace ID Propagation System

```python
# shared/observability_middleware.py
from fastapi import Request, Response
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode
import uuid

class TraceIDMiddleware:
    """Middleware to propagate chat.session_id as global trace ID"""
    
    async def __call__(self, request: Request, call_next):
        # Extract or generate trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            # For WhatsApp webhooks, extract from message
            if request.url.path == "/webhook/whatsapp":
                body = await request.body()
                # Parse WhatsApp payload for session_id
                trace_id = extract_session_id(body) or str(uuid.uuid4())
        
        # Set trace context
        ctx = trace.set_span_in_context(trace_id)
        token = context.attach(ctx)
        
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            context.detach(token)
```

### 2. Enhanced Auto-instrumentation

```python
# shared/observability_setup.py
def initialize_observability(settings: Settings):
    """Initialize all observability integrations"""
    
    # 1. Configure Logfire with auto-instrumentation
    if settings.logfire_token:
        import logfire
        logfire.configure(
            token=settings.logfire_token,
            project_name=settings.logfire_project_name,
            environment=settings.logfire_environment,
        )
        
        # Auto-instrument packages
        logfire.instrument_fastapi()
        logfire.instrument_httpx()
        logfire.instrument_pydantic()
        
        # Google GenAI instrumentation
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        GoogleGenAIInstrumentor().instrument()
    
    # 2. Configure Arize with OpenTelemetry
    if settings.arize_api_key:
        from arize.otel import register
        tracer_provider = register(
            space_id=settings.arize_space_key,
            api_key=settings.arize_api_key,
            project_name=settings.arize_model_id,
        )
        
        # Additional OpenInference instrumentors
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

### 3. Cost Telemetry Implementation

```python
# shared/observability_cost.py
from typing import Dict, Optional
from pydantic import BaseModel
from tokencost import calculate_prompt_cost, calculate_completion_cost
import logfire

class CostCalculator:
    """Calculate and track costs for LLM and WhatsApp usage"""
    
    # WhatsApp rates (per message, USD)
    WHATSAPP_RATES = {
        "service": 0.0,  # Free as of July 2025
        "marketing": {"US": 0.025, "BR": 0.018, "IN": 0.008},
        "utility": {"US": 0.020, "BR": 0.015, "IN": 0.006},
        "authentication": {"US": 0.015, "BR": 0.010, "IN": 0.004},
    }
    
    @staticmethod
    def calculate_llm_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate LLM token costs"""
        try:
            # Use tokencost library with fallback
            prompt_cost = calculate_prompt_cost(prompt_tokens, model)
            completion_cost = calculate_completion_cost(completion_tokens, model)
            return prompt_cost + completion_cost
        except:
            # Fallback for Gemini 1.5 Flash
            if "gemini-1.5-flash" in model.lower():
                prompt_cost = (prompt_tokens / 1_000_000) * 0.075
                completion_cost = (completion_tokens / 1_000_000) * 0.30
                return prompt_cost + completion_cost
            return 0.0
    
    @staticmethod
    def calculate_whatsapp_cost(
        message_type: str,
        country: str = "US"
    ) -> float:
        """Calculate WhatsApp message cost"""
        if message_type == "service":
            return 0.0
        
        rates = CostCalculator.WHATSAPP_RATES.get(message_type, {})
        return rates.get(country, rates.get("US", 0.0))
    
    @staticmethod
    def track_cost(
        cost_usd: float,
        cost_type: str,
        metadata: Dict
    ):
        """Track cost with alerting"""
        with logfire.span(
            "cost_tracking",
            _tags=["cost", cost_type],
            cost_usd=cost_usd,
            **metadata
        ):
            # Check alert thresholds
            if cost_type == "llm" and cost_usd > 0.05:
                logfire.warn(
                    "High LLM cost per call",
                    cost_usd=cost_usd,
                    **metadata
                )
            
            # Track cumulative cost per session
            session_cost = metadata.get("session_cost", 0) + cost_usd
            if session_cost > 1.0:
                logfire.error(
                    "High session cost",
                    session_cost=session_cost,
                    session_id=metadata.get("session_id")
                )
```

### 4. Phoenix Evaluations Integration

```python
# shared/observability_evals.py
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel
import phoenix as px
from phoenix.evals import QACorrectnessEvaluator, HallucinationEvaluator
from phoenix.evals.models import OpenAIModel
import logfire

class EvaluationResult(BaseModel):
    """Evaluation result model"""
    eval_type: str
    score: float
    label: str
    explanation: Optional[str] = None

class PhoenixEvaluator:
    """Online LLM evaluation using Phoenix"""
    
    def __init__(self, model: str = "gpt-4"):
        self.eval_model = OpenAIModel(model=model)
        self.qa_evaluator = QACorrectnessEvaluator(self.eval_model)
        self.hallucination_evaluator = HallucinationEvaluator(self.eval_model)
    
    async def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str,
        trace_id: str
    ) -> List[EvaluationResult]:
        """Run online evaluations asynchronously"""
        
        # Don't block main flow
        asyncio.create_task(self._run_evaluations(
            question, answer, context, trace_id
        ))
        
        return []  # Return immediately
    
    async def _run_evaluations(
        self,
        question: str,
        answer: str, 
        context: str,
        trace_id: str
    ):
        """Execute evaluations and log results"""
        try:
            # Run evaluations concurrently
            qa_task = asyncio.create_task(
                self.qa_evaluator.evaluate(
                    query=question,
                    response=answer,
                    reference=context
                )
            )
            
            hallucination_task = asyncio.create_task(
                self.hallucination_evaluator.evaluate(
                    response=answer,
                    reference=context
                )
            )
            
            qa_result, hallucination_result = await asyncio.gather(
                qa_task, hallucination_task
            )
            
            # Log evaluation results
            with logfire.span(
                "llm_evaluation",
                trace_id=trace_id,
                _tags=["evaluation", "phoenix"]
            ):
                logfire.info(
                    "QA Correctness",
                    score=qa_result.score,
                    label=qa_result.label,
                    explanation=qa_result.explanation
                )
                
                logfire.info(
                    "Hallucination Check",
                    score=hallucination_result.score,
                    label=hallucination_result.label,
                    explanation=hallucination_result.explanation
                )
                
                # Log to Arize/Phoenix
                px.log_evaluations(
                    evaluation_results=[qa_result, hallucination_result],
                    trace_id=trace_id
                )
                
        except Exception as e:
            logfire.error(f"Evaluation failed: {e}", trace_id=trace_id)
```

### 5. Enhanced Observability Updates

```python
# Updates to shared/observability_enhanced.py
class EnhancedObservability:
    """Enhanced observability with all features"""
    
    def __init__(self, settings: Settings):
        # ... existing initialization ...
        
        # Add cost calculator and evaluator
        self.cost_calculator = CostCalculator()
        self.evaluator = PhoenixEvaluator() if settings.phoenix_enabled else None
    
    async def trace_llm_interaction_with_cost_and_eval(
        self,
        trace_data: LLMTrace,
        context: Optional[str] = None
    ):
        """Enhanced LLM tracing with cost and evaluations"""
        
        # Calculate cost
        cost = self.cost_calculator.calculate_llm_cost(
            model=trace_data.model,
            prompt_tokens=trace_data.prompt_tokens,
            completion_tokens=trace_data.completion_tokens
        )
        
        # Track cost
        self.cost_calculator.track_cost(
            cost_usd=cost,
            cost_type="llm",
            metadata={
                "model": trace_data.model,
                "session_id": trace_data.trace_id,
                "operation": trace_data.operation_name
            }
        )
        
        # Original tracing
        await self.trace_llm_interaction(trace_data)
        
        # Run evaluations if enabled
        if self.evaluator and context:
            await self.evaluator.evaluate_response(
                question=trace_data.prompt,
                answer=trace_data.response,
                context=context,
                trace_id=trace_data.trace_id
            )
```

## Implementation Tasks

### Phase 1: Foundation (Day 1)
1. Create `shared/observability_middleware.py` with TraceIDMiddleware
2. Create `shared/observability_setup.py` with auto-instrumentation
3. Update FastAPI apps to use middleware
4. Update A2A protocol to propagate trace IDs
5. Write unit tests for trace propagation

### Phase 2: Cost Telemetry (Day 2)
1. Create `shared/observability_cost.py` with CostCalculator
2. Integrate tokencost library
3. Add cost tracking to LLM interactions
4. Implement WhatsApp cost calculation
5. Add alert rules and thresholds
6. Write cost calculation tests

### Phase 3: Evaluations (Day 3)
1. Create `shared/observability_evals.py` with PhoenixEvaluator
2. Set up Phoenix/Arize evaluation pipeline
3. Implement async evaluation execution
4. Add evaluation result logging
5. Create operator feedback mechanism
6. Write evaluation tests

### Phase 4: Integration (Day 4)
1. Update `shared/observability_enhanced.py` with new features
2. Update all agents to use enhanced observability
3. Add configuration for new features
4. Create monitoring dashboards
5. Implement health checks

### Phase 5: Testing & Validation (Day 5)
1. Create end-to-end trace validation tests
2. Test cost aggregation accuracy
3. Verify evaluation pipeline
4. Performance benchmarking
5. Create validation scripts

## Validation Gates

```bash
# 1. Syntax and style checks
ruff check . && black .

# 2. Type checking
mypy shared/observability*.py

# 3. Unit tests
pytest tests/shared/test_observability*.py -v

# 4. Integration tests
pytest tests/integration/test_observability_e2e.py -v

# 5. Trace propagation verification
python scripts/validate_trace_propagation.py

# 6. Cost calculation accuracy
python scripts/test_cost_calculations.py

# 7. Evaluation pipeline test
python scripts/test_phoenix_evaluations.py

# 8. Performance benchmark
python scripts/benchmark_observability_overhead.py

# 9. End-to-end validation
make test-observability
```

## Validation Scripts

### scripts/validate_trace_propagation.py
```python
"""Validate trace ID propagation across services"""
import httpx
import asyncio
from uuid import uuid4

async def test_trace_propagation():
    trace_id = str(uuid4())
    
    # Send request to orchestrator
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/webhook/whatsapp",
            headers={"X-Trace-ID": trace_id},
            json={"message": "test"}
        )
    
    # Verify trace in logs
    # Check classifier received same trace ID
    # Validate trace appears in Logfire/Arize
    
    assert response.headers.get("X-Trace-ID") == trace_id
    print(f"✓ Trace {trace_id} propagated successfully")

if __name__ == "__main__":
    asyncio.run(test_trace_propagation())
```

### scripts/test_cost_calculations.py
```python
"""Test cost calculation accuracy"""
from shared.observability_cost import CostCalculator

def test_llm_costs():
    # Test Gemini 1.5 Flash
    cost = CostCalculator.calculate_llm_cost(
        "gemini-1.5-flash",
        prompt_tokens=1_000_000,
        completion_tokens=100_000
    )
    expected = 0.075 + (0.30 * 0.1)  # $0.105
    assert abs(cost - expected) < 0.001
    print(f"✓ LLM cost calculation: ${cost:.4f}")

def test_whatsapp_costs():
    # Test service conversation (free)
    assert CostCalculator.calculate_whatsapp_cost("service") == 0.0
    
    # Test marketing message
    cost = CostCalculator.calculate_whatsapp_cost("marketing", "US")
    assert cost == 0.025
    print(f"✓ WhatsApp cost calculation: ${cost:.4f}")

if __name__ == "__main__":
    test_llm_costs()
    test_whatsapp_costs()
```

## Success Criteria
1. **Trace Propagation**: 100% of requests have consistent trace IDs
2. **Performance**: Observability overhead < 5ms per request
3. **Cost Accuracy**: Cost calculations within 0.1% of actual
4. **Evaluation Coverage**: 95% of LLM responses evaluated
5. **Alert Latency**: Cost alerts fire within 1 minute

## Gotchas & Considerations
1. **Trace ID Format**: Must be OpenTelemetry compatible (32 hex chars)
2. **Phoenix Rate Limits**: Implement exponential backoff for evaluations
3. **Cost Cache**: Cache model pricing to avoid repeated lookups
4. **Async Safety**: Ensure evaluations don't block main flow
5. **PII Redaction**: Implement before logging customer data

## Dependencies
- `arize-otel>=0.3.0` (NOT arize pandas client)
- `openinference-instrumentation-openai>=0.1.0`
- `openinference-instrumentation-google-genai>=0.1.0`
- `phoenix-evals>=0.8.0`
- `tokencost>=0.1.0`

## Configuration Updates

Add to `config/settings.py`:
```python
# Phoenix Evaluations
phoenix_enabled: bool = Field(default=False, env="PHOENIX_ENABLED")
phoenix_eval_model: str = Field(default="gpt-4", env="PHOENIX_EVAL_MODEL")
phoenix_eval_concurrency: int = Field(default=4, env="PHOENIX_EVAL_CONCURRENCY")

# Cost Thresholds
cost_alert_per_call_usd: float = Field(default=0.05, env="COST_ALERT_PER_CALL_USD")
cost_alert_per_session_usd: float = Field(default=1.0, env="COST_ALERT_PER_SESSION_USD")

# Trace Configuration  
trace_id_header: str = Field(default="X-Trace-ID", env="TRACE_ID_HEADER")
```

## Confidence Score: 9/10

The implementation leverages existing patterns and infrastructure, making it low-risk. The only uncertainty is Phoenix evaluation performance at scale, mitigated by async execution and sampling.

## References
- Logfire Integrations: https://logfire.pydantic.dev/docs/integrations/
- Arize OpenTelemetry: https://docs.arize.com/arize/observe/tracing-integrations-auto/opentelemetry-arize-otel
- Phoenix Evaluations: https://docs.arize.com/phoenix/evaluation/concepts-evals/evaluation
- Token Cost Library: https://github.com/AgentOps-AI/tokencost
- WhatsApp Pricing: https://developers.facebook.com/docs/whatsapp/pricing/
- Gemini Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing#text-models