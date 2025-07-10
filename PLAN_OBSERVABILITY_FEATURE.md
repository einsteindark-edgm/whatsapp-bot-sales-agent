## FEATURE:
Implement end-to-end observability for every WhatsApp conversation in **ChatCommerce Bot**.  
Goals:

1. **Global trace-ID** – propagate a single `chat.session_id` from the first webhook hit through every micro-service, span, and log until the transaction closes.
2. **Structured Logfire spans & logs**  
   * Use `logfire.instrument` on `classifier.predict`, `orchestrator.decision`, `agent.sales.flow`, `agent.pqr.flow`.  
   * Enrich with `_tags`, `agent.role`, `conversation.step`, `template_category`, `price_usd`.
   * Auto-instrument **FastAPI**, **HTTPX**, **Google GenAI OTEL**, **Pydantic AI**.
3. **Cost telemetry**  
   * `gen_ai.usage.price_usd` via **TokenCost** for LLM tokens (include Gemini 1.5 Flash rates).  
   * `messaging.whatsapp.price_usd` from Meta’s conversation-based rate card.
   * Alert rules: `gen_ai.usage.cost_usd > $0.05` per call, `SUM(price_usd) > $1` per conversation.
4. **Online LLM evaluations** in Phoenix (Arize): QA Correctness, Hallucination/Toxicity, Readability.  
   * Log `eval.*` scores and operator thumbs-up / thumbs-down annotations.
5. **Dashboards & monitors** – metrics only; visual layout will be built later.  
6. **Checklist** – provide scripts/tests to verify trace propagation, cost math, eval logging, and alert firing.

## EXAMPLES:
- `examples/logfire_fastapi_httpx.py` – minimal FastAPI + HTTPX auto-instrumentation.
- `examples/logfire_manual_span.py` – manual span with `_tags` and log message.
- `examples/arize_otel_register.py` – bootstrapping OpenTelemetry export to Arize with OpenInference.
- `examples/arize_evals_qa_correctness.py` – running QA & hallucination evals and logging results online.
- `examples/tokencost_example.py` – estimating USD cost for prompt and completion tokens.

## DOCUMENTATION:
- https://logfire.pydantic.dev/docs/integrations/ – auto-instrumenting FastAPI & HTTPX  
- https://logfire.pydantic.dev/docs/reference/api/logfire/ – manual spans and `_tags`  
- https://arize.com/docs/ax/integrations/opentelemetry/opentelemetry-arize-otel – `arize.otel.register` quick-start  
- https://arize.com/docs/phoenix/retrieval/quickstart-retrieval – Phoenix evals with `llm_classify`  
- https://github.com/AgentOps-AI/tokencost – Token & price estimation helper  
- https://cloud.google.com/vertex-ai/generative-ai/pricing#text-models – Gemini 1.5 Flash pricing reference  
- https://docs.wcapi.io/docs-whatsapp-business-platform-pricing – WhatsApp conversation pricing reference  

## OTHER CONSIDERATIONS:
* **Performance** – keep tracing overhead < 5 ms per request; batch span exporter where possible.  
* **Security** – redact PII headers before emitting to Logfire/OTel.  
* **Config** – all API keys and rate tables loaded via `shared/settings.py`; defaults in `/config/observability.toml`.  
* **Async safety** – decorators must preserve function signatures and `async` support.  
* **Concurrency** – Phoenix evals run with `concurrency=4`; tune if CPU-bound.  
* **Testing** – add pytest fixtures that:  
  1. hit `/webhook/whatsapp` with sample payload,  
  2. follow trace across micro-services,  
  3. assert cost metrics and eval scores exist,  
  4. trigger mock alerts when thresholds breached.

> **Folder structure note:**  
> Follow the **“Code structure & modularity”** section in `CLAUDE.md`; observability helpers go in `/shared/observability_enhanced.py`, decorators in `/shared/observability_decorators.py`, and tests in `/shared/tests/test_observability.py`.
