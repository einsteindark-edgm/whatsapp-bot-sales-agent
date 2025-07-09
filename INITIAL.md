## FEATURE:
Build a local proof-of-concept for a **multi-agent WhatsApp sales assistant** with:

* **CLI front-end** that sends/receives messages (async, lightweight, loading spinner).
* **Workflow Orchestrator Agent** (Google ADK) living in its own Docker container.
* **Classifier Agent** (PydanticAI + Gemini Flash 2.5) in another container that labels each inbound text as  
  `product_information` or `PQR`.
* Agents exchange messages via the **A2A protocol**; payloads:  
  `{"text": "<original>", "label": "<classification>"}`.
* HTTP communication between containers is implemented with **FastAPI**.
* **> 90 % unit-test coverage** using `pytest`, ADK’s `AgentEvaluator`, and PydanticAI’s `TestModel`.

## EXAMPLES:
- `examples/adk_orchestrator_snippet.py` – minimal custom orchestrator definition (ADK docs).  
- `examples/pydanticai_bank_support_snippet.py` – shows `Agent`, `deps_type`, `output_type` pattern (PydanticAI docs).  
- `examples/pydanticai_gemini_model.py` – one-liner to initialise Gemini Flash via PydanticAI.  
- `examples/fastapi_request_body.py` – canonical POST handler with Pydantic schema (FastAPI docs).  
- `examples/cli_spinner.py` – threaded CLI spinner while awaiting async I/O.  
- `examples/adk_pytest_evaluate.py` – ADK evaluation test wired into `pytest`.  
- `examples/pydanticai_test_fixture.py` – fixture overriding the LLM with `TestModel` for fast tests.

## DOCUMENTATION:
- https://google.github.io/adk-docs/agents/custom-agents/ – ADK agent patterns.  
- https://ai.pydantic.dev/ – PydanticAI guides, Gemini integration & testing utilities.  
- https://fastapi.tiangolo.com/ – FastAPI request/response handling.  
- https://google.github.io/adk-docs/evaluate/ – Programmatic evaluation with `pytest`.  
- https://stackoverflow.com/q/48854567 – Simple async CLI spinner pattern.
- https://google.github.io/adk-docs/ - all ADK docs
- https://ai.pydantic.dev/ - all PydanticAI docs
- https://fastapi.tiangolo.com/ - all FastAPI docs

## OTHER CONSIDERATIONS:
* **A2A message envelope** should include a trace-ID for observability (Logfire).
* **MCP stubs** can be mocked until payment/RAG services are ready.
* Keep Docker images slim (`python:3.12-slim`, multi-stage build).
* Remember WhatsApp Business API rate limits; plan pagination/backoff later.
* Async tests: mark with `pytest.mark.anyio` to integrate with FastAPI/ADK coroutines.

> **Folder structure note:**  
> Follow `/agents/<agent_name>/` layout and keep any file ≤ 500 LOC as per `CLAUDE.md`.
