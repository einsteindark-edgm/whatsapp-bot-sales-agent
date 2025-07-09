# CLAUDE.md

## Project description
ChatCommerce Bot is a multi-agent WhatsApp assistant that closes sales, answers product questions and triages complaints while remaining indistinguishable from a human.  
Agents cooperate via an **Agent-to-Agent (A2A) protocol** and call external tools—such as a product RAG or payment service—through the **Model Context Protocol (MCP)**.  
A **Workflow Agent** built with **Google ADK** orchestrates the conversation, while specialised domain agents run as Docker micro-services.  
Whenever the bot detects an intent to purchase or a complaint, it hands the thread to a human operator, summarising context and suggesting replies; the human replies back through the bot, which relays the final message to the customer.  
**Comprehensive observability** is implemented using **Pydantic Logfire** for structured logging and **Arize AX with OpenTelemetry** for LLM monitoring, with data stored in Neon Postgres.

## Tech stack
- **Python 3.12**
- **LangChain 0.2** · **LangGraph 0.1**
- **Google ADK 0.4**
- **Pydantic 2.6** (+ Pydantic Logfire 0.5)
- **Arize AX 1.3**
- **Docker 26 + Docker Compose**
- **Neon Postgres 16**
- Optional: **FastAPI 0.111** for HTTP adapters

### 🧱 Code structure & modularity (path → purpose)
> **Never create a file longer than 500 lines.** Split into helper modules when approaching the limit.

| Path | Purpose |
|------|---------|
| `/agents/<agent_name>/` | Self-contained agent (e.g. `sales`, `support`) |
| &nbsp;&nbsp;`domain/` | Entities, value objects, business rules |
| &nbsp;&nbsp;`application/` | Use-cases, orchestrators |
| &nbsp;&nbsp;`ports/inbound/` | CLI, REST, Webhook, Events |
| &nbsp;&nbsp;`ports/outbound/` | Repos, LLM, RAG, DB, MQ |
| &nbsp;&nbsp;`adapters/inbound/` | FastAPI routers, LangGraph nodes, Pub/Sub consumers |
| &nbsp;&nbsp;`adapters/outbound/` | Neon Postgres repo, OpenAI client, AX exporter |
| &nbsp;&nbsp;`tests/` | Unit & contract tests for the agent |
| `/shared/` | Cross-cutting models & helpers |
| `/config/` | YAML/TOML configs, `.env`, Helm templates |
| `/infra/` | Dockerfiles, IaC, CI/CD, observability |
| `/scripts/` | Maintenance CLI, seeds, migrations |
| `/docs/` | ADRs, diagrams, user manuals |
| `/tests/` | End-to-end integration tests |

## Commands
| Action | Command |
|--------|---------|
| Build containers | `make build` or `docker compose build` |
| Run locally | `docker compose up` |
| Tests | `pytest -q` |
| Lint / format | `ruff check . && black .` |
| Deploy agents | `gcloud ai agents deploy` |
| DB migrations | `alembic upgrade head` |
| **Start classifier** | `python3 -m uvicorn agents.classifier.main:app --host 0.0.0.0 --port 8001` |
| **Start orchestrator** | `python3 -m uvicorn agents.orchestrator.main:app --host 0.0.0.0 --port 8000` |
| **Test observability** | `python3 test_arize_otel.py` |
| **Check observability** | `curl http://localhost:8001/api/v1/observability-metrics` |
| **Expose webhook (dev)** | `ngrok http 8000` |
| **Test WhatsApp webhook verification** | `curl -X GET "http://localhost:8000/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=your_token&hub.challenge=test123"` |
| **Test WhatsApp incoming message** | `curl -X POST http://localhost:8000/webhook/whatsapp -H "Content-Type: application/json" -d @examples/whatsapp_webhook_payload.json` |
| **Check WhatsApp health** | `curl http://localhost:8000/webhook/whatsapp/health` |

### 📎 Style & conventions
- **PEP 8** + type hints; auto-format with **black** & **ruff**.
- Semantic commit messages; trunk-based development with short-lived branches.
- **Docstrings for every function** (Google style).
- **Pydantic** models for all DTOs; validate at boundaries.
- Use **FastAPI** or LangGraph nodes as thin adapters—no business logic.

### 🧪 Testing & reliability
- **Pytest** unit tests for every use-case, edge case and failure path.
- **Contract tests** for each port adapter.
- **/tests** mirrors application structure.
- Use **LangSmith** traces for LLM chain assertions.
- Gate CI on `pytest --cov >= 90 %`.

## Environment setup
1. Install Docker ≥ 26 and `make`.  
2. Clone repo & copy `.env.example` → `.env`.  
3. Run `make build && docker compose up`.  
4. Neon Postgres credentials injected via `DATABASE_URL`.  
5. Optional local dev: `poetry install && pre-commit install`.

## Allowed / forbidden tools
- **Allowed:** `make`, `docker`, `poetry`, `gcloud`, `pytest`, `ruff`, `black`.
- **Forbidden:** Any command requiring root access or destructive operations (`rm -rf /`, `DROP DATABASE`, etc.).

### 📚 Glossary
| Term | Meaning |
|------|---------|
| **A2A** | Agent-to-Agent cooperative protocol |
| **MCP** | Model Context Protocol for tool calls |
| **RAG** | Retrieval-Augmented Generation |
| **ADK** | Google’s Agent Development Kit |
| **AX** | Arize AI observability platform |
| **WABA** | WhatsApp Business Account |
| **Webhook** | HTTP callback for real-time notifications |
| **OTel** | OpenTelemetry for distributed tracing |

### ⚠️ Warnings & notes
- Be mindful of WhatsApp Business API rate limits and GDPR compliance.  
- LLM context window: max 32 k tokens; chunk messages accordingly.  
- Do **not** expose internal agent prompts to end-users.
- **✅ Completed (2025-07-09):** WhatsApp Business API integration - webhook endpoint + message processing implemented

## Tips
- Keep this file concise; update whenever architecture or processes change.  
- Ask questions if something is unclear—never assume missing context.
- every Agent should have a clear role and responsibility and have its own docker container.
- Every agent implement the protocol A2A to communicate with other agents.
- Every agent implement the protocol MCP to communicate with external tools.
- Every agent cloud use differents LLMs make a layer of abstraction.

## 🚨 Critical Lessons Learned - ALWAYS VERIFY THESE

### 🔧 Configuration & Service Management
1. **ALWAYS check actual running ports** - Use `ps aux | grep` to verify which ports services are actually running on
2. **ALWAYS verify service URLs in configuration files** - Don't assume defaults are correct
3. **ALWAYS check both hardcoded defaults AND settings files** for configuration mismatches
4. **ALWAYS restart services after configuration changes** - Configuration is not hot-reloaded
5. **ALWAYS load .env explicitly with load_dotenv()** - Add at the beginning of main.py files before any imports

### 🔗 A2A Protocol Implementation
1. **ALWAYS verify A2A message types use correct enum values** - Use lowercase with underscores: `"classification_request"` NOT `"CLASSIFICATION_REQUEST"`
2. **ALWAYS pass API keys explicitly in A2A requests** - Don't assume services will use fallback authentication
3. **ALWAYS test A2A endpoints directly with curl before integration** - Isolate protocol issues from business logic
4. **ALWAYS check A2A message parsing in both sender and receiver** - Ensure consistent message structure
5. **A2A is for INTER-SERVICE communication only** - Don't use A2A within the same service (e.g., webhook to orchestrator)

### 🐛 Debugging & Error Handling
1. **NEVER claim a task is complete without end-to-end testing** - User explicitly warned: "no vuelvas a dar por terminada una tarea si la funcionalidad no esta correcta"
2. **ALWAYS check service logs when debugging** - Look at both services in communication chain
3. **ALWAYS test with actual data flows, not just health checks** - Health endpoints don't test full functionality
4. **ALWAYS verify error messages are meaningful** - Generic errors hide the real problem
5. **ALWAYS read the actual implementation before using functions** - Don't assume method names or signatures

### 🏗️ Architecture & Development
1. **NEVER create simplified test services** - User explicitly stated: "creaste start_classifier y start_orchestrator, pero asi no debe funcionar el sistema... no lo vuelvas a hacer"
2. **ALWAYS use the real agent architecture** - Use actual agents in `/agents/` folder, not shortcuts
3. **ALWAYS trace the complete data flow** - From CLI → Orchestrator → Classifier → back to CLI
4. **ALWAYS verify dependencies and imports** - Missing imports cause runtime failures
5. **UNDERSTAND architectural boundaries** - Know when components are in the same service vs separate services

### 📝 Testing Strategy
1. **Test sequence: Direct → A2A → End-to-End** - Start with direct service calls, then A2A protocol, finally full CLI flow
2. **ALWAYS test both classification types** - Verify `product_information` AND `PQR` classifications work
3. **ALWAYS check confidence scores and processing times** - Ensure realistic classification results
4. **NEVER assume one test covers all scenarios** - Test edge cases and different message types
5. **Create debug scripts when complex flows fail** - Isolate and test each component

### 🔐 Authentication & Security
1. **ALWAYS verify API key propagation through the entire chain** - CLI → Orchestrator → Classifier
2. **ALWAYS check authentication fallbacks** - Ensure graceful handling when headers are missing
3. **ALWAYS validate that settings are loaded correctly** - Don't assume environment variables are available

### ⚡ Performance & Reliability
1. **ALWAYS monitor processing times** - Classification should complete in reasonable time (< 5 seconds)
2. **ALWAYS check for retry mechanisms** - Network calls should have proper retry logic  
3. **ALWAYS verify observability works** - Traces and logs should provide meaningful debugging info

### 📊 Observability & Monitoring (Updated: 2025-07-09)
1. **ALWAYS verify observability integration status before implementing** - Check if packages and credentials are available
2. **Arize requires OpenTelemetry (OTel) integration** - Use `arize.otel.register()` NOT pandas client for console visibility
3. **ALWAYS test observability integrations with real data** - Health checks don't validate data collection
4. **Logfire parameter conflicts** - Use non-reserved parameter names (avoid `agent_name`, `operation_name`)
5. **ALWAYS check integration logs for successful data transmission** - Look for endpoint confirmations and status codes
6. **Install arize-otel package for proper Arize integration** - Standard arize package uses deprecated pandas approach
7. **Verify dashboard registration** - Projects must appear in web console to confirm successful integration
8. **trace_method decorator doesn't exist** - Use `trace_context` async context manager from `observability_enhanced.py`
9. **ALWAYS use enhanced_observability for LLM tracking** - Call `trace_llm_interaction` after getting classification results
10. **Import observability correctly** - `from shared.observability import get_logger, trace_a2a_message` and `from shared.observability_enhanced import trace_context, enhanced_observability`
11. **Track WhatsApp interactions** - Include channel metadata, sender info, and message IDs in LLM traces

### 💻 Code Implementation Errors (Added: 2025-07-09)
1. **VERIFY imports exist before using them** - Check actual files for function/class names (e.g., trace_method doesn't exist, use trace_context)
2. **CHECK function signatures before calling** - Don't assume parameter names or return types (e.g., trace_a2a_message params)
3. **AsyncHTTPClient returns Response objects** - Always use `.json()` to get dict data
4. **Enum with use_enum_values=True returns strings** - Don't use `.value` on enum fields
5. **Read constructor parameters carefully** - Don't pass `config=` if constructor expects individual params
6. **Understand data flow before implementing** - Draw or document the flow first
7. **Import missing types** - Always import List, Dict, etc. from typing when needed
8. **Don't create non-existent models** - OrchestrationRequest didn't exist in A2A protocol

### 🚫 NEVER DO THESE AGAIN
- ❌ Create simplified services instead of using real agents
- ❌ Assume configuration defaults are correct without verification
- ❌ Mark tasks complete without full end-to-end testing
- ❌ Ignore user feedback about broken functionality
- ❌ Use wrong case for enum values in A2A protocol
- ❌ Skip API key authentication in service-to-service calls
- ❌ Use pandas client for Arize without verifying console registration
- ❌ Assume observability is working based on lack of errors alone
- ❌ Import functions/classes without verifying they exist
- ❌ Call methods without checking their actual signatures
- ❌ Use A2A for intra-service communication
- ❌ Assume Response objects are dictionaries
- ❌ Access .value on enums configured with use_enum_values=True
- ❌ Create models that don't exist (OrchestrationRequest)
- ❌ Pass wrong constructor parameters (config= when not expected)