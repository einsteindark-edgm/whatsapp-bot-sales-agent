# PRP: WhatsApp Business API Integration

## 📋 Feature Overview
Implement complete WhatsApp Business API integration to receive and respond to messages through the existing multi-agent orchestrator system.

**Generated**: 2025-07-09  
**Status**: Ready for Implementation  
**Confidence Level**: 9/10 - Comprehensive context with clear patterns and examples

## 🎯 Objectives
1. Create webhook endpoints to receive WhatsApp messages
2. Implement WhatsApp API client for sending messages
3. Integrate with existing orchestrator workflow
4. Maintain A2A protocol compatibility
5. Add comprehensive observability for WhatsApp flows

## 🔍 Context & Research

### Existing Codebase Patterns

#### 1. **FastAPI Router Pattern** (Reference: `/agents/orchestrator/adapters/inbound/fastapi_router.py`)
```python
# Authentication extraction (lines 47-64)
def get_api_key(request: Request) -> Optional[str]:
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]
    return api_key.strip() if api_key else None

# Trace ID pattern (lines 67-82)
def get_trace_id(request: Request) -> str:
    trace_id = request.headers.get("X-Trace-Id")
    if not trace_id:
        trace_id = generate_trace_id()
    return trace_id
```

#### 2. **HTTP Client Pattern** (Reference: `/agents/orchestrator/adapters/outbound/http_a2a_client.py`)
```python
# Configuration pattern (lines 28-50)
class A2AClientConfig(BaseModel):
    classifier_url: str = Field(..., description="URL of the classifier service")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    retries: int = Field(default=3, ge=0, le=10)
    
# AsyncHTTPClient usage (lines 148-153)
async with AsyncHTTPClient(
    timeout_config=self.timeout_config, 
    retry_config=self.retry_config, 
    headers=headers
) as client:
    response = await client.post(url=url, json_data=request.model_dump(mode='json'))
```

#### 3. **A2A Protocol** (Reference: `/shared/a2a_protocol.py`)
- Message structure with trace_id, sender_agent, receiver_agent
- Validation and creation methods
- Error handling patterns

#### 4. **Domain Models** (Reference: `/agents/orchestrator/domain/models.py`)
- WorkflowRequest pattern with metadata
- Validation using Pydantic
- UUID generation for IDs

### External Documentation
- **Official Meta WhatsApp API**: https://developers.facebook.com/docs/whatsapp/cloud-api/
- **Webhook Setup Guide**: https://developers.facebook.com/docs/whatsapp/cloud-api/guides/set-up-webhooks/
- **Message Format Reference**: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples/
- **API Rate Limits**: https://developers.facebook.com/docs/whatsapp/cloud-api/rate-limits/

### WhatsApp Webhook Format
```json
{
  "object": "whatsapp_business_account",
  "entry": [{
    "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
    "changes": [{
      "value": {
        "messaging_product": "whatsapp",
        "metadata": {
          "display_phone_number": "PHONE_NUMBER",
          "phone_number_id": "PHONE_NUMBER_ID"
        },
        "messages": [{
          "from": "CUSTOMER_PHONE_NUMBER",
          "id": "MESSAGE_ID",
          "timestamp": "TIMESTAMP",
          "text": {
            "body": "USER_MESSAGE_TEXT"
          },
          "type": "text"
        }]
      },
      "field": "messages"
    }]
  }]
}
```

## 📐 Implementation Blueprint

### Phase 1: Environment Setup
1. Add WhatsApp credentials to `.env`:
```bash
WHATSAPP_ACCESS_TOKEN=your_permanent_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token
WHATSAPP_API_VERSION=v18.0
```

2. Update `/config/settings.py` to include WhatsApp settings

### Phase 2: Domain Models (`/agents/orchestrator/domain/whatsapp_models.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class WhatsAppMetadata(BaseModel):
    display_phone_number: str
    phone_number_id: str

class WhatsAppTextMessage(BaseModel):
    body: str

class WhatsAppMessage(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[WhatsAppTextMessage] = None

class WhatsAppContact(BaseModel):
    profile: Dict[str, str]
    wa_id: str

class WhatsAppValue(BaseModel):
    messaging_product: str
    metadata: WhatsAppMetadata
    messages: Optional[List[WhatsAppMessage]] = None
    contacts: Optional[List[WhatsAppContact]] = None

class WhatsAppChange(BaseModel):
    value: WhatsAppValue
    field: str

class WhatsAppEntry(BaseModel):
    id: str
    changes: List[WhatsAppChange]

class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[WhatsAppEntry]

    def extract_message(self) -> Optional[tuple[str, str, str]]:
        """Extract message text, sender, and message_id from webhook payload"""
        # Implementation details...
```

### Phase 3: WhatsApp API Client (`/agents/orchestrator/adapters/outbound/whatsapp_api_client.py`)
```python
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import httpx
from shared.utils import AsyncHTTPClient, RetryConfig, TimeoutConfig
from shared.observability import trace_method, logger
import json

class WhatsAppConfig(BaseModel):
    access_token: str = Field(..., description="WhatsApp access token")
    phone_number_id: str = Field(..., description="WhatsApp phone number ID")
    api_version: str = Field(default="v18.0")
    base_url: str = Field(default="https://graph.facebook.com")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    retries: int = Field(default=3, ge=0, le=10)

class WhatsAppAPIClient:
    def __init__(self, config: Optional[WhatsAppConfig] = None):
        self.config = config or WhatsAppConfig(
            access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
            phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
        )
        self.timeout_config = TimeoutConfig(
            connect=10.0,
            read=self.config.timeout,
            write=10.0,
            pool=5.0
        )
        self.retry_config = RetryConfig(
            max_attempts=self.config.retries,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2,
            jitter=True
        )

    @trace_method("send_whatsapp_message")
    async def send_text_message(
        self, 
        to: str, 
        message: str,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send text message via WhatsApp API"""
        url = f"{self.config.base_url}/{self.config.api_version}/{self.config.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.config.access_token}",
            "Content-Type": "application/json",
            "X-Trace-Id": trace_id or ""
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "body": message
            }
        }
        
        async with AsyncHTTPClient(
            timeout_config=self.timeout_config,
            retry_config=self.retry_config,
            headers=headers
        ) as client:
            response = await client.post(url=url, json_data=payload)
            return response

    @trace_method("send_typing_indicator")
    async def send_typing_indicator(self, to: str) -> Dict[str, Any]:
        """Send typing indicator to user"""
        # Similar implementation...

    @trace_method("mark_message_read")
    async def mark_message_as_read(self, message_id: str) -> Dict[str, Any]:
        """Mark message as read"""
        # Similar implementation...
```

### Phase 4: Webhook Handler (`/agents/orchestrator/adapters/inbound/whatsapp_webhook_router.py`)
```python
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from typing import Optional, Dict, Any
import hmac
import hashlib
from shared.observability import trace_method, logger
from agents.orchestrator.domain.whatsapp_models import WhatsAppWebhookPayload
from agents.orchestrator.domain.models import WorkflowRequest
from agents.orchestrator.adapters.inbound.fastapi_router import get_api_key, get_trace_id
from shared.a2a_protocol import A2AMessage, MessageType, create_a2a_message
import os
import json

router = APIRouter(prefix="/webhook", tags=["whatsapp"])

def verify_webhook_signature(request: Request, payload: bytes) -> bool:
    """Verify webhook signature from Meta"""
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature:
        return False
    
    expected_signature = hmac.new(
        os.getenv("WHATSAPP_APP_SECRET", "").encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, f"sha256={expected_signature}")

@router.get("/whatsapp")
async def verify_webhook(
    hub_mode: str,
    hub_verify_token: str,
    hub_challenge: str
) -> Response:
    """Webhook verification endpoint for Meta"""
    verify_token = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        logger.info("WhatsApp webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")
    
    logger.error("WhatsApp webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@router.post("/whatsapp")
@trace_method("process_whatsapp_webhook")
async def process_webhook(
    request: Request,
    trace_id: str = Depends(get_trace_id)
) -> Dict[str, Any]:
    """Process incoming WhatsApp messages"""
    # Read raw body for signature verification
    body = await request.body()
    
    # Verify webhook signature
    if not verify_webhook_signature(request, body):
        logger.error("Invalid webhook signature", extra={"trace_id": trace_id})
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse webhook payload
    try:
        data = json.loads(body)
        webhook_payload = WhatsAppWebhookPayload(**data)
    except Exception as e:
        logger.error(f"Failed to parse webhook: {e}", extra={"trace_id": trace_id})
        return {"status": "error", "message": "Invalid payload"}
    
    # Extract message
    message_data = webhook_payload.extract_message()
    if not message_data:
        return {"status": "ok", "message": "No message to process"}
    
    text, sender, message_id = message_data
    
    # Create workflow request
    workflow_request = WorkflowRequest(
        user_message=text,
        user_id=sender,
        session_id=f"whatsapp_{sender}",
        metadata={
            "source": "whatsapp",
            "message_id": message_id,
            "phone_number": sender
        }
    )
    
    # Create A2A message for orchestration
    a2a_message = create_a2a_message(
        sender_agent="whatsapp_adapter",
        receiver_agent="orchestrator",
        message_type=MessageType.ORCHESTRATION_REQUEST,
        payload=workflow_request.model_dump(),
        trace_id=trace_id
    )
    
    # Process through orchestrator (import the orchestrate function)
    from agents.orchestrator.agent import WhatsAppOrchestrator
    orchestrator = WhatsAppOrchestrator()
    
    try:
        response = await orchestrator.process_whatsapp_message(a2a_message)
        
        # Send response back via WhatsApp
        from agents.orchestrator.adapters.outbound.whatsapp_api_client import WhatsAppAPIClient
        whatsapp_client = WhatsAppAPIClient()
        
        await whatsapp_client.send_typing_indicator(sender)
        await whatsapp_client.mark_message_as_read(message_id)
        await whatsapp_client.send_text_message(
            to=sender,
            message=response.get("response", "Lo siento, no pude procesar tu mensaje."),
            trace_id=trace_id
        )
        
        return {"status": "ok", "processed": True}
        
    except Exception as e:
        logger.error(f"Failed to process WhatsApp message: {e}", extra={"trace_id": trace_id})
        return {"status": "error", "message": str(e)}
```

### Phase 5: Orchestrator Integration
Update `/agents/orchestrator/main.py` to include WhatsApp routes:
```python
from agents.orchestrator.adapters.inbound.whatsapp_webhook_router import router as whatsapp_router

# In create_app function:
app.include_router(whatsapp_router)
```

## 🧪 Validation Gates

### 1. Syntax and Code Quality
```bash
# Lint and format
ruff check . --fix
black .

# Type checking
mypy agents/orchestrator/
```

### 2. Unit Tests
Create `/agents/orchestrator/tests/test_whatsapp_integration.py`:
```python
import pytest
from httpx import AsyncClient
from agents.orchestrator.main import create_app
import json

@pytest.mark.asyncio
async def test_webhook_verification():
    """Test WhatsApp webhook verification"""
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/webhook/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test_token",
                "hub.challenge": "test_challenge"
            }
        )
        assert response.status_code == 200
        assert response.text == "test_challenge"

@pytest.mark.asyncio
async def test_webhook_message_processing():
    """Test processing WhatsApp messages"""
    # Test implementation...
```

### 3. Integration Tests
```bash
# Start services
docker compose up -d

# Run integration tests
pytest tests/integration/test_whatsapp_flow.py -v

# Check logs
docker compose logs orchestrator | grep whatsapp
```

### 4. End-to-End Testing
```bash
# Use ngrok for local testing
ngrok http 8000

# Configure webhook URL in Meta Developer Console
# Send test message via WhatsApp
# Verify response received
```

## 🚨 Error Handling Strategy

### 1. Webhook Errors
- Always return 200 OK to prevent retries
- Log errors with trace_id for debugging
- Use try-catch blocks for all processing

### 2. API Client Errors
- Implement exponential backoff with jitter
- Handle rate limits (80 msg/sec)
- Log all API errors with response details

### 3. Message Processing Errors
- Fallback messages for classification failures
- Queue failed messages for retry
- Alert on repeated failures

## 🔐 Security Considerations

1. **Webhook Signature Validation**
   - Verify X-Hub-Signature-256 header
   - Use constant-time comparison
   - Reject unsigned requests

2. **Token Security**
   - Store tokens in environment variables
   - Never log tokens
   - Rotate tokens regularly

3. **Rate Limiting**
   - Implement per-user rate limits
   - Track message frequency
   - Prevent abuse

## 📊 Observability Requirements

1. **Metrics to Track**
   - Messages received/sent per minute
   - Response time distribution
   - Classification distribution
   - API error rates

2. **Logging**
   - Log all webhook events with trace_id
   - Log API calls and responses
   - Structured logging with metadata

3. **Tracing**
   - Full trace: WhatsApp → Webhook → Orchestrator → Classifier → Response
   - Include WhatsApp message_id in traces
   - Track processing time at each step

## 📝 Implementation Checklist

- [ ] Environment variables configured
- [ ] Domain models created
- [ ] WhatsApp API client implemented
- [ ] Webhook endpoints created
- [ ] Orchestrator integration complete
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Webhook verified with Meta
- [ ] End-to-end test successful
- [ ] Observability implemented
- [ ] Documentation updated

## 🎯 Success Criteria

1. Webhook receives and verifies Meta callbacks
2. Messages are processed through existing orchestrator flow
3. Responses are sent back via WhatsApp API
4. Full observability of WhatsApp flows
5. All tests passing with >90% coverage

## 📚 Additional Resources

- [WhatsApp Cloud API Postman Collection](https://github.com/WhatsApp/WhatsApp-Business-API-Postman-Collection)
- [Meta WhatsApp Business API Reference](https://developers.facebook.com/docs/whatsapp/api)
- [Webhook Security Best Practices](https://developers.facebook.com/docs/messenger-platform/webhook)

## 🏆 Confidence Score: 9/10

This PRP provides comprehensive context with:
- ✅ Clear references to existing codebase patterns
- ✅ Complete implementation blueprint with code examples
- ✅ Executable validation gates
- ✅ Detailed error handling strategy
- ✅ Security considerations
- ✅ Testing approach

The only missing element (-1 point) is actual Meta Developer account credentials for live testing, which the implementer will need to provide.