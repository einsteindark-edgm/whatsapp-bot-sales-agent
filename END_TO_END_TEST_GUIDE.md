# End-to-End Testing Guide

## Prerequisites

### 1. Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Update your `.env` file:
```bash
GEMINI_API_KEY=your_real_api_key_here
```

### 2. Install Docker (if not already installed)
```bash
# macOS with Homebrew
brew install docker docker-compose

# Or download Docker Desktop from docker.com
```

## Quick E2E Test (Without WhatsApp)

### Step 1: Build the system
```bash
# From project root
make build
# or
docker compose build
```

### Step 2: Start services
```bash
make run
# or  
docker compose up -d
```

### Step 3: Verify services are running
```bash
# Check orchestrator
curl http://localhost:8080/health

# Check classifier  
curl http://localhost:8001/health

# Should return: {"status": "healthy", "service": "orchestrator/classifier"}
```

### Step 4: Test CLI interface
```bash
# Test product information classification
python -m cli.main "What's the price of iPhone 15?"

# Test complaint/problem classification  
python -m cli.main "My order is delayed and I want to cancel it"

# Test general conversation
python -m cli.main "Hello, how are you today?"
```

### Step 5: Test direct API calls
```bash
# Test classifier directly
curl -X POST http://localhost:8001/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What is the price of iPhone 15?", "user_id": "test_user", "session_id": "test_session"}'

# Test orchestrator directly
curl -X POST http://localhost:8080/api/v1/orchestrate-direct \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What is the price of iPhone 15?", "user_id": "test_user", "session_id": "test_session"}'
```

## Full WhatsApp Integration (Production)

### 1. WhatsApp Business API Setup
- Register for WhatsApp Business API
- Get webhook URL and verification token
- Configure webhook endpoints in your FastAPI app

### 2. Add WhatsApp webhook handler
```python
# Add to orchestrator FastAPI router
@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: WhatsAppWebhookRequest):
    # Process WhatsApp message
    # Send to orchestrator
    # Return response to WhatsApp
    pass
```

### 3. Environment variables for WhatsApp
```bash
# Add to .env
WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_ACCESS_TOKEN=your_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using ports 8080/8001
   lsof -i :8080
   lsof -i :8001
   ```

2. **Docker build fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   # Rebuild
   docker compose build --no-cache
   ```

3. **Gemini API errors**
   - Verify API key is correct
   - Check quota limits in Google Cloud Console
   - Ensure API is enabled

4. **Import errors**
   ```bash
   # Install dependencies locally
   pip install -r requirements-dev.txt
   ```

## Performance Testing

```bash
# Load test the system
make test-performance

# Or manually with curl
for i in {1..10}; do
  curl -X POST http://localhost:8080/api/v1/orchestrate-direct \
    -H "Content-Type: application/json" \
    -d '{"user_message": "Test message '$i'", "user_id": "test_user", "session_id": "test_session"}' &
done
```

## Monitoring

### View logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f orchestrator
docker compose logs -f classifier
```

### Metrics endpoint
```bash
# Get system metrics
curl http://localhost:8080/api/v1/metrics
```

## Success Criteria

✅ All services start without errors  
✅ Health checks return "healthy"  
✅ CLI can classify messages correctly  
✅ API endpoints respond with valid JSON  
✅ Different message types get classified properly:
   - Product questions → "product_information"
   - Complaints → "PQR" 
   - General chat → "other"