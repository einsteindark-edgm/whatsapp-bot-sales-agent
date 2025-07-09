# Start WhatsApp Test Environment

Start all services needed for WhatsApp API testing including classifier, orchestrator, and prepare for ngrok.

## Steps

1. Check if .env file exists and has WhatsApp credentials configured
2. Start the classifier service on port 8001
3. Start the orchestrator service on port 8000
4. Verify both services are healthy
5. Provide instructions for ngrok setup

## Implementation

```bash
#!/bin/bash
# Start WhatsApp test environment

echo "🚀 Starting WhatsApp test environment..."

# Check .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please copy .env.example to .env and configure credentials"
    exit 1
fi

# Load and check WhatsApp credentials
set -a
source .env
set +a

if [ -z "$WHATSAPP_ACCESS_TOKEN" ] || [ "$WHATSAPP_ACCESS_TOKEN" = "your_permanent_access_token_here" ]; then
    echo "❌ Error: WhatsApp credentials not configured in .env"
    echo "Please set:"
    echo "  - WHATSAPP_ACCESS_TOKEN"
    echo "  - WHATSAPP_PHONE_NUMBER_ID"
    echo "  - WHATSAPP_WEBHOOK_VERIFY_TOKEN"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Stopping services..."
    if [ ! -z "$CLASSIFIER_PID" ]; then
        kill $CLASSIFIER_PID 2>/dev/null
    fi
    if [ ! -z "$ORCHESTRATOR_PID" ]; then
        kill $ORCHESTRATOR_PID 2>/dev/null
    fi
    exit
}
trap cleanup EXIT INT TERM

# Start classifier
echo "1️⃣ Starting Classifier service on port 8001..."
python3 -m uvicorn agents.classifier.main:app --host 0.0.0.0 --port 8001 > classifier.log 2>&1 &
CLASSIFIER_PID=$!
echo "   PID: $CLASSIFIER_PID"
sleep 3

# Start orchestrator
echo "2️⃣ Starting Orchestrator service on port 8000..."
python3 -m uvicorn agents.orchestrator.main:app --host 0.0.0.0 --port 8000 > orchestrator.log 2>&1 &
ORCHESTRATOR_PID=$!
echo "   PID: $ORCHESTRATOR_PID"
sleep 3

# Verify services
echo -e "\n3️⃣ Verifying services..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Classifier is running"
else
    echo "❌ Classifier failed to start - check classifier.log"
fi

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Orchestrator is running"
else
    echo "❌ Orchestrator failed to start - check orchestrator.log"
fi

# Test WhatsApp health endpoint
if curl -s http://localhost:8000/webhook/whatsapp/health > /dev/null; then
    echo "✅ WhatsApp webhook endpoint is ready"
else
    echo "❌ WhatsApp webhook endpoint not responding"
fi

echo -e "\n📱 Services started successfully!"
echo -e "\n🔗 Next steps:"
echo "1. In a new terminal, run: ngrok http 8000"
echo "2. Copy the HTTPS URL from ngrok (e.g., https://abc123.ngrok-free.app)"
echo "3. Configure webhook in Meta Developer Console:"
echo "   - Callback URL: https://YOUR-NGROK-ID.ngrok-free.app/webhook/whatsapp"
echo "   - Verify Token: $WHATSAPP_WEBHOOK_VERIFY_TOKEN"
echo "4. Subscribe to 'messages' webhook field"
echo -e "\n📊 Services are running. Press Ctrl+C to stop all services."
echo -e "\n📝 Logs:"
echo "   - Classifier: tail -f classifier.log"
echo "   - Orchestrator: tail -f orchestrator.log"
echo -e "\n"

# Keep running and show logs
tail -f orchestrator.log | grep -E "(WhatsApp|ERROR|WARNING)"
```

## Success Criteria

- Both classifier and orchestrator services start successfully
- Health endpoints respond with 200 OK
- WhatsApp webhook endpoint is accessible
- Clear instructions provided for ngrok setup
- Graceful shutdown on Ctrl+C

## Error Handling

- Check for .env file existence
- Validate WhatsApp credentials are configured
- Verify services start successfully
- Provide clear error messages
- Clean shutdown of all processes