#!/bin/bash
# Script to check ngrok status and restart if needed

check_ngrok() {
    if pgrep -x "ngrok" > /dev/null; then
        echo "✅ ngrok is running"
        
        # Get current URL
        URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$URL" ]; then
            echo "📡 Webhook URL: $URL/webhook/whatsapp"
            echo ""
            echo "Remember to update this URL in Meta Developer Console if it changed!"
        else
            echo "⚠️  Could not get ngrok URL"
        fi
    else
        echo "❌ ngrok is NOT running"
        echo ""
        echo "To start ngrok:"
        echo "  ngrok http 8000"
        echo ""
        echo "Or run in background:"
        echo "  nohup ngrok http 8000 > ngrok.log 2>&1 &"
    fi
}

# Check services status too
echo "=== Service Status Check ==="
echo ""

check_ngrok
echo ""

# Check if orchestrator is running
if pgrep -f "uvicorn.*orchestrator" > /dev/null; then
    echo "✅ Orchestrator is running on port 8000"
else
    echo "❌ Orchestrator is NOT running"
fi

# Check if classifier is running
if pgrep -f "uvicorn.*classifier" > /dev/null; then
    echo "✅ Classifier is running on port 8001"
else
    echo "❌ Classifier is NOT running"
fi

echo ""
echo "=== Ngrok Activity (last 10 requests) ==="
if [ -f ngrok.log ]; then
    tail -n 10 ngrok.log 2>/dev/null | grep -E "POST|GET" || echo "No recent activity"
fi