#!/bin/bash
# Quick Start Script for Testing Observability

echo "🚀 WhatsApp Bot Observability - Quick Start"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env and add your API keys before continuing${NC}"
    echo "   Required keys:"
    echo "   - GEMINI_API_KEY"
    echo "   - ARIZE_API_KEY (optional)"
    echo "   - ARIZE_SPACE_ID (optional)"
    echo "   - LOGFIRE_TOKEN (optional)"
    echo ""
    read -p "Press Enter after configuring .env..."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check requirements
echo "📋 Checking requirements..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${YELLOW}⚠️  Docker not found - some tests will be limited${NC}"
fi

if ! command_exists make; then
    echo -e "${YELLOW}⚠️  Make not found - running commands directly${NC}"
fi

echo -e "${GREEN}✅ Basic requirements met${NC}"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt >/dev/null 2>&1 || {
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
}
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Menu
echo "🎯 What would you like to test?"
echo ""
echo "1) Run ALL observability tests (recommended)"
echo "2) Test cost calculations only"
echo "3) Test decision tracking only"
echo "4) Test Arize integration only"
echo "5) Start services and test end-to-end"
echo "6) Generate observability report"
echo "7) Exit"
echo ""

read -p "Select an option (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running all observability tests..."
        python3 test_observability_complete.py
        ;;
    2)
        echo ""
        echo "💰 Testing cost calculations..."
        python3 test_cost_calculations.py
        ;;
    3)
        echo ""
        echo "🎯 Testing decision tracking..."
        python3 test_decision_tracking.py
        ;;
    4)
        echo ""
        echo "📊 Testing Arize integration..."
        python3 test_observability.py
        ;;
    5)
        echo ""
        echo "🐳 Starting services with Docker..."
        if command_exists docker; then
            docker compose up -d
            echo "Waiting for services to start..."
            sleep 10
            
            echo ""
            echo "🧪 Running end-to-end tests..."
            
            # Test health
            echo "Checking service health..."
            curl -s http://localhost:8001/health >/dev/null && echo -e "${GREEN}✅ Classifier healthy${NC}" || echo -e "${RED}❌ Classifier unhealthy${NC}"
            curl -s http://localhost:8000/health >/dev/null && echo -e "${GREEN}✅ Orchestrator healthy${NC}" || echo -e "${RED}❌ Orchestrator unhealthy${NC}"
            
            # Run CLI tests
            echo ""
            echo "Testing with CLI..."
            python3 -m cli.main "¿Cuál es el precio del iPhone 15?"
            sleep 2
            python3 -m cli.main "Mi pedido no ha llegado"
            
            # Check metrics
            echo ""
            echo "📊 Checking metrics..."
            curl -s http://localhost:8001/api/v1/observability-metrics | python3 -m json.tool
            
            echo ""
            read -p "Press Enter to stop services..."
            docker compose down
        else
            echo -e "${RED}Docker not available - skipping service tests${NC}"
        fi
        ;;
    6)
        echo ""
        echo "📊 Generating observability report..."
        python3 -c "
from shared.observability import get_metrics_summary
from shared.observability_cost import session_cost_aggregator
import json
from datetime import datetime

print('📊 OBSERVABILITY REPORT')
print(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('=' * 60)

metrics = get_metrics_summary()
print('\n🔢 Metrics Summary:')
print(json.dumps(metrics, indent=2, default=str))

sessions = session_cost_aggregator.get_all_sessions()
if sessions:
    print('\n💰 Session Costs:')
    total_cost = 0
    for sid, costs in sessions.items():
        print(f'  {sid}: Total=\${costs[\"total\"]:.6f} (LLM=\${costs[\"llm\"]:.6f}, WhatsApp=\${costs[\"whatsapp\"]:.6f})')
        total_cost += costs['total']
    print(f'  TOTAL: \${total_cost:.6f}')
else:
    print('\n💰 No session costs recorded yet')

print('\n✅ Report complete')
"
        ;;
    7)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
echo ""
echo "📚 Next steps:"
echo "   - Check the test results above"
echo "   - Review logs for any errors"
echo "   - Visit Arize dashboard: https://app.arize.com"
echo "   - Read the full guide: OBSERVABILITY_TEST_GUIDE.md"