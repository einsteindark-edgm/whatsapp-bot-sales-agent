#!/bin/bash
# Script para iniciar todos los servicios para prueba de WhatsApp

echo "🚀 Iniciando servicios para prueba de WhatsApp..."

# Verificar que existe .env
if [ ! -f .env ]; then
    echo "❌ Error: No se encontró archivo .env"
    echo "Por favor copia .env.example a .env y configura las credenciales"
    exit 1
fi

# Verificar credenciales WhatsApp
source .env
if [ -z "$WHATSAPP_ACCESS_TOKEN" ] || [ "$WHATSAPP_ACCESS_TOKEN" = "your_permanent_access_token_here" ]; then
    echo "❌ Error: WHATSAPP_ACCESS_TOKEN no está configurado en .env"
    exit 1
fi

# Función para matar procesos al salir
cleanup() {
    echo "🛑 Deteniendo servicios..."
    kill $CLASSIFIER_PID $ORCHESTRATOR_PID 2>/dev/null
    exit
}
trap cleanup EXIT

# Iniciar classifier
echo "1️⃣ Iniciando Classifier en puerto 8001..."
python3 -m uvicorn agents.classifier.main:app --host 0.0.0.0 --port 8001 &
CLASSIFIER_PID=$!
sleep 3

# Iniciar orchestrator
echo "2️⃣ Iniciando Orchestrator en puerto 8000..."
python3 -m uvicorn agents.orchestrator.main:app --host 0.0.0.0 --port 8000 &
ORCHESTRATOR_PID=$!
sleep 3

# Verificar servicios
echo "3️⃣ Verificando servicios..."
curl -s http://localhost:8001/health > /dev/null && echo "✅ Classifier OK" || echo "❌ Classifier FAILED"
curl -s http://localhost:8000/health > /dev/null && echo "✅ Orchestrator OK" || echo "❌ Orchestrator FAILED"

echo ""
echo "📱 Servicios iniciados!"
echo ""
echo "Próximos pasos:"
echo "1. En otra terminal ejecuta: ngrok http 8000"
echo "2. Copia la URL HTTPS de ngrok"
echo "3. Configura el webhook en Meta Developer Console:"
echo "   - URL: https://TU-NGROK-ID.ngrok-free.app/webhook/whatsapp"
echo "   - Token: $WHATSAPP_WEBHOOK_VERIFY_TOKEN"
echo ""
echo "📊 Logs activos. Presiona Ctrl+C para detener..."

# Mantener el script corriendo
wait