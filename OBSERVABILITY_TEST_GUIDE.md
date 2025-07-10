# Guía de Pruebas de Observabilidad

## 1. Preparación del Entorno

### Verificar Variables de Entorno
```bash
# Asegúrate de tener estas variables en tu .env
cat .env | grep -E "GEMINI_API_KEY|ARIZE_API_KEY|ARIZE_SPACE_ID|LOGFIRE_TOKEN|TRACE_ENABLED"
```

### Instalar Dependencias
```bash
pip install -r requirements.txt
```

## 2. Pruebas de Componentes Individuales

### 2.1 Probar Cálculo de Costos
```bash
python test_cost_calculations.py
```
**Qué esperar:**
- ✅ Cálculos correctos para Gemini 1.5/2.0 Flash
- ✅ Costos de WhatsApp por tipo y país
- ✅ Agregación de costos por sesión

### 2.2 Probar Tracking de Decisiones
```bash
# Prueba simple y directa
python test_decision_tracking_simple.py

# Prueba completa end-to-end
python test_decision_tracking.py
```
**Qué esperar:**
- ✅ Tracking de decisiones del clasificador
- ✅ Tracking de decisiones del orquestador
- ✅ Tags y atributos correctamente registrados

### 2.3 Probar Observabilidad con Arize
```bash
python test_observability.py
```
**Qué esperar:**
- ✅ Conexión exitosa con Arize
- ✅ Spans enviados correctamente
- ✅ Métricas LLM registradas

## 3. Pruebas de Integración

### 3.1 Iniciar los Servicios
```bash
# Opción 1: Con Docker
docker compose up

# Opción 2: Localmente (en terminales separadas)
# Terminal 1 - Clasificador
python -m uvicorn agents.classifier.main:app --host 0.0.0.0 --port 8001

# Terminal 2 - Orquestador
python -m uvicorn agents.orchestrator.main:app --host 0.0.0.0 --port 8000
```

### 3.2 Verificar Health Checks
```bash
# Clasificador
curl http://localhost:8001/health

# Orquestador
curl http://localhost:8000/health
```

### 3.3 Probar Flujo Completo con CLI
```bash
# Mensaje de información de producto
python -m cli.main "¿Cuál es el precio del iPhone 15?"

# Mensaje de PQR (queja)
python -m cli.main "Mi pedido no ha llegado y estoy muy molesto"
```

### 3.4 Probar Webhook de WhatsApp
```bash
# Verificación del webhook
curl -X GET "http://localhost:8000/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=test_verify_token&hub.challenge=test123"

# Enviar mensaje de prueba
curl -X POST http://localhost:8000/webhook/whatsapp \
  -H "Content-Type: application/json" \
  -d '{
    "entry": [{
      "id": "ENTRY_ID",
      "changes": [{
        "value": {
          "messaging_product": "whatsapp",
          "metadata": {
            "display_phone_number": "15550555555",
            "phone_number_id": "PHONE_NUMBER_ID"
          },
          "messages": [{
            "from": "521234567890",
            "id": "wamid.ID",
            "timestamp": "1234567890",
            "text": {
              "body": "¿Tienen audífonos inalámbricos en stock?"
            },
            "type": "text"
          }]
        },
        "field": "messages"
      }]
    }]
  }'
```

## 4. Verificar Métricas y Trazas

### 4.1 Ver Métricas Locales
```bash
curl http://localhost:8001/api/v1/observability-metrics
```

### 4.2 Verificar en Arize Console
1. Ve a https://app.arize.com
2. Busca el proyecto "whatsapp-bot-agent"
3. Verifica:
   - Spans de clasificación
   - Métricas de LLM (tokens, latencia, costo)
   - Decisiones con tags

### 4.3 Script de Verificación Completa
```bash
python -c "
import asyncio
from cli.main import main as cli_main
import time

async def test_flow():
    print('🧪 Probando flujo completo...')
    
    # Test 1: Clasificación de producto
    print('\n1️⃣ Mensaje de producto:')
    await cli_main(['¿Cuánto cuesta el Samsung Galaxy S24?'])
    time.sleep(2)
    
    # Test 2: Clasificación de PQR
    print('\n2️⃣ Mensaje de queja:')
    await cli_main(['No he recibido mi pedido #12345'])
    time.sleep(2)
    
    # Test 3: Mensaje ambiguo
    print('\n3️⃣ Mensaje ambiguo:')
    await cli_main(['Hola, necesito ayuda'])
    
    print('\n✅ Pruebas completadas!')
    print('📊 Revisa las métricas en: http://localhost:8001/api/v1/observability-metrics')
    print('🔍 Revisa Arize en: https://app.arize.com')

asyncio.run(test_flow())
"
```

## 5. Validar Resultados Esperados

### ✅ En los Logs deberías ver:
- Trace IDs propagándose entre servicios
- Decisiones de clasificación con confianza
- Costos calculados por llamada
- Tags y atributos en las operaciones

### ✅ En las Métricas deberías ver:
- `agent_operations_total` con diferentes operaciones
- `classification_decision` para el clasificador
- `classification_received` y `request_classification` para el orquestador
- Tiempos de procesamiento

### ✅ En Arize deberías ver:
- Proyecto "whatsapp-bot-agent" activo
- Spans con el modelo "google-gla:gemini-2.0-flash"
- Métricas de tokens y latencia
- Evaluaciones de calidad (si están habilitadas)

## 6. Troubleshooting

### Si no ves datos en Arize:
```bash
# Verifica las credenciales
echo $ARIZE_API_KEY
echo $ARIZE_SPACE_ID

# Prueba conexión directa
python test_arize_otel.py
```

### Si hay errores de Logfire:
```bash
# Desactiva temporalmente Logfire
export TRACE_ENABLED=false
```

### Si los servicios no se conectan:
```bash
# Verifica los puertos
lsof -i :8000
lsof -i :8001

# Verifica las URLs en la configuración
grep -r "localhost:8001" agents/
```

## 7. Generar Reporte de Pruebas
```bash
python -c "
from shared.observability import get_metrics_summary
from shared.observability_cost import session_cost_aggregator
import json

print('📊 REPORTE DE OBSERVABILIDAD')
print('=' * 50)

# Métricas
metrics = get_metrics_summary()
print('\n🔢 Operaciones Rastreadas:')
if 'counters' in metrics['metrics']:
    for op, count in sorted(metrics['metrics']['counters'].items()):
        if 'agent_operations_total' in op:
            print(f'  {op}: {count}')

# Costos
print('\n💰 Costos por Sesión:')
all_sessions = session_cost_aggregator.get_all_sessions()
for session_id, costs in all_sessions.items():
    print(f'  {session_id}: Total=${costs[\"total\"]:.6f} (LLM=${costs[\"llm\"]:.6f})')

print('\n✅ Reporte completado')
"
```

## Comandos Rápidos

```bash
# Todo en uno - prueba completa
make test-observability

# O manualmente:
python test_cost_calculations.py && \
python test_decision_tracking.py && \
python test_observability.py && \
echo "✅ Todas las pruebas de observabilidad pasaron!"
```