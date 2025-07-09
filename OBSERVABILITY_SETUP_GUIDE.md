# 📊 Guía de Configuración de Observabilidad

Esta guía te muestra cómo configurar **Logfire** y **Arize AX** para monitoreo completo de tu ChatCommerce Bot.

## 🎯 ¿Qué incluye la observabilidad?

### **Métricas Automáticas:**
- ✅ **LLM Traces**: Prompts, respuestas, latencia, confianza
- ✅ **Agent Operations**: Tiempos de ejecución, errores, estados
- ✅ **Classification Metrics**: Precisión, distribución de categorías
- ✅ **Performance Monitoring**: Latencia end-to-end, throughput
- ✅ **Error Tracking**: Fallos de clasificación, timeouts, excepciones

### **Integraciones Disponibles:**
1. **Pydantic Logfire** - Logging estructurado y trazas
2. **Arize AX** - Monitoreo de LLM y evaluación de modelos
3. **OpenTelemetry** - Trazas distribuidas (opcional)

## 🚀 Configuración Básica (Sin Credenciales Externas)

El sistema ya funciona con observabilidad básica **sin configuración adicional**:

```bash
# Ver métricas del clasificador
curl http://localhost:8001/api/v1/metrics

# Ver métricas del orquestador  
curl http://localhost:8080/api/v1/metrics
```

### **Ejemplo de Métricas Básicas:**
```json
{
  "timestamp": "2025-07-08T20:52:42.566025+00:00",
  "total_llm_traces": 1,
  "integrations": {
    "logfire_enabled": false,
    "arize_enabled": false,
    "otel_enabled": false
  },
  "recent_traces": [
    {
      "trace_id": "6aa7a352-859d-4736-9635-d8269cc98ef3",
      "model_name": "gemini-2.0-flash-exp",
      "prompt": "¿Cuánto cuesta el iPhone 15 Pro?",
      "response": "product_information",
      "latency_ms": 961.93,
      "classification_label": "product_information",
      "confidence_score": 0.85,
      "agent_name": "classifier",
      "timestamp": "2025-07-08T20:52:29.352332+00:00"
    }
  ]
}
```

## 🔥 Configuración Avanzada con Logfire

### **1. Crear Cuenta en Logfire**
1. Ve a: https://logfire.pydantic.dev/
2. Crea una cuenta gratuita
3. Crea un nuevo proyecto: `whatsapp-sales-assistant`
4. Obtén tu **token de API**

### **2. Configurar Variables de Entorno**
Agrega a tu archivo `.env`:

```bash
# Logfire Configuration
LOGFIRE_TOKEN="your_logfire_token_here"
LOGFIRE_PROJECT_NAME="whatsapp-sales-assistant"
LOGFIRE_ENVIRONMENT="development"  # o "production"
```

### **3. Reiniciar Servicios**
```bash
# Detener servicios
pkill -f "start_classifier.py"
pkill -f "start_orchestrator.py"

# Reiniciar con nueva configuración
python3 start_classifier.py &
python3 start_orchestrator.py &
```

### **4. Verificar Integración**
```bash
# Las métricas ahora mostrarán logfire_enabled: true
curl http://localhost:8001/api/v1/metrics
```

## 📈 Configuración Avanzada con Arize AX

### **1. Crear Cuenta en Arize**
1. Ve a: https://app.arize.com/
2. Crea una cuenta (tienen tier gratuito)
3. Crea un nuevo **Space** para el proyecto
4. Obtén tus credenciales:
   - **API Key**
   - **Space Key**

### **2. Configurar Variables de Entorno**
Agrega a tu archivo `.env`:

```bash
# Arize AX Configuration
ARIZE_API_KEY="your_arize_api_key_here"
ARIZE_SPACE_KEY="your_arize_space_key_here"
ARIZE_MODEL_ID="whatsapp-chatcommerce-bot"
ARIZE_MODEL_VERSION="1.0.0"
```

### **3. Reiniciar y Verificar**
```bash
# Reiniciar servicios
pkill -f "start_classifier.py" && pkill -f "start_orchestrator.py"
python3 start_classifier.py &
python3 start_orchestrator.py &

# Verificar integración
curl http://localhost:8001/api/v1/metrics
# Debería mostrar arize_enabled: true
```

### **4. Ver en Arize Dashboard**
- Ve a tu dashboard de Arize
- Deberías ver datos de LLM llegando en tiempo real
- Métricas incluyen: latencia, distribución de clasificaciones, confidence scores

## 🔧 Configuración OpenTelemetry (Opcional)

Para trazas distribuidas avanzadas:

```bash
# OpenTelemetry Configuration
OTEL_SERVICE_NAME="whatsapp-sales-assistant"
OTEL_EXPORTER_ENDPOINT="https://your-otlp-endpoint.com"
OTEL_HEADERS="authorization=Bearer your_token"
```

## 📊 Dashboard y Monitoreo

### **Métricas Clave a Monitorear:**

1. **Latencia de Clasificación**
   - Target: < 2 segundos
   - Alertar si > 5 segundos

2. **Precisión de Clasificación**
   - Target: > 90%
   - Alertar si < 85%

3. **Distribución de Categorías**
   - `product_information`: ~40%
   - `PQR`: ~35%
   - `other`: ~25%

4. **Tasa de Errores**
   - Target: < 1%
   - Alertar si > 5%

### **Consultas Útiles de Logfire:**

```python
# Buscar errores de clasificación
level:ERROR AND agent_name:classifier

# Latencia alta
latency_ms:>3000

# Clasificaciones con baja confianza
confidence_score:<0.7
```

### **Consultas Útiles de Arize:**

```sql
-- Promedio de latencia por día
SELECT DATE(timestamp), AVG(latency_ms) 
FROM predictions 
GROUP BY DATE(timestamp)

-- Distribución de clasificaciones
SELECT classification_label, COUNT(*) as count
FROM predictions 
GROUP BY classification_label
```

## 🚨 Alertas Recomendadas

### **Alertas Críticas:**
- Latencia > 5 segundos
- Tasa de errores > 5%
- Servicio no disponible

### **Alertas de Advertencia:**
- Latencia > 2 segundos
- Confianza promedio < 80%
- Clasificaciones desbalanceadas

## 🔍 Troubleshooting

### **Problema: logfire_enabled: false**
```bash
# Verificar token
echo $LOGFIRE_TOKEN

# Verificar conectividad
curl -H "Authorization: Bearer $LOGFIRE_TOKEN" https://logfire-api.pydantic.dev/v1/info
```

### **Problema: arize_enabled: false**
```bash
# Verificar credenciales
echo $ARIZE_API_KEY
echo $ARIZE_SPACE_KEY

# Test de conectividad con Arize
python3 -c "
from arize.pandas.logger import Client
client = Client(api_key='$ARIZE_API_KEY', space_key='$ARIZE_SPACE_KEY')
print('Arize connection OK')
"
```

### **Problema: No aparecen métricas**
```bash
# Verificar que los servicios estén corriendo
lsof -i :8001
lsof -i :8080

# Hacer una prueba para generar métricas
python3 -m cli.main -m "test message"

# Verificar logs
tail -f /var/log/classifier.log  # si tienes logs configurados
```

## 🎯 Próximos Pasos

1. **Configurar Alertas**: Set up alertas en Logfire/Arize
2. **Dashboards Personalizados**: Crear vistas para tu equipo
3. **A/B Testing**: Usar Arize para experimentar con diferentes prompts
4. **Performance Optimization**: Usar métricas para optimizar latencia
5. **Quality Monitoring**: Implementar evaluación continua de calidad

## 📚 Recursos Adicionales

- [Documentación Logfire](https://logfire.pydantic.dev/docs/)
- [Documentación Arize](https://docs.arize.com/)
- [OpenTelemetry Guía](https://opentelemetry.io/docs/)
- [CLI del Sistema](CLI_USAGE_GUIDE.md)

¡Tu ChatCommerce Bot ahora tiene observabilidad de nivel enterprise! 🚀