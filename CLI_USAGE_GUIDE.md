# 🚀 Guía Completa del CLI - ChatCommerce Bot

Esta guía te enseña cómo usar el CLI para hacer tus propias pruebas del sistema multi-agente.

## 📋 Prerrequisitos

1. **Servicios corriendo:**
   ```bash
   # Verificar que los servicios estén activos
   lsof -i :8001  # Classifier
   lsof -i :8080  # Orchestrator
   ```

2. **Variables de entorno configuradas:**
   ```bash
   # Verificar que GEMINI_API_KEY esté configurado
   cat .env | grep GEMINI_API_KEY
   ```

## 🎯 Formas de Usar el CLI

### 1. **Modo Interactivo** (Recomendado para sesiones largas)

```bash
cd "context-engineering-intro"
python3 -m cli.main
```

**Características:**
- Conversación continua
- Historial de mensajes
- Comandos especiales
- Sesión persistente

**Comandos disponibles en modo interactivo:**
- `help` - Mostrar ayuda
- `status` - Estado del servicio
- `health` - Verificar salud del sistema
- `metrics` - Métricas de rendimiento
- `history` - Historial de conversación
- `clear` - Limpiar historial
- `quit` o `exit` - Salir

### 2. **Mensaje Único** (Para pruebas rápidas)

```bash
# Enviar un mensaje específico
python3 -m cli.main -m "¿Cuál es el precio del iPhone 15?"

# Con usuario personalizado
python3 -m cli.main -m "Mi pedido está retrasado" --user-id "cliente_123"
```

### 3. **Pruebas Automatizadas** (Script de pruebas)

```bash
# Ejecutar script de pruebas automatizadas
python3 quick_test.py
```

### 4. **Verificación de Conexión**

```bash
# Solo probar la conexión
python3 -m cli.main --test-connection
```

## 🧪 Casos de Prueba Sugeridos

### **Categoría: Información de Productos** 
(Debería clasificarse como `product_information`)

```
- "¿Cuál es el precio del iPhone 15?"
- "Necesito auriculares inalámbricos buenos"
- "¿Tienen laptops disponibles para estudiantes?"
- "Quiero comprar una tablet Samsung"
- "¿Cuáles son las especificaciones del MacBook Pro?"
- "Busco zapatos deportivos talla 42"
- "¿Hay promociones en televisores?"
```

### **Categoría: PQR** (Problemas, Quejas, Reclamos)
(Debería clasificarse como `PQR`)

```
- "Mi pedido está retrasado desde hace una semana"
- "Quiero devolver este producto defectuoso"
- "El envío nunca llegó a mi dirección"
- "La calidad del producto no es la esperada"
- "Necesito cambiar la dirección de entrega"
- "El producto llegó dañado en el empaque"
- "No estoy satisfecho con mi compra"
```

### **Categoría: General** 
(Debería clasificarse como `other`)

```
- "Hola, ¿cómo estás?"
- "¿Cuál es el horario de atención?"
- "Gracias por tu ayuda"
- "Buenos días"
- "¿Cómo puedo contactar con soporte?"
- "¿Dónde están ubicados?"
```

## 📊 Interpretando los Resultados

### **Respuesta Típica:**

```
🤖 Assistant Response
┌─────────────────────────────────────────────────────────────┐
│ Gracias por tu consulta sobre productos. Te ayudo con      │
│ información sobre precios y características.               │
└─────────────────────────────────────────────────────────────┘

🛍️ product_information
📊 Confidence: 85.00%
⏱️ Processing Time: 1.20s
```

### **Elementos a Verificar:**

1. **Clasificación Correcta:**
   - 🛍️ = product_information 
   - ❓ = PQR
   - ❌ = other

2. **Confianza Alta:** >70%

3. **Tiempo de Respuesta:** <3 segundos

4. **Respuesta Contextual:** Apropiada para la categoría

## 🔧 Opciones Avanzadas

### **Configuración Personalizada:**

```bash
# URL diferente
python3 -m cli.main --url "http://localhost:8080"

# Timeout personalizado
python3 -m cli.main --timeout 60

# Más reintentos
python3 -m cli.main --retries 5

# Modo verbose para debug
python3 -m cli.main --verbose
```

### **Para Desarrollo:**

```bash
# Ejecutar con logs detallados
python3 -m cli.main --verbose --user-id "dev_user"

# Probar con API key específica
python3 -m cli.main --api-key "tu_api_key_aqui"
```

## 🚨 Solución de Problemas

### **Error: "Connection failed"**
```bash
# Verificar servicios
lsof -i :8001 :8080

# Reiniciar si es necesario
python3 start_classifier.py &
python3 start_orchestrator.py &
```

### **Error: "Classification failed"**
```bash
# Verificar API key
grep GEMINI_API_KEY .env

# Probar conexión directa
curl http://localhost:8001/api/v1/health
```

### **Respuestas Lentas**
```bash
# Aumentar timeout
python3 -m cli.main --timeout 60
```

## 📈 Métricas de Éxito

Para una implementación exitosa, busca:

- ✅ **Precisión:** >85% en clasificación correcta
- ✅ **Velocidad:** <3 segundos de respuesta
- ✅ **Disponibilidad:** Servicios siempre activos
- ✅ **Consistencia:** Resultados estables entre pruebas

## 🎯 Flujo de Prueba Recomendado

1. **Verificar Conexión:** `--test-connection`
2. **Probar Casos Básicos:** Script automatizado
3. **Sesión Interactiva:** Pruebas manuales detalladas
4. **Verificar Métricas:** Comando `metrics` en CLI
5. **Documentar Resultados:** Casos que fallan/funcionan

¡Ahora puedes hacer todas las pruebas que necesites! 🚀