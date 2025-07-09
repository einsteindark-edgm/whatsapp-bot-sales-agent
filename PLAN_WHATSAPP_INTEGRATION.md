# 📱 Plan de Integración WhatsApp Business API
## Fecha: 2025-07-09

### 🎯 **Objetivo del Día**
Implementar integración completa con WhatsApp Business API para recibir y responder mensajes automáticamente a través del orquestador.

---

## 📋 **FASE 1: Setup & Configuración (30 min)**

### 1.1 Configuración WhatsApp Business API
- [ ] **Crear/verificar Meta Business Account**
- [ ] **Configurar WhatsApp Business App** en Meta Developers
- [ ] **Obtener credenciales clave:**
  - `ACCESS_TOKEN` (permanente, no temporal)
  - `PHONE_NUMBER_ID` 
  - `WHATSAPP_BUSINESS_ACCOUNT_ID`
  - `WEBHOOK_VERIFY_TOKEN`

### 1.2 Configuración de Entorno
- [ ] **Agregar variables al .env:**
```bash
# WhatsApp Business API
WHATSAPP_ACCESS_TOKEN=your_permanent_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token
WHATSAPP_API_VERSION=v18.0
```

---

## 📋 **FASE 2: Análisis de Formato de Datos (30 min)**

### 2.1 Webhook Incoming Format (WhatsApp → Orquestador)
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
        }],
        "contacts": [{
          "profile": {
            "name": "CUSTOMER_NAME"
          },
          "wa_id": "CUSTOMER_WHATSAPP_ID"
        }]
      },
      "field": "messages"
    }]
  }]
}
```

### 2.2 Response Format (Orquestador → WhatsApp)
```json
{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "CUSTOMER_PHONE_NUMBER",
  "type": "text",
  "text": {
    "body": "RESPONSE_MESSAGE"
  }
}
```

---

## 📋 **FASE 3: Implementación Backend (120 min)**

### 3.1 Crear Webhook Endpoint en Orquestador (45 min)
- [ ] **Crear `whatsapp_webhook_router.py`** en `/agents/orchestrator/adapters/inbound/`
- [ ] **Implementar endpoints:**
  - `GET /webhook/whatsapp` - Verificación de webhook
  - `POST /webhook/whatsapp` - Recepción de mensajes

### 3.2 Extractor de Mensajes WhatsApp (30 min)
- [ ] **Crear `whatsapp_message_extractor.py`** en `/agents/orchestrator/domain/`
- [ ] **Funciones principales:**
  - `extract_message_from_webhook(webhook_data)` 
  - `validate_whatsapp_webhook(data)`
  - `extract_user_info(webhook_data)`

### 3.3 Cliente WhatsApp API (45 min)
- [ ] **Crear `whatsapp_api_client.py`** en `/agents/orchestrator/adapters/outbound/`
- [ ] **Implementar métodos:**
  - `send_text_message(to, message)`
  - `send_typing_indicator(to)`
  - `mark_message_as_read(message_id)`

---

## 📋 **FASE 4: Integración con Flujo Existente (60 min)**

### 4.1 Modificar Orquestador (30 min)
- [ ] **Crear `whatsapp_workflow_handler.py`**
- [ ] **Integrar con workflow existente:**
  - Recibir webhook → Extraer mensaje → Procesar con clasificador → Enviar respuesta

### 4.2 Actualizar Modelos de Dominio (30 min)
- [ ] **Extender `WorkflowRequest`** para incluir datos WhatsApp
- [ ] **Crear `WhatsAppMessage` model**
- [ ] **Actualizar observabilidad** para incluir métricas WhatsApp

---

## 📋 **FASE 5: Estructura de Archivos a Crear**

```
/agents/orchestrator/
├── adapters/inbound/
│   └── whatsapp_webhook_router.py       # 🆕 Webhook endpoints
├── adapters/outbound/
│   └── whatsapp_api_client.py           # 🆕 WhatsApp API client
├── domain/
│   ├── whatsapp_message_extractor.py    # 🆕 Message extraction
│   └── whatsapp_models.py               # 🆕 WhatsApp domain models
├── application/
│   └── whatsapp_workflow_handler.py     # 🆕 WhatsApp workflow
└── main.py                              # ✏️ Actualizar rutas
```

---

## 📋 **FASE 6: Testing & Verificación (60 min)**

### 6.1 Testing Local (30 min)
- [ ] **Usar ngrok** para exponer localhost
- [ ] **Configurar webhook URL** en Meta Developers
- [ ] **Test de verificación:** `GET /webhook/whatsapp?hub.challenge=test`

### 6.2 Testing End-to-End (30 min)
- [ ] **Enviar mensaje WhatsApp real** al número de prueba
- [ ] **Verificar recepción** en webhook
- [ ] **Verificar clasificación** funciona
- [ ] **Verificar respuesta** llega a WhatsApp

---

## 📋 **FASE 7: Observabilidad WhatsApp (30 min)**

### 7.1 Métricas Específicas
- [ ] **Agregar traces WhatsApp** a observabilidad existente
- [ ] **Métricas a capturar:**
  - Mensajes recibidos por hora
  - Tiempo de respuesta WhatsApp
  - Tipos de clasificación por WhatsApp
  - Errores de API WhatsApp

---

## 🔧 **Configuración Técnica Clave**

### Endpoints WhatsApp API:
- **Send Message:** `https://graph.facebook.com/v18.0/{phone-number-id}/messages`
- **Authentication:** `Authorization: Bearer {access-token}`

### Webhook Requirements:
- **HTTPS obligatorio** (usar ngrok para desarrollo)
- **Verificación token** requerida
- **Respuesta 200** obligatoria para confirmación

### Rate Limits:
- **80 mensajes/segundo** máximo
- **250 usuarios únicos** en 24h (incrementa por tiers)

---

## ⚠️ **Puntos Críticos - NO OLVIDAR**

1. **🔐 SEGURIDAD**
   - Validar webhook signature
   - No exponer tokens en logs
   - Verificar origen de webhooks

2. **🚀 RENDIMIENTO**
   - Procesar webhooks asincrónicamente
   - Responder 200 inmediatamente
   - Queue para procesamiento en background

3. **📊 OBSERVABILIDAD**
   - Trace completo: WhatsApp → Clasificador → WhatsApp
   - Monitorear errores de API
   - Alertas por volumen inusual

4. **🔄 MANEJO ERRORES**
   - Retry automático para errores temporales
   - Fallback para errores de clasificación
   - Log detallado para debugging

---

## 🎯 **Resultado Esperado Final**

Al final del día tendremos:
✅ **Sistema completo funcionando:** WhatsApp → Webhook → Orquestador → Clasificador → Respuesta → WhatsApp  
✅ **Testing verificado** con números reales  
✅ **Observabilidad completa** de flujo WhatsApp  
✅ **Documentación actualizada** en CLAUDE.md  

---

## 📞 **Casos de Prueba Planificados**

1. **Mensaje producto:** "Quiero información sobre smartphones"
   - **Esperado:** Clasificación `product_information` + respuesta informativa

2. **Mensaje PQR:** "Tengo un problema con mi pedido"
   - **Esperado:** Clasificación `PQR` + respuesta de escalación

3. **Mensaje ambiguo:** "Hola"
   - **Esperado:** Mensaje de clarificación

---

**⏰ Tiempo Total Estimado: 6 horas**  
**🎯 Prioridad: ALTA - Feature core para producción**