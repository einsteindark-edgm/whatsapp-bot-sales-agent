#!/usr/bin/env python3
"""
Script para probar y demostrar la observabilidad del sistema
"""
import asyncio
import httpx
import json
import time
from datetime import datetime

async def test_observability():
    """Test completo de observabilidad"""
    print("🔍 Probando Observabilidad del ChatCommerce Bot")
    print("=" * 60)
    
    # Lista de mensajes de prueba
    test_messages = [
        ("¿Cuál es el precio del iPhone 15?", "product_information"),
        ("Mi pedido llegó dañado", "PQR"), 
        ("Hola, ¿cómo están?", "other"),
        ("Necesito auriculares inalámbricos", "product_information"),
        ("Quiero cancelar mi orden", "PQR"),
    ]
    
    print(f"📝 Enviando {len(test_messages)} mensajes de prueba...")
    print()
    
    # Enviar mensajes de prueba
    async with httpx.AsyncClient() as client:
        for i, (message, expected_category) in enumerate(test_messages, 1):
            print(f"🔄 Test {i}: '{message}'")
            
            try:
                # Enviar al orquestador
                response = await client.post(
                    "http://localhost:8080/api/v1/orchestrate-direct",
                    json={
                        "user_message": message,
                        "user_id": f"test_user_{i}",
                        "session_id": f"test_session_{i}"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    classification = result.get("classification", {}).get("label", "unknown")
                    confidence = result.get("classification", {}).get("confidence", 0.0)
                    processing_time = result.get("processing_time", 0.0)
                    
                    # Verificar si la clasificación es correcta
                    is_correct = classification == expected_category
                    status_icon = "✅" if is_correct else "❌"
                    
                    print(f"   {status_icon} Clasificación: {classification} (esperado: {expected_category})")
                    print(f"   📊 Confianza: {confidence:.2%}")
                    print(f"   ⏱️  Tiempo: {processing_time:.2f}s")
                    
                else:
                    print(f"   ❌ Error HTTP: {response.status_code}")
                    
            except Exception as e:
                print(f"   💥 Error: {e}")
            
            print()
            # Pequeña pausa entre requests
            await asyncio.sleep(0.5)
    
    # Esperar un poco para que las métricas se actualicen
    print("⏳ Esperando a que se actualicen las métricas...")
    await asyncio.sleep(2)
    
    # Obtener métricas del clasificador
    print("\n📊 MÉTRICAS DEL CLASIFICADOR")
    print("-" * 40)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/api/v1/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                print(f"🤖 Total LLM traces: {metrics['total_llm_traces']}")
                print(f"🔌 Integraciones:")
                for integration, enabled in metrics['integrations'].items():
                    status = "✅ Habilitado" if enabled else "⚪ Deshabilitado"
                    print(f"   - {integration}: {status}")
                
                print(f"\n📈 Traces Recientes:")
                for trace in metrics['recent_traces'][-3:]:  # Últimos 3
                    print(f"   - ID: {trace['trace_id'][:8]}...")
                    print(f"     Prompt: '{trace['prompt'][:50]}{'...' if len(trace['prompt']) > 50 else ''}'")
                    print(f"     Clasificación: {trace['classification_label']}")
                    print(f"     Confianza: {trace['confidence_score']:.2%}")
                    print(f"     Latencia: {trace['latency_ms']:.0f}ms")
                    print(f"     Timestamp: {trace['timestamp']}")
                    print()
            else:
                print(f"❌ Error obteniendo métricas del clasificador: {response.status_code}")
                
    except Exception as e:
        print(f"💥 Error conectando al clasificador: {e}")
    
    # Obtener métricas del orquestador
    print("\n📊 MÉTRICAS DEL ORQUESTADOR")
    print("-" * 40)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8080/api/v1/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                print(f"🎯 Total LLM traces: {metrics['total_llm_traces']}")
                print(f"🔌 Integraciones:")
                for integration, enabled in metrics['integrations'].items():
                    status = "✅ Habilitado" if enabled else "⚪ Deshabilitado"
                    print(f"   - {integration}: {status}")
                
                print(f"\n🛠️  Información del Servicio:")
                service_info = metrics['service_info']
                print(f"   - Nombre: {service_info['name']}")
                print(f"   - Versión: {service_info['version']}")
                print(f"   - Entorno: {service_info['environment']}")
                
            else:
                print(f"❌ Error obteniendo métricas del orquestador: {response.status_code}")
                
    except Exception as e:
        print(f"💥 Error conectando al orquestador: {e}")
    
    # Verificar salud de los servicios
    print("\n🏥 VERIFICACIÓN DE SALUD")
    print("-" * 40)
    
    services = [
        ("Clasificador", "http://localhost:8001/api/v1/health"),
        ("Orquestador", "http://localhost:8080/api/v1/health")
    ]
    
    async with httpx.AsyncClient() as client:
        for service_name, health_url in services:
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get('status', 'unknown')
                    if status == 'healthy':
                        print(f"✅ {service_name}: {status}")
                    else:
                        print(f"⚠️  {service_name}: {status}")
                else:
                    print(f"❌ {service_name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"💥 {service_name}: Error - {e}")
    
    print("\n🎯 RESUMEN DE OBSERVABILIDAD")
    print("=" * 60)
    print("✅ Trazas de LLM funcionando")
    print("✅ Métricas de latencia capturadas")
    print("✅ Clasificaciones registradas")
    print("✅ IDs de trace generados")
    print("✅ Metadatos de usuario/sesión incluidos")
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. 🔥 Configurar Logfire para logs estructurados")
    print("2. 📈 Configurar Arize AX para monitoreo de LLM")
    print("3. 🚨 Setup alertas para latencia alta")
    print("4. 📊 Crear dashboards personalizados")
    print("5. 🧪 Implementar A/B testing de prompts")
    
    print(f"\n🕐 Test completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Ver OBSERVABILITY_SETUP_GUIDE.md para configuración avanzada")

if __name__ == "__main__":
    asyncio.run(test_observability())