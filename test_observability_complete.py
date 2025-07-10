#!/usr/bin/env python3
"""
Script completo para probar toda la funcionalidad de observabilidad.
Ejecuta pruebas unitarias, de integración y genera un reporte.
"""

import asyncio
import subprocess
import sys
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ObservabilityTester:
    """Ejecutor de pruebas de observabilidad."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        
    def print_header(self, title: str):
        """Imprime un encabezado formateado."""
        print(f"\n{'=' * 60}")
        print(f"🧪 {title}")
        print(f"{'=' * 60}")
    
    def print_step(self, step: str, number: int = None):
        """Imprime un paso de la prueba."""
        if number:
            print(f"\n{number}️⃣ {step}")
        else:
            print(f"\n▶️  {step}")
    
    def run_command(self, cmd: str, description: str) -> Tuple[bool, str]:
        """Ejecuta un comando y captura el resultado."""
        self.print_step(description)
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"   ✅ {description} - EXITOSO")
                return True, result.stdout
            else:
                print(f"   ❌ {description} - FALLÓ")
                print(f"   Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ {description} - TIMEOUT")
            return False, "Timeout"
        except Exception as e:
            print(f"   ❌ {description} - ERROR: {str(e)}")
            return False, str(e)
    
    async def test_components(self):
        """Prueba componentes individuales."""
        self.print_header("PRUEBAS DE COMPONENTES")
        
        # Test 1: Cálculo de costos
        success, output = self.run_command(
            "python3 test_cost_calculations.py",
            "Prueba de cálculo de costos"
        )
        self.results.append(("Cálculo de costos", success))
        
        # Test 2: Tracking de decisiones simple
        success, output = self.run_command(
            "python3 test_decision_tracking_simple.py",
            "Prueba de tracking de decisiones (simple)"
        )
        self.results.append(("Tracking simple", success))
        
        # Test 3: Tracking de decisiones completo
        success, output = self.run_command(
            "python3 test_decision_tracking.py",
            "Prueba de tracking de decisiones (completo)"
        )
        self.results.append(("Tracking completo", success))
        
        # Test 4: Integración con Arize
        success, output = self.run_command(
            "python3 test_observability.py",
            "Prueba de integración con Arize"
        )
        self.results.append(("Integración Arize", success))
    
    async def test_services(self):
        """Prueba los servicios si están ejecutándose."""
        self.print_header("PRUEBAS DE SERVICIOS")
        
        # Check classifier health
        success, output = self.run_command(
            "curl -s http://localhost:8001/health",
            "Health check del clasificador"
        )
        if success and output:
            try:
                health = json.loads(output)
                print(f"   📊 Estado: {health.get('status', 'unknown')}")
            except:
                pass
        self.results.append(("Health clasificador", success))
        
        # Check orchestrator health
        success, output = self.run_command(
            "curl -s http://localhost:8000/health",
            "Health check del orquestador"
        )
        if success and output:
            try:
                health = json.loads(output)
                print(f"   📊 Estado: {health.get('status', 'unknown')}")
            except:
                pass
        self.results.append(("Health orquestador", success))
        
        # Check metrics endpoint
        success, output = self.run_command(
            "curl -s http://localhost:8001/api/v1/observability-metrics",
            "Endpoint de métricas"
        )
        self.results.append(("Endpoint métricas", success))
    
    async def test_cli_flow(self):
        """Prueba el flujo completo con CLI."""
        self.print_header("PRUEBAS DE FLUJO CLI")
        
        # Solo ejecutar si los servicios están disponibles
        health_check = subprocess.run(
            "curl -s http://localhost:8000/health", 
            shell=True, 
            capture_output=True
        )
        
        if health_check.returncode != 0:
            print("⚠️  Servicios no disponibles - saltando pruebas de CLI")
            return
        
        # Test diferentes tipos de mensajes
        test_messages = [
            ("Mensaje de producto", "¿Cuál es el precio del iPhone 15 Pro Max?"),
            ("Mensaje de PQR", "Mi pedido #12345 no ha llegado"),
            ("Mensaje ambiguo", "Hola, necesito información"),
        ]
        
        for desc, message in test_messages:
            cmd = f'python3 -m cli.main "{message}"'
            success, output = self.run_command(cmd, desc)
            self.results.append((f"CLI: {desc}", success))
            time.sleep(1)  # Dar tiempo entre llamadas
    
    def generate_report(self):
        """Genera un reporte final de las pruebas."""
        self.print_header("REPORTE FINAL DE PRUEBAS")
        
        # Resumen de resultados
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success in self.results if success)
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Resumen de Pruebas:")
        print(f"   Total: {total_tests}")
        print(f"   ✅ Exitosas: {passed_tests}")
        print(f"   ❌ Fallidas: {failed_tests}")
        print(f"   📈 Tasa de éxito: {(passed_tests/total_tests*100):.1f}%")
        
        # Detalle de pruebas
        print(f"\n📋 Detalle de Pruebas:")
        for test_name, success in self.results:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}")
        
        # Métricas de observabilidad
        try:
            from shared.observability import get_metrics_summary
            from shared.observability_cost import session_cost_aggregator
            
            print(f"\n📈 Métricas de Observabilidad:")
            metrics = get_metrics_summary()
            
            if "counters" in metrics["metrics"]:
                counters = metrics["metrics"]["counters"]
                
                # Operaciones de decisión
                decision_ops = {
                    k: v for k, v in counters.items() 
                    if any(op in k for op in ["classification_decision", "classification_received", "request_classification"])
                }
                
                if decision_ops:
                    print(f"\n   🎯 Operaciones de Decisión:")
                    for op, count in decision_ops.items():
                        print(f"      {op}: {count}")
                
                # Total de operaciones
                total_ops = sum(v for k, v in counters.items() if "agent_operations_total" in k)
                print(f"\n   📊 Total de operaciones rastreadas: {total_ops}")
            
            # Costos
            all_sessions = session_cost_aggregator.get_all_sessions()
            if all_sessions:
                print(f"\n   💰 Costos por Sesión:")
                total_cost = 0
                for session_id, costs in all_sessions.items():
                    print(f"      {session_id[:20]}...: ${costs['total']:.6f}")
                    total_cost += costs['total']
                print(f"      Total acumulado: ${total_cost:.6f}")
            
        except Exception as e:
            print(f"\n   ⚠️  No se pudieron obtener métricas: {str(e)}")
        
        # Duración
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n⏱️  Tiempo total de pruebas: {duration:.2f} segundos")
        
        # Recomendaciones
        print(f"\n💡 Próximos Pasos:")
        if failed_tests > 0:
            print("   1. Revisar los logs de las pruebas fallidas")
            print("   2. Verificar que todas las dependencias estén instaladas")
            print("   3. Confirmar que las variables de entorno estén configuradas")
        else:
            print("   1. Revisar el dashboard de Arize en https://app.arize.com")
            print("   2. Verificar las métricas en http://localhost:8001/api/v1/observability-metrics")
            print("   3. Probar con el webhook de WhatsApp real")
        
        # Estado final
        if passed_tests == total_tests:
            print(f"\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        elif passed_tests > total_tests * 0.7:
            print(f"\n✅ La mayoría de las pruebas pasaron")
        else:
            print(f"\n❌ Varias pruebas fallaron - se requiere revisión")
        
        return passed_tests == total_tests


async def main():
    """Función principal."""
    tester = ObservabilityTester()
    
    print("🚀 INICIANDO PRUEBAS COMPLETAS DE OBSERVABILIDAD")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ejecutar pruebas
        await tester.test_components()
        await tester.test_services()
        await tester.test_cli_flow()
        
        # Generar reporte
        success = tester.generate_report()
        
        # Salir con código apropiado
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())