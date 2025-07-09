#!/usr/bin/env python3
"""
Script rápido para probar el CLI con algunos ejemplos
"""
import subprocess
import sys
import time

def run_test_message(message):
    """Ejecuta una prueba con un mensaje específico"""
    print(f"\n🧪 Probando: '{message}'")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "cli.main", 
        "--message", message,
        "--user-id", "test_user"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Resultado:")
            print(result.stdout)
        else:
            print("❌ Error:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - el servicio tardó demasiado en responder")
    except Exception as e:
        print(f"💥 Error inesperado: {e}")

def main():
    print("🚀 Iniciando pruebas rápidas del CLI...")
    
    # Primero verificar conexión
    print("\n🔍 Verificando conexión...")
    cmd = [sys.executable, "-m", "cli.main", "--test-connection"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Conexión exitosa!")
        else:
            print("❌ Error de conexión:")
            print(result.stderr)
            return
    except Exception as e:
        print(f"💥 No se pudo conectar: {e}")
        return
    
    # Casos de prueba
    test_cases = [
        # Información de productos
        "¿Cuál es el precio del iPhone 15?",
        "Necesito auriculares inalámbricos",
        "¿Tienen laptops disponibles?",
        
        # PQR (Problemas, Quejas, Reclamos)
        "Mi pedido está retrasado",
        "Quiero devolver este producto",
        "El producto llegó dañado",
        
        # Casos generales
        "Hola, ¿cómo estás?",
        "Gracias por tu ayuda",
        "¿Cuál es el horario de atención?",
    ]
    
    for message in test_cases:
        run_test_message(message)
        time.sleep(1)  # Pausa entre pruebas
    
    print("\n🎉 ¡Pruebas completadas!")
    print("\n📋 Para usar el modo interactivo:")
    print("python3 -m cli.main")

if __name__ == "__main__":
    main()