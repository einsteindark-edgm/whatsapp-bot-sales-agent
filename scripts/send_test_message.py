#!/usr/bin/env python3
"""
Script to send a test WhatsApp message.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.orchestrator.adapters.outbound.whatsapp_api_client import WhatsAppAPIClient
from shared.utils import generate_trace_id


async def send_test_message(phone_number: str, message: str):
    """Send a test message via WhatsApp API."""
    try:
        # Initialize WhatsApp client
        client = WhatsAppAPIClient()
        
        # Generate trace ID
        trace_id = generate_trace_id()
        
        print(f"Sending message to {phone_number}...")
        print(f"Message: {message}")
        print(f"Trace ID: {trace_id}")
        
        # Send message
        response = await client.send_text_message(
            to=phone_number,
            message=message,
            trace_id=trace_id
        )
        
        print("\nResponse:")
        print(response)
        
        if "messages" in response:
            print(f"\n✅ Message sent successfully!")
            print(f"Message ID: {response['messages'][0].get('id', 'unknown')}")
        else:
            print(f"\n❌ Failed to send message")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Your phone number with country code
    phone_number = "+573125671604"
    
    # Test message
    message = """¡Hola! 👋

Este es un mensaje de prueba desde el WhatsApp Business API.

Tu bot de ventas está funcionando correctamente. Ahora puedes enviarme mensajes y te responderé automáticamente.

Prueba preguntando:
• Información sobre productos
• Precios y disponibilidad
• Cualquier pregunta o reclamo

¡Estoy aquí para ayudarte!"""
    
    # Run async function
    asyncio.run(send_test_message(phone_number, message))