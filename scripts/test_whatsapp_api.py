#!/usr/bin/env python3
"""
Direct test of WhatsApp API.
"""

import httpx
from pathlib import Path
from dotenv import load_dotenv
import os
import json

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def test_send_message():
    """Test sending a message directly to WhatsApp API."""
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    api_version = os.getenv("WHATSAPP_API_VERSION", "v22.0")
    base_url = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com")
    
    url = f"{base_url}/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    # Message payload
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "573125671604",  # Without +
        "type": "text",
        "text": {
            "preview_url": False,
            "body": "Hola! Este es un mensaje de prueba directo desde la API de WhatsApp."
        }
    }
    
    print(f"URL: {url}")
    print(f"Phone Number ID: {phone_number_id}")
    print(f"Sending to: {payload['to']}")
    print("\nPayload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = httpx.post(url, headers=headers, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Message sent successfully!")
        else:
            print("\n❌ Failed to send message")
            
    except Exception as e:
        print(f"\nError: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    test_send_message()