#!/usr/bin/env python3
"""
Script to check WhatsApp message status.
"""

import httpx
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_message_status(message_id: str):
    """Check the status of a WhatsApp message."""
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    api_version = os.getenv("WHATSAPP_API_VERSION", "v18.0")
    base_url = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com")
    
    # Extract the actual message ID (remove prefix)
    clean_message_id = message_id.replace("wamid.", "")
    
    url = f"{base_url}/{api_version}/{message_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        response = httpx.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    # The message ID from the previous send
    message_id = "wamid.HBgMNTczMTI1NjcxNjA0FQIAERgSNjdGOUQxMjdBMUMzNEJDNDZBAA=="
    
    print(f"Checking status for message: {message_id}")
    check_message_status(message_id)