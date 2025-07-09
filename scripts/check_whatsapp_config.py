#!/usr/bin/env python3
"""
Script to check WhatsApp configuration and permissions.
"""

import httpx
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_phone_number_status():
    """Check the WhatsApp phone number configuration."""
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    api_version = os.getenv("WHATSAPP_API_VERSION", "v18.0")
    base_url = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com")
    
    print("=== WhatsApp Configuration Check ===\n")
    
    # Check phone number details
    url = f"{base_url}/{api_version}/{phone_number_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = httpx.get(url, headers=headers)
        data = response.json()
        
        print(f"Phone Number ID: {phone_number_id}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Display Phone Number: {data.get('display_phone_number', 'N/A')}")
            print(f"Verified Name: {data.get('verified_name', {}).get('name', 'N/A')}")
            print(f"Quality Rating: {data.get('quality_rating', 'N/A')}")
            print(f"Platform Type: {data.get('platform_type', 'N/A')}")
            print(f"Messaging Limit: {data.get('messaging_limit', {}).get('tier', 'N/A')}")
        else:
            print(f"Error: {data}")
            
    except Exception as e:
        print(f"Error checking phone number: {e}")
    
    # Check message templates
    print("\n=== Message Templates ===")
    templates_url = f"{base_url}/{api_version}/{phone_number_id}/message_templates"
    
    try:
        response = httpx.get(templates_url, headers=headers)
        data = response.json()
        
        if "data" in data:
            print(f"Total templates: {len(data['data'])}")
            for template in data['data'][:5]:  # Show first 5
                print(f"- {template.get('name')} ({template.get('status')})")
        else:
            print("No templates found or error:", data)
            
    except Exception as e:
        print(f"Error checking templates: {e}")


if __name__ == "__main__":
    check_phone_number_status()