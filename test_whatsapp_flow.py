#!/usr/bin/env python3
"""
Test WhatsApp message flow to debug the integration.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.orchestrator.domain.whatsapp_models import WhatsAppWebhookPayload
from agents.orchestrator.domain.models import WorkflowRequest
from agents.orchestrator.agent import process_workflow_request_async
from shared.utils import generate_trace_id


async def test_whatsapp_flow():
    """Test the complete WhatsApp message flow."""
    
    print("=== Testing WhatsApp Message Flow ===\n")
    
    # 1. Simulate incoming WhatsApp webhook payload
    webhook_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "12345",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "573142399044",
                        "phone_number_id": "471379852715548"
                    },
                    "contacts": [{
                        "profile": {"name": "Test User"},
                        "wa_id": "573125671604"
                    }],
                    "messages": [{
                        "from": "573125671604",
                        "id": "test_message_id",
                        "timestamp": "1234567890",
                        "text": {"body": "Hola, ¿cuáles son los precios de sus productos?"},
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    
    print("1. Parsing webhook payload...")
    try:
        parsed_payload = WhatsAppWebhookPayload(**webhook_payload)
        message_data = parsed_payload.extract_message()
        if message_data:
            text, sender, message_id = message_data
            print(f"   ✓ Message: {text}")
            print(f"   ✓ From: {sender}")
            print(f"   ✓ ID: {message_id}")
        else:
            print("   ✗ No message found in payload")
            return
    except Exception as e:
        print(f"   ✗ Error parsing payload: {e}")
        return
    
    # 2. Create workflow request
    print("\n2. Creating workflow request...")
    try:
        workflow_request = WorkflowRequest(
            user_message=text,
            user_id=sender,
            session_id=f"whatsapp_{sender}",
            conversation_id=f"whatsapp_{sender}_{message_id}",
            metadata={
                "source": "whatsapp",
                "message_id": message_id,
                "phone_number": sender,
                "channel": "whatsapp",
            }
        )
        print(f"   ✓ Workflow request created")
    except Exception as e:
        print(f"   ✗ Error creating workflow request: {e}")
        return
    
    # 3. Process through orchestrator directly (not via A2A)
    print("\n3. Processing through orchestrator...")
    trace_id = generate_trace_id()
    print(f"   Trace ID: {trace_id}")
    
    try:
        # Call orchestrator directly since we're in the same service
        response = await process_workflow_request_async(
            request=workflow_request,
            trace_id=trace_id
        )
        
        print(f"   ✓ Response received:")
        print(f"     - Success: {response.success}")
        print(f"     - Response: {response.response[:100]}...")
        print(f"     - Type: {response.response_type}")
        print(f"     - Status: {response.workflow_status}")
        
        if response.classification:
            print(f"     - Classification: {response.classification}")
            
    except Exception as e:
        print(f"   ✗ Error processing workflow: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Send response back (would be via WhatsApp API)
    print("\n4. Response would be sent via WhatsApp API:")
    print(f"   To: {sender}")
    print(f"   Message: {response.response}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_whatsapp_flow())