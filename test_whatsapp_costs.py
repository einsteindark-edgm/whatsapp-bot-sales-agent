#!/usr/bin/env python3
"""Test WhatsApp cost tracking through webhook."""

import httpx
import asyncio
import json
from datetime import datetime

async def test_whatsapp_webhook():
    """Send test messages through WhatsApp webhook to track costs."""
    
    print("🧪 Testing WhatsApp Cost Tracking")
    print("=" * 60)
    
    # Test messages from different countries
    test_cases = [
        {
            "from": "521234567890",  # Mexico
            "message": "Hola, ¿cuánto cuesta el iPhone 15?",
            "expected_country": "MX",
            "expected_type": "service"  # Product info = service message (free)
        },
        {
            "from": "551198765432",  # Brazil
            "message": "Meu pedido não chegou ainda",
            "expected_country": "BR", 
            "expected_type": "utility"  # PQR = utility message (paid)
        },
        {
            "from": "447700900123",  # UK
            "message": "What products do you have?",
            "expected_country": "GB",
            "expected_type": "service"
        },
        {
            "from": "14155552345",  # US
            "message": "I want to cancel my order",
            "expected_country": "US",
            "expected_type": "utility"
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for i, test in enumerate(test_cases):
            print(f"\n📱 Test {i+1}: {test['expected_country']} - {test['expected_type']}")
            print(f"   From: {test['from']}")
            print(f"   Message: {test['message']}")
            
            # Create WhatsApp webhook payload
            payload = {
                "object": "whatsapp_business_account",  # Required field
                "entry": [{
                    "id": "ENTRY_ID",
                    "changes": [{
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550555555",
                                "phone_number_id": "PHONE_NUMBER_ID"
                            },
                            "messages": [{
                                "from": test["from"],
                                "id": f"wamid.TEST_{datetime.now().timestamp()}",
                                "timestamp": str(int(datetime.now().timestamp())),
                                "text": {
                                    "body": test["message"]
                                },
                                "type": "text"
                            }]
                        },
                        "field": "messages"
                    }]
                }]
            }
            
            try:
                # Add a dummy signature header to bypass validation in dev
                headers = {
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": "sha256=dummy_signature"
                }
                
                response = await client.post(
                    "http://localhost:8000/webhook/whatsapp/test",  # Using test endpoint
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Response: {result}")
                    
                    if "cost" in result:
                        print(f"   💰 WhatsApp Cost: {result['cost']['whatsapp']}")
                        print(f"   📨 Message Type: {result['cost']['message_type']}")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
            
            # Wait between requests
            await asyncio.sleep(2)
    
    print("\n\n📊 Checking Cost Summary...")
    print("=" * 60)
    
    # Check the cost aggregator
    from shared.observability_cost import session_cost_aggregator, CostCalculator
    
    all_sessions = session_cost_aggregator.get_all_sessions()
    
    if all_sessions:
        whatsapp_total = 0
        llm_total = 0
        
        for session_id, costs in all_sessions.items():
            if "whatsapp" in session_id:
                print(f"\n📱 Session: {session_id}")
                print(f"   Total: {CostCalculator.format_cost(costs['total'])}")
                print(f"   LLM: {CostCalculator.format_cost(costs['llm'])}")
                print(f"   WhatsApp: {CostCalculator.format_cost(costs['whatsapp'])}")
                
                whatsapp_total += costs['whatsapp']
                llm_total += costs['llm']
        
        print(f"\n💰 Total WhatsApp Costs: {CostCalculator.format_cost(whatsapp_total)}")
        print(f"💰 Total LLM Costs: {CostCalculator.format_cost(llm_total)}")
        print(f"💰 Grand Total: {CostCalculator.format_cost(whatsapp_total + llm_total)}")
    else:
        print("❌ No session costs found")

if __name__ == "__main__":
    asyncio.run(test_whatsapp_webhook())