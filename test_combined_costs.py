#!/usr/bin/env python3
"""Test combined LLM and WhatsApp cost tracking."""

import httpx
import asyncio
import json
from datetime import datetime

async def test_full_flow():
    """Test full flow showing both LLM and WhatsApp costs."""
    
    print("🧪 Testing Combined Cost Tracking (LLM + WhatsApp)")
    print("=" * 60)
    
    # Test one complex case
    test_case = {
        "from": "551198765432",  # Brazil
        "message": "My order #12345 hasn't arrived and I'm very upset. I need a refund!",
        "expected_country": "BR",
        "expected_type": "utility"  # PQR = utility message (paid)
    }
    
    async with httpx.AsyncClient() as client:
        print(f"\n📱 Processing message from {test_case['expected_country']}")
        print(f"   From: {test_case['from']}")
        print(f"   Message: {test_case['message']}")
        
        # Create WhatsApp webhook payload
        payload = {
            "object": "whatsapp_business_account",
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
                            "from": test_case["from"],
                            "id": f"wamid.TEST_{datetime.now().timestamp()}",
                            "timestamp": str(int(datetime.now().timestamp())),
                            "text": {
                                "body": test_case["message"]
                            },
                            "type": "text"
                        }]
                    },
                    "field": "messages"
                }]
            }]
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=dummy_signature"
            }
            
            response = await client.post(
                "http://localhost:8000/webhook/whatsapp/test",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n   ✅ Response received")
                print(f"   📊 Classification: {result.get('classification', 'N/A')}")
                
                if "cost" in result:
                    print(f"\n   💰 WhatsApp Message Cost: {result['cost']['whatsapp']}")
                    print(f"   📨 Message Type: {result['cost']['message_type']}")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    print("\n\n🔍 Checking Logfire for detailed costs...")
    print("=" * 60)
    
    # Give Logfire time to process
    await asyncio.sleep(3)
    
    # Check cost aggregator to show session totals
    from shared.observability_cost import session_cost_aggregator, CostCalculator
    
    sessions = session_cost_aggregator.get_all_sessions()
    
    if sessions:
        print("\n📊 Session Cost Summary:")
        for session_id, costs in sessions.items():
            if "whatsapp" in session_id and costs['total'] > 0:
                print(f"\n   Session: {session_id}")
                print(f"   ├─ LLM Cost: {CostCalculator.format_cost(costs['llm'])}")
                print(f"   ├─ WhatsApp Cost: {CostCalculator.format_cost(costs['whatsapp'])}")
                print(f"   └─ Total: {CostCalculator.format_cost(costs['total'])}")
                
                # Show percentage breakdown
                if costs['total'] > 0:
                    llm_pct = (costs['llm'] / costs['total']) * 100
                    wa_pct = (costs['whatsapp'] / costs['total']) * 100
                    print(f"\n   Cost Breakdown:")
                    print(f"   ├─ LLM: {llm_pct:.1f}%")
                    print(f"   └─ WhatsApp: {wa_pct:.1f}%")
    
    print("\n\n💡 To see costs in Logfire:")
    print("   1. Go to https://logfire-us.pydantic.dev/einsteindark/whatsapp-bot-agent")
    print("   2. Look for spans with tags [cost, llm] or [cost, whatsapp]")
    print("   3. Check the cost_tracking events for detailed cost breakdowns")

if __name__ == "__main__":
    asyncio.run(test_full_flow())