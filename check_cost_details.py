#!/usr/bin/env python3
"""Check cost details in recent traces."""

import httpx
import asyncio
import json
from shared.observability_cost import CostCalculator

async def check_cost_details():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8001/api/v1/observability-metrics")
        
        if response.status_code == 200:
            data = response.json()
            
            print("📊 Recent Traces with Cost Information")
            print("=" * 60)
            
            if 'recent_traces' in data:
                traces_with_cost = []
                
                for trace in data['recent_traces']:
                    if 'metadata' in trace and 'cost_usd' in trace['metadata']:
                        traces_with_cost.append(trace)
                
                if traces_with_cost:
                    for trace in traces_with_cost[-5:]:  # Last 5 traces with cost
                        print(f"\nTrace ID: {trace['trace_id'][:30]}...")
                        print(f"  Timestamp: {trace['timestamp']}")
                        print(f"  Model: {trace['model_name']}")
                        print(f"  Prompt: {trace['prompt'][:50]}...")
                        print(f"  Tokens:")
                        print(f"    Input: {trace['token_count_input']}")
                        print(f"    Output: {trace['token_count_output']}")
                        print(f"    Total: {trace['metadata'].get('total_tokens', 'N/A')}")
                        print(f"  Cost:")
                        print(f"    USD: ${trace['metadata']['cost_usd']:.8f}")
                        print(f"    Formatted: {trace['metadata']['cost_formatted']}")
                        print(f"  Classification: {trace['classification_label']} ({trace['confidence_score']:.2%})")
                    
                    # Calculate total cost
                    total_cost = sum(t['metadata']['cost_usd'] for t in traces_with_cost)
                    print(f"\n💰 Total cost from {len(traces_with_cost)} traces: {CostCalculator.format_cost(total_cost)}")
                else:
                    print("❌ No traces found with cost information")
            else:
                print("❌ No recent traces found")
        else:
            print(f"❌ Failed to get metrics: {response.status_code}")

if __name__ == "__main__":
    asyncio.run(check_cost_details())