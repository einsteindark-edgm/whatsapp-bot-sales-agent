#!/usr/bin/env python3
"""Check recorded costs."""

from shared.observability_cost import session_cost_aggregator, CostCalculator
from shared.observability import get_metrics_summary
import json

print('💰 Session Costs Report')
print('=' * 50)

# Get all sessions
all_sessions = session_cost_aggregator.get_all_sessions()

if all_sessions:
    total_all = 0
    for session_id, costs in all_sessions.items():
        print(f'\nSession: {session_id[:40]}...')
        print(f'  Total: {CostCalculator.format_cost(costs["total"])}')
        print(f'  LLM: {CostCalculator.format_cost(costs["llm"])}')
        print(f'  WhatsApp: {CostCalculator.format_cost(costs["whatsapp"])}')
        total_all += costs['total']
    
    print(f'\n🔢 Total across all sessions: {CostCalculator.format_cost(total_all)}')
    print(f'   Sessions tracked: {len(all_sessions)}')
else:
    print('❌ No costs recorded yet')

# Check metrics for cost tracking
print('\n\n📊 Cost Tracking Metrics')
print('=' * 50)

metrics = get_metrics_summary()
if 'counters' in metrics['metrics']:
    cost_counters = {k: v for k, v in metrics['metrics']['counters'].items() if 'cost' in k.lower()}
    if cost_counters:
        print('Cost-related counters:')
        for counter, value in cost_counters.items():
            print(f'  {counter}: {value}')
    else:
        print('No cost-related counters found')

# Check latest traces for cost info
print('\n\n🔍 Checking recent traces for cost data...')
print('=' * 50)

try:
    import httpx
    import asyncio
    
    async def check_metrics():
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/api/v1/observability-metrics")
            if response.status_code == 200:
                data = response.json()
                if 'recent_traces' in data:
                    traces_with_cost = 0
                    for trace in data['recent_traces']:
                        if 'metadata' in trace and 'cost_usd' in trace['metadata']:
                            traces_with_cost += 1
                            print(f"✅ Trace {trace['trace_id'][:20]}... has cost: {trace['metadata']['cost_formatted']}")
                    
                    if traces_with_cost == 0:
                        print("❌ No traces found with cost data")
                    else:
                        print(f"\n✅ Found {traces_with_cost} traces with cost data")
    
    asyncio.run(check_metrics())
except Exception as e:
    print(f"Error checking metrics endpoint: {e}")