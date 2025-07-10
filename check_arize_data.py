#!/usr/bin/env python3
"""Check Arize data including tokens and costs."""

import time
import json
from shared.observability_enhanced import enhanced_observability

def check_arize_status():
    """Check Arize integration status and recent traces."""
    
    print("🔍 Checking Arize Integration Status")
    print("=" * 60)
    
    # Get metrics summary
    metrics = enhanced_observability.get_metrics_summary()
    
    print(f"\n📊 Observability Status:")
    print(f"   Logfire: {'✅' if metrics['integrations']['logfire_enabled'] else '❌'}")
    print(f"   Arize: {'✅' if metrics['integrations']['arize_enabled'] else '❌'}")
    print(f"   Arize OTel: {'✅' if metrics['integrations']['arize_otel_enabled'] else '❌'}")
    print(f"   OpenTelemetry: {'✅' if metrics['integrations']['otel_enabled'] else '❌'}")
    
    print(f"\n📈 Total LLM Traces: {metrics['total_llm_traces']}")
    print(f"💰 Total LLM Cost: ${metrics['total_llm_cost_usd']:.6f}")
    
    if metrics['recent_traces']:
        print("\n🔍 Recent LLM Traces Sent to Arize:")
        for i, trace in enumerate(metrics['recent_traces'][-3:], 1):
            print(f"\n   Trace {i}:")
            print(f"   ├─ Model: {trace.get('model_name', 'N/A')}")
            print(f"   ├─ Tokens In: {trace.get('token_count_input', 0)}")
            print(f"   ├─ Tokens Out: {trace.get('token_count_output', 0)}")
            print(f"   ├─ Total Tokens: {(trace.get('token_count_input', 0) + trace.get('token_count_output', 0))}")
            print(f"   ├─ Latency: {trace.get('latency_ms', 0):.2f}ms")
            print(f"   ├─ Cost: ${trace.get('cost_usd', 0):.6f}")
            print(f"   ├─ Classification: {trace.get('classification_label', 'N/A')}")
            print(f"   └─ Confidence: {trace.get('confidence_score', 0):.2%}")
    
    print("\n\n💡 To verify in Arize:")
    print("   1. Go to https://app.arize.com")
    print("   2. Navigate to your 'whatsapp-bot-agent' project")
    print("   3. Check the 'Traces' tab for LLM interactions")
    print("   4. Look for:")
    print("      - Token counts in span attributes")
    print("      - Cost information in metadata")
    print("      - Classification results")
    
    # Check if Arize client is sending data
    if enhanced_observability.arize.enabled:
        print("\n\n✅ Arize is enabled and should be receiving data")
        if enhanced_observability.arize.otel_enabled:
            print("   Using OpenTelemetry integration (recommended)")
        else:
            print("   Using pandas client (legacy)")
    else:
        print("\n\n❌ Arize is NOT enabled - no data being sent")

if __name__ == "__main__":
    check_arize_status()