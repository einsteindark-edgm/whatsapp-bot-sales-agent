#!/usr/bin/env python3
"""
Test script to verify Arize OpenTelemetry integration is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.observability_enhanced import enhanced_observability
from shared.observability_enhanced import LLMTrace
import uuid
from datetime import datetime, timezone
import time

def test_arize_integration():
    """Test Arize integration with real data."""
    print("🧪 Testing Arize OpenTelemetry integration...")
    
    # Check integration status
    metrics = enhanced_observability.get_metrics_summary()
    print(f"📊 Current integrations:")
    print(f"  - Logfire enabled: {metrics['integrations']['logfire_enabled']}")
    print(f"  - Arize enabled: {metrics['integrations']['arize_enabled']}")
    print(f"  - Arize OTel enabled: {metrics['integrations']['arize_otel_enabled']}")
    print(f"  - Total traces: {metrics['total_llm_traces']}")
    
    if not metrics['integrations']['arize_enabled']:
        print("❌ Arize integration not enabled")
        return False
    
    # Create test trace
    trace_id = str(uuid.uuid4())
    print(f"\n📡 Creating test trace: {trace_id}")
    
    test_trace = LLMTrace(
        trace_id=trace_id,
        model_name="test-model-gemini-2.0-flash",
        prompt="Test prompt for Arize verification",
        response="Test response: product_information classification",
        latency_ms=1500.0,
        token_count_input=8,
        token_count_output=15,
        classification_label="product_information", 
        confidence_score=0.92,
        agent_name="test-classifier",
        metadata={
            "test_run": True,
            "verification": "arize_otel_integration",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    
    print(f"📤 Sending test trace to Arize...")
    
    # Send to Arize
    try:
        enhanced_observability.arize.log_llm_trace(test_trace)
        print(f"✅ Test trace sent successfully")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Check updated metrics
        updated_metrics = enhanced_observability.get_metrics_summary()
        print(f"\n📊 Updated metrics:")
        print(f"  - Total traces: {updated_metrics['total_llm_traces']}")
        print(f"  - Recent traces: {len(updated_metrics['recent_traces'])}")
        
        if updated_metrics['total_llm_traces'] > metrics['total_llm_traces']:
            print(f"✅ Trace count increased from {metrics['total_llm_traces']} to {updated_metrics['total_llm_traces']}")
            
        # Show recent trace info
        if updated_metrics['recent_traces']:
            latest = updated_metrics['recent_traces'][-1]
            print(f"\n📋 Latest trace info:")
            print(f"  - Trace ID: {latest['trace_id']}")
            print(f"  - Classification: {latest['classification_label']}")
            print(f"  - Confidence: {latest['confidence_score']}")
            print(f"  - Agent: {latest['agent_name']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to send test trace: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_arize_integration()
    if success:
        print(f"\n🎉 Arize OpenTelemetry integration test PASSED!")
        print(f"📈 Check your Arize dashboard for the test data")
        sys.exit(0)
    else:
        print(f"\n💥 Arize OpenTelemetry integration test FAILED!")
        sys.exit(1)