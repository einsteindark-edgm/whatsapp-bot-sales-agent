"""
Simple test to verify decision tracking operations are recorded.
"""

import asyncio
import sys
import os
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.observability import trace_agent_operation, get_metrics_summary


def test_trace_operations():
    """Test that trace operations are recorded correctly."""
    
    print("🧪 Testing trace_agent_operation directly")
    print("=" * 50)
    
    # Test 1: Classifier decision tracking
    trace_agent_operation(
        agent_name="classifier",
        operation_name="classification_decision",
        trace_id="test_001",
        status="completed",
        duration=1.5,
        metadata={
            "decision_label": "product_information",
            "decision_confidence": 0.95,
            "_tags": ["classification", "product_information", "confidence_95"],
        }
    )
    print("✅ Traced classifier decision")
    
    # Test 2: Orchestrator classification received
    trace_agent_operation(
        agent_name="orchestrator",
        operation_name="classification_received",
        trace_id="test_002",
        status="completed",
        duration=0.1,
        metadata={
            "classification_label": "PQR",
            "classification_confidence": 0.87,
            "response_type_decision": "pqr_response",
            "_tags": ["orchestration", "classification_received", "pqr_response"],
        }
    )
    print("✅ Traced orchestrator classification received")
    
    # Test 3: Orchestrator request classification
    trace_agent_operation(
        agent_name="orchestrator",
        operation_name="request_classification",
        trace_id="test_003",
        status="started",
        metadata={
            "message_length": 42,
            "session_id": "test_session",
            "_tags": ["orchestration", "classification_request"],
        }
    )
    print("✅ Traced orchestrator request classification")
    
    # Get and display metrics
    metrics = get_metrics_summary()
    
    print("\n📊 Metrics Summary")
    print("-" * 50)
    
    if "counters" in metrics["metrics"]:
        counters = metrics["metrics"]["counters"]
        
        # Filter for our specific operations
        decision_ops = {k: v for k, v in counters.items() 
                       if "classification_decision" in k 
                       or "classification_received" in k 
                       or "request_classification" in k}
        
        if decision_ops:
            print("✅ Decision tracking operations found:")
            for op, count in decision_ops.items():
                print(f"   {op}: {count}")
        else:
            print("❌ No decision tracking operations found")
            print("\nAll operations:")
            for op, count in counters.items():
                print(f"   {op}: {count}")
    
    # Also check timing metrics
    if "timers" in metrics["metrics"]:
        timers = metrics["metrics"]["timers"]
        decision_timers = {k: v for k, v in timers.items() 
                          if "classification_decision" in k 
                          or "classification_received" in k}
        
        if decision_timers:
            print("\n⏱️ Decision operation timings:")
            for timer, values in decision_timers.items():
                print(f"   {timer}: {values}")


if __name__ == "__main__":
    test_trace_operations()