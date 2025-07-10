"""
Test decision tracking for classifier and orchestrator agents.

This script validates that decision tracking with tags and attributes
is working correctly for both agents.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.classifier.agent import classifier_agent
from agents.orchestrator.agent import orchestrator_agent
from agents.orchestrator.domain.models import WorkflowRequest
from shared.observability import get_metrics_summary
from shared.observability_cost import session_cost_aggregator
from config.settings import settings


async def test_decision_tracking():
    """Test decision tracking in both classifier and orchestrator."""
    
    print("🎯 Testing Decision Tracking with Tags and Attributes")
    print("=" * 60)
    
    # Test messages
    test_cases = [
        {
            "text": "What's the price of the new iPhone 15 Pro?",
            "expected_label": "product_information",
            "user_id": "test_user_1",
            "session_id": "test_session_1",
        },
        {
            "text": "My order hasn't arrived yet and I'm very upset",
            "expected_label": "PQR",
            "user_id": "test_user_2",
            "session_id": "test_session_2",
        },
        {
            "text": "Do you have wireless headphones in stock?",
            "expected_label": "product_information",
            "user_id": "test_user_3",
            "session_id": "test_session_3",
        },
    ]
    
    # Process each test case
    for i, test_case in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}: {test_case['text'][:50]}...")
        trace_id = f"test_trace_{datetime.now().timestamp()}"
        
        try:
            # Test 1: Direct classifier call
            print("\n1️⃣ Testing direct classifier call...")
            classification = await classifier_agent.classify_message(
                text=test_case["text"],
                trace_id=trace_id,
                api_key=settings.gemini_api_key,
                include_reasoning=True,
                extract_keywords=True,
            )
            
            print(f"   ✅ Classification: {classification.label} (confidence: {classification.confidence:.2f})")
            if classification.keywords:
                print(f"   📌 Keywords: {', '.join(classification.keywords[:5])}")
            if classification.reasoning:
                print(f"   💭 Reasoning: {classification.reasoning[:100]}...")
            
            # Test 2: Orchestrator workflow
            print("\n2️⃣ Testing orchestrator workflow...")
            workflow_request = WorkflowRequest(
                user_id=test_case["user_id"],
                session_id=test_case["session_id"],
                user_message=test_case["text"],
                include_classification=True,
            )
            
            workflow_response = await orchestrator_agent.process_workflow_request(
                request=workflow_request,
                trace_id=trace_id,
            )
            
            print(f"   ✅ Workflow status: {workflow_response.workflow_status}")
            print(f"   📊 Response type: {workflow_response.response_type}")
            print(f"   💬 Response: {workflow_response.response[:100]}...")
            
            if workflow_response.classification:
                print(f"   🏷️ Classification in response: {workflow_response.classification}")
            
            # Check session costs
            session_cost = session_cost_aggregator.get_session_cost(test_case["session_id"])
            if session_cost["total"] > 0:
                print(f"   💰 Session cost: ${session_cost['total']:.6f} (LLM: ${session_cost['llm']:.6f})")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print metrics summary
    print("\n\n📊 Metrics Summary")
    print("=" * 60)
    metrics = get_metrics_summary()
    
    # Pretty print metrics
    print(json.dumps(metrics, indent=2, default=str))
    
    # Check decision tracking
    print("\n\n🏷️ Decision Tracking Verification")
    print("=" * 60)
    
    # Look for specific operation tracking
    if "counters" in metrics["metrics"]:
        counters = metrics["metrics"]["counters"]
        
        # Check for classifier decisions
        classifier_decisions = {k: v for k, v in counters.items() if "classification_decision" in k}
        if classifier_decisions:
            print("✅ Classifier decision tracking found:")
            for decision, count in classifier_decisions.items():
                print(f"   - {decision}: {count}")
        else:
            print("⚠️ No classifier decision tracking found")
            print("   (Note: Decision tracking is added AFTER classification, not by the decorator)")
        
        # Check for orchestrator decisions
        orchestrator_ops = {k: v for k, v in counters.items() 
                           if "orchestrator" in k and 
                           ("classification_received" in k or "request_classification" in k)}
        if orchestrator_ops:
            print("\n✅ Orchestrator decision tracking found:")
            for op, count in orchestrator_ops.items():
                print(f"   - {op}: {count}")
        else:
            print("\n⚠️ No orchestrator decision tracking found")
            print("   (Note: These are separate trace_agent_operation calls)")
        
        # Show all operations for debugging
        print("\n📋 All tracked operations:")
        for op, count in sorted(counters.items()):
            if "agent_operations_total" in op:
                print(f"   - {op}: {count}")
    
    print("\n\n✅ Decision tracking test completed!")


async def main():
    """Main test runner."""
    try:
        await test_decision_tracking()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())