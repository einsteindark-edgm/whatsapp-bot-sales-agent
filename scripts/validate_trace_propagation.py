"""
Validate trace ID propagation across services.

This script tests that trace IDs are correctly propagated through all
service calls in the system, from the initial request through all
agent interactions.
"""

import httpx
import asyncio
from uuid import uuid4
import sys
from typing import Dict, Any, Optional
import json
from datetime import datetime


class TraceValidator:
    """Validator for trace ID propagation."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_trace_propagation(self) -> bool:
        """Test trace propagation through the system."""
        trace_id = str(uuid4())
        print(f"\n🔍 Testing trace propagation with ID: {trace_id}")
        
        # Test 1: Direct orchestrator request
        print("\n1️⃣ Testing direct orchestrator request...")
        result1 = await self._test_orchestrator_direct(trace_id)
        self.results.append(("Orchestrator Direct", result1))
        
        # Test 2: WhatsApp webhook
        print("\n2️⃣ Testing WhatsApp webhook...")
        result2 = await self._test_whatsapp_webhook(trace_id)
        self.results.append(("WhatsApp Webhook", result2))
        
        # Test 3: A2A communication
        print("\n3️⃣ Testing A2A communication...")
        result3 = await self._test_a2a_propagation(trace_id)
        self.results.append(("A2A Communication", result3))
        
        # Test 4: Check logs for trace
        print("\n4️⃣ Checking logs for trace ID...")
        result4 = await self._check_logs_for_trace(trace_id)
        self.results.append(("Log Verification", result4))
        
        # Summary
        self._print_summary()
        
        return all(result["success"] for _, result in self.results)
    
    async def _test_orchestrator_direct(self, trace_id: str) -> Dict[str, Any]:
        """Test direct orchestrator endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/orchestrate-direct",
                    headers={"X-Trace-ID": trace_id},
                    json={"user_message": "Test trace propagation"}
                )
                
                # Check response headers
                response_trace_id = response.headers.get("X-Trace-ID")
                
                result = {
                    "success": response_trace_id == trace_id,
                    "status_code": response.status_code,
                    "sent_trace_id": trace_id,
                    "received_trace_id": response_trace_id,
                    "headers": dict(response.headers),
                }
                
                if result["success"]:
                    print(f"✅ Trace ID correctly returned in response headers")
                else:
                    print(f"❌ Trace ID mismatch: sent {trace_id}, received {response_trace_id}")
                
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sent_trace_id": trace_id,
            }
    
    async def _test_whatsapp_webhook(self, trace_id: str) -> Dict[str, Any]:
        """Test WhatsApp webhook endpoint."""
        try:
            # WhatsApp webhook payload
            payload = {
                "entry": [{
                    "id": "123456789",
                    "changes": [{
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15555555555",
                                "phone_number_id": "123456"
                            },
                            "messages": [{
                                "from": "1234567890",
                                "id": "msg_" + trace_id[:8],
                                "timestamp": str(int(datetime.now().timestamp())),
                                "text": {"body": "Test trace propagation"},
                                "type": "text"
                            }]
                        }
                    }]
                }]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/webhook/whatsapp",
                    headers={"X-Trace-ID": trace_id},
                    json=payload
                )
                
                # Check response headers
                response_trace_id = response.headers.get("X-Trace-ID")
                
                result = {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "sent_trace_id": trace_id,
                    "received_trace_id": response_trace_id,
                    "response": response.json() if response.status_code == 200 else None,
                }
                
                if result["success"]:
                    print(f"✅ WhatsApp webhook processed successfully")
                else:
                    print(f"❌ WhatsApp webhook failed: {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sent_trace_id": trace_id,
            }
    
    async def _test_a2a_propagation(self, trace_id: str) -> Dict[str, Any]:
        """Test A2A protocol trace propagation."""
        try:
            # Create A2A message
            a2a_message = {
                "sender_agent": "test_client",
                "receiver_agent": "orchestrator",
                "message_type": "orchestration_request",
                "trace_id": trace_id,
                "payload": {
                    "user_message": "Test A2A trace propagation"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/orchestrate",
                    headers={"X-Trace-ID": trace_id},
                    json=a2a_message
                )
                
                result = {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                    "sent_trace_id": trace_id,
                }
                
                if response.status_code == 200:
                    response_data = response.json()
                    result["response_trace_id"] = response_data.get("trace_id")
                    result["trace_match"] = response_data.get("trace_id") == trace_id
                    
                    if result["trace_match"]:
                        print(f"✅ A2A trace ID correctly propagated")
                    else:
                        print(f"❌ A2A trace ID mismatch")
                else:
                    print(f"❌ A2A request failed: {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sent_trace_id": trace_id,
            }
    
    async def _check_logs_for_trace(self, trace_id: str) -> Dict[str, Any]:
        """Check if trace ID appears in service logs."""
        # In a real implementation, this would check log files or log aggregation service
        # For now, we'll check the observability metrics endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/observability-metrics")
                
                if response.status_code == 200:
                    metrics = response.json()
                    recent_traces = metrics.get("recent_traces", [])
                    
                    # Check if our trace ID appears in recent traces
                    trace_found = any(
                        trace.get("trace_id") == trace_id 
                        for trace in recent_traces
                    )
                    
                    result = {
                        "success": trace_found,
                        "trace_found": trace_found,
                        "total_traces": len(recent_traces),
                    }
                    
                    if trace_found:
                        print(f"✅ Trace ID found in observability metrics")
                    else:
                        print(f"❌ Trace ID not found in observability metrics")
                    
                    return result
                else:
                    return {
                        "success": False,
                        "error": f"Failed to get metrics: {response.status_code}"
                    }
                    
        except Exception as e:
            print(f"❌ Error checking logs: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TRACE PROPAGATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, result in self.results if result.get("success", False))
        
        for test_name, result in self.results:
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                print(f"  Error: {error}")
        
        print("-" * 60)
        print(f"Total: {passed_tests}/{total_tests} passed")
        
        if passed_tests == total_tests:
            print("\n🎉 All trace propagation tests passed!")
        else:
            print("\n⚠️ Some trace propagation tests failed!")


async def main():
    """Main test function."""
    validator = TraceValidator()
    
    print("🚀 Starting Trace Propagation Validation")
    print("=" * 60)
    
    try:
        success = await validator.test_trace_propagation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())