"""
Workflow Orchestrator Agent Implementation.

This module implements the workflow orchestrator agent for managing conversations
and coordinating with the classifier agent using A2A protocol.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from .domain.models import (
    ConversationContext,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowStatus,
    ConversationState,
    ResponseType,
    OrchestrationMetrics,
    OrchestrationConfig,
)
from .adapters.outbound.http_a2a_client import A2AHttpClient
from shared.observability import get_logger
from shared.observability_enhanced import trace_agent_operation
from shared.a2a_protocol import ClassificationRequest


# Configure logger
logger = get_logger(__name__)


class WorkflowOrchestrator:
    """
    Custom orchestrator for WhatsApp sales workflow.

    This agent implements custom orchestration logic for managing conversations 
    and coordinating with the classifier agent using A2A protocol.
    """

    def __init__(
        self,
        name: str,
        config: Optional[OrchestrationConfig] = None,
        a2a_client: Optional[A2AHttpClient] = None,
    ):
        """
        Initialize the workflow orchestrator.

        Args:
            name: Agent name
            config: Optional orchestration configuration
            a2a_client: Optional A2A HTTP client
        """
        # Set our attributes
        self.name = name
        self._config = config or OrchestrationConfig()
        self.a2a_client = a2a_client or A2AHttpClient(
            classifier_url=self._config.classifier_url,
            timeout=self._config.classification_timeout,
            retries=self._config.classification_retries,
        )
        self.metrics = OrchestrationMetrics()
        self.conversations: Dict[str, ConversationContext] = {}

        logger.info(
            "Workflow orchestrator initialized",
            name=name,
            classifier_url=self._config.classifier_url,
            model_name=self._config.model_name,
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the orchestrator.

        Returns:
            Health check results
        """
        try:
            # Check A2A client connection (simplified)
            classifier_healthy = True  # Simplified for now
            
            return {
                "status": "healthy" if classifier_healthy else "degraded",
                "orchestrator": "healthy",
                "classifier_connection": "healthy" if classifier_healthy else "unhealthy",
                "active_conversations": len(self.conversations),
                "total_requests": self.metrics.total_requests,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def process_workflow_request(
        self, request: WorkflowRequest, trace_id: str
    ) -> WorkflowResponse:
        """
        Process a workflow request.

        Args:
            request: Workflow request
            trace_id: Trace ID for observability

        Returns:
            Workflow response
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Processing workflow request",
                user_id=request.user_id,
                session_id=request.session_id,
                message_length=len(request.user_message),
                trace_id=trace_id,
            )

            # Get or create conversation context
            conversation_key = f"{request.user_id}:{request.session_id}"
            if conversation_key not in self.conversations:
                self.conversations[conversation_key] = ConversationContext(
                    user_id=request.user_id, session_id=request.session_id
                )

            conversation = self.conversations[conversation_key]
            conversation.add_message("user", request.user_message)
            conversation.update_state(ConversationState.WAITING_FOR_CLASSIFICATION)

            # Send message to classifier via A2A protocol
            classification_request = ClassificationRequest(
                sender_agent="orchestrator",
                receiver_agent="classifier",
                trace_id=trace_id,
                payload={"text": request.user_message},
            )

            # Get classification from classifier service
            from config.settings import settings
            classification_response = await self.a2a_client.send_classification_request(
                classification_request, api_key=settings.gemini_api_key
            )

            # Store classification in conversation
            classification_data = classification_response.payload
            conversation.update_classification(classification_data)
            conversation.update_state(ConversationState.PROCESSING_RESPONSE)

            logger.info(
                "Classification received",
                label=classification_data.get("label"),
                confidence=classification_data.get("confidence"),
                trace_id=trace_id,
            )

            # Generate response based on classification
            response_text = self._generate_response(
                user_message=request.user_message, classification=classification_data
            )

            # Add response to conversation
            conversation.add_message("assistant", response_text)
            conversation.update_state(ConversationState.COMPLETED)

            # Calculate final processing time
            final_processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(
                classification_data=classification_data,
                processing_time=final_processing_time,
                success=True,
            )

            # Enhanced observability - trace workflow completion
            trace_agent_operation(
                agent_name="orchestrator",
                operation_name="process_workflow",
                trace_id=trace_id,
                status="completed",
                duration=final_processing_time * 1000,  # Convert to milliseconds
                metadata={
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "classification_label": classification_data.get("label"),
                    "classification_confidence": classification_data.get("confidence"),
                    "response_type": self._determine_response_type(classification_data).value,
                    "message_length": len(request.user_message),
                    "response_length": len(response_text),
                }
            )

            # Create workflow response
            return WorkflowResponse(
                request_id=request.request_id,
                response=response_text,
                response_type=self._determine_response_type(classification_data),
                classification=classification_data if request.include_classification else None,
                workflow_status=WorkflowStatus.COMPLETED,
                conversation_state=ConversationState.COMPLETED,
                processing_time=final_processing_time,
                success=True,
            )

        except Exception as e:
            # Calculate processing time for error case
            error_processing_time = time.time() - start_time
            
            logger.error(
                "Workflow processing failed",
                error=str(e),
                trace_id=trace_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )

            # Enhanced observability - trace workflow failure
            trace_agent_operation(
                agent_name="orchestrator",
                operation_name="process_workflow",
                trace_id=trace_id,
                status="failed",
                duration=error_processing_time * 1000,  # Convert to milliseconds
                error=str(e),
                metadata={
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "message_length": len(request.user_message),
                    "error_type": type(e).__name__,
                }
            )

            # Update metrics for failure
            self._update_metrics(
                classification_data=None,
                processing_time=error_processing_time,
                success=False,
            )

            # Return error response
            return WorkflowResponse(
                request_id=request.request_id,
                response="I apologize, but I'm having trouble processing your message right now. Please try again.",
                response_type=ResponseType.ERROR_RESPONSE,
                workflow_status=WorkflowStatus.FAILED,
                conversation_state=ConversationState.ERROR,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def _generate_response(self, user_message: str, classification: Dict[str, Any]) -> str:
        """
        Generate response based on classification.

        Args:
            user_message: Original user message
            classification: Classification result

        Returns:
            Generated response text
        """
        label = classification.get("label", "unknown")
        confidence = classification.get("confidence", 0.0)

        # Generate response based on classification
        if label == "product_information":
            response = f"Gracias por tu consulta sobre productos. Te ayudo con información sobre precios y características. Tu mensaje: '{user_message}'"
        elif label == "PQR":
            response = f"Entiendo tu inquietud. Un agente especializado te contactará pronto para resolver: '{user_message}'"
        else:
            response = "¡Hola! Soy tu asistente de WhatsApp. ¿En qué puedo ayudarte hoy?"

        # Add confidence information if configured
        if self._config.include_classification_in_response and confidence > 0:
            if confidence < self._config.handoff_threshold:
                response += "\n\n[Note: This message may require human assistance for better support]"

        return response

    def _determine_response_type(self, classification: Dict[str, Any]) -> ResponseType:
        """
        Determine response type based on classification.

        Args:
            classification: Classification result

        Returns:
            Response type
        """
        if not classification:
            return ResponseType.ERROR_RESPONSE

        label = classification.get("label")
        confidence = classification.get("confidence", 0.0)

        if confidence < self._config.handoff_threshold:
            return ResponseType.HANDOFF_RESPONSE
        elif label == "product_information":
            return ResponseType.PRODUCT_INFORMATION_RESPONSE
        elif label == "PQR":
            return ResponseType.PQR_RESPONSE
        else:
            return ResponseType.CLARIFICATION_REQUEST

    def _update_metrics(
        self, classification_data: Optional[Dict[str, Any]], processing_time: float, success: bool
    ):
        """
        Update orchestration metrics.

        Args:
            classification_data: Classification result
            processing_time: Processing time in seconds
            success: Whether the operation was successful
        """
        if not self._config.enable_metrics:
            return

        # Update request metrics
        self.metrics.total_requests += 1

        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        # Update response type metrics
        if classification_data:
            label = classification_data.get("label")
            if label == "product_information":
                self.metrics.product_information_responses += 1
            elif label == "PQR":
                self.metrics.pqr_responses += 1
        else:
            self.metrics.error_responses += 1

        # Update processing time
        current_avg = self.metrics.average_processing_time
        total = self.metrics.total_requests
        self.metrics.average_processing_time = (current_avg * (total - 1) + processing_time) / total

    def get_conversation(self, user_id: str, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Conversation context if found, None otherwise
        """
        conversation_key = f"{user_id}:{session_id}"
        return self.conversations.get(conversation_key)

    def clear_conversation(self, user_id: str, session_id: str) -> bool:
        """
        Clear conversation context.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            True if conversation was found and cleared, False otherwise
        """
        conversation_key = f"{user_id}:{session_id}"
        if conversation_key in self.conversations:
            del self.conversations[conversation_key]
            return True
        return False

    def get_metrics(self) -> OrchestrationMetrics:
        """
        Get orchestration metrics.

        Returns:
            Current metrics
        """
        return self.metrics

    def reset_metrics(self):
        """Reset orchestration metrics."""
        self.metrics = OrchestrationMetrics()


# Global orchestrator instance
orchestrator_agent = WorkflowOrchestrator(name="WorkflowOrchestrator", config=OrchestrationConfig())


# Functions for use by the router
async def process_workflow_request_async(request: WorkflowRequest, trace_id: str) -> WorkflowResponse:
    """Process a workflow request asynchronously."""
    return await orchestrator_agent.process_workflow_request(request, trace_id)


def get_orchestrator_metrics() -> OrchestrationMetrics:
    """Get orchestrator metrics."""
    return orchestrator_agent.get_metrics()


def reset_orchestrator_metrics():
    """Reset orchestrator metrics."""
    orchestrator_agent.reset_metrics()


async def orchestrator_health_check() -> Dict[str, Any]:
    """Perform orchestrator health check."""
    return await orchestrator_agent.health_check()