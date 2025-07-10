"""
Cost telemetry for LLM and WhatsApp usage tracking.

This module provides cost calculation and tracking functionality for both
LLM token usage and WhatsApp message costs, with alerting capabilities.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from shared.observability import get_logger
import structlog

# Try to import logfire, but don't fail if not available
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    
# Try to import tokencost, but don't fail if not available
try:
    from tokencost import calculate_prompt_cost, calculate_completion_cost
    TOKENCOST_AVAILABLE = True
except ImportError:
    TOKENCOST_AVAILABLE = False

logger = get_logger(__name__)


class CostMetadata(BaseModel):
    """Metadata for cost tracking."""
    
    model: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    agent_name: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    channel: Optional[str] = None
    country: Optional[str] = "US"
    message_type: Optional[str] = None
    

class CostCalculator:
    """Calculate and track costs for LLM and WhatsApp usage."""
    
    # WhatsApp rates (per message, USD) - Updated for 2025 pricing
    WHATSAPP_RATES = {
        "service": 0.0,  # Free as of July 2025
        "marketing": {
            "US": 0.025,
            "BR": 0.018,
            "IN": 0.008,
            "GB": 0.022,
            "DE": 0.023,
            "FR": 0.023,
            "ES": 0.021,
            "IT": 0.021,
            "MX": 0.015,
            "AR": 0.016,
            "CO": 0.014,
            "CL": 0.017,
            "PE": 0.015,
            "DEFAULT": 0.025,  # Fallback rate
        },
        "utility": {
            "US": 0.020,
            "BR": 0.015,
            "IN": 0.006,
            "GB": 0.018,
            "DE": 0.019,
            "FR": 0.019,
            "ES": 0.017,
            "IT": 0.017,
            "MX": 0.012,
            "AR": 0.013,
            "CO": 0.011,
            "CL": 0.014,
            "PE": 0.012,
            "DEFAULT": 0.020,
        },
        "authentication": {
            "US": 0.015,
            "BR": 0.010,
            "IN": 0.004,
            "GB": 0.013,
            "DE": 0.014,
            "FR": 0.014,
            "ES": 0.012,
            "IT": 0.012,
            "MX": 0.008,
            "AR": 0.009,
            "CO": 0.007,
            "CL": 0.010,
            "PE": 0.008,
            "DEFAULT": 0.015,
        },
    }
    
    # Model-specific rates (fallback when tokencost not available)
    LLM_RATES = {
        # Gemini models
        "gemini-1.5-flash": {
            "prompt": 0.075 / 1_000_000,  # $0.075 per 1M tokens
            "completion": 0.30 / 1_000_000,  # $0.30 per 1M tokens
        },
        "gemini-1.5-pro": {
            "prompt": 1.25 / 1_000_000,  # $1.25 per 1M tokens
            "completion": 5.00 / 1_000_000,  # $5.00 per 1M tokens
        },
        "gemini-2.0-flash": {
            "prompt": 0.075 / 1_000_000,  # Same as 1.5 flash
            "completion": 0.30 / 1_000_000,
        },
        # OpenAI models (for reference)
        "gpt-4": {
            "prompt": 30.00 / 1_000_000,
            "completion": 60.00 / 1_000_000,
        },
        "gpt-3.5-turbo": {
            "prompt": 0.50 / 1_000_000,
            "completion": 1.50 / 1_000_000,
        },
    }
    
    @staticmethod
    def calculate_llm_cost(
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate LLM token costs.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Total cost in USD
        """
        if TOKENCOST_AVAILABLE:
            try:
                # Use tokencost library for accurate pricing
                prompt_cost = calculate_prompt_cost(prompt_tokens, model)
                completion_cost = calculate_completion_cost(completion_tokens, model)
                total_cost = prompt_cost + completion_cost
                
                logger.debug(
                    "Calculated LLM cost using tokencost",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_cost=prompt_cost,
                    completion_cost=completion_cost,
                    total_cost=total_cost,
                )
                
                return total_cost
            except Exception as e:
                logger.warning(
                    f"Failed to calculate cost using tokencost: {e}",
                    model=model,
                    error=str(e),
                )
                # Fall through to manual calculation
        
        # Manual calculation with fallback rates
        model_lower = model.lower()
        
        # Find matching rate
        rates = None
        for model_key, model_rates in CostCalculator.LLM_RATES.items():
            if model_key in model_lower:
                rates = model_rates
                break
        
        if not rates:
            # Unknown model
            logger.warning(
                f"Unknown model '{model}', returning 0 cost",
                model=model,
            )
            return 0.0
        
        prompt_cost = prompt_tokens * rates["prompt"]
        completion_cost = completion_tokens * rates["completion"]
        total_cost = prompt_cost + completion_cost
        
        logger.debug(
            "Calculated LLM cost using fallback rates",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
        )
        
        return total_cost
    
    @staticmethod
    def calculate_whatsapp_cost(
        message_type: str,
        country: str = "US"
    ) -> float:
        """
        Calculate WhatsApp message cost.
        
        Args:
            message_type: Type of message (service, marketing, utility, authentication)
            country: Country code for pricing
            
        Returns:
            Cost in USD
        """
        # Service messages are free
        if message_type == "service":
            return 0.0
        
        # Get rates for message type
        rates = CostCalculator.WHATSAPP_RATES.get(message_type, {})
        
        if not isinstance(rates, dict):
            # Fixed rate (shouldn't happen with current structure)
            return rates
        
        # Get country-specific rate with fallback
        cost = rates.get(country, rates.get("DEFAULT", 0.0))
        
        logger.debug(
            "Calculated WhatsApp cost",
            message_type=message_type,
            country=country,
            cost=cost,
        )
        
        return cost
    
    @staticmethod
    def track_cost(
        cost_usd: float,
        cost_type: str,
        metadata: Dict[str, Any],
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Track cost with alerting.
        
        Args:
            cost_usd: Cost in USD
            cost_type: Type of cost (llm, whatsapp)
            metadata: Additional metadata
            alert_thresholds: Optional custom alert thresholds
        """
        # Default alert thresholds
        if alert_thresholds is None:
            alert_thresholds = {
                "llm_per_call": 0.05,
                "session_total": 1.0,
            }
        
        # Log cost with structured logging
        log_data = {
            "cost_usd": cost_usd,
            "cost_type": cost_type,
            **metadata
        }
        
        if LOGFIRE_AVAILABLE:
            # Use Logfire span for enhanced tracking
            with logfire.span(
                "cost_tracking",
                _tags=["cost", cost_type],
                **log_data
            ):
                # Check alert thresholds
                if cost_type == "llm" and cost_usd > alert_thresholds.get("llm_per_call", 0.05):
                    logfire.warn(
                        "High LLM cost per call",
                        cost_usd=cost_usd,
                        threshold=alert_thresholds["llm_per_call"],
                        **metadata
                    )
                
                # Track cumulative cost per session
                session_cost = metadata.get("session_cost", 0) + cost_usd
                if session_cost > alert_thresholds.get("session_total", 1.0):
                    logfire.error(
                        "High session cost",
                        session_cost=session_cost,
                        threshold=alert_thresholds["session_total"],
                        session_id=metadata.get("session_id")
                    )
        else:
            # Use standard logging
            logger.info("Cost tracked", **log_data)
            
            # Check alerts with standard logging
            if cost_type == "llm" and cost_usd > alert_thresholds.get("llm_per_call", 0.05):
                logger.warning(
                    "High LLM cost per call",
                    cost_usd=cost_usd,
                    threshold=alert_thresholds["llm_per_call"],
                    **metadata
                )
            
            session_cost = metadata.get("session_cost", 0) + cost_usd
            if session_cost > alert_thresholds.get("session_total", 1.0):
                logger.error(
                    "High session cost",
                    session_cost=session_cost,
                    threshold=alert_thresholds["session_total"],
                    session_id=metadata.get("session_id")
                )
    
    @staticmethod
    def estimate_conversation_cost(
        num_messages: int,
        avg_tokens_per_message: int = 150,
        model: str = "gemini-1.5-flash",
        message_types: Optional[Dict[str, int]] = None,
        country: str = "US"
    ) -> Dict[str, float]:
        """
        Estimate cost for a conversation.
        
        Args:
            num_messages: Number of messages in conversation
            avg_tokens_per_message: Average tokens per message
            model: LLM model name
            message_types: Optional breakdown of WhatsApp message types
            country: Country for WhatsApp pricing
            
        Returns:
            Dictionary with cost breakdown
        """
        # Default message type distribution
        if message_types is None:
            message_types = {
                "service": int(num_messages * 0.8),  # 80% service messages
                "utility": int(num_messages * 0.15),  # 15% utility
                "marketing": int(num_messages * 0.05),  # 5% marketing
            }
        
        # Calculate LLM costs (assuming equal prompt/completion)
        total_tokens = num_messages * avg_tokens_per_message
        llm_cost = CostCalculator.calculate_llm_cost(
            model=model,
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens // 2
        )
        
        # Calculate WhatsApp costs
        whatsapp_cost = 0.0
        whatsapp_breakdown = {}
        
        for msg_type, count in message_types.items():
            cost_per_msg = CostCalculator.calculate_whatsapp_cost(msg_type, country)
            type_cost = cost_per_msg * count
            whatsapp_cost += type_cost
            whatsapp_breakdown[msg_type] = type_cost
        
        return {
            "total_cost": llm_cost + whatsapp_cost,
            "llm_cost": llm_cost,
            "whatsapp_cost": whatsapp_cost,
            "whatsapp_breakdown": whatsapp_breakdown,
            "num_messages": num_messages,
            "total_tokens": total_tokens,
            "model": model,
            "country": country,
        }
    
    @staticmethod
    def format_cost(cost_usd: float) -> str:
        """
        Format cost for display.
        
        Args:
            cost_usd: Cost in USD
            
        Returns:
            Formatted cost string
        """
        if cost_usd < 0.01:
            return f"${cost_usd:.6f}"
        elif cost_usd < 1.0:
            return f"${cost_usd:.4f}"
        else:
            return f"${cost_usd:.2f}"


# Session cost aggregator for tracking cumulative costs
class SessionCostAggregator:
    """Aggregate costs per session for monitoring."""
    
    def __init__(self):
        self._session_costs: Dict[str, Dict[str, float]] = {}
    
    def add_cost(
        self,
        session_id: str,
        cost_usd: float,
        cost_type: str
    ) -> float:
        """
        Add cost to session.
        
        Args:
            session_id: Session identifier
            cost_usd: Cost to add
            cost_type: Type of cost
            
        Returns:
            Total session cost
        """
        if session_id not in self._session_costs:
            self._session_costs[session_id] = {
                "total": 0.0,
                "llm": 0.0,
                "whatsapp": 0.0,
            }
        
        self._session_costs[session_id]["total"] += cost_usd
        self._session_costs[session_id][cost_type] = (
            self._session_costs[session_id].get(cost_type, 0.0) + cost_usd
        )
        
        return self._session_costs[session_id]["total"]
    
    def get_session_cost(self, session_id: str) -> Dict[str, float]:
        """Get cost breakdown for session."""
        return self._session_costs.get(
            session_id,
            {"total": 0.0, "llm": 0.0, "whatsapp": 0.0}
        )
    
    def clear_session(self, session_id: str):
        """Clear session cost data."""
        if session_id in self._session_costs:
            del self._session_costs[session_id]
    
    def get_all_sessions(self) -> Dict[str, Dict[str, float]]:
        """Get all session costs."""
        return self._session_costs.copy()


# Global session cost aggregator instance
session_cost_aggregator = SessionCostAggregator()


__all__ = [
    "CostCalculator",
    "CostMetadata",
    "SessionCostAggregator",
    "session_cost_aggregator",
]