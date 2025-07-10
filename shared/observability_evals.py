"""
Phoenix LLM evaluation integration for online quality assessment.

This module provides online evaluation capabilities for LLM responses
using Phoenix evals, including QA correctness and hallucination detection.
"""

import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from shared.observability import get_logger
from datetime import datetime, timezone

# Try to import Phoenix evals, but don't fail if not available
try:
    import phoenix as px
    from phoenix.evals import (
        QACorrectnessEvaluator,
        HallucinationEvaluator,
        ToxicityEvaluator,
        run_evals
    )
    from phoenix.evals.models import OpenAIModel, BaseEvalModel
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    QACorrectnessEvaluator = None
    HallucinationEvaluator = None
    ToxicityEvaluator = None
    OpenAIModel = None
    BaseEvalModel = None

# Try to import logfire for enhanced logging
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

logger = get_logger(__name__)


class EvaluationResult(BaseModel):
    """Evaluation result model."""
    
    eval_type: str = Field(..., description="Type of evaluation")
    score: float = Field(..., description="Evaluation score (0-1)")
    label: str = Field(..., description="Evaluation label")
    explanation: Optional[str] = Field(None, description="Explanation for the evaluation")
    trace_id: str = Field(..., description="Associated trace ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Configuration for Phoenix evaluations."""
    
    eval_model: str = Field(default="gpt-4", description="Model to use for evaluations")
    concurrency: int = Field(default=4, ge=1, le=10, description="Concurrent evaluation limit")
    timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="Evaluation timeout in seconds")
    retry_attempts: int = Field(default=2, ge=0, le=5, description="Retry attempts for failed evaluations")
    
    # Feature flags
    enable_qa_correctness: bool = Field(default=True, description="Enable QA correctness evaluation")
    enable_hallucination: bool = Field(default=True, description="Enable hallucination detection")
    enable_toxicity: bool = Field(default=False, description="Enable toxicity detection")
    
    # Sampling
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Evaluation sampling rate")


class PhoenixEvaluator:
    """Online LLM evaluation using Phoenix."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize Phoenix evaluator.
        
        Args:
            config: Optional evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.enabled = False
        self.eval_model = None
        self.qa_evaluator = None
        self.hallucination_evaluator = None
        self.toxicity_evaluator = None
        
        # Evaluation results buffer
        self.evaluation_results: List[EvaluationResult] = []
        
        # Initialize evaluators
        self._initialize()
    
    def _initialize(self):
        """Initialize Phoenix evaluators."""
        if not PHOENIX_AVAILABLE:
            logger.warning(
                "Phoenix evals not available. Install with: pip install arize-phoenix[evals]"
            )
            return
        
        try:
            # Initialize evaluation model
            if hasattr(OpenAIModel, '__init__'):
                self.eval_model = OpenAIModel(model=self.config.eval_model)
            else:
                logger.warning("OpenAIModel not properly imported")
                return
            
            # Initialize evaluators based on config
            if self.config.enable_qa_correctness:
                self.qa_evaluator = QACorrectnessEvaluator(self.eval_model)
                logger.info(f"QA Correctness evaluator initialized with {self.config.eval_model}")
            
            if self.config.enable_hallucination:
                self.hallucination_evaluator = HallucinationEvaluator(self.eval_model)
                logger.info(f"Hallucination evaluator initialized with {self.config.eval_model}")
            
            if self.config.enable_toxicity:
                self.toxicity_evaluator = ToxicityEvaluator(self.eval_model)
                logger.info(f"Toxicity evaluator initialized with {self.config.eval_model}")
            
            self.enabled = True
            logger.info(
                "Phoenix evaluators initialized successfully",
                extra={
                    "eval_model": self.config.eval_model,
                    "concurrency": self.config.concurrency,
                    "enabled_evaluators": {
                        "qa_correctness": self.config.enable_qa_correctness,
                        "hallucination": self.config.enable_hallucination,
                        "toxicity": self.config.enable_toxicity,
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix evaluators: {e}")
            self.enabled = False
    
    async def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str,
        trace_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """
        Run online evaluations asynchronously.
        
        Args:
            question: User question/prompt
            answer: LLM response
            context: Context/reference information
            trace_id: Trace ID for correlation
            metadata: Optional metadata
            
        Returns:
            List of evaluation results (empty list returned immediately,
            actual evaluations run asynchronously)
        """
        if not self.enabled:
            logger.debug("Phoenix evaluator not enabled, skipping evaluations")
            return []
        
        # Check sampling rate
        import random
        if random.random() > self.config.sample_rate:
            logger.debug(
                f"Skipping evaluation due to sampling (rate: {self.config.sample_rate})"
            )
            return []
        
        # Don't block main flow - run evaluations asynchronously
        asyncio.create_task(
            self._run_evaluations(question, answer, context, trace_id, metadata)
        )
        
        return []  # Return immediately
    
    async def _run_evaluations(
        self,
        question: str,
        answer: str,
        context: str,
        trace_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Execute evaluations and log results."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create evaluation tasks based on config
            tasks = []
            
            if self.qa_evaluator and self.config.enable_qa_correctness:
                tasks.append(
                    self._run_single_evaluation(
                        "qa_correctness",
                        self.qa_evaluator,
                        query=question,
                        response=answer,
                        reference=context
                    )
                )
            
            if self.hallucination_evaluator and self.config.enable_hallucination:
                tasks.append(
                    self._run_single_evaluation(
                        "hallucination",
                        self.hallucination_evaluator,
                        response=answer,
                        reference=context
                    )
                )
            
            if self.toxicity_evaluator and self.config.enable_toxicity:
                tasks.append(
                    self._run_single_evaluation(
                        "toxicity",
                        self.toxicity_evaluator,
                        text=answer
                    )
                )
            
            # Run evaluations concurrently with timeout
            if tasks:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout
                )
                
                # Process results
                for eval_type, result in results:
                    if isinstance(result, Exception):
                        logger.error(
                            f"Evaluation failed for {eval_type}",
                            extra={
                                "error": str(result),
                                "trace_id": trace_id
                            }
                        )
                        continue
                    
                    # Create evaluation result
                    eval_result = EvaluationResult(
                        eval_type=eval_type,
                        score=result.score if hasattr(result, 'score') else 0.0,
                        label=result.label if hasattr(result, 'label') else "unknown",
                        explanation=result.explanation if hasattr(result, 'explanation') else None,
                        trace_id=trace_id,
                        metadata=metadata or {}
                    )
                    
                    # Store result
                    self.evaluation_results.append(eval_result)
                    
                    # Log result
                    self._log_evaluation_result(eval_result)
                
                # Log completion
                duration = asyncio.get_event_loop().time() - start_time
                logger.info(
                    f"Evaluations completed in {duration:.2f}s",
                    extra={
                        "trace_id": trace_id,
                        "num_evaluations": len(results),
                        "duration_seconds": duration
                    }
                )
                
        except asyncio.TimeoutError:
            logger.error(
                f"Evaluations timed out after {self.config.timeout}s",
                extra={"trace_id": trace_id}
            )
        except Exception as e:
            logger.error(
                f"Evaluation pipeline failed: {e}",
                extra={"trace_id": trace_id},
                exc_info=True
            )
    
    async def _run_single_evaluation(
        self,
        eval_type: str,
        evaluator: Any,
        **kwargs
    ) -> tuple[str, Any]:
        """Run a single evaluation with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                # Run evaluation
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: evaluator.evaluate(**kwargs)
                )
                return (eval_type, result)
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    logger.debug(
                        f"Retrying {eval_type} evaluation (attempt {attempt + 1})",
                        extra={"error": str(e)}
                    )
        
        # All retries failed
        raise last_error
    
    def _log_evaluation_result(self, result: EvaluationResult):
        """Log evaluation result to observability platforms."""
        log_data = {
            "eval_type": result.eval_type,
            "score": result.score,
            "label": result.label,
            "trace_id": result.trace_id,
            "timestamp": result.timestamp.isoformat(),
        }
        
        # Log with structured logging
        logger.info(f"Evaluation result: {result.eval_type}", extra=log_data)
        
        # Log to Logfire if available
        if LOGFIRE_AVAILABLE:
            try:
                logfire.info(
                    "LLM evaluation completed",
                    _tags=["evaluation", result.eval_type],
                    **log_data
                )
            except Exception as e:
                logger.debug(f"Failed to log to Logfire: {e}")
        
        # TODO: Log to Arize Phoenix when dashboard integration is available
        # px.log_evaluations(
        #     evaluation_results=[result],
        #     trace_id=result.trace_id
        # )
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        if not self.evaluation_results:
            return {
                "total_evaluations": 0,
                "enabled": self.enabled,
                "config": self.config.model_dump()
            }
        
        # Calculate summary statistics
        summary = {
            "total_evaluations": len(self.evaluation_results),
            "enabled": self.enabled,
            "config": self.config.model_dump(),
            "results_by_type": {},
            "recent_results": []
        }
        
        # Group by evaluation type
        from collections import defaultdict
        by_type = defaultdict(list)
        
        for result in self.evaluation_results:
            by_type[result.eval_type].append(result)
        
        # Calculate stats per type
        for eval_type, results in by_type.items():
            scores = [r.score for r in results]
            summary["results_by_type"][eval_type] = {
                "count": len(results),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
            }
        
        # Recent results
        summary["recent_results"] = [
            r.model_dump() for r in self.evaluation_results[-10:]
        ]
        
        return summary
    
    def clear_results(self):
        """Clear evaluation results buffer."""
        self.evaluation_results.clear()
        logger.debug("Evaluation results buffer cleared")


# Global evaluator instance (lazy initialization)
_phoenix_evaluator: Optional[PhoenixEvaluator] = None


def get_phoenix_evaluator(config: Optional[EvaluationConfig] = None) -> PhoenixEvaluator:
    """Get or create Phoenix evaluator instance."""
    global _phoenix_evaluator
    
    if _phoenix_evaluator is None:
        _phoenix_evaluator = PhoenixEvaluator(config)
    
    return _phoenix_evaluator


# Convenience functions
async def evaluate_llm_response(**kwargs) -> List[EvaluationResult]:
    """Evaluate LLM response using Phoenix."""
    evaluator = get_phoenix_evaluator()
    return await evaluator.evaluate_response(**kwargs)


def get_evaluation_summary() -> Dict[str, Any]:
    """Get evaluation summary."""
    evaluator = get_phoenix_evaluator()
    return evaluator.get_evaluation_summary()


__all__ = [
    "PhoenixEvaluator",
    "EvaluationResult",
    "EvaluationConfig",
    "get_phoenix_evaluator",
    "evaluate_llm_response",
    "get_evaluation_summary",
]