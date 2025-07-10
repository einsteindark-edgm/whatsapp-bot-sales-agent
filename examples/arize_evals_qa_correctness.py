# Source: https://arize.com/docs/phoenix/retrieval/quickstart-retrieval

from phoenix.trace import SpanEvaluations
from phoenix.evals import (
    HALLUCINATION_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    QA_PROMPT_RAILS_MAP,
    QA_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)
import phoenix as px

# `queries_df` must contain question / answer pairs (see docs)
hallucination_eval = llm_classify(
    dataframe=queries_df,
    model=OpenAIModel("gpt-4-turbo-preview", temperature=0.0),
    template=HALLUCINATION_PROMPT_TEMPLATE,
    rails=list(HALLUCINATION_PROMPT_RAILS_MAP.values()),
    provide_explanation=True,
    concurrency=4,
)
hallucination_eval["score"] = (
    hallucination_eval.label[~hallucination_eval.label.isna()] == "factual"
).astype(int)

qa_correctness_eval = llm_classify(
    dataframe=queries_df,
    model=OpenAIModel("gpt-4-turbo-preview", temperature=0.0),
    template=QA_PROMPT_TEMPLATE,
    rails=list(QA_PROMPT_RAILS_MAP.values()),
    provide_explanation=True,
    concurrency=4,
)
qa_correctness_eval["score"] = (
    qa_correctness_eval.label[~qa_correctness_eval.label.isna()] == "correct"
).astype(int)

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval),
)
