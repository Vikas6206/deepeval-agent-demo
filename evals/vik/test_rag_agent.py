import sys
import os
# PRECISION - wethre the actual answer is present in top (k) documents
#RECALL - 10 docs returned but answer was found in Top 2 documents = 80% noise 
# evals/vik/ → project root (3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, AnswerRelevancyMetric,FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.contextvars import get_current_golden
from deepeval.tracing import observe,update_current_trace


from rag_agent import rag_support_agent as _rag_support_agent

# in order to observe the tracing, we need to set tracing by @observe before calling the agent
# Trace have actual and expected values
@observe(name="rag_support_agent")
def rag_support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        # in case if we are expecting any expected output, we need to update the trace with the expected output [N/A for now]
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _rag_support_agent(user_input)
    # this is the actual agent call with observibility already applied and CallbackHandler is used to handle the tracing



dataset = EvaluationDataset(
    goldens =[
        Golden(
            input = "What is the return policy for electronics?",
            expected_output = "You can return the electronics within 15 days of delivery if unopened and in original packaging. Refunds take 5-7 business days."
        ),
        Golden(
            input = "How long the express delivery takes and what does it cost?",
            expected_output = "Express delivery typically takes 2-3 business days within the US."
        )
    ]
);

precisionMetric = ContextualPrecisionMetric(
    threshold = 0.7,
    model = AnthropicModel(model="claude-sonnet-4-6")
)

contextualRecallMetric = ContextualRecallMetric(
    threshold = 0.7,
    model = AnthropicModel(model="claude-sonnet-4-6")
)

answerRelevanceMetric = AnswerRelevancyMetric(
    threshold = 0.7,
    model = AnthropicModel(model="claude-sonnet-4-6")
)

faithfulnessMetric = FaithfulnessMetric(
    threshold = 0.7,
    model = AnthropicModel(model="claude-sonnet-4-6")
)

for golden in dataset.evals_iterator(metrics=[precisionMetric,contextualRecallMetric,answerRelevanceMetric,faithfulnessMetric]):
    rag_support_agent(golden.input)