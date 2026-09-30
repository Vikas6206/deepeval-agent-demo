import sys
import os

# evals/vik/ → project root (3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe
from deepeval.models import AnthropicModel
from agent_instrumented import support_agent as _support_agent

judge = AnthropicModel(model="claude-sonnet-4-6")

# in order to observe the tracing, we need to set tracing by @observe before calling the agent
# Trace have actual and expected values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)
    # this is the actual agent call with observibility already applied and CallbackHandler is used to handle the tracing



prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ], threshold= 0.7, model = judge
)

step_efficiency = StepEfficiencyMetric(threshold= 0.5, model = judge)
answer_relevance = AnswerRelevancyMetric(threshold= 0.7, model = judge)

evals_dataset = EvaluationDataset(goldens = [
     Golden(input = "Where is my order ORD-1042?"),
    Golden( input="What's the refund policy for electronics?" ),
    Golden( input="I want to return order ORD-2099, what should I do?" )
])

for golden in evals_dataset.evals_iterator(metrics = [prompt_alignment,step_efficiency,answer_relevance]):
    support_agent(golden.input)