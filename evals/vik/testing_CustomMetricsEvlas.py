import sys
import os

# evals/vik/ → project root (3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import SingleTurnParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.contextvars import get_current_golden
from deepeval.tracing import observe,update_current_trace
from agent_instrumented import support_agent as _support_agent


correctness = GEval(
    name = "Correctness",
    criteria =(
        "Determine wether the actual output conveys the same factual information"
        "as the expected output. Minor wording differences are acceptable."
        "missing or wrong facts are not acceptable."
    ),
    model = AnthropicModel(model="claude-sonnet-4-6"),
    threshold = 0.8,
    evaluation_params =[
        # only single turn conversation with agent and hence used SingleTurnParams
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ]
)


# in order to observe the tracing, we need to set tracing by @observe before calling the agent
# Trace have actual and expected values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()

    if golden:
        # in case if we are expecting any expected output, we need to update the trace with the expected output [N/A for now]
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)
    # this is the actual agent call with observibility already applied and CallbackHandler is used to handle the tracing



#provide the input[goldens], expected output
dataSet = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_output="The order ORD-1042 is shipped and will be delivered by 2026-05-13."),

    Golden(input="What is refund policy for electronics?",
           expected_output="The refund policy for electronics is 15 days return policy, if unopened")
])  

for golden in dataSet.evals_iterator(metrics=[correctness]):
    support_agent(golden.input) 