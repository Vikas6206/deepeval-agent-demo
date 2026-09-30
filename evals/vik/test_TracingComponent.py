import sys
import os

# evals/vik/ → project root (3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent

# Judge LLM — must pass AnthropicModel; a plain "claude-..." string defaults to OpenAI
judge = AnthropicModel(model="claude-sonnet-4-6")

# create metrics
task_completion_metric = TaskCompletionMetric(threshold=0.7, model=judge)
tool_correctness_metric = ToolCorrectnessMetric(model=judge)

# in order to observe the tracing, we need to set tracing by @observe before calling the agent
# Trace have actual and expected values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()

    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)

        # in case if we are expecting any expected output, we need to update the trace with the expected output [N/A for now]
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)

    return _support_agent(user_input)
    # this is the actual agent call with observibility already applied and CallbackHandler is used to handle the tracing

# create data set
dataSet = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),

    Golden(input="What is refund policy for electronics?",
           expected_tools=[ToolCall(name="get_refund_policy")])
])

# loop through data set, call the agent and evaluate metrics
for golden in dataSet.evals_iterator(metrics=[task_completion_metric, tool_correctness_metric]):
    support_agent(golden.input)  # calling the above agent function with the input
    # tracing is happening when the agent is called, @observe is observing the tracing


# We go for LLMtestcase( style of testing when access to agent is not provided)
