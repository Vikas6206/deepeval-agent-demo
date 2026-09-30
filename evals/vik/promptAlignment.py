#2. Prompt allignment i.e.
#wether agent followed the system prompt instructions or not while responding to 
# the user input

#  system_prompt=(
#         "You are a friendly customer-support agent. "
#         "Use the available tools to answer order and refund questions. "
#         "Keep replies short and helpful."
#     )

import sys
import os

# evals/vik/ → project root (3 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



