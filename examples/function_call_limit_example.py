"""This example illustrates creating an agent with function call limits and raising an exception"""

import vertexai

from config import (PROJECT_ID, REGION, SIMPLE_MODEL)
from gemini_agents_toolkit import agent, history_utils
from gemini_agents_toolkit.agent import TooManyFunctionCallsException


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


vertexai.init(project=PROJECT_ID, location=REGION)

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions,
                                                          model_name=SIMPLE_MODEL)

try:
    msg, history = duck_comms_agent.send_message("say to the duck message: I am hungry", limit_function_calls_to = 0)
except TooManyFunctionCallsException as e:
    print(e)
    print(e.call_history)
