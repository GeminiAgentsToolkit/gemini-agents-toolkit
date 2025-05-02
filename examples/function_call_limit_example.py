"""This example illustrates creating an agent with function call limits and raising an exception"""

from gemini_agents_toolkit.config import SIMPLE_MODEL
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.agent import TooManyFunctionCallsException
from google.adk.agents import LlmAgent


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"



all_functions = [say_to_duck]
duck_comms_agent = agent.ADKAgenService(
    agent=LlmAgent(model=SIMPLE_MODEL, name="test_agent", tools=all_functions),
    function_call_limit_per_chat=0)

try:
    msg, history = duck_comms_agent.send_message("say to the duck message: I am hungry")
    print(msg)
except TooManyFunctionCallsException as e:
    print(e)
    print(e.call_history)
