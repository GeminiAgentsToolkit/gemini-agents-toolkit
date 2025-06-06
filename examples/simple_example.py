"""This example illustrates creating an agent to execute instructions"""

from gemini_agents_toolkit.config import SIMPLE_MODEL
from gemini_agents_toolkit import agent, history_utils
from google.adk.agents import LlmAgent


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"

all_functions = [say_to_duck]
duck_comms_agent = agent.ADKAgentService(agent=LlmAgent(model=SIMPLE_MODEL, name="test_agent", tools=all_functions))

msg, history = duck_comms_agent.send_message("say to the duck message: I am hungry")
print(msg)
history_list = history_utils.to_serializable_list(history)
history_restored = history_utils.from_serializable_list(history_list)
msg, _ = duck_comms_agent.send_message("can you repeat, what did the duck sad?", history=history_restored)
print(msg)


# print(duck_comms_agent.get_history()[-1])
# print(type(duck_comms_agent.get_history()[-1]))
# print(type(duck_comms_agent.get_history()))
# print(duck_comms_agent.get_history()[-1]['raw'])
# print(type(duck_comms_agent.get_history()[-1]['raw']))