"""This example illustrates usage several agents which execute different instructions"""

import datetime
import vertexai
from config import (PROJECT_ID, REGION, SIMPLE_MODEL, DEFAULT_MODEL)
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.history_utils import summarize

vertexai.init(project=PROJECT_ID, location=REGION)


def generate_duck_comms_agent():
    """create an agent to say to a duck"""

    def say_to_duck(say: str):
        """say something to a duck"""
        return f"duck answer is: duck duck {say} duck duck duck"

    return agent.create_agent_from_functions_list(
        functions=[say_to_duck],
        delegation_function_prompt=("""Agent can communicat to ducks and can say something to them.
                                    And provides the answer from the duck."""),
        model_name=DEFAULT_MODEL)


def generate_time_checker_agent():
    """create an agent to get the time"""

    def get_local_time():
        """get the current local time"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return agent.create_agent_from_functions_list(
        functions=[get_local_time],
        delegation_function_prompt="Agent can provide the current local time.",
        model_name=SIMPLE_MODEL)


duck_comms_agent = generate_duck_comms_agent()
time_checker_agent = generate_time_checker_agent()

main_agent = agent.create_agent_from_functions_list(
    delegates=[time_checker_agent, duck_comms_agent],
    model_name=SIMPLE_MODEL)

result_say_operation, history_say_operation = main_agent.send_message("say to the duck message: I am hungry")
result_time_operation, history_time_operation = main_agent.send_message("can you tell me what time it is?")

print(result_say_operation)
print(result_time_operation)
print(summarize(agent=main_agent, history=history_say_operation + history_time_operation))
