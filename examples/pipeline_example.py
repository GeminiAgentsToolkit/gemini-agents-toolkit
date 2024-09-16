"""This example illustrates creating a pipeline with several basic steps"""

import vertexai
from config import (PROJECT_ID, REGION, DEFAULT_MODEL, SIMPLE_MODEL)
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline.basic_step import BasicStep


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


vertexai.init(project=PROJECT_ID, location=REGION)

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions,
                                                          model_name=SIMPLE_MODEL)
computation_agent = agent.create_agent_from_functions_list(
    model_name=DEFAULT_MODEL,
    system_instruction="you are agent design to do computation"
)

PRINTING_AGENT_PROMPT = """
You are agent design to do nice prints in the shell.
You do not have to be producing BASH commands,
your output will be printed by other developer in the bash.
You are ONLY in charge of the formatting ASCII characters  that will be printed.
No need to even use characters like ``` before/after you response.
"""
printing_agent = agent.create_agent_from_functions_list(model_name=DEFAULT_MODEL,
                                                        system_instruction=PRINTING_AGENT_PROMPT)

pipeline = (BasicStep(duck_comms_agent, "say to duck: I am hungry") |
            BasicStep(
                computation_agent,
                "calculate how many times word duck was used in the response"
            ) |
            BasicStep(
                printing_agent,
                "print number in a nice format, go nuts with how you want it to look"
            ))
print(pipeline.execute())
