from gemini_agents_toolkit import agent

import vertexai


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


vertexai.init(project="gemini-trading-backend", location="us-west1")

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name="gemini-1.5-flash")

print(duck_comms_agent.send_message("say to the duck message: I am hungry"))