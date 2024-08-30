SDK For Simplified usage of Gemini Agents. Give Gemini Ability to use your custom functions in seveal code lines:

```python
from gemini_agents_toolkit import agent

import vertexai


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


vertexai.init(project="gemini-trading-backend", location="us-west1")

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name="gemini-1.5-flash")

print(duck_comms_agent.send_message("say to the duck message: I am hungry"))
```

Here is more complex example of several agents that delegating tasks to each other:

```python
from gemini_agents_toolkit import agent

import vertexai


vertexai.init(project="gemini-trading-backend", location="us-west1")


def generate_duck_comms_agent():
    def say_to_duck(say: str):
        """say something to a duck"""
        return f"duck answer is: duck duck {say} duck duck duck"
    return agent.create_agent_from_functions_list(
        functions=[say_to_duck], 
        delegation_function_prompt="Agent can communicat to ducks and can say something to them. And provides the answer from the duck.", 
        model_name="gemini-1.5-flash")


def generate_time_checker_agent():
    def get_local_time():
        """get the current local time"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return agent.create_agent_from_functions_list(
        functions=[get_local_time], 
        delegation_function_prompt="Agent can provide the current local time.", 
        model_name="gemini-1.5-flash")


duck_comms_agent = generate_duck_comms_agent()
time_checker_agent = generate_time_checker_agent()

main_agent = agent.create_agent_from_functions_list(delegates=[time_checker_agent, duck_comms_agent], model_name="gemini-1.5-flash")

print(main_agent.send_message("say to the duck message: I am hungry"))
print(main_agent.send_message("can you tell me what time it is?"))
```

Example of how you can call periodic task:

```python
from gemini_agents_toolkit import agent
import time

import vertexai


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"


def print_msg_from_agent(msg: str):
    print(msg)


vertexai.init(project="gemini-trading-backend", location="us-west1")

all_functions = [say_to_duck]
duck_comms_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name="gemini-1.5-flash", add_scheduling_functions=True, on_message=print_msg_from_agent)

# no need to print result directly since we passed to agent on_message
duck_comms_agent.send_message("can you be saying, each minute, to the duck that I am hungry")

# wait 3 min to see results
time.sleep(180)
```