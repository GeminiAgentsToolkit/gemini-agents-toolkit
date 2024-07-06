SDK For Simplified usage of Gemini Agents. Give Gemini Ability to use your custom functions in seveal code lines:

```python
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
)
from gemini_toolbox import declarations
from gemini_toolbox import client

def get_current_time():
    """returns current time"""
    return "6pm PST"


def say_to_duck(say):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"

all_functions_tools = declarations.generate_tool_from_functions([get_current_time, say_to_duck])

vertexai.init(project="model-registry-v2", location="us-west1")

model = GenerativeModel(model_name="gemini-1.5-pro", tools=[all_functions_tools])

client = client.GeminiChatClient([get_current_time, say_to_duck], model, debug=True)

print(client.send_message("say to the duck message: I am hungry"))
```