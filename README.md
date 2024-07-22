SDK For Simplified usage of Gemini Agents. Give Gemini Ability to use your custom functions in seveal code lines:

```python
import vertexai
from gemini_toolbox import client

def get_current_time():
    """returns current time"""
    return "6pm PST"


def say_to_duck(say):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"

vertexai.init(project="gemini-trading-backend", location="us-west1")

all_functions = [get_current_time, say_to_duck]
clt = client.generate_chat_client_from_functions_list(all_functions, model_name="gemini-1.5-pro", debug=True)

print(clt.send_message("say to the duck message: I am hungry"))
```