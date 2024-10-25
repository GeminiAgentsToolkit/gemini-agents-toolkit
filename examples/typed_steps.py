import vertexai
from config import (PROJECT_ID, REGION, SIMPLE_MODEL)
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline import Pipeline


vertexai.init(project=PROJECT_ID, location=REGION)
fs_agent = agent.create_agent_from_functions_list()

pipeline = Pipeline(default_agent=fs_agent, use_convert_agent_helper=True)
msg, _ = pipeline.float_step("60 - 5% is?")

print(msg)

msg, _ = pipeline.int_step("60 - 5?")

print(msg)

msg, _ = pipeline.char_step("first character of the world 'hello' is?")

print(msg)

msg, _ = pipeline.string_array_step("sort words(by alphabet): 'my', 'hello', 'pen'")

print(msg)
