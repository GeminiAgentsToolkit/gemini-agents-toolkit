from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline import Pipeline
from gemini_agents_toolkit.config import SIMPLE_MODEL
from google.adk.agents import LlmAgent


fs_agent = agent.ADKAgentService(agent=LlmAgent(model=SIMPLE_MODEL, name="test_agent"))

pipeline = Pipeline(default_agent=fs_agent, use_convert_agent_helper=True)
msg, _ = pipeline.float_step("60 - 5% is?")

print(msg)

msg, _ = pipeline.int_step("22 - 53?")

print(msg)

msg, _ = pipeline.char_step("first character of the word 'hello' is?")

print(msg)

msg, _ = pipeline.string_array_step("sort words(by alphabet): 'my', 'hello', 'pen'")

print(msg)
