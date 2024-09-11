from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline.eager_pipeline import EagerPipeline

from json_agent import create_agent as create_json_agent

import vertexai
import os


def pwd() -> str:
    """print current working directory
    
    Returns:
        str: current working directory
    """
    return os.getcwd()


def ls(path: str) -> str:
    """list files in a directory
    
    Args:
        path (str): path to directory
    
    Returns:
        str: list of files in directory
    """
    return "\n".join(os.listdir(path))


def read_file(path: str) -> str:
    """read content of a file
    
    Args:
        path (str): path to file
    
    Returns:
        str: content of file
    """
    with open(path, "r") as f:
        return "content of the file is: \n" + f.read()


def write_file(path: str, content: str):
    """write a file
    
    Args:
        path (str): path to file
        content (str): content to write
    """
    with open(path, "w") as f:
        f.write(content)
        return "DONE"


all_functions = [read_file, write_file, pwd, ls]
fs_agent_system_instruction = """
You are agent that can read/write data from the FS.
"""
vertexai.init(project="gemini-trading-backend", location="us-west1")

fs_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name="gemini-1.5-pro", system_instruction=fs_agent_system_instruction)
json_agent = create_json_agent(model_name="gemini-1.5-pro")

pipeline = EagerPipeline(default_agent=fs_agent)
pipeline.step("read the content of the file examples/test.yaml")
pipeline.step("convert yaml to json", agent=json_agent)
pipeline.step("save json to file examples/test_out.json")
print(pipeline.summary())