"""This example illustrates usage several agents in a pipeline"""

import os
import vertexai
# pylint: disable-next=import-error
from json_agent import create_agent as create_json_agent
from config import (PROJECT_ID, REGION, DEFAULT_MODEL)
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline.eager_pipeline import EagerPipeline


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
    with open(path, "r", encoding="utf-8") as f:
        return "content of the file is: \n" + f.read()


def write_file(path: str, content: str):
    """write a file
    
    Args:
        path (str): path to file
        content (str): content to write
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        return "DONE"
    

def notify_user(msg: str):
    """notify user when needed about something, only use this when user explicitly wants to be notified
    
    Args:
        msg (str): message to notify
    """
    print("NOTIFICATION: " + msg)


all_functions = [read_file, write_file, pwd, ls]
FS_AGENT_SYSTEM_INSTRUCTION = """
You are agent that can read/write data from the FS.
"""
vertexai.init(project=PROJECT_ID, location=REGION)

fs_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name=DEFAULT_MODEL,
                                                  system_instruction=FS_AGENT_SYSTEM_INSTRUCTION)
json_agent = create_json_agent(model_name=DEFAULT_MODEL)

pipeline = EagerPipeline(default_agent=fs_agent)
pipeline.step("read the content of the file examples/test.yaml")
pipeline.step("convert yaml to json", agent=json_agent)
pipeline.step("save json to file examples/test_out.json")
print(pipeline.summary())
