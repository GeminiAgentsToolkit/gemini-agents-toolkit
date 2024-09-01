from gemini_agents_toolkit.pipeline.abstract_step import AbstractStep
from gemini_agents_toolkit.agent import GeminiAgent


class BasicStep(AbstractStep):
    def __init__(self, agent: GeminiAgent, prompt: str, name:str = None, debug: bool = False):
        super(BasicStep, self).__init__(name=name, debug=debug)
        self.agent = agent
        self.prompt = prompt
        self.debug = debug

    def run(self, data):
        prompt = f"user prompt: {self.prompt}\n data from prev steps: {data}"
        if self.debug:
            print("prompt to send: " + prompt)
        return self.agent.send_message(prompt)
