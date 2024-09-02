from gemini_agents_toolkit.pipeline.abstract_step import AbstractStep
from gemini_agents_toolkit.agent import GeminiAgent


class BasicStep(AbstractStep):
    def __init__(self, agent: GeminiAgent, prompt: str, name:str = None, debug: bool = False):
        super().__init__(name=name, debug=debug)
        self.agent = agent
        self.prompt = prompt
        self.debug = debug

    def run(self, data):
        prompt = f"user prompt: {self.prompt}\n data from prev steps: {data}"
        if self.debug:
            print("prompt to send: " + prompt)
        return self.agent.send_message(prompt)
    
    def next_if_step(self, prompt, if_step, else_step):
        from gemini_agents_toolkit.pipeline.if_step import IfStep
        next_step = IfStep(self.agent, prompt, if_step, else_step, debug=self.debug)
        return self.set_next_step(next_step)

    def next_step(self, next_step):
        if isinstance(next_step, str):
            next_step = BasicStep(self.agent, next_step, debug=self.debug)
        return self.set_next_step(next_step)

    def final_step_summary(self):
        from gemini_agents_toolkit.pipeline.summary_step import SummaryStep
        return self.set_next_step(SummaryStep(self.agent, debug=self.debug))
