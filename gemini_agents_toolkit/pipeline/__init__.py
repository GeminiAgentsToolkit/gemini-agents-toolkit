from gemini_agents_toolkit.pipeline.summary_step import SummaryStep
from gemini_agents_toolkit.pipeline.basic_step import BasicStep
from gemini_agents_toolkit.pipeline.if_step import IfStep
from gemini_agents_toolkit.pipeline.terminal_step import TerminalStep


class PipelineBuilder:

    def __init__(self, agent, debug=False):
        self.agent = agent
        self.debug = debug

    def summary_step(self, name=None):
        return SummaryStep(self.agent, name, self.debug)

    def basic_step(self, prompt, name=None):
        return BasicStep(self.agent, prompt, name=name, debug=self.debug)

    def if_step(self, prompt, then_step, else_step, name=None):
        return IfStep(self.agent, prompt, then_step=then_step, else_step=else_step, debug=self.debug, name=name)

    def terminal_step(self, name=None):
        return TerminalStep(name, self.debug)