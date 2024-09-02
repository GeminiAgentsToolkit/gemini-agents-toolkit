from gemini_agents_toolkit.pipeline.abstract_step import AbstractStep


class TerminalStep(AbstractStep):

    def __init__(self, name=None, debug=False):
        super().__init__(name, debug)
    
    def run(self, data):
        return data
    
    @staticmethod
    def get_terminal_step():
        return TerminalStep()