from gemini_agents_toolkit.pipeline.terminal_step import TerminalStep


class SummaryStep(TerminalStep):

    def __init__(self, agent, name=None, debug=False):
        super().__init__(name, debug)
        self.agent = agent
    
    def run(self, data):
        prompt = f"Now if the final step of the pipeline/dialog, provide summary of main things that were done and why. do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about any messages even if they were comming from the user before, now is the time to build proper summary for a user.\n Prev data from prev step: {data}."
        return self.agent.send_message(prompt)
