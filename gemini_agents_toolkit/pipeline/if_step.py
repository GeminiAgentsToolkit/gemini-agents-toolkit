from gemini_agents_toolkit.pipeline.basic_step import BasicStep


class IfStep(BasicStep):

    def __init__(self, agent, prompt, name=None, debug=False, if_step=None, else_step=None):
        super().__init__(agent, prompt, name, debug)
        if isinstance(if_step, str):
            self.if_step = BasicStep(agent, if_step)
        while if_step.previous_step:
            if_step = if_step.previous_step
        if_step.previous_step = self
        self.if_step = if_step
        if isinstance(else_step, str):
            self.else_step = BasicStep(agent, else_step)
        while else_step.previous_step:
            else_step = else_step.previous_step
        else_step.previous_step = self
        self.else_step = else_step
        if self.debug:
            print(f"if step: {self.if_step}")
            print(f"else step: {self.else_step}")
    
    def run(self, data):
        prompt = f"Following prompt provided by user, and user expects this to compute in a boolean yes/no answer, you have to retunr True/False and nothing else in your response. Prompt: {self.prompt}\n data from prev steps: {data}.\n\n remember you ONLY can return True/False"
        bool_answer = self.agent.send_message(prompt)
        if self.debug:
            print(f"prompt to send: {prompt}")
            print(f"response from agent: {bool_answer}")
        if "true" in bool_answer.lower():
            self._next_step = self.if_step
            if self.debug:
                print(f"True - Chaining to next step: {self.if_step}")
            return data
        elif "false" in bool_answer.lower():
            if self.debug:
                print(f"False - Chaining to next step: {self.else_step}")
            self._next_step = self.else_step
            return data
        else:
            raise ValueError("Invalid response from user, expected True/False only")
