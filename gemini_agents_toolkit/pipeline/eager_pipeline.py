class EagerPipeline(object):
    def __init__(self, agent):
        self.agent = agent
        self.prev_step_data = None

    def if_step(self, prompt, then_steps, else_steps):
        if self.boolean_step(prompt):
            if then_steps:
                self.steps(then_steps)
        else:
            if else_steps:
                self.steps(else_steps)

    def steps(self, steps):
        if isinstance(steps, list):
            for step in steps:
                self.step(step)
        else:
            self.step(steps)

    def step(self, prompt):
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        user prompt: {prompt}
        data from prev steps: {self.prev_step_data}"""
        self.prev_step_data = self.agent.send_message(prompt)
        return self.prev_step_data
    
    def boolean_step(self, prompt):
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        Following prompt provided by user, and user expects this to compute in a boolean yes/no answer, you have to retunr
        True/False and nothing else in your response. 
        Prompt: {prompt}
        data from prev steps: {self.prev_step_data}.
        
        IMPORTANT: remember you ONLY can return True/False"""
        bool_answer = self.agent.send_message(prompt)
        if "true" in bool_answer.lower():
            return True
        elif "false" in bool_answer.lower():
            return False
        else:
            raise ValueError("Invalid response from user, expected True/False only")
        
    def summary(self):
        prompt = f"""Now if the final step of the pipeline/dialog, provide summary of main things that were done and why.
        do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about
        any messages even if they were comming from the user before, now is the time to build proper summary for a user.
        Prev data from prev step: {self.prev_step_data}."""
        return self.agent.send_message(prompt)
    
