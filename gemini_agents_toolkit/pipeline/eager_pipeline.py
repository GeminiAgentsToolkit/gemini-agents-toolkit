class EagerPipeline(object):
    def __init__(self, *, default_agent=None, logger=None):
        self.agent = default_agent
        self.prev_step_data = None
        self.logger = logger


    def _get_agent(self, agent):
        if agent:
            return agent
        if self.agent:
            return self.agent
        raise ValueError("either default agent or local(per ste) agent should be set")


    def if_step(self, prompt, then_steps, else_steps, *, agent=None):
        agent_to_use = self._get_agent(agent)
        if self.logger:
            self.logger.info(f"if_step: {prompt}, then_steps: {then_steps}, else_steps: {else_steps}") 
        if self.boolean_step(prompt, agent=agent_to_use):
            if then_steps:
                self.steps(then_steps, agent=agent_to_use)
        else:
            if else_steps:
                self.steps(else_steps, agent=agent_to_use)

    def steps(self, steps, *, agent=None):
        agent_to_use = self._get_agent(agent)
        if isinstance(steps, list):
            for step in steps:
                self.step(step, agent=agent_to_use)
        else:
            self.step(steps, agent=agent_to_use)

    def step(self, prompt, *, agent=None):
        if self.logger:
            self.logger.info(f"step: {prompt}")
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        user prompt: {prompt}
        data from prev steps: {self.prev_step_data}"""
        agent_to_use = self._get_agent(agent)
        self.prev_step_data = agent_to_use.send_message(prompt)
        return self.prev_step_data
    
    def boolean_step(self, prompt, *, agent=None):
        if self.logger:
            self.logger.info(f"boolean_step: {prompt}")
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        Following prompt provided by user, and user expects this to compute in a boolean yes/no answer, you have to retunr
        True/False and nothing else in your response. 
        Prompt: {prompt}
        data from prev steps: {self.prev_step_data}.
        
        IMPORTANT: remember you ONLY can return True/False"""
        agent_to_use = self._get_agent(agent)
        bool_answer = agent_to_use.send_message(prompt)
        if "true" in bool_answer.lower():
            if self.logger:
                self.logger.info(f"boolean_step: True")
            return True
        elif "false" in bool_answer.lower():
            if self.logger:
                self.logger.info(f"boolean_step: False")
            return False
        else:
            raise ValueError("Invalid response from user, expected True/False only")
        
    def summary(self, *, agent=None):
        if self.logger:
            self.logger.info(f"summary")
        prompt = f"""Now if the final step of the pipeline/dialog, provide summary of main things that were done and why.
        do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about
        any messages even if they were comming from the user before, now is the time to build proper summary for a user.
        Prev data from prev step: {self.prev_step_data}."""
        agent_to_use = self._get_agent(agent)
        return agent_to_use.send_message(prompt)
    
