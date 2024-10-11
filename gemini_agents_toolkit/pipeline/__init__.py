from vertexai.generative_models import GenerationConfig
from gemini_agents_toolkit.history_utils import summarize
from gemini_agents_toolkit import agent
from config import (SIMPLE_MODEL)


class Pipeline(object):
    def __init__(self, *, default_agent=None, logger=None, use_convert_to_bool_agent=False):
        self.agent = default_agent
        self.logger = logger
        self._full_history = []
        if use_convert_to_bool_agent:
            response_schema = {"type": "STRING", "enum": ["True", "False"]}
            generation_config = GenerationConfig(
                response_schema=response_schema,
                response_mime_type="application/json",
                temperature=0
            )
            self.convert_to_bool_agent = agent.create_agent_from_functions_list(model_name=SIMPLE_MODEL, generation_config=generation_config)

    def _get_agent(self, agent):
        if agent:
            return agent
        if self.agent:
            return self.agent
        raise ValueError("either default agent or local(per ste) agent should be set")

    def if_step(self, prompt, then_steps=None, else_steps=None, *, agent=None, history=None):
        agent_to_use = self._get_agent(agent)
        if self.logger:
            self.logger.info(f"if_step: {prompt}, then_steps: {then_steps}, else_steps: {else_steps}") 
        bool_result, updated_history = self.boolean_step(prompt, agent=agent_to_use, history=history)
        if bool_result:
            if then_steps:
                return self.steps(then_steps, agent=agent_to_use, history=updated_history)
        else:
            if else_steps:
                return self.steps(else_steps, agent=agent_to_use, history=updated_history)

    def steps(self, steps, *, agent=None, history=None):
        agent_to_use = self._get_agent(agent)
        if history:
            agent_to_use.set_history(history)
        final_result = None
        final_history = history
        if isinstance(steps, list):
            for step in steps:
                final_result, delta_history = self.step(step, agent=agent_to_use, history=final_history)
                final_history += delta_history
        else:
            final_result, final_history = self.step(steps, agent=agent_to_use, history=final_history)
        return final_result, final_history

    def step(self, prompt, *, agent=None, history=None):
        if self.logger:
            self.logger.info(f"step: {prompt}")
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        user prompt: {prompt}"""
        agent_to_use = self._get_agent(agent)
        result, updated_history = agent_to_use.send_message(prompt, history=history)
        self._full_history.extend(updated_history)
        return result, updated_history
    
    def boolean_step(self, prompt, *, agent=None, history=None):
        if self.logger:
            self.logger.info(f"boolean_step: {prompt}")

        #TODO think to rename puser prompt to simple user question. 
        prompt = f"""this is one step in the pipeline, this steps are user command but not comming direclty from the user:
        Following prompt provided by user, and user expects this to compute in a boolean yes/no answer, you have to retunr
        True/False and nothing else in your response. 
        Prompt: {prompt}
        
        IMPORTANT: remember you ONLY can return True/False, no print(False) or print(True) or any other print statement"""
        agent_to_use = self._get_agent(agent)
       
        bool_answer, history = agent_to_use.send_message(prompt, history=history)
        if self.convert_to_bool_agent:
            bool_answer, _ = self.convert_to_bool_agent.send_message(f"please convert to best fitting response True/False here is answer:{bool_answer}, \n question was: {prompt}")
        self._full_history.extend(history)
        if "true" in bool_answer.lower():
            if self.logger:
                self.logger.info(f"boolean_step: True")
            return True, history
        elif "false" in bool_answer.lower():
            if self.logger:
                self.logger.info(f"boolean_step: False")
            return False, history
        else:
            if self.logger:
                self.logger.debug(f"prompt:{prompt}\nanswer:{bool_answer}")
            raise ValueError("Invalid response from user, expected True/False only")
        
    def summarize_full_history(self, *, agent=None):
        agent_to_use = self._get_agent(agent)
        return summarize(agent=agent_to_use, history=self._full_history)
    
    def get_full_history(self):
        return self._full_history
    