from gemini_agents_toolkit.history_utils import summarize, print_history
from gemini_agents_toolkit.config import SIMPLE_MODEL
from gemini_agents_toolkit import agent
from google.adk.agents import LlmAgent


CONVERT_BOT_SYSTEM_INSTRUCTIONS = """There is another agent that produces answer, these answer should comply with the specific schema.
However time to time agent might product results that is NOT exactly following the schema, you main task will be to be fixing such cases.
You will be getting in the input - message from this other agent and a schema that answer should comply with. 
If the message already compy with the schema, you do not have to do anything, just return the message as it is.
If it does not, you have to do you best to convert the message to comply with the schema.
Do you best to contract the best answer or return None if you can not convert the message to comply with the schema.
Your output should be JSON, however under NO circumstances you should print anything before/after the JSON, or any other computational code.
Do not put, for example ```json or ``` in the output, just the JSON. Your output will be used AS IS as a json response."""
BOOLEAN_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "STRING", 
            "enum": ["True", "False"]
        }
    }
}
FLOAT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "number",
            "format": "float"
        }
    }
}
INT_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "INTEGER"
        }
    }
}
CHAR_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "STRING"
        }
    }
}
STRING_ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        }
    }
}


class Pipeline(object):
    def __init__(self, *, default_agent=None, logger=None, use_convert_to_bool_agent=False, use_convert_agent_helper=False, debug=False):
        self.agent = default_agent
        self.logger = logger
        self._full_history = []
        self.debug = debug
        if use_convert_agent_helper or use_convert_to_bool_agent:
            self.convert_agent = agent.ADKAgentService(agent=LlmAgent(
                model=SIMPLE_MODEL, name="convert_agent", instruction=CONVERT_BOT_SYSTEM_INSTRUCTIONS))\

    def _convert_to_type(self, message, return_type_schema):
        if not self.convert_agent:
            return message
        message_to_agent = f"response from other agent: {message}, expected schema: {return_type_schema}"
        response, events = self.convert_agent.send_message(message_to_agent)
        self._full_history.extend(events)
        if "```json" in response:
            response = response.replace("```json","").replace("```", "")
        if self.debug:
            print("#### response after json: " + str(eval(response)))
        return eval(response)["content"]

    def _get_agent(self, agent):
        if agent:
            return agent
        if self.agent:
            return self.agent
        raise ValueError("either default agent or local(per step) agent should be set")

    def if_step(self, prompt, then_steps=None, else_steps=None, *, agent=None, events=None, debug=False):
        agent_to_use = self._get_agent(agent)
        if self.logger:
            self.logger.info(f"if_step: {prompt}, then_steps: {then_steps}, else_steps: {else_steps}") 
        bool_result, updated_history = self.boolean_step(prompt, agent=agent_to_use, events=events)
        if bool_result:
            if then_steps:
                return self.steps(then_steps, agent=agent_to_use, events=updated_history, debug=debug)
        else:
            if else_steps:
                return self.steps(else_steps, agent=agent_to_use, events=updated_history, debug=debug)

    def steps(self, steps, *, agent=None, events=None, debug=False):
        agent_to_use = self._get_agent(agent)
        final_result = None
        final_history = events
        if isinstance(steps, list):
            for step in steps:
                final_result, delta_history = self.step(step, agent=agent_to_use, events=final_history, debug=debug)
                final_history += delta_history
        else:
            final_result, final_history = self.step(steps, agent=agent_to_use, events=final_history, debug=debug)
        return final_result, final_history

    def step(self, prompt, *, agent=None, events=None, debug=False):
        debug_mode = self.debug or debug
        if debug_mode:
            print(f"###### START OF\n=> user prompt: {prompt}")
            print_history("*** INPUT HISTORY ***\n\n")
            print_history(events)

        if self.logger:
            self.logger.info(f"step: {prompt}")
        prompt = f"""this is one step in the pipeline, this steps are user command but not coming directly from the user:
        user prompt: {prompt}"""
        agent_to_use = self._get_agent(agent)
        result, updated_history = agent_to_use.send_message(prompt, events=events)
        self._full_history.extend(updated_history)

        if debug_mode:
            print(f"###### => response from agent: {result}")
            print("@@@@@@@ updated history @@@@@@@ \n")
            print_history(updated_history)
            print(f"###### END OF\n=> user prompt: {prompt}\n#################\n\n\n")
        
        return result, updated_history
    
    def char_step(self, prompt, *, agent=None, events=None, debug=False):
         char_answer, events = self._typed_step(prompt, agent=agent, events=events, debug=debug, type_schema=CHAR_SCHEMA)
         return char_answer, events

    def float_step(self, prompt, *, agent=None, events=None, debug=False):
        float_answer, events = self._typed_step(prompt, agent=agent, events=events, debug=debug, type_schema=FLOAT_SCHEMA)
        return float(float_answer), events
    
    def boolean_step(self, prompt, *, agent=None, events=None, debug=False):
        bool_answer, events = self._typed_step(prompt, agent=agent, events=events, debug=debug, type_schema=BOOLEAN_SCHEMA)
        return eval(bool_answer), events

    def int_step(self, prompt, *, agent=None, events=None, debug=False):
        int_answer, events = self._typed_step(prompt, agent=agent, events=events, debug=debug, type_schema=INT_SCHEMA)
        return int(int_answer), events

    def string_array_step(self, prompt, *, agent=None, events=None, debug=False):
        string_array_answer, events = self._typed_step(prompt, agent=agent, events=events, debug=debug, type_schema=STRING_ARRAY_SCHEMA)
        return string_array_answer, events

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)

    def _typed_step(self, prompt, *, agent=None, events=None, debug=False, type_schema):
        debug_mode = self.debug or debug
        self._log_info(f"boolean_step: {prompt}")

        if debug_mode:
            print(f"###### START OF\n=> user prompt: {prompt}")
            print("*** INPUT HISTORY ***\n\n")
            print_history(events)

        #TODO think to rename user prompt to simple user question.
        prompt = f"""this is one step in the pipeline, this steps are user command but not coming directly from the user:
        Following prompt provided by user, and user expects this to have answer following the json schema:
        {type_schema}
        you have to return respones with the JSON that comply with the schema ONLY.
        Prompt: {prompt}
        
        IMPORTANT: remember you ONLY can return answer that comply with the schema, no print(...) or any computational code or any other print statement"""
        agent_to_use = self._get_agent(agent)

        original_typed_answer, events = agent_to_use.send_message(prompt, events=events)
        typed_answer = self._convert_to_type(original_typed_answer, type_schema)

        if debug_mode:
            print(f"###### => original response from agent: {original_typed_answer}")
            print(f"###### => upated response from agent: {typed_answer}")
            print(f"###### => enforced schema: {type_schema}")
            print("@@@@@@@ updated history @@@@@@@ \n")
            print_history(events)
            print(f"###### END OF\n=> user prompt: {prompt}\n#################\n\n\n")

        self._full_history.extend(events)
        return typed_answer, events
        
    def summarize_full_history(self, *, agent=None):
        agent_to_use = self._get_agent(agent)

        summary_text, events = summarize(agent=agent_to_use, events=self._full_history)
        self._full_history.extend(events)
        return f"SUMMARY:\n{summary_text}", events
    
    def get_full_history(self):
        return self._full_history
    
    def print_full_history(self):
        print_history(self.get_full_history())
