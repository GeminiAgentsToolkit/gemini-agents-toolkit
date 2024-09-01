from vertexai.generative_models import (
    Part, GenerativeModel,
)
from vertexai.generative_models import FunctionDeclaration, Tool
from gemini_agents_toolkit import scheduler

import logging
import inspect
import traceback


class GeminiAgent(object):

    def __init__(self, model_name="gemini-1.5-pro", *, functions=None, system_instruction=None, delegation_function_prompt=None, delegates=None, debug=False, recreate_client_each_time=False, history_depth=-1, on_message=None):
        if not functions:
            functions = []
        self.functions = {func.__name__: func for func in functions}
        func_declarations =  [_generate_function_declaration(func) for func in functions]
        if delegates is not None:
            for i, delegate in enumerate(delegates):
                if not isinstance(delegate, GeminiAgent):
                    raise ValueError("delegates must be a list of GeminiAgent instances")
                if not delegate.delegation_function_prompt:
                    raise ValueError("all delegates must have a delegation prompt specified set")
                
                # Get the signature of the original send_message method
                tool_name = f"delegate_{i}_send_message"
                self.functions[tool_name] = delegate.send_message
                func_declarations.append(_generate_function_declaration(delegate.send_message, user_set_name=tool_name, user_set_description=delegate.delegation_function_prompt))
        self._model = GenerativeModel(model_name=model_name, tools=[Tool(function_declarations=func_declarations)], system_instruction=system_instruction)
        self.chat = self._model.start_chat()
        self.debug = debug
        self.recreate_client_each_time = recreate_client_each_time
        self._delegation_prompt_set = False
        self.on_message = on_message
        self.delegation_function_prompt = delegation_function_prompt
        if delegation_function_prompt is not None:
            delegation_function_prompt = f"""Use this funciton to delegate task to the agent (your coworker). 
            For delegate pass natural language command to the agent via this method.
            Here is what this agent can be delegate to do: {delegation_function_prompt}

            Keep in mind that the agent does not have full context of the converstaion you have to construct the command in a way that agent can understand and execute the task.
            The agent will treat the message you send as if it comes from user and will respond to it as if it is a user message. In this case you are the user, and you need to ask it to do something, be clear and concise."""
            self.delegation_function_prompt = delegation_function_prompt
        if history_depth > 0 and recreate_client_each_time:
            raise ValueError("history_depth can only be used with recreate_client_each_time=False")
        
    def _call_function(self, function_call):
        if self.debug:
            print("call function initiated")
            print(str(function_call))
        # Find the function by name in self.functions
        func = self.functions.get(function_call.name)
        if func:
            try:
                # Extract the arguments from the function call
                args = function_call.args
                # Call the function with the extracted arguments
                return func(**args)
            except TypeError as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}"}
            except ValueError as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Value error during function call  {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Value error during function call {function_call.name}: {e}\n{stack_trace}"}
            except Exception as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}"}
        else:
            return {"error": "Function not found"}
        
    def _maybe_trim_history(self):
        if self.history_depth > 0:
            trimmed_history = []
            user_messages_count = 0
            
            # Iterate through the history in reverse
            for h in reversed(self.chat._history):
                if h.role == "user":
                    trimmed_history.append(h)
                    user_messages_count += 1
                    if user_messages_count == self.history_depth:
                        break
                else:
                    trimmed_history.append(h)
                    
            # Reverse the trimmed history to restore the original order
            self.chat._history = list(reversed(trimmed_history))
        
    def _maybe_recreate_client(self):
        if self.recreate_client_each_time:
            self.chat = self._model.start_chat()

    def send_message(self, msg: str) -> str:
        if self.debug:
            print(f"about to send msg: {msg}")
        response = self.chat.send_message(msg)
        if self.debug:
            print(f"msg: '{msg}' is out")
        
        # Process any function calls until there are no more function calls in the response
        while response.candidates[0].function_calls:
            if self.debug:
                print(f"function call found: {response.candidates[0].function_calls[0]}")
            function_call = response.candidates[0].function_calls[0]
            api_response = self._call_function(function_call)
            if self.debug:
                print(f"api response: {api_response}")
            
            # Return the API response to Gemini
            response = self.chat.send_message(
                Part.from_function_response(
                    name=function_call.name,
                    response={
                        "content": api_response,
                    },
                ),
            )

        self._maybe_recreate_client()
        if self.on_message is not None:
            self.on_message(response.text)
        # Extract the text from the final model response
        return response.text

    
def _generate_function_declaration(func, *, user_set_name=None, user_set_description=None):
    func_name = user_set_name
    if not func_name:
        # Extract function name
        func_name = func.__name__
    
    func_doc = user_set_description
    if not func_doc:
        # Extract function docstring
        func_doc = func.__doc__ if func.__doc__ else ""
    
    # Extract function parameters
    sig = inspect.signature(func)
    params = {
        "type": "object",
        "properties": {}
    }
    for name, param in sig.parameters.items():
        # Assuming all parameters are strings for simplicity; you can enhance this as needed
        params["properties"][name] = {"type": "string", "description": "No description provided"}
    
    # Create FunctionDeclaration
    function_declaration = FunctionDeclaration(
        name=func_name,
        description=func_doc,
        parameters=params
    )
    
    return function_declaration


def create_agent_from_functions_list(
        *, 
        model_name="gemini-1.5-pro",
        functions=[],
        system_instruction=None, 
        debug=False, 
        recreate_client_each_time=False, 
        history_depth=-1,
        add_scheduling_functions=False, 
        gcs_bucket=None, 
        gcs_blob=None,
        delegation_function_prompt=None, 
        delegates=None,
        on_message=None
    ):
    if (not gcs_bucket and gcs_blob) or (gcs_bucket and not gcs_blob):
        raise ValueError("Both gcs_bucket and gcs_blob must be provided")
    if add_scheduling_functions:
        scheduler_instance = scheduler.ScheduledTaskExecutor(debug=debug, gcs_bucket=gcs_bucket, gcs_blob=gcs_blob)
        functions.extend([
            scheduler_instance.add_task,
            scheduler_instance.get_all_jobs,
            scheduler_instance.delete_job,
        ])
    if add_scheduling_functions and not on_message:
        # print warning
        logging.warning("on_message is not set, you may not see the results of the scheduled tasks")
    if debug:
        print(f"all_functions: {functions}")
    agent = GeminiAgent(
        delegation_function_prompt=delegation_function_prompt,
        delegates=delegates,
        on_message=on_message,
        functions=functions, 
        model_name=model_name, 
        system_instruction=system_instruction, 
        debug=debug, 
        recreate_client_each_time=recreate_client_each_time, 
        history_depth=history_depth)
    if add_scheduling_functions:
        scheduler_instance.set_gemini_agent(agent)
        scheduler_instance.start_scheduler()
    return agent
