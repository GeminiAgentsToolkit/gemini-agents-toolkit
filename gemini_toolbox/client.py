from vertexai.generative_models import (
    Part, GenerativeModel,
)
import inspect
from gemini_toolbox import declarations


class GeminiChatClient(object):

    def __init__(self, functions, model, *, debug=False, recreate_client_each_time=True):
        self.functions = {func.__name__: func for func in functions}
        self.chat = model.start_chat()
        self._model = model
        self.debug = debug
        self.recreate_client_each_time = recreate_client_each_time

    def _call_function(self, function_call):
        if self.debug:
            print("call function initiated")
            print(str(function_call))
        # Find the function by name in self.functions
        func = self.functions.get(function_call.name)
        if func:
            # Extract the arguments from the function call
            args = function_call.args
            # Call the function with the extracted arguments
            return func(**args)
        else:
            return {"error": "Function not found"}
        
    def _maybe_recreate_client(self):
        if self.recreate_client_each_time:
            self.chat = self._model.start_chat()

    def send_message(self, msg):
        if self.debug:
            print(f"about to send msg: {msg}")
        response = self.chat.send_message(msg)
        if self.debug:
            print(f"msg: '{msg}' is out")
        # Process any function calls until there are no more function calls in the response
        while response.candidates[0].function_calls:
            function_call = response.candidates[0].function_calls[0]
            api_response = self._call_function(function_call)
            
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
        # Extract the text from the final model response
        return response.text
    

def generate_chat_client_from_functions_package(package, model_name="gemini-1.5-pro", *, system_instruction="", debug=False, recreate_client_each_time=True):
    all_functions = [
        func
        for name, func in inspect.getmembers(package, inspect.isfunction)
    ]
    all_functions_tools = declarations.generate_tool_from_functions(all_functions)
    model = GenerativeModel(model_name=model_name, tools=[all_functions_tools], system_instruction=system_instruction)
    return GeminiChatClient(all_functions, model, debug=debug, recreate_client_each_time=recreate_client_each_time)


def generate_chat_client_from_functions_list(all_functions, model_name="gemini-1.5-pro", *, system_instruction="", debug=False, recreate_client_each_time=True):
    all_functions_tools = declarations.generate_tool_from_functions(all_functions)
    model = GenerativeModel(model_name=model_name, tools=[all_functions_tools], system_instruction=system_instruction)
    return GeminiChatClient(all_functions, model, debug=debug, recreate_client_each_time=recreate_client_each_time)