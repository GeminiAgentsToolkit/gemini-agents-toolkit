from vertexai.generative_models import (
    Part,
)


class GeminiChatClient(object):

    def __init__(self, functions, model, debug=False):
        self.functions = {func.__name__: func for func in functions}
        self.chat = model.start_chat()
        self.debug = debug

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

        # Extract the text from the final model response
        return response.text