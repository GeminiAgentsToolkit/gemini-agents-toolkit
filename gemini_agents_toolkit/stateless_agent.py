"""An agent for executing user's instructions"""

import logging
import inspect
import traceback
import re
from vertexai.generative_models import Part, GenerativeModel
from vertexai.generative_models import (
    FunctionDeclaration,
    Tool,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
    GenerationConfig,
    Part,
    GenerativeModel
)
from config import (DEFAULT_MODEL)


# pylint: disable-next=too-many-instance-attributes
class StatelessGeminiAgent:
    """An agent to request LLM for executing users instructions with tools (custom functions) provided by user"""

    # pylint: disable-next=too-many-instance-attributes, too-many-arguments, too-many-locals
    def __init__(
            self,
            model_name=DEFAULT_MODEL,
            *,
            functions=None,
            system_instruction=None,
            delegation_function_prompt=None,
            delegates=None,
            debug=False,
            on_message=None
    ):
        if not functions:
            functions = []
        self.functions = {func.__name__: func for func in functions}
        func_declarations = [_generate_function_declaration(func) for func in functions]
        if delegates is not None:
            for i, delegate in enumerate(delegates):
                if not isinstance(delegate, StatelessGeminiAgent):
                    raise ValueError("delegates must be a list of StatelessGeminiAgent instances")
                if not delegate.delegation_function_prompt:
                    raise ValueError("all delegates must have a delegation prompt specified set")

                # Get the signature of the original send_message method
                tool_name = f"delegate_{i}_send_message"
                self.functions[tool_name] = delegate.send_message
                func_declarations.append(
                    _generate_function_declaration(
                        delegate.send_message,
                        user_set_name=tool_name,
                        user_set_description=delegate.delegation_function_prompt))
        tools = None
        if func_declarations:
            tools = [Tool(function_declarations=func_declarations)]
        safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        self._model = GenerativeModel(model_name=model_name,
                                      tools=tools,
                                      system_instruction=system_instruction,
                                      safety_settings=safety_config)
        self.debug = debug
        self._delegation_prompt_set = False
        self.on_message = on_message
        self.delegation_function_prompt = delegation_function_prompt
        if delegation_function_prompt is not None:
            delegation_function_prompt = f"""
            Use this function to delegate task to the agent (your coworker). 
            For delegate pass natural language command to the agent via this method.
            Here is what this agent can be delegate to do: {delegation_function_prompt}

            Keep in mind that the agent does not have full context of the conversation,
             you have to construct the command in a way that agent can understand
             and execute the task.
            The agent will treat the message you send as if it comes from user
             and will respond to it as if it is a user message. In this case you are the user,
             and you need to ask it to do something, be clear and concise."""
            self.delegation_function_prompt = delegation_function_prompt

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
                logging.error(
                    f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Invalid arguments for function {function_call.name}: {e}\n{stack_trace}"}
            except ValueError as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Value error during function call  {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Value error during function call {function_call.name}: {e}\n{stack_trace}"}
            # pylint: disable-next=broad-exception-caught
            except Exception as e:
                stack_trace = traceback.format_exc()
                logging.error(f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}")
                return {"error": f"Unexpected error during function call {function_call.name}: {e}\n{stack_trace}"}
        else:
            return {"error": "Function not found"}

    def send_message(self, msg: str, *, context = None, generation_config: GenerationConfig = None) -> str:
        """Initiate communication with LLM to execute user's instructions"""
        if self.debug:
            print(f"about to send msg: {msg}")
        chat = self._model.start_chat()
        if context:
            chat._history = context
        
        response = chat.send_message(msg)
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
            response = chat.send_message(
                Part.from_function_response(
                    name=function_call.name,
                    response={
                        "content": api_response,
                    },
                ),
                generation_config=generation_config
            )
        if self.on_message is not None:
            self.on_message(response.text)
        # Extract the text from the final model response
        return response.text, chat.history

# pylint: disable-next=too-many-locals
def _generate_function_declaration(func, *, user_set_name=None, user_set_description=None):
    func_name = user_set_name or func.__name__
    func_doc = user_set_description or (func.__doc__ or "")

    sig = inspect.signature(func)
    params = {
        "type": "object",
        "properties": {}
    }

    # Parse parameter descriptions from docstring using regex
    param_descriptions = {}
    if func_doc:
        param_pattern = re.compile(r":param\s+(\w+):\s*(.*)")
        for match in param_pattern.finditer(func_doc):
            param_name, description = match.groups()
            param_descriptions[param_name] = description.strip()

    for name, param in sig.parameters.items():
        type_mapping = {
            int: "integer",
            float: "number",
            str: "string",
            # Add more mappings as needed
        }
        param_type = type_mapping.get(param.annotation, "string")

        params["properties"][name] = {
            "type": param_type,
            "description": param_descriptions.get(name, "No description provided")
        }

    function_declaration = FunctionDeclaration(
        name=func_name,
        description=func_doc,
        parameters=params
    )

    return function_declaration


# pylint: disable-next=too-many-arguments
def create_agent_from_functions_list(
        *,
        model_name=DEFAULT_MODEL,
        functions=None,
        system_instruction=None,
        debug=False,
        delegation_function_prompt=None,
        delegates=None,
        on_message=None
):
    """Create an agent with custom functions and other parameters received from user"""
    if functions is None:
        functions = []
    if debug:
        print(f"all_functions: {functions}")
    agent = StatelessGeminiAgent(
        delegation_function_prompt=delegation_function_prompt,
        delegates=delegates,
        on_message=on_message,
        functions=functions,
        model_name=model_name,
        system_instruction=system_instruction,
        debug=debug)
    return agent
