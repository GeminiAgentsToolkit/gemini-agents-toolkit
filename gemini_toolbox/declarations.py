import inspect
from vertexai.generative_models import FunctionDeclaration, Tool


def generate_function_declaration(func):
    # Extract function name
    func_name = func.__name__
    
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


def generate_tool_from_functions(functions):
    function_declarations = [generate_function_declaration(func) for func in functions]
    tool = Tool(function_declarations=function_declarations)
    return tool
