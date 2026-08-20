from typing import Callable, Any, Dict, Optional, List
from .utils import analyze_tool_function 


class Tool:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        is_agent_tool: bool = False,
        is_group_tool: bool = False
    ):
        self.func = func
        
        analysis = analyze_tool_function(func)
        
        self.name: str = name or func.__name__
        self.description: str = description or analysis.get('docstring', 'No description provided.')
        self.parameters: List[Dict[str, Any]] = parameters or analysis.get('parameters', [])
        self.is_agent_tool = is_agent_tool
        self.is_group_tool = is_group_tool
        self._info = self.build_info()
    
    @property
    def info(self) -> Dict[str, Any]:
        return self._info
    
    def build_info(self) -> Dict[str, Any]:
        '''
        Generates a tool description dictionary compliant with the OpenAI Function Calling specification.
        
        Returns:
            A dictionary that can be directly serialized to JSON and sent to the LLM API.
        '''
        json_schema_properties = {}
        required_params = []
        py_to_json_type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object'
        }
        for param in self.parameters:
            param_name = param['name']
            
            param_type = py_to_json_type_map.get(param.get('annotation', 'str'), 'string')
            
            json_schema_properties[param_name] = {
                'type': param_type,
                'description': param.get('description', '')
            }
            
            if param.get('required', False):
                required_params.append(param_name)
        
        if not self.description.startswith('A tool:') and \
           not self.description.startswith('An Agent:') and \
           not self.description.startswith('A Group:'):
            self.description = f'A tool:\n{self.description}'
                
        tool_info = {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': json_schema_properties,
                }
            }
        }
        
        if required_params:
            tool_info['function']['parameters']['required'] = required_params
            
        return tool_info
    
    def __call__(self, **kwargs):
        '''Allows the tool instance to be called like a function.'''
        return self.execute(**kwargs)
        
    def execute(self, **kwargs: Any) -> Any:
        '''
        Executes the core logic of the tool.

        Args:
            **kwargs: Parameters passed from the Agent, where keys are parameter names and values are their values.

        Returns:
            The execution result of the tool's function.
        '''
        return self.func(**kwargs)

    def __repr__(self) -> str:
        '''Provides a string representation of the Tool instance.'''
        return f"Tool(name='{self.name}')"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
    is_agent_tool: bool = False,
    is_group_tool: bool = False
):
    def decorator(func: Callable) -> Tool:
        return Tool(
            func=func,
            name=name,
            description=description,
            parameters=parameters,
            is_agent_tool=is_agent_tool,
            is_group_tool=is_group_tool
        )
    return decorator