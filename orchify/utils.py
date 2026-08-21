from typing           import Callable, Any, Dict, List, Union
from docstring_parser import parse

import inspect
import os

import random
import string


def analyze_tool_function(func: Callable) -> Dict[str, Any]:
    '''Analyzes a function in detail, combining its signature and docstring parser results.

    Retrieves detailed information for each parameter. Can automatically identify
    multiple docstring styles such as Google, reST, NumPy, etc.

    Args:
        func (Callable): The target function to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'docstring': The combined summary description of the function.
            - 'parameters': A list of parameter dictionaries containing names,
              kinds, defaults, annotations, descriptions, and requirement status.
    '''
    if not callable(func):
        raise TypeError('The provided object is not callable. Please provide a valid function or callable object.')

    original_docstring = inspect.getdoc(func) or 'No docstring provided.'
    parsed_docstring = parse(original_docstring)

    param_descriptions = {
        param.arg_name: param.description for param in parsed_docstring.params
    }

    summary_parts = []
    if parsed_docstring.short_description:
        summary_parts.append(parsed_docstring.short_description)
    if parsed_docstring.long_description:
        summary_parts.append(parsed_docstring.long_description)
    summary = '\n\n'.join(summary_parts) if summary_parts else ''

    parameters: List[Dict[str, Any]] = []
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        signature = None

    if signature is not None:
        for name, param in signature.parameters.items():
            parameters.append({
                'name': name,
                'kind': str(param.kind.description),  # e.g., 'positional or keyword'
                'default': param.default if param.default is not inspect.Parameter.empty else 'N/A',
                'annotation': param.annotation.__name__ if hasattr(param.annotation, '__name__') \
                              and param.annotation is not inspect.Parameter.empty else 'N/A',
                'description': param_descriptions.get(name, 'No description found in docstring.').replace('\n', ' '),
                'required': param.default is inspect.Parameter.empty,
            })

    return {
        'docstring': summary,
        'parameters': parameters
    }


def safecode(length: int = 4) -> str:
    '''Generates a random safe code consisting of letters and digits.

    Args:
        length (int): The length of the code to generate. Defaults to 4.

    Returns:
        str: The generated random code.
    '''
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def req_file(path: str, mode: str = 'r', encoding: str = 'utf-8') -> Union[str, bytes]:
    '''Reads and returns the content of a file.

    Args:
        path (str): The path to the file.
        mode (str): The mode in which the file is opened (e.g. 'r', 'rb'). Defaults to 'r'.
        encoding (str): The encoding used to decode the file when reading in text mode. Defaults to 'utf-8'.

    Returns:
        Union[str, bytes]: The content of the file. Returns an empty string or empty bytes if the file does not exist.
    '''
    if not os.path.isfile(path): return b'' if 'b' in mode else ''
    if 'b' in mode:
        with open(path, mode) as f:
            return f.read()
    else:
        with open(path, mode, encoding=encoding) as f:
            return f.read()
