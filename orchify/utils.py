from typing           import Callable, Any, Dict, List, Optional, Union
from docstring_parser import parse

import inspect
import logging
import os

import random
import string
import sys


def analyze_tool_function(func: Callable) -> Dict[str, Any]:
    '''Analyze a callable and merge its signature with its parsed docstring.

    Args:
        func (Callable): The function to inspect. Google, reST and NumPy style
            docstrings are understood.

    Returns:
        Dict[str, Any]: ``{'docstring': str, 'parameters': List[dict]}`` where
        every parameter dictionary carries the name, kind, default, annotation,
        description and required flag of one argument.
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
                'kind': str(param.kind.description),
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
    '''Generate a short random code of letters and digits.

    Args:
        length (int): Number of characters to generate. Defaults to 4.

    Returns:
        str: The generated code, usable as a name suffix.
    '''
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def req_file(path: str, mode: str = 'r', encoding: str = 'utf-8') -> Union[str, bytes]:
    '''Read a file and return its content.

    Args:
        path (str): File to read.
        mode (str): Open mode, e.g. ``'r'`` or ``'rb'``. Defaults to ``'r'``.
        encoding (str): Text encoding used for text modes. Defaults to
            ``'utf-8'``.

    Returns:
        Union[str, bytes]: The file content, or an empty string/bytes when the
        file does not exist or cannot be read.
    '''
    if not os.path.isfile(path): return b'' if 'b' in mode else ''
    if 'b' in mode:
        with open(path, mode) as f:
            return f.read()
    else:
        with open(path, mode, encoding=encoding) as f:
            return f.read()


_LOGGER_NAME = 'orchify'
_HANDLER_MARK = '_orchify_echo_handler'

_logger = logging.getLogger(_LOGGER_NAME)
_logger.addHandler(logging.NullHandler())
_echo_state: Optional[bool] = None


def _read_log_echo(default: bool = True) -> bool:
    '''ORCHIFY_LOG_ECHO=0 (false/no/off/none) silences the default stdout echo.'''
    raw = os.getenv('ORCHIFY_LOG_ECHO')
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() not in ('0', 'false', 'no', 'off', 'none')


def _resolve_level(level: Optional[Union[int, str]]) -> int:
    raw = level if level is not None else os.getenv('ORCHIFY_LOG_LEVEL', 'INFO')
    if isinstance(raw, str):
        resolved = getattr(logging, raw.strip().upper(), None)
        return resolved if isinstance(resolved, int) else logging.INFO
    try:
        return int(raw)
    except (TypeError, ValueError):
        return logging.INFO


def get_logger(name: Optional[str] = None) -> logging.Logger:
    '''Return the orchify logger, or a child of it.

    Called as ``get_logger(__name__)`` it returns children of
    ``logging.getLogger('orchify')``, so applications can filter per module.

    The library never prints: with the default ``ORCHIFY_LOG_ECHO=1`` the
    records the library used to print are echoed to stdout (exactly once,
    propagation is disabled while the echo is on). ``ORCHIFY_LOG_ECHO=0`` hands
    the records to the application's own logging configuration instead.

    Args:
        name (Optional[str]): Module name; omitted for the root orchify logger.

    Returns:
        logging.Logger: The resolved logger.
    '''
    if _echo_state is None:
        _apply_echo(_read_log_echo())
        _logger.setLevel(_resolve_level(None))
    if not name:
        return _logger
    short = name.split('.', 1)[1] if name.startswith(_LOGGER_NAME + '.') else name
    return _logger.getChild(short)


def _apply_echo(echo: bool) -> None:
    global _echo_state
    _echo_state = bool(echo)
    for handler in list(_logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            _logger.removeHandler(handler)
            handler.close()
    if not echo:
        _logger.propagate = True
        return
    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARK, True)
    handler.setFormatter(logging.Formatter('%(message)s'))
    _logger.addHandler(handler)
    _logger.propagate = False


def configure_logging(level: Optional[Union[int, str]] = None,
                      echo: Optional[bool] = None) -> logging.Logger:
    '''Configure orchify's logger and return it.

    Safe to call repeatedly, and to call after importing orchify.

    Args:
        level (Optional[Union[int, str]]): Level name or number; defaults to
            ``ORCHIFY_LOG_LEVEL`` (INFO).
        echo (Optional[bool]): True echoes records to stdout, False leaves them
            to the application's logging setup; defaults to ``ORCHIFY_LOG_ECHO``
            (1).

    Returns:
        logging.Logger: The configured orchify logger.
    '''
    _apply_echo(_read_log_echo() if echo is None else bool(echo))
    _logger.setLevel(_resolve_level(level))
    return _logger
