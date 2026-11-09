'''JSON-Schema structured output with a parse-failure retry.

Ask a model for a JSON object that matches a schema, parse it, and when the
reply does not validate, tell the model what was wrong and ask again::

    from orchify import Agent, ask_structured

    schema = {
        'type': 'object',
        'properties': {'title': {'type': 'string'}, 'tags': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['title', 'tags'],
    }
    data = ask_structured(agent, 'Summarize this changelog as JSON.', schema)

The validator is deliberately small (``type``, ``properties``, ``required``,
``items``, ``enum``, ``const``, ``minimum``/``maximum``, ``pattern``,
``minLength``/``maxLength``, ``anyOf``/``oneOf``) and has no third-party
dependency. Use a full validator downstream when the schema is complex.
'''

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_FENCE = re.compile(r'```(?:json|JSON)?\s*(.*?)```', re.DOTALL)


class StructuredOutputError(RuntimeError):
    '''Raised when the model never returned data matching the schema.'''

    def __init__(self, message: str, attempts: int = 0, last_error: str = '',
                 last_text: str = ''):
        '''Create the error.

        Args:
            message (str): Human-readable summary.
            attempts (int): How many replies were tried.
            last_error (str): Validation error of the final reply.
            last_text (str): The final reply text.
        '''
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.last_text = last_text


def schema_instruction(schema: Dict[str, Any], *, name: str = 'result') -> str:
    '''Build the system instruction that asks for JSON matching ``schema``.

    Args:
        schema (Dict[str, Any]): JSON Schema of the expected object.
        name (str): What the object represents, used in the wording.

    Returns:
        str: Instruction text for the system prompt.
    '''
    return (
        'Reply with a single JSON %s that matches this JSON Schema, and nothing '
        'else: no prose, no markdown fence.\n\n%s'
        % (name, json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True))
    )


def extract_json(text: str) -> Any:
    '''Pull the first JSON value out of a model reply.

    Args:
        text (str): Raw model output, possibly wrapped in a markdown fence.

    Returns:
        Any: The parsed value.

    Raises:
        ValueError: When no JSON value can be found or parsed.
    '''
    if not isinstance(text, str):
        raise ValueError('structured output expects text, got %s' % type(text).__name__)
    body = text.strip()
    if not body:
        raise ValueError('the reply was empty')
    fence = _FENCE.search(body)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except ValueError:
        pass
    # Tolerate prose around the object: try every balanced-looking window.
    for opener, closer in (('{', '}'), ('[', ']')):
        start = body.find(opener)
        end = body.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(body[start:end + 1])
            except ValueError:
                continue
    raise ValueError('the reply contains no parseable JSON')


def _type_ok(value: Any, expected: str) -> bool:
    '''Check one JSON Schema ``type`` against a Python value.'''
    if expected == 'object':
        return isinstance(value, dict)
    if expected == 'array':
        return isinstance(value, list)
    if expected == 'string':
        return isinstance(value, str)
    if expected == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == 'boolean':
        return isinstance(value, bool)
    if expected == 'null':
        return value is None
    return True


def validate(value: Any, schema: Dict[str, Any], path: str = '') -> List[str]:
    '''Validate ``value`` against the supported subset of JSON Schema.

    Args:
        value (Any): Parsed JSON value.
        schema (Dict[str, Any]): Schema to validate against.
        path (str): Dotted path used in error messages.

    Returns:
        List[str]: Validation errors; empty when the value matches.
    '''
    errors: List[str] = []
    if not isinstance(schema, dict) or not schema:
        return errors
    where = path or 'value'

    if 'const' in schema and value != schema['const']:
        errors.append('%s must equal %r' % (where, schema['const']))
    if 'enum' in schema and value not in (schema['enum'] or []):
        errors.append('%s must be one of %r' % (where, schema['enum']))
    expected = schema.get('type')
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, item) for item in types):
            errors.append('%s must be of type %s' % (where, '/'.join(map(str, types))))
            return errors

    for key in ('anyOf', 'oneOf'):
        options = schema.get(key)
        if not options:
            continue
        matched = [option for option in options if not validate(value, option, path)]
        if key == 'anyOf' and not matched:
            errors.append('%s matches none of the anyOf schemas' % where)
        if key == 'oneOf' and len(matched) != 1:
            errors.append('%s must match exactly one oneOf schema, matched %d'
                          % (where, len(matched)))

    if isinstance(value, dict):
        properties = schema.get('properties') or {}
        for required in schema.get('required') or []:
            if required not in value:
                errors.append('%s is missing required property %r' % (where, required))
        if schema.get('additionalProperties') is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append('%s has unexpected propert%s %s'
                              % (where, 'y' if len(extra) == 1 else 'ies',
                                 ', '.join(map(repr, extra))))
        for key, child in properties.items():
            if key in value:
                errors.extend(validate(value[key], child,
                                       (path + '.' + key) if path else key))

    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        for index, item in enumerate(value):
            errors.extend(validate(item, schema['items'], '%s[%d]' % (where, index)))

    if isinstance(value, str):
        pattern = schema.get('pattern')
        if pattern and re.search(pattern, value) is None:
            errors.append('%s does not match pattern %r' % (where, pattern))
        if 'minLength' in schema and len(value) < int(schema['minLength']):
            errors.append('%s is shorter than %d characters' % (where, schema['minLength']))
        if 'maxLength' in schema and len(value) > int(schema['maxLength']):
            errors.append('%s is longer than %d characters' % (where, schema['maxLength']))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            errors.append('%s must be at least %r' % (where, schema['minimum']))
        if 'maximum' in schema and value > schema['maximum']:
            errors.append('%s must be at most %r' % (where, schema['maximum']))

    return errors


def parse_structured(text: str, schema: Dict[str, Any]) -> Tuple[Any, List[str]]:
    '''Parse and validate one reply.

    Args:
        text (str): Raw model reply.
        schema (Dict[str, Any]): Expected JSON Schema.

    Returns:
        Tuple[Any, List[str]]: ``(value, errors)``; ``value`` is None when the
        reply could not be parsed at all.
    '''
    try:
        value = extract_json(text)
    except ValueError as exc:
        return None, [str(exc)]
    return value, validate(value, schema)


def ask_structured(
    agent: Any,
    prompt: str,
    schema: Dict[str, Any],
    *,
    retries: int = 2,
    name: str = 'result',
    max_steps: Optional[int] = None,
    timeout: Optional[float] = None,
    key: Optional[str] = None,
) -> Any:
    '''Ask ``agent`` for JSON matching ``schema``, retrying on invalid replies.

    Args:
        agent (Any): Agent (or anything with ``ask``/``messages``).
        prompt (str): The user request.
        schema (Dict[str, Any]): JSON Schema the answer must satisfy.
        retries (int): Extra attempts after the first reply.
        name (str): What the JSON represents, for the instruction text.
        max_steps (Optional[int]): Step budget for each attempt.
        timeout (Optional[float]): Timeout per attempt.
        key (Optional[str]): When set, return ``value[key]`` instead of the value.

    Returns:
        Any: The validated value (or ``value[key]``).

    Raises:
        StructuredOutputError: When no attempt produced a valid value.
    '''
    instruction = schema_instruction(schema, name=name)
    ask = getattr(agent, 'ask')
    last_error = ''
    last_text = ''
    attempt = 0
    for attempt in range(1, max(0, int(retries)) + 2):
        if attempt == 1:
            message = '\n\n%s' % instruction
        else:
            message = (
                'That reply did not match the schema.\n\nProblems:\n- %s\n\n'
                'Return ONLY corrected JSON that matches the schema below.\n\n%s'
                % (last_error, instruction)
            )
        kwargs: Dict[str, Any] = {}
        if max_steps is not None:
            kwargs['max_steps'] = max_steps
        if timeout is not None:
            kwargs['timeout'] = timeout
        text = ask(prompt + message if attempt == 1 else message, **kwargs)
        last_text = str(text or '')
        value, errors = parse_structured(last_text, schema)
        if not errors:
            return value[key] if key is not None and isinstance(value, dict) else value
        last_error = '; '.join(errors[:6])
        _remember(agent, prompt, message, last_text, last_error)
    raise StructuredOutputError(
        'no valid JSON after %d attempt(s): %s' % (attempt, last_error),
        attempts=attempt,
        last_error=last_error,
        last_text=last_text,
    )


def _remember(agent: Any, prompt: str, message: str, text: str, error: str) -> None:
    '''Record a failed attempt on the agent, when it accepts attributes.'''
    failures = getattr(agent, 'structured_failures', None)
    if failures is None:
        failures = []
        try:
            setattr(agent, 'structured_failures', failures)
        except Exception:  # noqa: BLE001 - e.g. a slotted stand-in
            return
    failures.append({'prompt': prompt[:200], 'message': message[-400:],
                     'reply': text[:400], 'error': error})
    del failures[:-10]
