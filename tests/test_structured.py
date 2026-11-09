"""Structured output: JSON extraction, validation and the retry loop."""

import json

import pytest

from orchify import Agent
from orchify.structured import (StructuredOutputError, ask_structured, extract_json,
                                parse_structured, schema_instruction, validate)
from orchify.testing import FakeLLM, final

_SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string', 'minLength': 2},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'score': {'type': 'integer', 'minimum': 0, 'maximum': 10},
    },
    'required': ['title', 'tags'],
    'additionalProperties': False,
}


def _agent(*texts):
    '''Agent whose replies are the given texts, one per request.'''
    return Agent(name='structured', llm=FakeLLM([[final(text)] for text in texts]))


def test_extract_json_handles_plain_fenced_and_wrapped_replies():
    '''Every common shape of model reply parses.'''
    assert extract_json('{"a": 1}') == {'a': 1}
    assert extract_json('```json\n{"a": 2}\n```') == {'a': 2}
    assert extract_json('Sure, here it is:\n{"a": 3}\nThanks!') == {'a': 3}
    assert extract_json('[1, 2, 3]') == [1, 2, 3]


def test_extract_json_rejects_empty_and_text_only_replies():
    '''No JSON at all is an error the caller can report.'''
    for text in ('', '   ', 'I cannot help with that'):
        try:
            extract_json(text)
        except ValueError as exc:
            assert 'JSON' in str(exc) or 'empty' in str(exc)
        else:  # pragma: no cover - every case must raise
            raise AssertionError('extract_json(%r) should have raised' % text)


def test_validate_reports_every_supported_problem():
    '''The validator names the offending paths.'''
    assert validate({'title': 'ok', 'tags': ['a']}, _SCHEMA) == []

    errors = validate({'tags': [], 'score': 11, 'extra': True}, _SCHEMA)
    assert any("missing required property 'title'" in item for item in errors)
    assert any('score must be at most 10' in item for item in errors)
    assert any('unexpected property' in item for item in errors)

    errors = validate({'title': 'x', 'tags': ['a', 3]}, _SCHEMA)
    assert any('tags[1] must be of type string' in item for item in errors)
    assert any('title is shorter than 2' in item for item in errors)


def test_validate_supports_enum_anyof_and_pattern():
    '''Small combinators work without a third-party validator.'''
    schema = {'anyOf': [{'type': 'string', 'pattern': '^[a-z]+$'},
                        {'type': 'integer', 'const': 7}]}
    assert validate('abc', schema) == []
    assert validate(7, schema) == []
    assert validate('ABC', schema)
    assert validate(8, schema)


def test_parse_structured_returns_errors_instead_of_raising():
    '''Parse failures come back as a message, not an exception.'''
    value, errors = parse_structured('not json', _SCHEMA)
    assert value is None and errors
    value, errors = parse_structured(json.dumps({'title': 'ok', 'tags': []}), _SCHEMA)
    assert value == {'title': 'ok', 'tags': []} and errors == []


def test_ask_structured_retries_with_the_validation_error():
    '''A bad first reply is corrected on the second request.'''
    agent = _agent('{"title": 3}', json.dumps({'title': 'good', 'tags': ['a']}))
    value = ask_structured(agent, 'describe it', _SCHEMA)

    assert value == {'title': 'good', 'tags': ['a']}
    assert len(agent.messages) >= 3
    sent = [m['content'] for m in agent.llm.calls[-1]['messages']]
    assert any('did not match the schema' in str(text) for text in sent)
    assert any('title must be of type string' in str(text) for text in sent)


def test_ask_structured_gives_up_after_the_retries():
    '''A model that never complies raises with diagnostics.'''
    agent = _agent('nope', 'still nope', '{"title": "ok"}')
    with pytest.raises(StructuredOutputError) as caught:
        ask_structured(agent, 'describe it', _SCHEMA, retries=2)

    error = caught.value
    assert error.attempts == 3
    assert error.last_text == '{"title": "ok"}'
    assert 'missing required property' in error.last_error
    assert agent.structured_failures[-1]['reply'] == '{"title": "ok"}'


def test_ask_structured_can_return_one_key():
    '''``key=`` unwraps a single field.'''
    agent = _agent(json.dumps({'title': 'only', 'tags': []}))
    assert ask_structured(agent, 'describe it', _SCHEMA, key='title') == 'only'


def test_schema_instruction_embeds_the_schema():
    '''The instruction carries the schema verbatim.'''
    instruction = schema_instruction(_SCHEMA, name='article')
    assert 'JSON article' in instruction
    assert '"required"' in instruction
    assert '"title"' in instruction
