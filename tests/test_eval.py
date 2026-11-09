"""Evaluation datasets, scorers and the runner."""

import json
import os
import uuid

import pytest

from orchify import Agent
from orchify.cli import EXIT_FAILED, EXIT_OK, _eval
from orchify.eval import (Case, EvalError, EvalReport, Result, contains,
                          exact_match, json_match, load_dataset, regex_scorer,
                          run_dataset, save_dataset)
from orchify.testing import FakeLLM, final


def _directory():
    '''A writable scratch directory inside the test tree.'''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data',
                        'eval-%s' % uuid.uuid4().hex[:8])
    os.makedirs(path, exist_ok=True)
    return path


def _drop(directory):
    '''Remove a scratch directory, ignoring a locked leftover.'''
    import shutil
    shutil.rmtree(directory, ignore_errors=True)


def _agent(answer):
    '''An agent that answers with a fixed string.'''
    return Agent(name='eval', llm=FakeLLM([[final(answer)]]))


def test_case_defaults_and_round_trip():
    '''A case derives a readable name and serialises without empty fields.'''
    case = Case(input='What is 2+2?', expected='4')
    assert case.name == 'case-what-is-2-2'
    data = case.as_dict()
    assert data['input'] == 'What is 2+2?' and 'pattern' not in data
    again = Case.from_dict(data)
    assert again.expected == '4'
    with pytest.raises(EvalError, match='input'):
        Case.from_dict({'name': 'broken'})


def test_load_dataset_reads_arrays_and_jsonl():
    '''Both file shapes map to the same cases.'''
    directory = _directory()
    try:
        array_path = os.path.join(directory, 'array.json')
        with open(array_path, 'w', encoding='utf-8') as handle:
            json.dump([{'input': 'a', 'expected': '1'}, {'input': 'b', 'pattern': 'x+'}], handle)
        cases = load_dataset(array_path)
        assert [case.input for case in cases] == ['a', 'b']
        assert cases[1].pattern == 'x+'

        jsonl_path = os.path.join(directory, 'cases.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as handle:
            handle.write('{"input": "c", "expected": "3"}\n\n{"input": "d"}\n')
        assert [case.input for case in load_dataset(jsonl_path)] == ['c', 'd']

        saved = save_dataset(cases, os.path.join(directory, 'saved.json'))
        assert len(load_dataset(saved)) == 2
    finally:
        _drop(directory)


def test_load_dataset_reports_missing_and_broken_files():
    '''A missing file and a malformed line both raise EvalError.'''
    directory = _directory()
    try:
        with pytest.raises(EvalError, match='not found'):
            load_dataset(os.path.join(directory, 'nope.json'))
        broken = os.path.join(directory, 'broken.jsonl')
        with open(broken, 'w', encoding='utf-8') as handle:
            handle.write('{"input": "a"}\nnot json\n')
        with pytest.raises(EvalError, match='invalid JSON'):
            load_dataset(broken)
    finally:
        _drop(directory)


def test_scorers_cover_exact_contains_json_and_regex():
    '''Each built-in scorer accepts the answers it is meant to accept.'''
    numbers = Case(input='q', expected='four')
    assert exact_match(numbers, ' four ') is True
    assert exact_match(numbers, 'five') is False
    assert contains(numbers, 'the answer is FOUR') is True
    assert contains(Case(input='q'), 'anything') is False

    payload = Case(input='q', expected={'a': 1})
    assert json_match(payload, '{"a": 1}') is True
    assert json_match(payload, 'not json') is False

    digits = Case(input='q', pattern=r'^\d{3}$')
    from orchify.eval import _case_regex
    assert _case_regex(digits, '123') == (True, '')
    assert _case_regex(digits, '12') == (False, '')
    assert _case_regex(Case(input='q'), '123') == (False, 'case has no pattern')
    assert regex_scorer('ok')(Case(input='q'), 'looks ok') is True


def test_run_dataset_scores_and_summarises():
    '''Passing, failing and crashing cases are all reported.'''

    def factory(case):
        if case.input == 'boom':
            return _agent('never used')
        return _agent('the answer is ' + str(case.expected))

    class Exploding(Agent):
        '''Fails the run on purpose.'''

        def ask(self, *args, **kwargs):
            raise RuntimeError('provider is down')

    def crashing_factory(case):
        if case.input == 'boom':
            return Exploding(name='boom', llm=FakeLLM([[final('x')]]))
        return factory(case)

    cases = [Case(input='q1', expected='one'), Case(input='q2', expected='two'),
             Case(input='boom')]
    report = run_dataset(cases, agent_factory=crashing_factory, scorer='contains')
    assert len(report) == 3
    assert report.passed == 2 and report.failed == 1
    assert report.pass_rate == pytest.approx(2 / 3)
    assert 'provider is down' in report.failures()[0].error
    summary = report.summary()
    assert summary['total'] == 3
    assert summary['cases'] == {'case-q1': True, 'case-q2': True, 'case-boom': False}
    rendered = report.render()
    assert '2/3 passed' in rendered and '[FAIL]' in rendered


def test_run_dataset_validates_its_inputs():
    '''Missing agents and unknown scorers are rejected up front.'''
    with pytest.raises(EvalError, match='agent'):
        run_dataset([Case(input='q')])
    with pytest.raises(EvalError, match='unknown scorer'):
        run_dataset([Case(input='q')], agent=_agent('x'), scorer='nope')


def test_report_saves_every_case():
    '''The JSON report keeps the summary and per-case rows.'''
    directory = _directory()
    try:
        report = EvalReport([Result(case=Case(input='q', expected='4'),
                                    answer='4', passed=True, seconds=0.5)],
                            scorer='exact')
        path = report.save(os.path.join(directory, 'report.json'))
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        assert payload['summary']['pass_rate'] == 1.0
        assert payload['results'][0]['answer'] == '4'
    finally:
        _drop(directory)


def test_cli_eval_runs_a_dataset_and_reports_the_exit_code():
    '''`orchify eval` prints a table and fails when a case fails.'''
    directory = _directory()
    try:
        path = os.path.join(directory, 'cases.json')
        save_dataset([Case(input='q1', expected='one')], path)
        assert _eval(data=path, agent=_agent('the answer is one')) == EXIT_OK
        assert _eval(data=path, agent=_agent('wrong'), scorer='exact') == EXIT_FAILED
        assert _eval(data=os.path.join(directory, 'missing.json')) == EXIT_ERROR_CODE
    finally:
        _drop(directory)


EXIT_ERROR_CODE = 1
