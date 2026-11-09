"""Datasets and scorers for checking an agent before deploying it.

An evaluation is a list of :class:`Case` objects plus a scorer. The runner
builds an agent per case (or reuses one), asks the question and records the
answer, the score and the elapsed time.

Example:
    >>> from orchify import Agent, load_dataset, run_dataset
    >>> cases = load_dataset('evals/support.jsonl')
    >>> report = run_dataset(cases, agent_factory=Agent.from_env, scorer='contains')
    >>> print(report.render())

Cases are JSON objects with ``input`` (required), ``expected``, ``name``,
``pattern`` and ``metadata``. One JSON array or one JSON object per line both
work, which keeps small datasets editable by hand.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .utils import get_logger

__all__ = [
    'Case', 'EvalError', 'EvalReport', 'Result', 'SCORERS', 'Scorer',
    'contains', 'exact_match', 'json_match', 'load_dataset', 'regex_scorer',
    'run_dataset', 'save_dataset', 'score',
]

logger = get_logger(__name__)

Scorer = Callable[['Case', str], Union[bool, tuple]]


class EvalError(RuntimeError):
    '''Raised when a dataset cannot be read or written.'''


@dataclass
class Case:
    '''One evaluation question and its expected answer.

    Attributes:
        input (str): Prompt sent to the agent.
        expected (Any): Expected answer, compared by the scorer.
        name (str): Stable label; defaults to ``case-N``.
        pattern (str): Regular expression for the ``regex`` scorer.
        metadata (Dict[str, Any]): Free-form labels (tags, source, notes).
    '''

    input: str
    expected: Any = None
    name: str = ''
    pattern: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.input:
            raise EvalError('case input cannot be empty')
        if not self.name:
            slug = re.sub(r'[^a-z0-9]+', '-', str(self.input).strip().lower())[:24].strip('-')
            self.name = 'case-' + (slug or 'unnamed')

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int = 0) -> 'Case':
        '''Build a case from a JSON object.

        Args:
            data (Dict[str, Any]): ``input``, ``expected``, ``name``, ``pattern``,
                ``metadata``.
            index (int): Position, used for the default name.

        Returns:
            Case: The case.

        Raises:
            EvalError: When the object has no ``input``.
        '''
        if not isinstance(data, dict) or 'input' not in data:
            raise EvalError(f'case #{index + 1} needs an "input" field')
        metadata = data.get('metadata') or {}
        if not isinstance(metadata, dict):
            raise EvalError(f'case #{index + 1} has non-object metadata')
        return cls(
            input=str(data['input']),
            expected=data.get('expected'),
            name=str(data.get('name') or f'case-{index + 1}'),
            pattern=str(data.get('pattern') or ''),
            metadata=dict(metadata),
        )

    def as_dict(self) -> Dict[str, Any]:
        '''Return a JSON-able representation.

        Returns:
            Dict[str, Any]: The case without empty optional fields.
        '''
        data: Dict[str, Any] = {'input': self.input, 'name': self.name}
        if self.expected is not None:
            data['expected'] = self.expected
        if self.pattern:
            data['pattern'] = self.pattern
        if self.metadata:
            data['metadata'] = dict(self.metadata)
        return data


def load_dataset(path: str) -> List[Case]:
    '''Read a dataset from a JSON array or a JSONL file.

    Args:
        path (str): File to read.

    Returns:
        List[Case]: The cases, in file order.

    Raises:
        EvalError: When the file is missing or malformed.
    '''
    if not os.path.exists(path):
        raise EvalError(f'dataset not found: {path}')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            raw = handle.read()
    except OSError as exc:
        raise EvalError(f'cannot read dataset {path}: {exc}') from exc
    stripped = raw.strip()
    if not stripped:
        return []
    if stripped[0] in ('[', '{'):
        try:
            document = json.loads(stripped)
        except ValueError:
            document = None
        if document is not None:
            entries = document if isinstance(document, list) else document.get('cases', [])
            return [Case.from_dict(entry, index) for index, entry in enumerate(entries)]
    cases = []
    for index, line in enumerate(stripped.splitlines()):
        if not line.strip():
            continue
        try:
            cases.append(Case.from_dict(json.loads(line), index))
        except ValueError as exc:
            raise EvalError(f'{path}:{index + 1}: invalid JSON ({exc})') from exc
    return cases


def save_dataset(cases: Iterable[Case], path: str) -> str:
    '''Write cases as a JSON array.

    Args:
        cases (Iterable[Case]): Cases to write.
        path (str): Destination file.

    Returns:
        str: The path written.
    '''
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump([case.as_dict() for case in cases], handle, ensure_ascii=False, indent=2)
    return path


def _expected_text(case: Case) -> str:
    '''Render ``case.expected`` as text for the text scorers.'''
    expected = case.expected
    if isinstance(expected, str):
        return expected
    if expected is None:
        return ''
    return json.dumps(expected, ensure_ascii=False, sort_keys=True)


def exact_match(case: Case, answer: str) -> bool:
    '''Score 1 when the answer equals the expected text, ignoring edge spaces.'''
    return (answer or '').strip() == _expected_text(case).strip()


def contains(case: Case, answer: str) -> bool:
    '''Score 1 when the answer contains the expected text.'''
    expected = _expected_text(case).strip().lower()
    return bool(expected) and expected in (answer or '').lower()


def json_match(case: Case, answer: str) -> bool:
    '''Score 1 when both sides parse as equal JSON values.'''
    expected = case.expected
    if expected is None:
        return False
    try:
        given = json.loads(answer or '')
    except ValueError:
        return False
    return given == expected


def regex_scorer(pattern: str) -> Scorer:
    '''Build a scorer that searches the answer for ``pattern``.

    Args:
        pattern (str): Regular expression searched with :func:`re.search`.

    Returns:
        Scorer: A callable taking ``(case, answer)``.
    '''
    compiled = re.compile(pattern)

    def scorer(case: Case, answer: str) -> bool:
        '''Match one answer against the pattern.'''
        return compiled.search(answer or '') is not None

    return scorer


def score(scorer: Scorer, case: Case, answer: str) -> tuple:
    '''Apply a scorer and normalise its return value.

    Args:
        scorer (Scorer): A scorer callable.
        case (Case): The case being scored.
        answer (str): The agent's answer.

    Returns:
        tuple: ``(passed, detail)``.
    '''
    result = scorer(case, answer)
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1] or '')
    return bool(result), ''


def _case_regex(case: Case, answer: str) -> tuple:
    '''Match the answer against ``case.pattern``.'''
    if not case.pattern:
        return False, 'case has no pattern'
    return re.search(case.pattern, answer or '') is not None, ''


SCORERS: Dict[str, Scorer] = {
    'exact': exact_match,
    'contains': contains,
    'json': json_match,
    'regex': _case_regex,
}


def resolve_scorer(scorer: Union[str, Scorer]) -> Scorer:
    '''Turn a scorer name into a callable.

    Args:
        scorer (Union[str, Scorer]): ``'exact'``, ``'contains'``, ``'json'``,
            ``'regex'`` or a callable.

    Returns:
        Scorer: The callable to use.

    Raises:
        EvalError: When the name is unknown.
    '''
    if callable(scorer):
        return scorer
    name = str(scorer).strip().lower()
    if name not in SCORERS:
        raise EvalError(f"unknown scorer {scorer!r}; use one of {sorted(SCORERS)} or a callable")
    return SCORERS[name]


@dataclass
class Result:
    '''Outcome of one case.

    Attributes:
        case (Case): The case that ran.
        answer (str): What the agent answered.
        passed (bool): Scorer verdict.
        detail (str): Scorer explanation, when it gave one.
        error (str): Exception text when the run failed.
        seconds (float): Wall clock time of the run.
    '''

    case: Case
    answer: str = ''
    passed: bool = False
    detail: str = ''
    error: str = ''
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        '''Return a JSON-able representation.

        Returns:
            Dict[str, Any]: The result plus the case input.
        '''
        return {
            'name': self.case.name,
            'input': self.case.input,
            'expected': self.case.expected,
            'answer': self.answer,
            'passed': self.passed,
            'detail': self.detail,
            'error': self.error,
            'seconds': round(self.seconds, 4),
            'metadata': dict(self.case.metadata),
        }


class EvalReport:
    '''Results of a whole dataset.

    Attributes:
        results (List[Result]): One entry per case, in run order.
        scorer (str): Scorer name used.
    '''

    def __init__(self, results: List[Result], scorer: str = 'contains') -> None:
        '''Wrap the results.

        Args:
            results (List[Result]): Per-case outcomes.
            scorer (str): Label of the scorer used.
        '''
        self.results = list(results)
        self.scorer = scorer

    def __len__(self) -> int:
        '''Number of cases run.'''
        return len(self.results)

    def __iter__(self):
        '''Iterate over the results.'''
        return iter(self.results)

    @property
    def passed(self) -> int:
        '''Cases that scored 1.'''
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        '''Cases that scored 0.'''
        return len(self.results) - self.passed

    @property
    def pass_rate(self) -> float:
        '''Fraction of passing cases, 0.0 for an empty report.'''
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def seconds(self) -> float:
        '''Total wall clock time spent in agents.'''
        return sum(result.seconds for result in self.results)

    def failures(self) -> List[Result]:
        '''Only the cases that did not pass.

        Returns:
            List[Result]: Failing results.
        '''
        return [result for result in self.results if not result.passed]

    def summary(self) -> Dict[str, Any]:
        '''Return aggregate counters.

        Returns:
            Dict[str, Any]: ``total``, ``passed``, ``failed``, ``pass_rate``,
            ``seconds`` and per-name verdicts.
        '''
        return {
            'total': len(self.results),
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': round(self.pass_rate, 4),
            'seconds': round(self.seconds, 3),
            'scorer': self.scorer,
            'cases': {result.case.name: result.passed for result in self.results},
        }

    def render(self, *, limit: int = 50) -> str:
        '''Render a compact table for a terminal.

        Args:
            limit (int): Maximum rows printed.

        Returns:
            str: The table plus a one-line summary.
        '''
        lines = []
        for result in self.results[:limit]:
            mark = 'PASS' if result.passed else 'FAIL'
            line = f'[{mark}] {result.case.name:<28} {result.answer[:60]!r}'
            if result.error:
                line += f'  error={result.error}'
            elif not result.passed and result.detail:
                line += f'  ({result.detail})'
            lines.append(line)
        if len(self.results) > limit:
            lines.append(f'... {len(self.results) - limit} more case(s)')
        lines.append(f'{self.passed}/{len(self.results)} passed '
                     f'({self.pass_rate:.0%}) in {self.seconds:.2f}s')
        return '\n'.join(lines)

    def save(self, path: str) -> str:
        '''Write the full report as JSON.

        Args:
            path (str): Destination file.

        Returns:
            str: The path written.
        '''
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {'summary': self.summary(),
                   'results': [result.as_dict() for result in self.results]}
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path


def run_dataset(cases: Iterable[Case], *, agent: Any = None,
                agent_factory: Optional[Callable[[Case], Any]] = None,
                scorer: Union[str, Scorer] = 'contains',
                max_steps: int = 8,
                reset: bool = True,
                timeout: Optional[float] = None) -> EvalReport:
    '''Run every case through an agent and score the answers.

    Args:
        cases (Iterable[Case]): Cases to run.
        agent (Any): One agent reused for every case.
        agent_factory (Optional[Callable[[Case], Any]]): Builds an agent per case;
            takes precedence over ``agent`` when both are given.
        scorer (Union[str, Scorer]): Scorer name (``'exact'``, ``'contains'``,
            ``'json'``) or callable.
        max_steps (int): Step budget per case.
        reset (bool): Reset a shared agent between cases.
        timeout (Optional[float]): Per-case timeout in seconds.

    Returns:
        EvalReport: Every result plus aggregate counters.

    Raises:
        EvalError: Without an agent or factory, or with an unknown scorer.
    '''
    if agent is None and agent_factory is None:
        raise EvalError('run_dataset needs an agent or an agent_factory')
    case_list = list(cases)
    function = resolve_scorer(scorer)
    label = scorer if isinstance(scorer, str) else getattr(scorer, '__name__', 'scorer')
    results: List[Result] = []
    for case in case_list:
        instance = agent_factory(case) if agent_factory is not None else agent
        if reset and agent_factory is None and hasattr(instance, 'reset'):
            instance.reset()
        started = time.perf_counter()
        answer = ''
        error = ''
        try:
            answer = instance.ask(case.input, max_steps=max_steps, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            error = f'{type(exc).__name__}: {exc}'
        elapsed = time.perf_counter() - started
        passed, detail = (False, '') if error else score(function, case, answer)
        results.append(Result(case=case, answer=str(answer or ''), passed=passed,
                              detail=detail, error=error, seconds=elapsed))
        logger.debug('eval %s: %s (%.2fs)', case.name, 'pass' if passed else 'fail', elapsed)
    return EvalReport(results, scorer=str(label))
