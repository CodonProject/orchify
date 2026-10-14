import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify.event import AgentEvent, ToolEventAssembler
from orchify.llm.base import Chunk, FinalStatus, Response


def _call(call_id='', name='', arguments=''):
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def _assembly(tool_calls, chunk_arg=''):
    return Response(
        current_chunk=Chunk(is_cot=False, content='', is_assembly_tool=True,
                            assembly_chunk=chunk_arg, total_tool_call=list(tool_calls)),
        final_status=None,
        is_final=False,
    )


def _build(assembler, response):
    return assembler.build(response, agent_name='a', agent_code='c', turn_id='t', turn=2)


def _types(events):
    return [event.event_type for event in events]


def test_ignores_final_and_plain_chunks():
    '''Only assembly chunks produce events.'''
    assembler = ToolEventAssembler()
    plain = Response(current_chunk=Chunk(is_cot=False, content='hi'), final_status=None, is_final=False)
    assert _build(assembler, plain) == []

    final = Response(current_chunk=None,
                     final_status=FinalStatus(content='done', reasoning=''), is_final=True)
    assert _build(assembler, final) == []

    empty = Response(current_chunk=None, final_status=None, is_final=False)
    assert _build(assembler, empty) == []

    not_assembly = Response(current_chunk=Chunk(is_cot=False, content='', is_assembly_tool=False),
                            final_status=None, is_final=False)
    assert _build(assembler, not_assembly) == []


def test_assembly_start_step_and_finish():
    '''Split arguments yield start, incremental steps, then one finish with parsed args.'''
    assembler = ToolEventAssembler()
    first = _build(assembler, _assembly([_call('call_1', 'get_weather', '{"ci')], chunk_arg='{"ci'))
    assert _types(first) == ['tool:assembly:start', 'tool:assembly:step']
    assert first[0].tool_id == 'call_1' and first[0].tool_name == 'get_weather'
    assert first[0].args is None
    assert first[1].chunk_arg == '{"ci' and first[1].turn == 2

    second = _build(assembler, _assembly([_call('call_1', 'get_weather', '{"city": "Tokyo"}')],
                                         chunk_arg='ty": "Tokyo"}'))
    assert _types(second) == ['tool:assembly:step', 'tool:assembly:finish']
    assert second[0].chunk_arg == 'ty": "Tokyo"}'
    assert second[1].tool_id == 'call_1' and second[1].args == {'city': 'Tokyo'}

    third = _build(assembler, _assembly([_call('call_1', 'get_weather', '{"city": "Tokyo"} extra')],
                                        chunk_arg=' extra'))
    assert third == []


def test_one_finish_per_call_across_chunks():
    '''A call that already parsed is never reported again, even if it grows.'''
    assembler = ToolEventAssembler()
    complete = _build(assembler, _assembly([_call('c', 'now', '{}')]))
    assert _types(complete) == ['tool:assembly:start', 'tool:assembly:step', 'tool:assembly:finish']
    assert complete[2].args == {}

    again = _build(assembler, _assembly([_call('c', 'now', '{}')]))
    assert again == []


def test_multiple_calls_assemble_independently():
    '''Each interleaved tool call keeps its own start/step/finish state.'''
    assembler = ToolEventAssembler()
    first = _build(assembler, _assembly([_call('a', 'one', '{"x"'),
                                         _call('b', 'two', '{"y"')]))
    assert _types(first) == ['tool:assembly:start', 'tool:assembly:step',
                             'tool:assembly:start', 'tool:assembly:step']
    assert [event.tool_id for event in first] == ['a', 'a', 'b', 'b']

    second = _build(assembler, _assembly([_call('a', 'one', '{"x": 1}'),
                                          _call('b', 'two', '{"y": 2}')]))
    assert _types(second) == ['tool:assembly:step', 'tool:assembly:finish',
                              'tool:assembly:step', 'tool:assembly:finish']
    finished = {event.tool_id: event.args for event in second if event.event_type == 'tool:assembly:finish'}
    assert finished == {'a': {'x': 1}, 'b': {'y': 2}}


def test_call_without_id_is_tracked_by_index():
    '''A provider that omits the call id still gets one start and sequential steps.'''
    assembler = ToolEventAssembler()
    first = _build(assembler, _assembly([_call('', 'lookup', '{"q"')]))
    assert _types(first) == ['tool:assembly:start', 'tool:assembly:step']
    assert first[0].tool_id == ''

    second = _build(assembler, _assembly([_call('', 'lookup', '{"q": 3}')]))
    assert _types(second) == ['tool:assembly:step', 'tool:assembly:finish']
    assert second[0].chunk_arg == ': 3}'
    assert second[1].args == {'q': 3}


def test_unparsable_and_empty_arguments_never_finish():
    '''Partial JSON keeps streaming; blank arguments produce no step.'''
    assembler = ToolEventAssembler()
    first = _build(assembler, _assembly([_call('c', 'slow', 'not json')]))
    assert _types(first) == ['tool:assembly:start', 'tool:assembly:step']

    second = _build(assembler, _assembly([_call('c', 'slow', 'not json yet')]))
    assert _types(second) == ['tool:assembly:step']

    unchanged = _build(assembler, _assembly([_call('c', 'slow', 'not json yet')]))
    assert unchanged == []

    blank_after_start = _build(assembler, _assembly([_call('d', 'empty', '')]))
    assert _types(blank_after_start) == ['tool:assembly:start']


def test_build_from_resp_rejects_an_empty_response():
    '''A response with neither a final status nor a narratable chunk is rejected.'''
    response = Response(current_chunk=Chunk(is_cot=False, content='', is_assembly_tool=True))
    try:
        AgentEvent.build_from_resp(response, 'a', 'code', 't1')
    except ValueError as exc:
        assert 'current_chunk' in str(exc)
    else:
        raise AssertionError('build_from_resp should reject an empty response')


def main():
    '''Run every test in this module and report a summary.'''
    tests = [
        test_ignores_final_and_plain_chunks,
        test_assembly_start_step_and_finish,
        test_one_finish_per_call_across_chunks,
        test_multiple_calls_assemble_independently,
        test_call_without_id_is_tracked_by_index,
        test_unparsable_and_empty_arguments_never_finish,
        test_build_from_resp_rejects_an_empty_response,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            failures.append(fn.__name__)
            print(f'FAIL {fn.__name__}: {exc}')
            traceback.print_exc()
    print()
    if failures:
        print(f'{len(failures)} failed: {failures}')
        return 1
    print(f'all {len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
