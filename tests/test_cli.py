import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchify.cli import EXIT_MISSING_KEY, EXIT_OK, _masked_key
from orchify.cli import main as cli_main

KEY_NAMES = ('API_KEY', 'OPENAI_API_KEY')


def _capture(argv):
    '''Run the CLI while capturing its stdout.

    Args:
        argv (list): Command line arguments.

    Returns:
        tuple: The exit code and the captured text.
    '''
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli_main(list(argv))
    return code, buffer.getvalue()


@contextlib.contextmanager
def _without_keys():
    '''Temporarily remove every credential from the environment.

    Yields:
        None: While the keys are unset.
    '''
    saved = {name: os.environ.pop(name, None) for name in KEY_NAMES}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def test_doctor_reports_ok():
    '''The environment doctor runs and prints a verdict line.'''
    code, text = _capture(['doctor'])
    assert code == EXIT_OK, text
    assert 'web backend' in text
    assert 'verdict: OK' in text


def test_doctor_json_is_machine_readable():
    '''The JSON doctor report carries the verdict and every check.'''
    code, text = _capture(['doctor', '--json'])
    payload = json.loads(text)
    assert payload['ok'] is True
    assert code == EXIT_OK
    assert payload['failed'] == 0
    assert payload['passed'] >= 1
    for check in payload['checks']:
        assert set(check) == {'status', 'label', 'detail'}


def test_version_and_help():
    '''The version subcommand and the bare invocation both work.'''
    code, text = _capture(['version'])
    assert code == EXIT_OK
    assert text.startswith('orchify ')
    code, text = _capture([])
    assert code == EXIT_OK
    assert 'usage: orchify' in text


def test_offline_demo_runs_every_stage():
    '''The offline demo answers, calls a tool and clears an approval.'''
    code, text = _capture(['demo'])
    assert code == EXIT_OK, text
    assert '[1] scripted answer: Hello from the offline demo.' in text
    assert '[2] tool round trip: 2 + 3 = 5' in text
    assert '[3] approval requested: deploy' in text
    assert '[3] approved answer: deployed to prod' in text
    assert '== offline demo done ==' in text


def test_live_demo_without_credentials_is_reported():
    '''The live demo refuses to run without an API key.'''
    with _without_keys():
        code, text = _capture(['demo', '--live'])
    assert code == EXIT_MISSING_KEY
    assert 'No API_KEY' in text


def test_live_probe_without_credentials_fails_the_doctor():
    '''The live probe reports a failure instead of touching the network.'''
    with _without_keys():
        code, text = _capture(['doctor', '--json', '--live'])
    payload = json.loads(text)
    assert code != EXIT_OK
    assert payload['ok'] is False
    probed = [check for check in payload['checks'] if check['label'] == 'provider probe']
    assert probed and probed[0]['status'] == 'fail'


def test_masked_key_hides_the_middle():
    '''Keys are shortened for printing and short keys are not echoed.'''
    assert _masked_key('sk-1234567890abcdef') == 'sk-123...cdef'
    assert _masked_key('short') == 'set'


def test_module_entry_point_runs_the_cli():
    '''`python -m orchify` reaches the same entry point as the script.'''
    import runpy

    saved = sys.argv
    sys.argv = ['orchify', 'version']
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            try:
                runpy.run_module('orchify', run_name='__main__', alter_sys=True)
            except SystemExit as exc:
                assert exc.code == EXIT_OK
    finally:
        sys.argv = saved
    assert 'orchify ' in buffer.getvalue()


def main():
    '''Run every CLI test in this file.

    Returns:
        int: 0 when all tests passed, 1 when any failed.
    '''
    tests = [
        test_doctor_reports_ok,
        test_doctor_json_is_machine_readable,
        test_version_and_help,
        test_offline_demo_runs_every_stage,
        test_live_demo_without_credentials_is_reported,
        test_live_probe_without_credentials_fails_the_doctor,
        test_masked_key_hides_the_middle,
        test_module_entry_point_runs_the_cli,
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
