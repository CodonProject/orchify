# Orchify

A Python framework for building LLM-powered AI agent orchestration systems.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from orchify import Agent, OpenAICompat

llm = OpenAICompat()
agent = Agent(name="MyAgent", llm=llm, system_prompt="You are a helpful assistant.")

agent.run(user_input="Hello!")
```

## With Tools

```python
from orchify import Agent, OpenAICompat, tool

@tool(cache=True)
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

llm = OpenAICompat()
agent = Agent(
    name="WeatherBot",
    llm=llm,
    system_prompt="You are a weather assistant.",
    tools=[get_weather]
)

agent.run(user_input="What's the weather in Tokyo?")
```

## Environment Variables

Create `.env` file:

```
API_KEY=your_api_key
BASE_URL=https://api.deepseek.com
MODEL_ID=deepseek-chat
```

Optional runtime tuning (all have safe defaults):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ORCHIFY_TOOL_WORKERS` | `32` | Worker threads in the shared tool executor. |
| `ORCHIFY_POLL_INTERVAL` | `0.05` | Seconds between control-channel checks; also the responsiveness of `pause`/`stop`. |
| `ORCHIFY_PAUSE_TIMEOUT` | unset (wait forever) | Default `Agent(pause_timeout=...)`. |
| `ORCHIFY_APPROVAL_TIMEOUT` | unset (wait forever) | Default `Agent(approval_timeout=...)`; on expiry the request is **denied** (fail closed). |
| `ORCHIFY_RUN_TIMEOUT` | unset (wait forever) | Default timeout for `Agent.wait()` / `Group.wait()`. |
| `ORCHIFY_VOTE_TIMEOUT` | `300` | Seconds a `mode='voting'` group waits for its members; `0` dispatches without waiting. |
| `ORCHIFY_HTTP_TIMEOUT` | connect `15` / read `600` / write `60` / pool `60` | Shared HTTP client timeout, in seconds (one number applies to every phase). `0`/`off`/`none`/`inf` disables it. |
| `ORCHIFY_LLM_MAX_RETRIES` | `2` | Retries when the provider stream fails **before** producing any data. |
| `ORCHIFY_LLM_RETRY_BACKOFF` | `0.5` | Base seconds of the exponential backoff (capped at `30`); a `Retry-After` header wins. |

## Timeouts, Retries and Cancellation

A provider that stalls used to hang a run forever. Now:

* the shared HTTP client has a real timeout (`ORCHIFY_HTTP_TIMEOUT`);
* `orchify.llm.openai` retries `408/409/425/429/5xx` and transport errors with
  exponential backoff (`ORCHIFY_LLM_MAX_RETRIES`, `ORCHIFY_LLM_RETRY_BACKOFF`,
  `Retry-After`) — but only until the first chunk arrives, so a partial answer
  is never silently restarted from the beginning;
* callers can stop waiting and get a handle back instead of busy-waiting
  (see *Waiting for a Run* below).

## Event Monitoring

```python
from orchify.broker import orchify_broker
from orchify.event import BaseEvent

@orchify_broker.hook('*')
def on_event(event: BaseEvent):
    print(f"{event.event_type}: {event.content}")

agent.run(user_input="Hi!")
```

Hooks are selected under the broker lock but **invoked outside it**, so a slow
hook delays only its own event, not every other run (or the emitting thread).

## Runs and Turns

Every `Agent.run()` / `Group.run()` is one **run**, identified by a `run_id`.
`run()` returns it, every event carries it, and it is the key for feedback,
hooks, approvals and run contexts:

```python
run_id = agent.run("Hello!")            # e.g. 'MyAgent#a1b2_c3d4'
orchify_broker.feedback('control:pause', run_id=run_id)
orchify_broker.pending_approvals(run_id)   # approvals of this run
orchify_broker.active_runs()               # every run currently alive
```

A run makes one or more **turns**. One turn is a single LLM round trip — one
step of the agent loop. Every event carries:

| Field | Meaning |
| --- | --- |
| `run_id` | one execution: unique, stable for the whole run. |
| `turn` | the 1-based turn (LLM round trip) index inside the run. |
| `turn_id` | `f'{run_id}#{turn}'` — a step-scoped label. |

So the identity flows `run_id` → `turn` → `turn_id`, and a run contains many
turns. Address control feedback and approvals at the **run** (`run_id=`), never
at a turn: a turn is only a fraction of a run, and a caller-supplied label may
be shared by several runs. `feedback(turn_id=...)` is still accepted for a
single turn: it is resolved back to its run via
`orchify_broker.run_id_of_turn(turn_id)`, and refused (returns `False`) when it
maps to no active run. Caller-supplied ids are honoured —
`agent.run(..., run_id='my-run')` — and the legacy `turn_id=` keyword now simply
sets the run id.

A `Group` follows the same rule: the group has its own `run_id`, every member
run keeps its own, and all member handles are retained in `group.last_run_ids`
(voting/competition no longer drop the runners-up).

## Waiting for a Run

`run()` returns as soon as the run is scheduled, so the caller keeps a handle
instead of busy-waiting:

```python
run_id = agent.run("Hello!")
agent.is_running()              # True while that run is in flight
agent.wait(run_id, timeout=30)  # True when it finished, False on timeout
agent.cancel(run_id)            # ask the run to stop (control:stop)
agent.last_answer               # the final answer of the last run
```

`Agent.wait()` / `Group.wait()` default their timeout to `ORCHIFY_RUN_TIMEOUT`
(unset = wait forever) and raise `RuntimeError` when called from the
orchestrator loop thread, where waiting would deadlock the run being waited for.

One `Agent` handles one run at a time by default: a second `run()` while the
first is still in flight raises `RuntimeError`, because both would share
`agent.messages` and `agent.last_run_id`. Use a second instance, or pass
`Agent(..., allow_concurrent_runs=True)` to opt into the overlap. `Group`
follows the same rule.

## Groups

`Group(mode=...)` accepts `broadcast`, `manager_delegation`, `voting` and
`competition`. (`round_robin` was removed — chain agents yourself with
`Group.call_agent()` / `Agent.call_agent()`, or let a manager drive them.)

A `voting` group waits for its members and tallies their answers:

```python
group = Group(name="panel", agents=[a, b, c], mode="voting")
run_id = group.run("Is this safe to ship?")
group.last_votes   # [Vote(agent_name='a', vote='yes'), ...]
group.last_winner  # the agent whose (normalised) answer got the most votes
group.last_answer  # that answer, and the 'vote' event payload
```

Answers are compared after whitespace/case normalisation; ties go to the first
agent in the group's order. The wait is bounded by `ORCHIFY_VOTE_TIMEOUT`
(default 300 s) or `Group(..., vote_timeout=...)`; `vote_timeout=0` restores
fire-and-forget dispatch. `competition` and `manager_delegation` stay
dispatch-only: read the members' `last_answer`s yourself.

## Pause / Resume / Stop

Every run listens on a feedback channel that may be driven from any thread —
including from inside a hook:

```python
orchify_broker.feedback('control:pause',  run_id=run_id)
orchify_broker.feedback('control:resume', run_id=run_id)
orchify_broker.feedback('control:stop',   run_id=run_id)
```

* `control:pause` is honoured at **four** points: before the model request, while
  the response is still streaming, after the stream, and after the last step's
  tools (so a pause arriving on the final step is not lost). The run emits
  `run:paused` then parks.
* `control:resume` is never lost: the run arms its resume waiter *before*
  emitting `run:paused`, so a resume sent directly in reaction to the event is
  still delivered.
* `control:stop` interrupts a parked run and finishes it with `stopped=True`.
* `Agent(pause_timeout=...)` (or `ORCHIFY_PAUSE_TIMEOUT`) makes a parked run
  self-resume with `run:resumed` `status='timeout'` instead of waiting forever.

## Manual Approval Pool

Let a human confirm a tool call before it runs. Opt in per tool, per agent, or
with a callable policy:

```python
from orchify import Agent, OpenAICompat, tool, orchify_broker

@tool(require_approval=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"

agent = Agent(name="Ops", llm=OpenAICompat(), tools=[delete_file])
# agent-level gate for every tool:  Agent(..., require_approval=True)
# selective gate:                   Agent(..., require_approval=lambda tool, args: args.get('force'))

@orchify_broker.hook('run:approval:required')
def on_approval(event):
    pending = orchify_broker.get_approval(event.payload['approval_id'])
    print(f"{pending.tool_name} wants {pending.args}")
    orchify_broker.approve(pending.approval_id, reason='looks fine')
    # or: orchify_broker.reject(pending.approval_id, reason='too risky')

agent.run("Clean up the build directory")
```

* `run:approval:required` carries `{approval_id, kind, tool_id, tool_name, args, step, timeout}`;
  the pending request also exposes its `run_id` and `turn_id`.
* `orchify_broker.pending_approvals()` lists everything still awaiting a decision;
  `pending_approvals(run_id)` narrows it to one run.
* `run:approval:resolved` carries `{approved, reason, timed_out}`; an approved
  call executes normally, a denied one never runs and the model receives
  `[approval denied] <reason>` as the tool result.
* On `Agent(approval_timeout=...)` expiry the request is **rejected** with
  `timed_out=True` — fail closed, never fail open.
* A `control:stop` that lands while a request is parked is consumed as a stop:
  the call is denied and the run ends instead of continuing (`wait_approval`
  raises `RunStopped`).

## Tests

```bash
python tests/test_pause_approval.py   # or: python -m pytest tests
```

The suite is fully offline: a scripted fake LLM stands in for the provider.
30 tests, no network — the four pause points (including mid-stream and the last
step), the lost-resume race, stop-while-parked, approval approve/reject/timeout,
run/turn identity, group member handles and the voting tally, the per-instance
run guard (`allow_concurrent_runs`), run-completion handles and their reuse,
hooks-outside-the-lock, the HTTP timeout and the retry/backoff helpers.

CI (`.github/workflows/ci.yml`) runs the same suite on Python 3.9/3.11/3.13 and
builds the sdist + wheel.

The package is configured in `pyproject.toml` (setuptools). `setup.py` remains
only as a shim, because `install.py` / `publish.py` shell out to
`setup.py sdist bdist_wheel`.
