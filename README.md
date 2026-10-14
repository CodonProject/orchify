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
| `ORCHIFY_MAX_CALL_DEPTH` | `8` | How deep nested `Agent`/`Group` calls may nest before a call is refused; `0` disables the cap. |
| `ORCHIFY_BUDGET_TOKENS` | unset (`0` = off) | Default `Agent(budget_tokens=...)`: total tokens a run may use. |
| `ORCHIFY_BUDGET_USD` | unset | Default `Agent(budget_usd=...)`: estimated cost a run may spend (needs a price entry). |
| `ORCHIFY_BUDGET_SECONDS` | unset | Default `Agent(budget_seconds=...)`: wall-clock budget for a run. |
| `ORCHIFY_PRICE_TABLE` | built-in snapshot | Inline JSON or a JSON file path with per-model USD-per-1M-token prices, merged over the built-ins, e.g. `{"my-model": {"prompt": 0.5, "completion": 1.5, "cache_hit": 0.05}}`. |
| `ORCHIFY_MAX_CONCURRENT_RUNS` | `0` (off) | Runs executing at once, globally; the surplus waits in a `queued` run state. |
| `ORCHIFY_MAX_CONCURRENT_LLM_CALLS` | `0` (off) | In-flight LLM requests across every run. |
| `ORCHIFY_LLM_RPM` | `0` (off) | LLM requests allowed per rolling minute. |
| `ORCHIFY_TOOL_CACHE_SIZE` | `256` | Entries per `Tool` result cache (LRU); `0` disables caching. |
| `ORCHIFY_TOOL_TIMEOUT` | unset (wait forever) | Seconds an async tool may run before it is abandoned. |
| `ORCHIFY_HTTP_TIMEOUT` | connect `15` / read `600` / write `60` / pool `60` | Shared HTTP client timeout, in seconds (one number applies to every phase). `0`/`off`/`none`/`inf` disables it. |
| `ORCHIFY_LLM_MAX_RETRIES` | `2` | Retries when the provider stream fails **before** producing any data. |
| `ORCHIFY_LLM_RETRY_BACKOFF` | `0.5` | Base seconds of the exponential backoff (capped at `30`); a `Retry-After` header wins. |
| `ORCHIFY_MANAGER_ENABLED` | `1` | `0` disables the global manager entirely: no event tap, no registry, no cost. |
| `ORCHIFY_MANAGER_HISTORY` | `200` | Finished runs (and decided approvals) the manager retains; `0` keeps none. |
| `ORCHIFY_MANAGER_EVENTS` | `500` | Events kept in the manager's ring buffer; `0` keeps none. |
| `ORCHIFY_MANAGER_SINK` | unset (no file IO) | Path of a JSONL file every event and finished run is appended to; replay with `orchify.manager.read_sink()`. |
| `ORCHIFY_LOG_ECHO` | `1` | Echo library diagnostics to stdout; `0` leaves them to your own logging config. |
| `ORCHIFY_LOG_LEVEL` | `INFO` | Level of the `orchify` logger (an invalid value falls back to `INFO`). |

## Timeouts, Retries and Cancellation

A provider that stalls used to hang a run forever. Now:

* the shared HTTP client has a real timeout (`ORCHIFY_HTTP_TIMEOUT`);
* `orchify.llm.openai` retries `408/409/425/429/5xx` and transport errors with
  exponential backoff (`ORCHIFY_LLM_MAX_RETRIES`, `ORCHIFY_LLM_RETRY_BACKOFF`,
  `Retry-After`) — but only until the first chunk arrives, so a partial answer
  is never silently restarted from the beginning;
* callers can stop waiting and get a handle back instead of busy-waiting
  (see *Waiting for a Run* below).

## Budgets, Gates and Call Depth

Three guards keep a run from running away — each is off by default except the
depth cap:

```python
agent = Agent(name="Ops", llm=OpenAICompat(),
              budget_tokens=200_000,      # total tokens for one run
              budget_usd=1.50,            # estimated spend (needs a price entry)
              budget_seconds=120,         # wall-clock budget
              max_call_depth=8)           # how deep Agent/Group calls may nest
```

* Budgets are checked after every response; on breach the run emits
  `run:budget:exceeded` (payload: `reason`/`reasons`, `usage`, `limits`), then
  stops gracefully through the normal `run:finish` path, so hooks and the
  manager see an ordinary stopped run. `run:finish` payloads carry a `budget`
  dict with `tokens`/`cost_usd`/`priced`/`elapsed`/`limits`; the manager's totals
  sum `cost_usd`.
* Cost uses `orchify.budget.DEFAULT_PRICE_TABLE` (USD per 1M tokens, longest
  model-prefix wins) merged with `ORCHIFY_PRICE_TABLE`. `budget_usd` with a
  model that has no price is a configuration error: the run fails loudly instead
  of guessing.
* `ORCHIFY_MAX_CONCURRENT_RUNS` queues the surplus: a run that cannot start
  emits `run:queued` (payload `limit`), shows up as `queued` in the manager, and
  emits `run:dequeued` when a slot frees. `ORCHIFY_MAX_CONCURRENT_LLM_CALLS` and
  `ORCHIFY_LLM_RPM` bound the LLM side; gates never fail a run, they wait (a
  queued `control:stop` still stops it immediately).
* The depth cap refuses a call that would exceed `max_call_depth` or re-enter a
  call path already in progress (`A → B → A`), emitting `run:call:blocked` with
  `{reason, caller, callee, call_path, max_call_depth}`; the model receives the
  refusal as the tool result and can choose another route.

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

## Open Questions (`ask_user`)

The same pool also carries free-text questions, so an agent can ask a human for
information instead of a yes/no decision:

```python
agent = Agent(name="Ops", llm=OpenAICompat(), enable_ask_user=True)

@orchify_broker.hook('run:approval:required')
def on_question(event):
    if event.payload['kind'] == 'question':
        orchify_broker.answer_approval(event.payload['approval_id'], 'staging')
        # or, through the manager:  orchify_manager.answer(approval_id, 'staging')
```

* `enable_ask_user=True` registers a built-in `ask_user(question)` tool; a
  custom tool opts in with `@tool(ask_question=True)` — its function is **never
  executed**, the human's answer is the tool result.
* Questions are `kind='question'` entries in the approval pool: the payload
  gains `question`, and `run:approval:resolved` gains `answer`.
* `orchify_broker.pending_questions()` / `orchify_manager.questions()` list the
  open ones; `orchify_broker.answer_approval(id, text)` /
  `orchify_manager.answer(id, text)` resolve them.
* An unanswered question times out into `[no answer] the user did not answer the
  question` — the run continues instead of hanging.

## Async Tools and the Result Cache

`async def` tools work: a coroutine returned by a tool is awaited on the
orchestrator loop, bounded by `Tool(timeout=...)` / `ORCHIFY_TOOL_TIMEOUT`
(default: wait forever).

```python
@tool(cache=True)            # or cache=30 for a 30-second TTL
async def fetch(url: str) -> str:
    ...
```

Caching is per `Tool` instance, keyed by the JSON-serialised arguments, LRU
bounded by `ORCHIFY_TOOL_CACHE_SIZE` (default `256`; `0` disables it), and
single-flight: concurrent calls with the same arguments wait for one execution.
`tool:call:finish` payloads carry `cached` so a caller can tell a hit from a
miss; `tool.clear_cache()` empties one tool.

## Global Manager

`orchify_manager` is a single global view over everything Orchify is doing —
agents, groups, runs, approvals, hooks, events and the backend threads:

```python
from orchify import orchify_manager as om

om.status()          # counts + live runs + totals + backend, one dict
om.runs()            # every live run, with status/step/elapsed
om.run(run_id)       # one run (live, else from history, else None)
om.history(limit=20) # finished runs: answer, usage, steps, stopped
om.agents()          # registered Agents (even idle ones)
om.groups()          # registered Groups + mode/members/last winner
om.approvals()       # the manual-confirmation pool, decided ones included
om.questions()       # open ask_user questions (kind='question')
om.events(limit=50, run_id=run_id, types=['agent:*'])
om.hooks()           # bindings by owner / event type
om.backend()         # loop + thread alive, HTTP timeout, tool-pool usage

print(om.report())   # human-readable tables
om.to_json()         # the same data as JSON (json.dumps-safe)
```

It is also a **control surface** — every call delegates to the existing
feedback/approval channel, so it adds no new semantics:

```python
om.pause(run_id); om.resume(run_id); om.stop(run_id)
om.pause_all(); om.stop_all()
om.approve(approval_id, reason='looks fine'); om.reject(approval_id)
om.answer(approval_id, 'staging')   # resolve an ask_user question
om.wait(run_id, timeout=30)   # like Agent.wait(): raises on the loop thread
```

Runs are reported as `queued`, `running`, `paused`, `awaiting_approval`, `done`,
`stopped` or `error`; group runs are tracked too (their own `run_id`, the
voting winner and every member `run_id`). Agents and groups register
themselves at construction, so an idle instance is visible before it ever
runs; the registry holds **weak references**, so simply dropping your last
reference to an agent removes it from the manager.

The manager is deliberately passive: it is a single hook that only appends to
bounded buffers (`ORCHIFY_MANAGER_HISTORY`, `ORCHIFY_MANAGER_EVENTS`) and never
blocks a run, touches the network, or starts a thread. `close()` removes that
hook. Set `ORCHIFY_MANAGER_ENABLED=0` to opt out of all of it.

The tap is also lazy: an event's payload is stored by reference (a shallow copy
freezes only the top level) and converted to JSON-safe data when you actually
read `events()`, so an event nobody inspects costs no conversion. Setting
`ORCHIFY_MANAGER_EVENTS=0` skips record building altogether, so it really is a
zero-cost switch.

Set `ORCHIFY_MANAGER_SINK=/path/runs.jsonl` to persist instead of (only)
remembering: every event and every finished run is appended as one JSON object
per line (`"record": "event"` / `"record": "run"`), the first write lazily opens
the file, and the tap never raises — a sink error disables the sink and is
logged. Replay it with:

```python
from orchify.manager import read_sink

read_sink('runs.jsonl', kinds=['run'], run_id=run_id, types=['agent:*'], limit=50)
# or, from a manager instance:  om.replay(limit=50)
```

## Logging

Diagnostics (hook failures, agent errors, skipped plugin modules) go through the
standard `logging` module on the `orchify` logger — the library makes no bare
`print()` calls. The default deliberately keeps the old console behaviour: level
`INFO` plus a stdout handler, so `Error executing hook ...` is still visible.

```python
from orchify import configure_logging, get_logger

configure_logging(level='DEBUG')   # or set ORCHIFY_LOG_LEVEL
configure_logging(echo=False)      # let your own logging config own the output
logger = get_logger(__name__)      # a child of the 'orchify' logger
```

`ORCHIFY_LOG_ECHO=0` is the environment equivalent of `echo=False`: no handler is
added and records propagate to your root logger.

## Tests

```bash
python tests/test_pause_approval.py      # or: python -m pytest tests
python tests/test_manager.py
python tests/test_logging_and_edges.py
python tests/test_budget_gate.py
python tests/test_ask_user.py
python tests/test_sink.py
python tests/test_tool_runtime.py
```

The suite is fully offline: a scripted fake LLM stands in for the provider.
82 tests, no network — the four pause points (including mid-stream and the last
step), the lost-resume race, stop-while-parked, approval approve/reject/timeout,
run/turn identity, group member handles and the voting tally, the per-instance
run guard (`allow_concurrent_runs`), run-completion handles and their reuse,
hooks-outside-the-lock, the HTTP timeout and the retry/backoff helpers, the
token/second/USD budgets (including the missing-price error and the price-table
overrides), the run/LLM gates and their queueing, the call-depth/cycle guard for
agents and groups, async tools and the tool result cache, the open-question
channel (`ask_user`), the JSONL sink and `read_sink()`, plus the
global manager (registration and weakref release, the disabled/zero-cost mode,
run lifecycle and history, the approval pool, stop/stop_all, group visibility,
event filters, bounded buffers, and `to_json()`/`report()`), and the edges:
logging configuration and the hook-failure record, `CallChain` (plus the
`Agent`/`Group` chain getters), plugin loading/`attach`/`detach_tools`, the
`WebBackend` lifecycle and `Vote`.

`tests/smoke_live_control.py`, `tests/smoke_live_manager.py` and
`tests/smoke_live_features.py` are opt-in live smokes: they talk to a real
provider from `.env` (pause/resume, the approval pool, the manager driving both,
and — for the features — a priced token budget, an async cached tool, `ask_user`
and the JSONL sink).

CI (`.github/workflows/ci.yml`) runs the same suite on Python 3.9/3.11/3.13 and
builds the sdist + wheel; on 3.13 it also prints a coverage report (`pytest-cov`)
for information — there is no coverage gate.

The package is configured in `pyproject.toml` (setuptools). `setup.py` remains
only as a shim, because `install.py` / `publish.py` shell out to
`setup.py sdist bdist_wheel`.
