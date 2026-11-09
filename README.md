# Orchify

A Python framework for building LLM-powered AI agent orchestration systems.

## Contents

- [Installation](#installation)
- [Command Line](#command-line)
- [Quick Start](#quick-start)
- [With Tools](#with-tools)
- [Examples](#examples)
- [HTTP API](#http-api)
- [Environment Variables](#environment-variables)
- [Configuration in Code](#configuration-in-code)
- [Timeouts, Retries and Cancellation](#timeouts-retries-and-cancellation)
- [Budgets, Gates and Call Depth](#budgets-gates-and-call-depth)
- [Event Monitoring](#event-monitoring)
- [Subscribing from Your Code](#subscribing-from-your-code)
- [Runs and Turns](#runs-and-turns)
- [Waiting for a Run](#waiting-for-a-run)
- [Groups](#groups)
- [Pause/Resume/Stop](#pauseresumestop)
- [Manual Approval Pool](#manual-approval-pool)
- [Tool Policies](#tool-policies)
- [Tool Retries and Errors](#tool-retries-and-errors)
- [Open Questions (`ask_user`)](#open-questions-ask_user)
- [Terminal Console](#terminal-console)
- [Errors](#errors)
- [Async Tools and the Result Cache](#async-tools-and-the-result-cache)
- [MCP Tools](#mcp-tools)
- [Context and Cost Plugins](#context-and-cost-plugins)
- [Cost Visibility](#cost-visibility)
- [Persistence and Resume](#persistence-and-resume)
- [Telemetry](#telemetry)
- [Model Providers and Fallback](#model-providers-and-fallback)
- [Multimodal Input](#multimodal-input)
- [Retrieval: Embeddings and RAG](#retrieval-embeddings-and-rag)
- [MCP Server](#mcp-server)
- [Run Store](#run-store)
- [Record and Replay](#record-and-replay)
- [Evaluation](#evaluation)
- [Production Server](#production-server)
- [Docker](#docker)
- [Tool Result Truncation](#tool-result-truncation)
- [Summarizing Context](#summarizing-context)
- [Sub-agent Delegation](#sub-agent-delegation)
- [Structured Output](#structured-output)
- [Tool Sandbox](#tool-sandbox)
- [Global Manager](#global-manager)
- [Logging](#logging)
- [Testing Your Agents](#testing-your-agents)
- [Tests](#tests)

## Installation

```bash
pip install -e .
```

## Command Line

Installing the package also adds an `orchify` console script; from a checkout,
`python -m orchify` reaches the same entry point:

```bash
orchify version         # orchify and Python versions
orchify doctor          # python, dependencies, credentials, backend, gates, plugins
orchify doctor --json   # the same report as JSON; exits 1 when a check fails
orchify doctor --live   # also probes the provider's /models endpoint over HTTP
orchify demo            # offline walkthrough: scripted answer, tool call, approval
orchify demo --live     # one real answer through the provider in .env
orchify runs            # finished runs: tokens and cost per run
orchify runs --sink var/runs.jsonl --json
orchify chat            # terminal REPL over .env (or --setup your_module.py)
orchify chat --prompt "Summarise this diff" --max-steps 4
orchify serve           # the HTTP API on 127.0.0.1:8010 (see Production Server)
orchify serve --setup setup.py --keys k1,k2 --tokens 200000
orchify eval --data evals/support.jsonl --scorer contains --json
orchify plugins         # installed orchify.plugins entry points and the registry
orchify plugins --load  # import and load them (reports broken ones)
orchify mcp             # expose .env's agent to an MCP client over stdio
orchify mcp --setup setup.py --expose tools --tool-prefix fs_
```

`doctor` reports each check as `ok`, `warn` or `fail` and exits non-zero only
for a `fail`. It never contacts the network unless you pass `--live`, so it is
safe to run anywhere, and it warns when an installed copy of orchify shadows the
checkout you are standing in.

## Quick Start

One call asks the question and returns the answer:

```python
from orchify import Agent

agent = Agent.from_env(
    name="MyAgent",
    system_prompt="You are a helpful assistant.",
)

print(agent.ask("Hello!"))
```

`Agent.from_env()` reads `API_KEY`/`OPENAI_API_KEY`, `BASE_URL` and `MODEL_ID` for
you; pass `llm=OpenAICompat(...)` instead when you want to build the client
yourself. `ask()` starts a run and blocks until it finishes, so it must not be
called from the orchestrator loop thread — use `await agent.aask(...)` there.
`run()`/`arun()` give you the raw run handle when you want to monitor or stop a
run yourself.

## With Tools

```python
from orchify import Agent, tool

@tool(cache=True)
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

agent = Agent.from_env(
    name="WeatherBot",
    system_prompt="You are a weather assistant.",
    tools=[get_weather],
)

print(agent.ask("What's the weather in Tokyo?"))
```

## Examples

Runnable scripts live in `examples/`:

| Script | What it shows |
| --- | --- |
| `examples/quickstart.py` | One `ask()` call against your provider. |
| `examples/tools.py` | A tool whose schema comes from its signature and docstring. |
| `examples/approval_console.py` | `console()` approving risky tool calls in the terminal. |
| `examples/group_vote.py` | A `mode='voting'` panel and its winner. |
| `examples/offline_demo.py` | The whole flow with `orchify.testing.FakeLLM` — no API key needed. |
| `examples/lit_review.py` | A literature-review agent over the arXiv API. |
| `examples/serve_api.py` | The whole engine behind HTTP (FastAPI) for a web frontend. |
| `examples/api_smoke_test.py` | 22 checks against a running `serve_api.py`. |

```bash
python examples/offline_demo.py
```

## HTTP API

The reusable service lives in `orchify/server.py`; `examples/serve_api.py` is one
deployment of it (arXiv tools + a note writer). The kernel is untouched — the
server only adds sessions, routes and event washing.

```python
from orchify.server import create_app

app = create_app(agent_factory=lambda session: Agent.from_env(name=session))
```

```bash
python examples/serve_api.py                 # 127.0.0.1:8010, docs at /docs
HOST=0.0.0.0 PORT=8010 python examples/serve_api.py
ORCHIFY_API_KEYS=dev-key python examples/serve_api.py   # require a key
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Process + engine state (the only unauthenticated data route) |
| `POST` | `/api/chat` | Submit a task, returns `run_id` immediately |
| `GET` | `/api/sessions` | Live sessions, LRU order |
| `DELETE` | `/api/sessions/{id}` | Forget one conversation (a running run keeps going) |
| `GET` | `/api/runs` | Live + finished runs, newest first, with `tokens`/`cost_usd` |
| `GET` | `/api/runs/{id}` | One run: status, answer, usage, budget, cost |
| `GET` | `/api/runs/{id}/events` | Incremental timeline (`?since=`), plus `finished` |
| `GET` | `/api/runs/{id}/events/stream` | The same timeline as server-sent events |
| `POST` | `/api/runs/{id}/control/{action}` | `stop` / `pause` / `resume` a live run |
| `GET` | `/api/approvals` | Pending approvals, with the exact `args` |
| `POST` | `/api/approvals/{id}/approve` | Approve |
| `POST` | `/api/approvals/{id}/reject` | Deny — the reason goes back to the agent |
| `POST` | `/api/approvals/{id}/answer` | Answer an `ask_user` question |
| `GET` | `/metrics` | Prometheus text exposition |

Production knobs come from `create_app(...)` arguments or the environment:
`ORCHIFY_API_KEYS` (comma separated, rotation friendly), `ORCHIFY_SESSION_RUNS` /
`ORCHIFY_SESSION_WINDOW` (per-session rate limit), `ORCHIFY_MAX_CONCURRENT_RUNS`
(global concurrency) and `ORCHIFY_SSE_TIMEOUT` (idle limit of the event stream).

Two things worth knowing before wiring a frontend:

- **`run_id` contains `#`**, which is a URL fragment delimiter. Every record also
  carries `run_id_url` (with `!` instead) — use that in paths, or the request
  silently 404s.
- **Denying does not stop the run.** The rejection becomes a tool result, and the
  agent keeps going — usually to explain that the action was blocked. Judge
  success by the `tool:call:finish` detail, not by `status`.

Full contract, state machine and known limits: [`docs/api-contract.md`](docs/api-contract.md).

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
| `ORCHIFY_POLL_INTERVAL` | `0.05` | Fallback control-channel tick: `pause`/`approval`/`resume`/run completion wake the waiter immediately, this only bounds how often a `stop` request is noticed. Trades idle wakeups for stop latency. |
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
| `ORCHIFY_TOOL_CACHE_TTL` | unset (never expires) | Seconds a `Tool(cache=True)` entry stays fresh; a numeric `cache=` value wins. |
| `ORCHIFY_MAX_RUN_TOOL_CALLS` | `8` | Tool calls one run may have in flight at once (one run cannot occupy the whole shared pool); `0` disables the cap. |
| `ORCHIFY_STREAM_RETRIES` | `1` | Automatic re-requests for a stream that failed **after** it already showed text; the replayed prefix is never shown twice. `0` disables it. |
| `ORCHIFY_RUN_OUTCOME_LIMIT` | `256` | Finished-run outcomes the broker remembers for late readers; `0` disables the registry. |
| `ORCHIFY_TOOL_TIMEOUT` | unset (wait forever) | Seconds an async tool may run before it is abandoned. |
| `ORCHIFY_TOOL_POLICY` | unset | Default tool permission policy for every agent, e.g. `read_*=allow,bash=deny,write_*=ask`. Rules match the tool name as a glob; the first match wins. |
| `ORCHIFY_HTTP_TIMEOUT` | connect `15` / read `600` / write `60` / pool `60` | Shared HTTP client timeout, in seconds (one number applies to every phase). `0`/`off`/`none`/`inf` disables it. |
| `ORCHIFY_LLM_MAX_RETRIES` | `2` | Retries when the provider stream fails **before** producing any data. |
| `ORCHIFY_LLM_RETRY_BACKOFF` | `0.5` | Base seconds of the exponential backoff (capped at `30`); a `Retry-After` header wins. |
| `ORCHIFY_MANAGER_ENABLED` | `1` | `0` disables the global manager entirely: no event tap, no registry, no cost. |
| `ORCHIFY_MANAGER_HISTORY` | `200` | Finished runs (and decided approvals) the manager retains; `0` keeps none. |
| `ORCHIFY_MANAGER_EVENTS` | `500` | Events kept in the manager's ring buffer; `0` keeps none. |
| `ORCHIFY_MANAGER_SINK` | unset (no file IO) | Path of a JSONL file every event and finished run is appended to; replay with `orchify.manager.read_sink()`. |
| `ORCHIFY_LOG_ECHO` | `1` | Echo library diagnostics to stdout; `0` leaves them to your own logging config. |
| `ORCHIFY_LOG_LEVEL` | `INFO` | Level of the `orchify` logger (an invalid value falls back to `INFO`). |
| `ORCHIFY_API_KEYS` | unset (open) | Comma-separated API keys accepted by `orchify.server`; accepting several makes rotation painless. |
| `ORCHIFY_SESSION_RUNS` | `0` (off) | Runs one session may submit per `ORCHIFY_SESSION_WINDOW`. |
| `ORCHIFY_SESSION_WINDOW` | `60` | Length of the per-session quota window, in seconds. |
| `ORCHIFY_SSE_TIMEOUT` | `1800` | Seconds an idle `/events/stream` connection is kept open before `event: timeout`. |
| `ORCHIFY_SESSION_TOKENS` | `0` (off) | Tokens one session may spend in total before `429`; charged from each run's usage. |
| `ORCHIFY_SESSION_COST` | `0` (off) | USD one session may spend (`ORCHIFY_PRICE_TABLE` prices); unbudgeted = off. |
| `ORCHIFY_RUN_STORE` | unset (memory) | Where finished runs live: `memory`, `sqlite:var/runs.db`, `redis://host:6379/0` (or any SQLite path). |
| `ORCHIFY_REDIS_URL` | `redis://127.0.0.1:6379/0` | Default URL for `RedisRunStore(url=None)`. |
| `ORCHIFY_SANDBOX_DIR` | unset (temp dir) | Working directory for `run_sandboxed()` when no `workdir=` is passed. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | unset / `https://api.anthropic.com` | Credentials for `AnthropicLLM`. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) / `GEMINI_BASE_URL` | unset / `https://generativelanguage.googleapis.com` | Credentials for `GeminiLLM`. |

## Configuration in Code

`configure()` sets the same knobs as the environment variables at runtime, for
every part of the library that reads them:

```python
from orchify import config, configure, gate_stats

configure(
    max_call_depth=4,
    budget_usd=0.50,
    max_concurrent_runs=2,
    llm_rpm=120,
    manager_sink="runs.jsonl",
    log_echo=False,
)

print(config().max_call_depth)      # 4
print(gate_stats()["llm_rpm"])      # per_minute 120
```

`configure()` writes the values into `os.environ`, refreshes the poll interval,
rebuilds the run/LLM gates, re-applies the logging settings, resizes the shared
HTTP timeout and re-configures the global manager. Values that only matter when
an object is constructed (`pause_timeout`, `approval_timeout`, `tool_timeout`,
`budget_*`, `vote_timeout`) apply to objects created afterwards. An unknown
setting raises `TypeError`, and `config().resolved()` shows the effective value
of every setting together with the environment variable it came from.

Two conveniences round it out:

```python
agent = Agent.from_env("MyAgent", system_prompt="You are terse.")  # llm built for you
agent.reset()                                                      # forget the conversation
```

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

Two options make that guarantee stronger:

```python
from orchify.broker import orchify_broker

orchify_broker.hook('run:finish', audit, priority=10)      # runs first
orchify_broker.hook('*', upload_trace, background=True)    # never blocks the run

orchify_broker.pending_hooks()        # queued background hooks
orchify_broker.drain_hooks(timeout=5) # wait until they all ran
```

`priority` orders the hooks of one event (higher first, ties keep registration
order); `background=True` hands the hook to one shared daemon dispatcher, so even
a hook that does network IO cannot slow the run. Errors from a background hook
are logged and never reach the run, and a hook removed before it runs is skipped.
`orchify_manager.hooks()` reports each binding's owner, mode, priority and
background flag, plus how many background hooks are still queued.

## Subscribing from Your Code

`orchify.on()` is the short form of the same thing, and it does not require your
hook to pretend to be a plugin:

```python
from orchify import on, off

@on('tool:call:start')
def log_tool(event):
    print(f"tool {event.tool_name}")

binding = on('run:finish', lambda event: print(event.payload), once=True)

off(binding)      # a HookBinding, an owner name or the callback itself
```

A hook registered by an application fires whatever `owner` it carries; only
hooks registered as plugins are filtered by an agent's `enabled_plugins`, so
plugin isolation still holds.

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
When you only want the answer, `ask()` combines `run()` + `wait()` for you and
`aask()`/`arun()` are its awaitable forms:

```python
answer = agent.ask("Hello!", timeout=30)      # blocking, raises RunTimeout/RunFailed
answer = await agent.aask("Hello!")           # safe on the orchestrator loop
run_id = await agent.arun("Hello!")           # same, but you get the run id
```

`Group.ask()` follows the same shape and resolves the answer of the member (or
nested group) that produced it. `agent.reset()` clears the conversation,
`agent.last_error` holds the traceback of the last aborted run, and
`Agent.from_env(...)` builds the LLM from the environment for you.

One `Agent` handles one run at a time by default: a second `run()` while the
first is still in flight raises `RunAlreadyActive` (a `RuntimeError`), because
both would share `agent.messages` and `agent.last_run_id`. Use a second instance,
or pass `Agent(..., allow_concurrent_runs=True)` to opt into the overlap. `Group`
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

## Tool Policies

`require_approval` is a boolean per tool or agent. When the decision needs to
depend on *which* tool and *which* arguments, use a `ToolPolicy`: an ordered list
of rules with a default.

```python
from orchify import Agent, ToolPolicy, ToolRule

policy = ToolPolicy(
    rules=[
        ToolRule('read_*', 'allow'),
        ToolRule({'write_file', 'save_note'}, 'ask', reason='writes to disk'),
        ToolRule('bash', 'deny', args={'cmd': 'rm *'}),   # only this shape is denied
        ToolRule('bash', 'ask'),
    ],
    default='allow',
)

agent = Agent(name="Ops", llm=OpenAICompat(), tools=tools, tool_policy=policy)
```

* Rules match the tool name as a glob (`fnmatch`) and optionally a subset of the
  arguments (`args={'cmd': 'rm *'}`, values may be globs too). The **first**
  matching rule wins; nothing matching falls back to `default`.
* Three actions: `allow` (run it), `ask` (go through the approval pool) and
  `deny` (never run it). A denial does not abort the run: the model receives
  `[policy denied] <reason>` as the tool result and carries on — the same shape
  as a human rejection.
* A policy decision and an agent/tool `require_approval` combine: an explicit
  `require_approval` still forces `ask` even when the policy says `allow`.
* The approval request now carries a `reason`, so the human sees *why* it was
  asked (`run:approval:required` payload, `reason` key).
* Set a default for every agent with `ORCHIFY_TOOL_POLICY=read_*=allow,bash=deny`
  (comma-separated `glob=action`), or build one in code with
  `orchify.policy.policy_from_env()`.
* The decision, including the rule that fired, lands in the
  `tool:call:finish` payload as `policy` (`{action, reason, tool_name}`) and
  `denied_by_policy: true` for denials.

## Tool Retries and Errors

Tools are fallible; a retry policy belongs on the tool, not in every call site.

```python
from orchify import tool

@tool(retry=2, retry_on=(httpx.TimeoutException,), backoff=0.5, on_error='return')
def fetch(url: str) -> str:
    ...
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `retry` | `0` | Extra attempts after the first failure. |
| `retry_on` | every exception | Exception type or tuple that may be retried. |
| `backoff` | `0.5` | Base delay; doubles per attempt. |
| `max_backoff` | `8.0` | Cap on the delay. |
| `on_error` | `'raise'` | `'return'` turns the final failure into `[tool error] <type>: <msg>` so the model can recover instead of the run dying. |

`KeyboardInterrupt` and `SystemExit` are never retried. Retries happen inside
the tool's single-flight slot, so coalesced callers see one outcome, and the
final error (or its text) is what the agent receives as the tool result.

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

## Terminal Console

`console()` watches the pool for you: it prints every pending approval and
question and sends the answer you type back to the waiting agent. That is all
`examples/offline_demo.py` needs to drive a full approval + question flow.

```python
from orchify import console

loop = console()          # inp=input, out=print
loop.start()              # a daemon thread
try:
    print(agent.ask("Ship the release."))
finally:
    loop.stop()
```

`console(inp=..., out=...)` accepts any callables, so a test or a GUI can answer
from a queue instead of a keyboard; `run_once()` handles at most one item and
returns whether it did anything. The top-level shortcut functions
`approvals()`, `pending_approvals()`, `questions()`, `approve()`, `reject()` and
`answer()` are thin wrappers over the global manager for the cases where you want
to build your own loop.

## Errors

Every failure the library raises on purpose derives from `OrchifyError`, while
still subclassing the built-in exception it used to raise, so existing
`except RuntimeError` / `except ValueError` blocks keep working:

| Exception | Base | Raised when |
| --- | --- | --- |
| `RunAlreadyActive` | `RuntimeError` | A second `run()` starts on an agent/group that is still running (`run_id`, `owner`). |
| `WaitOnLoopThread` | `RuntimeError` | `wait()`/`ask()` is called from the orchestrator loop thread; await `arun()`/`aask()` instead. |
| `RunTimeout` | `TimeoutError` | `ask()` gave up waiting; the run is asked to stop first. |
| `RunFailed` | `OrchifyError` | The run ended with `agent:abort`; `.error` carries the traceback. |
| `UnknownRun` | `ValueError` | Waiting for a run id that does not exist (yet). |

Message texts name the fix (use `allow_concurrent_runs=True`, `wait()`, a second
instance, `arun()`), so a failure tells you what to change instead of only what
broke.

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

## MCP Tools

An MCP server's tools are just tools. Point a client at a command (stdio) or a
URL (streamable HTTP), and hand the result to an agent:

```python
from orchify import Agent, MCPClient, OpenAICompat

client = MCPClient(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
)
tools = client.tools()            # names are prefixed 'mcp_'; prefix='fs_' renames
agent = Agent(name="Files", llm=OpenAICompat(), tools=tools)
agent.ask("What is in the working directory?")
client.close()                    # or: with MCPClient(...) as client:
```

The same works with `uvx`, which is how many MCP servers are published:

```python
client = MCPClient(command='uvx', args=['mcp-server-time'], name='time')
client.tools()[0].execute(timezone='Asia/Tokyo')
client.close()
```

`env=` is merged over the current environment, which is the place for whatever a
stdio server needs (for `uvx`: `UV_CACHE_DIR`, `UV_TOOL_DIR`, `UV_DEFAULT_INDEX`
behind a proxy or a mirror).

* Transport: `command=`/`args=`/`env=`/`cwd=` for stdio, or `url=`/`headers=`
  for streamable HTTP with session ids. Both do the `initialize` handshake.
  The client negotiates `2025-11-25`, `2025-06-18` or `2025-03-26`, rejects a
  version it does not speak, and sends `MCP-Protocol-Version` on every HTTP
  request after the handshake.
* HTTP sessions are handled for you: `Mcp-Session-Id` is captured from
  `initialize` and echoed afterwards, `404` triggers a transparent
  re-initialisation and retry, and `close()` sends `DELETE` to end the session.
  A timed-out request also sends `notifications/cancelled` (never for
  `initialize`, which the specification forbids cancelling).
* With `resume_stream=True` (the default) the client opens the optional `GET`
  SSE stream and resumes it with `Last-Event-ID` after a disconnect; pass
  `resume_stream=False` when the server answers `405` or you do not need it.
* Names are sanitised to `[A-Za-z0-9_-]` and capped at 64 characters (MCP allows
  arbitrary strings; OpenAI does not), so `tools(prefix='fs_')` is the escape
  hatch when two servers would collide.
* The server's raw JSON Schema is preserved via `Tool(parameters_schema=...)`,
  which keeps nested objects, enums and `required` intact instead of flattening
  them into strings.
* A tool that answers with `isError` raises `MCPError` (`.code`, `.data`), so
  the usual `on_error`/`retry` policy applies. `tools(require_approval=True)`
  routes every call through the approval pool. `structuredContent` comes back as
  the decoded Python object from `call_tool()`; the generated `Tool` renders it
  as JSON text so a tool message is always a string.
* A stdio server that dies during startup reports its last stderr lines in the
  `MCPError`, so a missing package, a bad interpreter or an unwritable cache
  directory is visible instead of a bare "closed the connection".
* `client.list_tools()` / `call_tool()` / `prompts()` / `resources()` are there
  when you want to drive the protocol yourself, pagination included.

## Context and Cost Plugins

Long conversations and money are policy decisions, not kernel features. Both are
opt-in plugin classes in `orchify/plugins/`, and an agent only picks them up when
it enables them by name — so nothing changes for agents that do not ask:

```python
from orchify import Agent, OpenAICompat, orchify_plugins
from orchify.plugins import ContextWindowPlugin, CostPlugin

orchify_plugins.register(ContextWindowPlugin(max_prompt_tokens=16000, keep_recent=2))
orchify_plugins.register(CostPlugin(max_run_cost=0.50))

agent = Agent(
    name="assistant",
    llm=OpenAICompat(),
    plugins=['context_window', 'cost'],      # opt in, per agent
)
```

`ContextWindowPlugin` installs itself as LLM middleware and, before every
request, estimates the prompt (CJK-aware, no tokenizer download), keeps the
system prefix plus the newest turns, and replaces everything older with one
marker message. Turns are never split, so an assistant message with `tool_calls`
always travels with its `tool` results. `keep_recent`, `reserve_tokens`,
`min_savings` and `marker` tune it; `plugin.compact(messages, tools)` is usable
standalone, and `plugin.snapshot()` reports how much was saved. With
`persist=True` the trimmed history is also pushed back into the run via
`control:update_messages`, so later steps start from the compacted view.

`CostPlugin` listens on `agent:finish`/`run:finish`, prices the exact token
counts with `budget.price_for()` and exposes `run_cost(run_id)`,
`total_cost()` and `snapshot()`. `max_run_cost` / `max_total_cost` ask the run
to stop when a cap is crossed and emit `cost:budget`. Both plugins are
registered once and enabled per agent; `orchify_plugins.attach(name, agent)`
binds one to an existing agent's LLM, and `plugin.cleanup()` detaches it again.

Third-party plugins do not need to be imported by hand: declare an entry point
in the `orchify.plugins` group and `Agent(plugins=['my-plugin'])` loads it on
first use (a relative import inside the package, so it stays lazy):

```toml
[project.entry-points."orchify.plugins"]
my-plugin = "my_package.plugin:MyPlugin"
```

The target may be a `Plugin` subclass, an instance, a `PLUGINS` list or a
zero-argument factory returning any of those. `orchify plugins` lists what is
installed, `orchify plugins --load` imports and loads them and reports broken
ones as `plugin:error` events instead of failing the whole app.

## Cost Visibility

Even without `CostPlugin`, every finished run records what it cost:

* the manager's run record carries `model`, `tokens` and `cost_usd`, and the
  totals aggregate `cost_usd` from the same field (no double counting);
* `orchify.report()` prints a `cost=$…` line and an `usd` column in the history
  table; `status()` exposes the same numbers as a dict;
* `orchify runs` prints the newest runs with `tokens`/`usd`, and
  `orchify runs --sink var/orchify.jsonl` reads a sink file after the process
  is gone (`--json` for machines).

Pricing comes from `ORCHIFY_PRICE_TABLE` (inline JSON or a file path), merged
over a built-in snapshot; a model that is not in the table counts as `0.0`
rather than guessing.

## Persistence and Resume

Run history and agent state survive the process with two stdlib-only stores:

```python
from orchify import Agent, CheckpointStore, EventStore, orchify_broker

events = EventStore('var/orchify')
events.attach(orchify_broker)          # append-only JSONL, one file per run

checkpoints = CheckpointStore('var/orchify')
agent = Agent(name="assistant", llm=OpenAICompat(), checkpoint=checkpoints.callback)

agent.ask("Draft the report")
checkpoints.resume(agent)              # newest run: messages are back in place
agent.ask("Now shorten it")            # continues the same conversation
```

* `EventStore` buffers events and flushes them per run (`flush_every` seconds;
  `events.flush()` forces it), keeps at most `max_events` per run, and skips a
  damaged line instead of failing the read. `events(run_id)` and `runs()` replay
  what happened.
* `CheckpointStore` writes one JSON line per step (`run_id`, `step`, `agent`,
  `model`, `messages`) and keeps the newest `keep` per run.
* `save_agent(agent, path)` / `load_agent(agent, path)` snapshot a whole agent
  (messages, model, `messages_mode`) when you want a single file instead of a
  run history.
* Resuming never re-calls the model: it restores `agent.messages`, and the next
  `ask()` continues from there.

## Telemetry

`orchify.telemetry` turns the event stream into metrics and spans without
adding a dependency:

```python
from orchify import setup_telemetry

metrics, tracer = setup_telemetry(prefix='orchify', service_name='my-orchestrator')
...
print(metrics.render())        # Prometheus text exposition
metrics.snapshot()             # the same data as a dict
```

* `Metrics` counts events (`events_total{type}`), runs (`runs_started_total`,
  `runs_active`, `runs_finished_total{status}`), steps, tokens
  (`llm_tokens_total{kind}`), run duration, tool calls/cache hits/errors,
  approvals, budget stops and LLM retries. Tool names are bounded to 64 distinct
  labels, the rest fold into `tool="other"`, so cardinality cannot explode.
* Tokens and cost are also labelled by model
  (`llm_tokens_by_model_total{model,kind}`, `llm_cost_usd_total{model}`), priced
  with `orchify.budget`; an unpriced model simply produces no cost sample.
* `Tracer` emits OpenTelemetry spans when `opentelemetry` is importable and
  keeps them in memory otherwise (`tracer.backend` says which); `tracer.span()`
  is a context manager for your own spans.
* Request-level tracing: wrap the LLM once and every provider call becomes a
  span with tokens, duration and `finish_reason`:

  ```python
  from orchify import Agent, Metrics, instrument_llm

  metrics, tracer = Metrics(prefix='orchify'), Tracer(service_name='orchestrator')
  agent = Agent.from_env()
  instrument_llm(agent.llm, metrics, tracer)   # or agent.llm.use(LLMInstrumentation(metrics, tracer))
  ```

  `Metrics.observe_request()` records `llm_requests_total{model,status}` and the
  `llm_request_seconds{model}` summary; failed calls close the span with an
  error status instead of losing it.
* `setup_telemetry()` attaches both to `orchify_broker` in the background and
  returns them; call `metrics.detach()` / `tracer.detach()` on shutdown, or let
  `orchify.server.create_app(metrics=..., tracer=...)` manage the lifecycle.

## Model Providers and Fallback

`OpenAICompat` speaks any OpenAI-compatible endpoint; the package also ships
native Anthropic and Gemini adapters that translate the same `messages` /
`tools` structures, plus a fallback chain:

```python
from orchify import AnthropicLLM, FallbackLLM, GeminiLLM, OpenAICompat

llm = FallbackLLM([
    OpenAICompat(model='deepseek-chat'),          # OPENAI_API_KEY / OPENAI_BASE_URL
    AnthropicLLM(model='claude-sonnet-4-5'),      # ANTHROPIC_API_KEY
    GeminiLLM(model='gemini-2.5-flash'),          # GEMINI_API_KEY / GOOGLE_API_KEY
])
agent = Agent.from_env(llm=llm)
```

* Every adapter produces the same `Chunk`/`FinalStatus` stream, including the
  reasoning (`is_cot`) channel and `finish_reason` (`stop`, `length`,
  `tool_calls`, `content_filter`).
* `FallbackLLM` walks the providers in order, skipping any whose credentials are
  missing, retrying the next one only while nothing has been yielded (so output
  is never duplicated), and cooling a provider down for `cooldown` seconds after
  `max_failures` failures. `snapshot()` reports per-provider failures, cooldown
  and which one answered last.
* Honouring proxy/envelope retries: `orchify.llm.retry.stream_sse()` retries
  408/409/425/429/500/502/503/504 and transport errors before the first token,
  then raises — a mid-stream failure is surfaced rather than replayed.

## Multimodal Input

Content parts let one message carry text and images. `ask`/`arun` accept either a
string or a list of parts:

```python
from orchify import image_part, text_part

answer = agent.ask([
    text_part('What is in this chart?'),
    image_part('chart.png', detail='high'),        # path, bytes or data:/http URL
])
```

The OpenAI adapter forwards the parts verbatim; Anthropic and Gemini convert them
to their own block formats (`image` / `inlineData`). `as_data_url()` inlines a
local file, `detect_mime()` sniffs the magic bytes, and `describe()` splits any
content value into its text and image halves — handy for logging.

## Retrieval: Embeddings and RAG

Two embedding backends ship in the box: `HashingEmbeddings` (deterministic,
dependency-free, good enough for keyword-ish recall) and `OpenAIEmbeddings`
(`/embeddings`, batched, needs `OPENAI_API_KEY`).

```python
from orchify import HashingEmbeddings, VectorIndex, index_event_store, retrieve_tool

index = VectorIndex(HashingEmbeddings(512))
index.add_texts(['Refunds within 30 days.', 'Shipping takes 3-5 days.'],
                metadatas=[{'source': 'refunds.md'}, {'source': 'shipping.md'}])

index.save('var/index.jsonl')                       # one vector per line
index = VectorIndex.load('var/index.jsonl', HashingEmbeddings(512))

agent = Agent.from_env(tools=[retrieve_tool(index, top_k=4)])
index_event_store(EventStore('var/runs'), index)    # index past runs
```

* `search(query, k=4)` returns `Hit(document, score)` sorted by cosine
  similarity; `min_score` filters noise.
* `retrieve_tool()` wraps an index as a normal tool and returns
  `[score] source\n<text>` blocks, truncated to `max_chars`.
* `index_event_store()` copies finished answers and tool results out of an
  `EventStore` into the index, so an agent can remember earlier runs.

## MCP Server

`orchify.mcp_server` is the other direction of [`MCP Tools`](#mcp-tools): expose
an agent (or plain tools) to Claude Desktop, Cursor or any MCP client over stdio.

```python
from orchify import Agent
from orchify.mcp_server import serve_stdio

serve_stdio(Agent.from_env(name='assistant'), expose='agent')
```

```json
{"mcpServers": {"orchify": {"command": "orchify", "args": ["mcp"]}}}
```

* `expose='agent'` publishes one `ask` tool (delegating to `agent.ask`),
  `expose='tools'` publishes your tool list, `expose='both'` publishes both with
  an `agent_` prefix so names cannot collide.
* A setup file (`--setup setup.py` defining `agent=` and/or `tools=`) avoids
  writing your own entry point; `--tool-prefix` renames the published tools.
* `initialize` negotiates the protocol version: a version the server supports is
  echoed, anything else falls back to `2025-06-18` instead of failing.
* `tools/list` is paginated (`page_size`, opaque `cursor`/`nextCursor`, an
  invalid cursor is `-32602`), tool entries carry `title`, `annotations` and
  `outputSchema` when the tool exposes them, and `tools/call` results that are
  dictionaries come back as `structuredContent` next to the serialised text.
* Requests are answered with the standard MCP methods (`initialize`, `ping`,
  `tools/list`, `tools/call`); unknown methods return `-32601`, tool failures
  come back as `isError: true` content instead of killing the connection, and
  `notifications/cancelled` is logged rather than answered.

The same server also speaks **streamable HTTP**, so it can live next to the
normal API instead of a child process:

```bash
orchify mcp --transport http --port 8010           # standalone: POST/GET/DELETE /mcp
orchify serve --mcp                                # mount /mcp on the API app
```

```python
from orchify import MCPHTTPServer
from orchify.mcp_server import MCPStdioServer
from orchify.server import create_app

app = create_app(mcp=MCPHTTPServer(MCPStdioServer(tools=my_tools, expose='tools')))
```

One endpoint answers `POST` (a message or a batch, JSON in, JSON out), `GET`
(the optional server-to-client SSE stream) and `DELETE` (end the session).
`initialize` returns `Mcp-Session-Id`; later requests must echo it, an expired
session gets `404`, an unsupported `MCP-Protocol-Version` gets `400`.
`create_app(mcp=..., mcp_path='/mcp')` mounts it next to the API and shares its
auth (a configured `api_keys` makes `/mcp` require `Authorization: Bearer` or
`X-API-Key`, like every other route).

Interoperability was checked against the official `mcp` Python SDK (1.29.0):
its streamable-HTTP client negotiates `2025-11-25` with this server, lists the
tools with their titles and calls them, and this package's client drives a
public streamable-HTTP server (`MCPClient(url='https://…/mcp')`) through the
same handshake, tool listing and structured tool call.

## Run Store

Finished runs live in memory by default. A `RunStore` keeps them across
processes and replicas:

```python
from orchify import OrchifyManager, SQLiteRunStore

manager = OrchifyManager(store=SQLiteRunStore('var/runs.db'))
manager.history(limit=20)     # served by the store, so other replicas' runs appear
```

| Store | Backend | Use |
| --- | --- | --- |
| `MemoryRunStore(max_runs=500)` | in-process `OrderedDict` | tests, single process |
| `SQLiteRunStore(path, max_runs=0)` | sqlite3 (WAL) | one host, several workers |
| `RedisRunStore(url=None)` | redis hash + sorted set | several hosts (`pip install redis`; the connection is lazy, so a missing package only shows up when a run is actually stored) |

`ORCHIFY_RUN_STORE` selects one without code: unset/`memory` → memory,
`sqlite:var/runs.db` → SQLite, `redis://host:6379/0` → Redis, any other value is
treated as a SQLite path. `run_store_from_env()` returns the same thing, and
`status()['store']` shows which store is active and how many records it holds.

## Record and Replay

Record a conversation once, replay it in CI without a network or an API key:

```python
from orchify import OpenAICompat, RecordLLM, ReplayLLM

llm = RecordLLM(OpenAICompat(model='deepseek-chat'))   # alias of RecordingLLM
agent = Agent.from_env(llm=llm, name='recorded')
agent.ask('Explain the diff.')
llm.save('recordings/diff.json')

agent = Agent.from_env(llm=ReplayLLM.from_file('recordings/diff.json'))
agent.ask('Explain the diff.')      # identical chunks, zero HTTP calls
```

* The recording is JSON: one entry per request (kwargs + every response) and it
  round-trips tool calls, reasoning and token counters.
* `ReplayLLM(strict=True)` fails when a request differs from the recording — use
  it to pin prompt templates. Without it, calls replay in order and a missing
  answer raises `ReplayError` instead of silently returning nothing.
* Non-underscore attributes (`model`, ...) delegate to the wrapped provider.

## Evaluation

`orchify.eval` scores an agent against a dataset:

```python
from orchify import EvalReport, load_dataset, run_dataset

cases = load_dataset('evals/support.jsonl')            # JSON array or JSONL
report = run_dataset(cases, agent=agent, scorer='contains', max_steps=4)
print(report.render())
print(report.summary())          # {'total':…, 'pass_rate':…, 'cases': {…}}
report.save('evals/report.json')
```

Scorers: `exact`, `contains` (default), `json` and `regex` (case `pattern=`).
`run_dataset(agent_factory=...)` gives every case a fresh agent; `agent=` reuses
one and calls `reset()` between cases unless `reset=False`. `orchify eval
--data evals/support.jsonl --json` runs the same thing from the CLI and exits
non-zero when any case fails, so it slots straight into CI.

## Production Server

`orchify/server.py` is the deployment layer: sessions, SSE, auth, quota, metrics
and run control, with the kernel untouched.

```python
from orchify import Agent, Metrics
from orchify.server import Quota, create_app, serve

app = create_app(
    agent_factory=lambda session: Agent.from_env(name=session),
    api_keys=['dev-key'],                                  # or ORCHIFY_API_KEYS
    quota=Quota(max_concurrent=8, max_per_window=30, window=60),
    metrics=Metrics(prefix='orchify'),
)
serve(app)                                                 # 127.0.0.1:8010
```

* `agent=` serves one shared agent (single conversation), `agent_factory=` gives
  each `session_id` its own — that is what a chat UI wants.
* Without `api_keys` (and without `ORCHIFY_API_KEYS`) every route is public; the
  lifespan logs a warning and `/api/health` reports `auth: "open"`.
* `GET /api/runs/{run_id}/events/stream` is a real event stream: frames arrive
  the moment an event is published (a condition variable, not a poll), idle
  connections get keep-alive comments, and the stream ends with `event: done`
  once the run reaches a terminal state. Reverse proxies need
  `proxy_buffering off` — the response already sends `X-Accel-Buffering: no`.
* `POST /api/runs/{run_id}/control/{stop|pause|resume}` drives the same control
  channel the library uses in-process; `DELETE /api/sessions/{id}` forgets a
  conversation without killing a run.
* `/metrics` serves the `Metrics` instance attached at startup as Prometheus
  text (and creates one if you did not pass it).
* Quotas: `Quota(max_concurrent=8, max_per_window=30, window=60)` limits
  throughput; `Quota(max_tokens=200000, max_cost=5.0)` limits spend per session
  and answers `429` with the reason (`used N tokens (limit N)`). Token and cost
  usage is charged from each run's `run:finish` usage, so it covers streaming
  runs the same way. `ORCHIFY_SESSION_TOKENS` / `ORCHIFY_SESSION_COST` set them
  from the environment.
* `orchify serve --setup setup.py --keys k1,k2 --sessions 32 --tokens 200000`
  starts this app from the shell; `orchify chat` is the same agent in a
  terminal REPL (`/help`, `/reset`, `/status`, `/report`, `/runs`, `/stop`).

The full HTTP contract — payloads, error codes, event taxonomy, deployment
checklist — lives in [`docs/api-contract.md`](docs/api-contract.md).

## Docker

The repository ships a `Dockerfile` that serves the API with a persistent run
store:

```bash
docker build -t orchify .
docker run -p 8010:8010 -v orchify-data:/data \
  -e OPENAI_API_KEY -e OPENAI_BASE_URL -e ORCHIFY_API_KEYS=changeme orchify
```

It installs the `web` extra, keeps runs in `/data/runs.db` plus a JSONL journal
in `/data/runs.jsonl`, exposes `8010`, and health-checks `/api/health`. Mount a
`setup.py` and pass `--setup` if you want your own tools instead of
`Agent.from_env()`.

## Tool Result Truncation

A huge tool result (a whole file, an API dump) otherwise eats the context window
on the next step:

```python
agent = Agent.from_env(
    tool_result_limit=4000,                  # characters kept per result
    tool_result_dir='var/tool-results',      # optional: offload the full text
)
```

Long results become `<head> ... <tail>` with a note giving the original size and
the path of the saved file; the tool message and the `tool:call:finish` event
(both the result and a `{truncated, original_chars, path}` payload) show it. The
saved file is `<run_id>-<tool_id>-<tool_name>.txt`, so nothing is lost — the model
can read it back with a file tool if it needs the middle.

## Summarizing Context

`ContextWindowPlugin` drops old turns; `SummarizingContextPlugin` replaces them
with a recap instead, which keeps the decisions and numbers a drop would lose:

```python
from orchify import orchify_plugins
from orchify.plugins.summarize import SummarizingContextPlugin

orchify_plugins.register(SummarizingContextPlugin(
    max_prompt_tokens=16000, keep_recent=2, persist=True))
agent = Agent.from_env(plugins=['summarizing_context'])
```

* When a prompt exceeds `max_prompt_tokens - reserve_tokens`, the oldest turns
  are summarized by `Summarizer` (any `llm`, usually the cheap one) and replaced
  by one `[summary of N earlier message(s)...]` user message. The newest
  `keep_recent` turns, the system prompt and any tool call/result pairing stay
  intact.
* A summarizer failure never breaks a run: the plugin falls back to a plain
  marker and increments `stats['fallbacks']`.
* `persist=True` sends the compacted history back to the running agent as
  `control:update_messages` (`op: 'replace'`), so later steps use it too — that
  needs the run id, so it happens on the event loop where one is known.
* `Summarizer.render()` / `.summarize()` are usable standalone;
  `plugin.compact()` is the sync equivalent of the middleware.

## Sub-agent Delegation

A capable model often splits a task itself; `Delegation` gives it a bounded,
auditable tool for that instead of letting it run wild:

```python
from orchify import Agent, Delegation

lead = Agent.from_env(name='lead')
delegation = Delegation(agent=lead, max_depth=2, max_concurrent=4, prefix='helper')
lead.add_tool(delegation.tool())

lead.ask('Research the three vendors and write one comparison table.')
```

* `delegate(task, name=None)` builds a child (a clone of the template, or your
  `factory(name, **overrides)`), runs it to completion and returns its answer as
  text; failures come back as `[delegate error] ...` so the parent can continue
  instead of crashing.
* Depth is bound per tool instance: a child at the limit gets no delegate tool,
  and a call beyond `max_depth` answers `[delegate limit] ...` without spawning
  anything.
* `max_concurrent` is a non-blocking slot pool — a busy parent gets
  `[delegate error] too many sub-agents are running; retry later` rather than
  queueing behind itself. Children share the broker, so they appear in
  `orchify_manager.runs()` like any other run.
* `require_approval=True` routes the delegate tool through the approval pool;
  `tool_name='subtask'` renames it.

## Structured Output

Json-mode plus a retry loop when the model breaks the contract:

```python
from orchify import ask_structured

schema = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string', 'minLength': 3},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
        'score': {'type': 'integer', 'minimum': 0, 'maximum': 10},
    },
    'required': ['title', 'tags', 'score'],
    'additionalProperties': False,
}

data = ask_structured(agent, 'Summarise this changelog.', schema, retries=2)
```

* The first attempt appends a schema instruction; a mismatch re-sends the
  problems as a correction and counts an attempt. After `retries` failures it
  raises `StructuredOutputError` carrying `attempts`, `last_error` and
  `last_text`, and the last few replies stay on
  `agent.structured_failures` for debugging.
* `extract_json()` tolerates prose and ```json fences; `validate()` implements
  the subset of JSON Schema that matters for tool-shaped output (`type`, `enum`,
  `const`, `anyOf`/`oneOf`, `properties`, `required`,
  `additionalProperties: false`, `items`, `pattern`, `minLength`/`maxLength`,
  `minimum`/`maximum`) and returns every error, not just the first.
* `key='title'` returns one field instead of the whole object.

## Tool Sandbox

Model-written code (or a risky tool) can run in a child process with a timeout
and a scrubbed environment:

```python
from orchify import Sandbox, sandbox_tool
from mytools import convert

safe = sandbox_tool(convert, timeout=20, workdir='var/sandbox')
agent = Agent.from_env(tools=[safe])
```

* Each call is `python -I -c <runner>` with the function's source and JSON
  arguments on stdin; the child's `cwd` is a scratch directory (or `workdir`)
  and its environment keeps only `PATH`/system/TEMP — API keys in the parent are
  invisible.
* Timeouts raise `SandboxTimeout`; a child exception comes back as a
  `SandboxError` with its type and message.
* Functions that close over local variables are rejected up front (a child
  cannot see them) — pass the values as arguments instead. Module imports the
  function uses are recreated in the child automatically.
* This is isolation for crashes, hangs and environment leaks, **not** a security
  boundary: the child runs as your user. Use containers/OS-level sandboxes when
  the code is untrusted.

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

The manager is deliberately passive: one hook appends to a bounded inbox, and a
background consumer thread (`OrchifyManagerTap`) turns those entries into run
records, history, totals and sink lines. The tap itself only enqueues, so the
emitting run never takes the manager lock, builds a record or writes a file; each
reader (`status()`, `runs()`, `events()`, `approvals()`, `totals()`, ...) first
drains what is queued, so a read always reflects events emitted before it.
`close()` flushes the inbox and stops that thread. The buffers stay bounded
(`ORCHIFY_MANAGER_HISTORY`, `ORCHIFY_MANAGER_EVENTS`). Set
`ORCHIFY_MANAGER_ENABLED=0` to opt out of all of it.

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

## Testing Your Agents

`orchify.testing` ships the same helpers the suite uses, so you can test an agent
without a provider, a key or a network:

```python
from orchify import Agent
from orchify.testing import FakeLLM, final, tool_call, Recorder

agent = Agent(
    name="Bot",
    llm=FakeLLM([
        [final("", [tool_call("lookup", '{"key": "alpha"}')])],
        [final("the answer")],
    ]),
    tools=[lookup],
)

with Recorder() as rec:
    assert agent.ask("ask me something") == "the answer"

print(rec.types())        # ['run:start', 'agent:start', ... , 'run:finish']
```

A script entry is either `{'content': '...'}` (a streamed chunk), `{'final':
...}` (the terminal response, built by `final()`), or `{'delay': 0.1}`;
`FakeLLM.calls` records the messages each request carried, which is how the
suite asserts on tool results and feedback.

## Tests

```bash
python tests/test_pause_approval.py      # or: python -m pytest tests
python tests/test_manager.py
python tests/test_logging_and_edges.py
python tests/test_budget_gate.py
python tests/test_ask_user.py
python tests/test_sink.py
python tests/test_tool_runtime.py
python tests/test_ergonomics.py
python tests/test_plugins.py
python tests/test_llm_middleware.py
python tests/test_run_outcome.py
python tests/test_group_async.py
python tests/test_cli.py
python tests/test_agent_controls.py
python tests/test_stream_retry.py
python tests/test_hooks.py
python tests/test_latency.py
python tests/test_plugins_and_policy.py
python tests/test_mcp.py
python tests/test_persistence.py
python tests/test_telemetry.py
python tests/test_server.py
python tests/test_entrypoints.py
python tests/test_mcp_server.py
python tests/test_mcp_http.py
python tests/test_llm_providers.py
python tests/test_rag.py
python tests/test_store.py
python tests/test_cli_chat.py
python tests/test_record.py
python tests/test_eval.py
python tests/test_tool_results.py
python tests/test_summarize.py
python tests/test_delegation.py
python tests/test_structured.py
python tests/test_sandbox.py
```

The suite is fully offline: a scripted fake LLM stands in for the provider, and
the MCP tests start a real (but local, inline) MCP server over stdio or mount the
streamable-HTTP transport in-process.
400 tests, no network — the four pause points (including mid-stream and the last
step), the lost-resume race, stop-while-parked, approval approve/reject/timeout,
run/turn identity, group member handles and the voting tally, the per-instance
run guard (`allow_concurrent_runs`), run-completion handles and their reuse,
hooks-outside-the-lock, the HTTP timeout and the retry/backoff helpers, the
token/second/USD budgets (including the missing-price error and the price-table
overrides), the run/LLM gates and their queueing, the call-depth/cycle guard for
agents and groups, async tools and the tool result cache (TTL, LRU bound,
single-flight, a shared `ToolCache`, the per-run concurrency cap), the
open-question channel (`ask_user`), the JSONL sink and `read_sink()`, plus the
global manager (registration and weakref release, the disabled/zero-cost mode,
run lifecycle and history, the approval pool, stop/stop_all, group visibility,
event filters, bounded buffers, and `to_json()`/`report()`), the ergonomics
surface (`ask()`/`aask()`/`arun()`, `reset()`, `from_env()`, the typed
exceptions, `on()`/`off()`, `configure()`/`config()`, `console()`), the edges:
logging configuration and the hook-failure record, `CallChain` (plus the
`Agent`/`Group` chain getters), plugin loading/`attach`/`detach_tools`, the
`WebBackend` lifecycle and `Vote`, the plugin manager (registration errors,
dependency order, discovery, load/unload closures), middleware ordering, scoping
and owner filtering in the LLM dispatch chain, the per-run terminal outcome
registry behind `ask()`, `Group.arun()`/`Group.aask()` and the non-blocking
`await_run()`, the `orchify` command line (`doctor`, `demo`, `version`, `runs`),
the in-run control channel (stop, override, message rewrites, inject/deny tool
results, malformed arguments, unknown tools), and the mid-answer stream retry
(duplicate-free replay, divergent-replay failure, the reasoning prefix, the
`ORCHIFY_STREAM_RETRIES` switch) and hook priority plus background dispatch.

The newer subsystems have their own files: `test_latency.py` pins the
event-driven waits (with `POLL_INTERVAL` forced to 5 seconds, a resume, an
approval and a run completion must still return in under one, and an emit must
visit only the bindings it matched), `test_plugins_and_policy.py` covers the
token estimator, context compaction, the cost plugin, `ToolPolicy` (including
agent-level deny/ask), `policy_from_env()` and tool retries,
`test_mcp.py` runs the handshake/list/call path against an inline MCP server,
`test_persistence.py` covers checkpoints, run replay and `save_agent`/
`load_agent`, `test_telemetry.py` covers the metric names, the Prometheus render
and the tool-label cardinality cap, and `test_server.py` drives the FastAPI app
in-process (auth, quota, control, SSE, sessions, `/metrics`).

Round three added: `test_entrypoints.py` (pip entry-point plugins, `orchify
plugins`), `test_mcp_server.py` (protocol plus a real subprocess round trip),
`test_llm_providers.py` (Anthropic/Gemini SSE translation and the fallback
chain, with a patched `post_stream_async`), `test_rag.py` (hashing embeddings,
ranking, index save/load, the retrieve tool and multimodal parts), `test_store.py`
(memory/SQLite/Redis stores and a second manager reading the first one's runs),
`test_cli_chat.py` (`orchify serve`/`chat`), `test_record.py` (record/replay,
strict mode, tool calls), `test_eval.py` (datasets, scorers, exit codes),
`test_tool_results.py` (truncation and file offload), `test_summarize.py`
(recap compaction, tool-call pairing, summarizer failure),
`test_delegation.py` (delegation round trip, depth limits, template cloning),
`test_structured.py` (JSON extraction, schema validation, corrective retries)
and `test_sandbox.py` (subprocess round trip, scrubbed environment, timeout,
closure rejection).

`tests/smoke_live_control.py`, `tests/smoke_live_manager.py` and
`tests/smoke_live_features.py` are opt-in live smokes: they talk to a real
provider from `.env` (pause/resume, the approval pool, the manager driving both,
and — for the features — a priced token budget, an async cached tool, `ask_user`
and the JSONL sink).

CI (`.github/workflows/ci.yml`) runs the same suite on Python 3.9/3.11/3.13 and
builds the sdist + wheel; on 3.13 it also prints a coverage report (`pytest-cov`)
for information — there is no coverage gate.

The package is configured in `pyproject.toml` (setuptools). `install.py` and
`publish.py` drive the standard frontends — `python -m build` for the sdist and
wheel, `python -m twine upload` over `dist/*` — and `setup.py` remains only as a
shim. The wheel ships `orchify/py.typed` and `orchify/cli.py`, and
`pip install -e ".[dev]"` adds `pytest`, `pytest-cov`, `mypy`, `build` and
`twine`; `mypy orchify` is wired through `[tool.mypy]` but stays advisory rather
than a CI gate.
