# Orchify HTTP API 契约

本文描述 orchify 内置 HTTP 服务（`orchify/server.py` 的 `create_app()`）暴露的
接口，供前端 / 网关对接。`examples/serve_api.py` 只是用它挂上了 arXiv 工具的一个
实例，接口一致。接口变动时请同步改本文。

- 启动：`python examples/serve_api.py`，或在代码里 `create_app(agent_factory=...)`
- 地址：`http://${HOST:-127.0.0.1}:${PORT:-8010}`，交互文档在 `/docs`
- 模型：由 `MODEL_ID` / `BASE_URL` / `API_KEY` 等环境变量决定（见 README「Environment Variables」）
- 编码：请求与响应均为 `application/json`；时间戳是 Unix 秒（浮点）；事件流例外（见 SSE）

服务只依赖 FastAPI（`pip install "orchify[web]"`），不需要额外的网关即可上生产：
自带 API key 鉴权、会话配额、`/metrics` 指标和 SSE 事件流。

## 1. 两个约定

### 1.1 run_id 与 URL 安全形式

orchify 生成的 `run_id` 形如 `会话名#8位码`。`#` 在 URL 里是 fragment 分隔符，
直接拼进路径会被截断（服务端只会收到前缀，返回 404）。因此每个携带 run_id 的
记录都会额外给出 `run_id_url`：把 `#` 换成 `!`。

- 前端只用 `run_id_url` 拼路径，不用自己 `encodeURIComponent`
- 服务端把路径里的 `!` 还原成 `#` 再查
- 需要展示或做 key 时用原始 `run_id`

### 1.2 校验与错误

| 状态码 | 含义 |
| --- | --- |
| 200 | 成功 |
| 400 | 控制动作未知（`control/` 只接受 `stop` / `pause` / `resume`） |
| 401 | 缺少或错误的 API key（仅在配置了 key 时出现） |
| 404 | 找不到该 run / 审批项（已处理或不存在） |
| 409 | 该 `session_id` 已有任务在跑（orchify 一个会话同时只允许一个 run） |
| 422 | 请求体未通过校验（FastAPI 标准格式） |
| 429 | 超出配额：全局并发已满，或该会话在窗口内提交次数超限 |
| 500 | 提交任务时内核抛错，`detail` 里带异常类型 |

错误体统一为 `{"detail": "<人话>"}`。

### 1.3 鉴权

配置了 key 之后，除 `GET /api/health` 外的所有接口都要带 key。两种写法等价：

```
Authorization: Bearer <key>
X-API-Key: <key>
```

key 从 `create_app(api_keys=[...])` 或环境变量 `ORCHIFY_API_KEYS`（逗号分隔，支持
轮换期间同时接受新旧两把）读取。没配置 key 时服务是公开的，启动日志会显式警告，
只适合本机开发。比较用 `secrets.compare_digest`，不受时序攻击影响。

### 1.4 配额

`Quota` 有五个维度，0 表示不限制：

| 参数 | 环境变量 | 说明 |
| --- | --- | --- |
| `max_concurrent` | `ORCHIFY_MAX_CONCURRENT_RUNS` | 全局同时在跑的 run 数 |
| `max_per_window` | `ORCHIFY_SESSION_RUNS` | 单个会话在一个窗口内可提交的 run 数 |
| `window` | `ORCHIFY_SESSION_WINDOW` | 窗口长度，默认 60 秒 |
| `max_tokens` | `ORCHIFY_SESSION_TOKENS` | 单个会话累计可消耗的 token 数 |
| `max_cost` | `ORCHIFY_SESSION_COST` | 单个会话累计可花费的美元（按价格表估算） |

超限返回 429，`detail` 说明是全局满、会话 run 数超限、token 超限还是费用超限。
token / 费用额度由 `run:finish` 事件的 usage 累加（后台 hook），所以流式 run 同样
计入；`GET /api/health` 的 `quota.usage` 给出每个会话已用的 run 数、token 与美元。

## 2. 端点

### GET /api/health

健康检查。`ok` 恒为 `true`（进程活着才能响应）。

```json
{
  "ok": true,
  "model": "deepseek-chat",
  "base_url": "https://api.deepseek.com/v1",
  "auth": "open",
  "sessions": 2,
  "sessions_created": 5,
  "events": {"runs_tracked": 3, "runs_finished": 1, "events_kept": 41, "events_dropped": 120},
  "quota": {"active": 0, "max_concurrent": 4, "max_per_window": 60, "window": 60.0, "sessions": {}},
  "orchify": {"enabled": true, "counts": {"runs": 1}, "totals": {}, "hooks": {}}
}
```

- `auth`：`open`（未配置 key）或 `api-key`
- `sessions`：当前会话池里的 Agent 数（容量 32，LRU 淘汰最久未用；淘汰只丢上下文，
  不影响正在跑的 run）
- `sessions_created`：启动以来新建过的会话总数（含已被淘汰的）
- `events`：事件缓存概况，`events_dropped` 是被刻意过滤掉的 token 级碎片数
- `quota`：配额实时占用
- `orchify`：`orchify.api.status()` 原样返回，含运行计数、用量汇总、hook 数

### POST /api/chat

提交一个任务，**立即返回**，不等结果。`agent.run` 本身非阻塞。

请求体：

| 字段 | 类型 | 默认 | 约束 |
| --- | --- | --- | --- |
| `text` | string | 必填 | 最短 1 |
| `session_id` | string | `"default"` | `^[A-Za-z0-9_-]{1,32}$`，同值共享上下文 |
| `max_steps` | int | `8` | 1–30 |

```json
{"text": "帮我查一下 2401.00001 这篇论文", "session_id": "u42", "max_steps": 6}
```

响应：

```json
{"run_id": "u42#a1b2c3d4", "run_id_url": "u42!a1b2c3d4", "session_id": "u42"}
```

同一 `session_id` 上一单还没跑完时返回 **409**；超出配额返回 **429**。

### GET /api/sessions

当前活着的会话（LRU 顺序，最久未用的在前）。用于运维观察，也可做"清理闲置会话"
的依据。

```json
{"total": 1, "items": ["u42"]}
```

### DELETE /api/sessions/{session_id}

丢掉一个会话的上下文（正在跑的 run 不受影响，它已被内核持有引用）。

→ `{"ok": true, "dropped": true}`；会话不存在时 `dropped` 为 `false`，仍返回 200。

### GET /api/runs

运行列表，新的在前。合并了 orchify 的两个视图：`api.runs()`（活跃）与
`api.history()`（已结束）。

查询参数：`status`（可选，按状态过滤）、`limit`（默认 20，夹在 1–100）。

```json
{"total": 3, "active": 1, "items": [{"run_id": "u42#a1b2c3d4", "run_id_url": "u42!a1b2c3d4", "status": "running", "...": "..."}]}
```

`items` 的每条是 orchify 运行记录，常见字段：`run_id`、`status`、`started_at`、
`finished_at`、`answer`、`step`、`usage`、`budget`，以及成本可见性字段：

| 字段 | 说明 |
| --- | --- |
| `model` | 该 run 使用的模型名（来自运行上下文），用于定价 |
| `tokens` | 结束时的总 token 数（优先 `total_tokens`，缺失时回退 prompt/completion） |
| `cost_usd` | 按 `model` 查价目表算出的美元成本；模型不在价目表里时为 0 |

最终答案取 `answer`，渲染进度用 `step`，成本看 `cost_usd`（前端可直接展示"本次
$0.0042"）。

### GET /api/runs/{run_id}

单个运行详情：状态、答案、进度、用量、预算。`run_id` 用 URL 安全形式。

- 200：运行记录（额外带 `run_id_url`）
- 404：没有这个运行

### GET /api/runs/{run_id}/events?since=0

增量拉取执行事件，前端轮询用。

```json
{"run_id": "u42#a1b2c3d4", "items": [], "last_seq": 12, "finished": false}
```

- 传回上次的 `last_seq` 作为 `since`，只拿到更新的部分
- 每个 run 最多缓存 300 条（超出丢最旧），足够轮询使用
- `items` 为空时 `last_seq` 原样返回 `since`，可安全续轮询
- `finished`：该 run 是否已收到终止事件（`run:finish` / `run:error` / `agent:abort`）。
  拿到 `true` 且 `items` 为空就可以停止轮询

事件对象（轮询与 SSE 完全相同）：

| 字段 | 说明 |
| --- | --- |
| `seq` | 全局自增序号，单调递增，作为游标 |
| `run_id` | 原始 run_id（非 URL 形式） |
| `type` | orchify 原始事件类型，如 `run:finish` |
| `kind` | 归一化分类：`status` / `tool` / `reason` / `approval` / `budget` / `error` |
| `step` | 轮次序号，可能为 `null` |
| `ts` | 事件时间（Unix 秒） |
| `detail` | 一句话摘要，可能缺省 |

`type` → `kind` 的映射见 `orchify/server.py` 里的 `KEEP_TYPES`。被刻意丢弃的
事件：`agent:answer`、`agent:reason:step`、`tool:assembly:*`（token 级碎片；
推理碎片会合并进 `agent:reason:finish` 那一条的 `detail`，完整答案看 `run:finish`）。

### GET /api/runs/{run_id}/events/stream?since=0

同一个事件流，改成 Server-Sent Events，服务端主动推，不需要轮询。

```
: connected

data: {"seq": 1, "run_id": "u42#a1b2c3d4", "type": "run:start", "kind": "status", "step": null, "ts": 1735660800.1, "detail": "帮我查一下 ..."}

: keep-alive

data: {"seq": 2, "type": "tool:call:start", "kind": "tool", "detail": "arxiv_search {\"query\": \"...\"}", "...": "..."}

event: done
data: {"run_id": "u42#a1b2c3d4", "status": "done", "tokens": 1520, "cost_usd": 0.0042}
```

- 每个事件一帧 `data: <json>`，字段与轮询接口的 `items` 完全一致
- 空闲时发 `: keep-alive` 注释帧，防止中间层掐连接
- 收到 `event: done` 即结束，`data` 里带最终 `status` / `tokens` / `cost_usd`
- `event: error` 表示 run_id 未知；`event: timeout` 表示空闲超过
  `ORCHIFY_SSE_TIMEOUT`（默认 1800 秒）
- 断线后带上最后的 `since` 重连即可补齐，前端不需要自己维护游标逻辑之外的缓冲

浏览器侧：

```js
const es = new EventSource(`/api/runs/${runIdUrl}/events/stream?since=${lastSeq}`);
es.onmessage = (e) => render(JSON.parse(e.data));
es.addEventListener('done', (e) => { showCost(JSON.parse(e.data)); es.close(); });
```

注意 `EventSource` 不能自定义 header，跨域或需要 key 时用 `fetch` + 流式读取，
或者把 key 放在 cookie / 网关层。

### POST /api/runs/{run_id}/control/stop

给运行中的 run 发控制指令，走 orchify 的控制通道（`control:stop|pause|resume`）。

→ `{"ok": true, "accepted": true, "action": "stop", "run_id": "u42#a1b2c3d4"}`

- `accepted` 为 `false` 表示该 run 已结束或拒绝接收（不影响 200）
- `pause` 之后要用 `control/resume` 恢复；`stop` 不可恢复
- run 不存在返回 404，动作拼错返回 400

### GET /metrics

Prometheus 文本格式（`text/plain; version=0.0.4`）。指标由
`orchify.telemetry.Metrics` 从事件流生成，前缀默认 `orchify`：

```
# TYPE orchify_runs_started_total counter
orchify_runs_started_total 3
# TYPE orchify_runs_active gauge
orchify_runs_active 1
# TYPE orchify_run_duration_seconds summary
orchify_run_duration_seconds_sum 12.5
orchify_run_duration_seconds_count 2
```

覆盖：`events_total{type}`、`runs_started_total`、`runs_finished_total{status}`、
`runs_active`、`steps_total`、`llm_tokens_total{kind}`、
`llm_tokens_by_model_total{model,kind}`、`llm_cost_usd_total{model}`、
`llm_requests_total{model,status}`、`llm_request_seconds{model}`、
`run_duration_seconds`、`tool_calls_total{tool}`、`tool_cache_hits_total`、
`tool_errors_total`、`approvals_total{decision}`、`budget_exceeded_total`、
`llm_retries_total`。
工具标签做了基数上限（64 个，超出归入 `other`），避免用户自定义工具名撑爆时间序列。

### POST|GET|DELETE /mcp

把 Agent 或工具按 MCP（Model Context Protocol）暴露给支持 MCP 的客户端
（Claude Desktop、Cursor、MCP Inspector 等）。只有在 `create_app(mcp=...)` 或
`orchify serve --mcp` 时才挂载，默认路径 `/mcp`（`mcp_path` 可改）。

- `POST`：一条消息或一批消息。初始化（`initialize`）成功时返回
  `Mcp-Session-Id` 响应头，之后的请求必须带回；会话过期返回 `404`
  （错误码 `-32000`）；`MCP-Protocol-Version` 不受支持返回 `400`（`-32600`）；
  JSON 解析失败返回 `400`（`-32700`）。只含通知的请求返回 `202`（空响应体）。
- `GET`：可选的服务器→客户端 SSE 流（本实现返回 `text/event-stream`），
  `Accept` 不匹配时返回 `405`。
- `DELETE`：结束会话，返回 `200`；缺少会话头 `400`，会话不存在 `404`。
- 支持的方法：`initialize`、`ping`、`tools/list`（游标分页，非法游标 `-32602`）、
  `tools/call`。未知方法返回 `-32601`；工具执行失败用 `isError: true` 而不是
  JSON-RPC 错误。
- 与 `/api/*` 共用鉴权：配置了 `ORCHIFY_API_KEYS` 时同样需要 `Authorization:
  Bearer` 或 `X-API-Key`。配额（token / 费用）只作用于 `/api/chat` 发起的 run，
  不限制 MCP 工具调用。

### GET /api/approvals?run_id=

待人工确认的操作。查询参数 `run_id` 可选（用 URL 安全形式），不传则返回全部。

```json
{"total": 1, "items": [{"approval_id": "ap_7", "run_id": "u42#a1b2c3d4", "run_id_url": "u42!a1b2c3d4", "tool_name": "save_note", "args": {"path": "a.md"}, "reason": "写文件需要确认"}]}
```

`args` 是 agent 想传给工具的参数，前端必须展示出来，这是人工确认的依据。

### POST /api/approvals/{approval_id}/approve

批准。批准后 agent 才会真正执行那个工具。

```json
{"reason": "没问题"}
```

→ `{"ok": true, "approval_id": "ap_7", "action": "approve"}`；不存在或已处理返回 404。

### POST /api/approvals/{approval_id}/reject

拒绝。`reason` 会回传给 agent（缺省 `"denied"`），agent 据此调整策略。

→ `{"ok": true, "approval_id": "ap_7", "action": "reject"}`；404 同上。

### POST /api/approvals/{approval_id}/answer

回答 agent 主动提出的问题（来自 `ask_user` 工具）。

```json
{"text": "用 3 号仓库"}
```

→ `{"ok": true, "approval_id": "ap_7", "action": "answer"}`；404 同上。

## 3. 前端典型流程

两条路任选：SSE（推荐，实时、无轮询开销）或轮询（最省事，任何环境都能跑）。

**SSE：**

1. `POST /api/chat` 拿到 `run_id_url`
2. `new EventSource('/api/runs/<run_id_url>/events/stream')`，逐条 `data:` 渲染：
   `status` 进时间线、`tool` 进步骤卡片、`reason` 折叠展示、`error` 高亮
3. 收到 `run:approval:required` 时拉 `GET /api/approvals?run_id=...`，展示 `args`
   后走 approve / reject / answer
4. 收到 `event: done` 时用帧里的 `status` / `tokens` / `cost_usd` 收尾，关闭连接

**轮询：**

1. `POST /api/chat` 拿到 `run_id_url`
2. `GET /api/runs/{run_id_url}/events?since=<last_seq>`，同上分区渲染
3. 同上处理审批
4. `finished` 为 `true` 且 `items` 为空时停轮询，取 `GET /api/runs/{run_id_url}` 拿
   完整 `answer`、`usage`、`tokens`、`cost_usd`

轮询间隔建议 500–1000 ms；事件缓存按 run 环形保留 300 条，长时间运行请依赖
`status` 与最终 `run:finish`，不要指望从 `seq=0` 还能补齐全部历史。

## 4. 部署清单

| 关注点 | 做法 |
| --- | --- |
| 鉴权 | 设 `ORCHIFY_API_KEYS`（或 `create_app(api_keys=...)`）；未设时服务公开 |
| 配额 | `ORCHIFY_MAX_CONCURRENT_RUNS` / `ORCHIFY_SESSION_RUNS` / `ORCHIFY_SESSION_WINDOW` / `ORCHIFY_SESSION_TOKENS` / `ORCHIFY_SESSION_COST` |
| 指标 | Prometheus 抓 `/metrics`；`orchify.telemetry.Tracer` 可另外接 OpenTelemetry，`instrument_llm()` 给每次模型请求加 span |
| 持久化 | `orchify.persistence.EventStore` 落事件、`CheckpointStore` 落步检查点，重启后可 `resume()`；`ORCHIFY_RUN_STORE` 让多副本共享 run 历史 |
| 启动 | `orchify serve --setup setup.py`（或仓库自带的 `Dockerfile`） |
| 反向代理 | SSE 需要关闭响应缓冲（nginx: `proxy_buffering off`；本服务已发 `X-Accel-Buffering: no`） |
| 健康检查 | `GET /api/health`（唯一免鉴权的数据接口） |

## 5. 与内核的对象对应

| HTTP 概念 | orchify 概念 |
| --- | --- |
| `run_id` | `Agent.run()` 的返回值，内核里唯一标识一次运行 |
| `session_id` | 一个 `Agent` 实例（同一会话共享 `messages`） |
| 事件 | broker 事件（`orchify.event`），`KEEP_TYPES` 里的子集 |
| 审批项 | `Approval`，对应 `requires_approval` 工具或 `ask_user` |
| `tokens` / `cost_usd` | 运行记录里按模型定价算出的用量与成本 |
| `/metrics` | `orchify.telemetry.Metrics` 的快照 |
| 控制指令 | `control:stop` / `control:pause` / `control:resume` 反馈通道 |
