"""把 orchify 包成 HTTP 服务：前端与编排内核之间的边界层。

这一层刻意做得很薄。内核（Agent 执行循环、审批门、运行记录、预算）
全部沿用 orchify 现有实现，一行不改；这里只负责三件事：

1. 把「会话」映射到 Agent 实例，让同一会话保留上下文
2. 把 orchify 的进程内函数暴露成 HTTP 路由
3. 订阅事件流，供前端展示「agent 此刻在做什么」

前端契约见 docs/api-contract.md。

启动::

    python examples/serve_api.py            # 默认 127.0.0.1:8010
    HOST=0.0.0.0 PORT=8010 python examples/serve_api.py

依赖：fastapi + uvicorn（已有）。orchify 本身仍只依赖 httpx。
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import orchify.api as api
from orchify import Agent, Tool

# ---------------------------------------------------------------------------
# 工具：文献检索 + 一个需要审批的写操作
# ---------------------------------------------------------------------------

_UA = 'orchify-http-demo/0.1 (+https://example.invalid)'
_ARXIV = 'https://export.arxiv.org/api/query'
_NS = {'a': 'http://www.w3.org/2005/Atom'}
_NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notes')


def arxiv_search(query: str, limit: int = 6) -> str:
    '''在 arXiv 上检索论文。

    Args:
        query (str): 检索词，英文效果最好。可用 AND / OR 组合。
        limit (int): 返回条数，1-15。

    Returns:
        str: 每行一条，格式为「arXiv id | 标题 | 年份」。
    '''
    limit = max(1, min(int(limit), 15))
    params = urllib.parse.urlencode({
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': limit,
        'sortBy': 'relevance',
    })
    req = urllib.request.Request(f'{_ARXIV}?{params}', headers={'User-Agent': _UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            xml = resp.read().decode('utf-8', 'replace')
    except Exception as exc:  # noqa: BLE001 - 网络问题一律回报给 agent，由它决定重试
        return f'检索失败：{exc}'

    root = ET.fromstring(xml)
    lines: List[str] = []
    for entry in root.findall('a:entry', _NS):
        raw_id = (entry.findtext('a:id', '', _NS) or '').rsplit('/', 1)[-1]
        arxiv_id = re.sub(r'v\d+$', '', raw_id)
        title = ' '.join((entry.findtext('a:title', '', _NS) or '').split())
        published = entry.findtext('a:published', '', _NS) or ''
        lines.append(f'{arxiv_id} | {title} | {published[:4]}')
    if not lines:
        return f'没有命中结果：{query}'
    return '\n'.join(lines)


def arxiv_abstract(arxiv_id: str) -> str:
    '''读取某篇 arXiv 论文的摘要。

    Args:
        arxiv_id (str): arXiv 编号，形如 2410.10781。

    Returns:
        str: 标题 + 摘要正文。
    '''
    arxiv_id = arxiv_id.strip()
    params = urllib.parse.urlencode({'id_list': arxiv_id, 'max_results': 1})
    req = urllib.request.Request(f'{_ARXIV}?{params}', headers={'User-Agent': _UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            xml = resp.read().decode('utf-8', 'replace')
    except Exception as exc:  # noqa: BLE001
        return f'读取失败：{exc}'

    entry = ET.fromstring(xml).find('a:entry', _NS)
    if entry is None:
        return f'找不到该编号：{arxiv_id}'
    title = ' '.join((entry.findtext('a:title', '', _NS) or '').split())
    summary = ' '.join((entry.findtext('a:summary', '', _NS) or '').split())
    return f'# {title}\n\n{summary}'


def save_note(path: str, content: str) -> str:
    '''把整理好的内容写入笔记文件。会写磁盘，所以需要人工确认。

    Args:
        path (str): 相对文件名，例如 review.md。
        content (str): 要写入的文本内容。

    Returns:
        str: 写入结果说明。
    '''
    safe = re.sub(r'[^A-Za-z0-9._\u4e00-\u9fff-]', '_', path.strip()) or 'note.md'
    os.makedirs(_NOTES_DIR, exist_ok=True)
    target = os.path.join(_NOTES_DIR, safe)
    with open(target, 'w', encoding='utf-8') as fh:
        fh.write(content)
    return f'已写入 {len(content)} 字符到 notes/{safe}'


_TOOLS = [
    Tool(func=arxiv_search, name='arxiv_search', description='在 arXiv 检索论文。'),
    Tool(func=arxiv_abstract, name='arxiv_abstract', description='读取某篇 arXiv 论文的摘要。'),
    Tool(
        func=save_note,
        name='save_note',
        description='把整理好的综述写入笔记文件。会写磁盘，需人工确认。',
        require_approval=True,
    ),
]

_SYSTEM_PROMPT = '''你是文献调研助手。工作方式：

1. 先想清楚要查哪几个方面，用不同的关键词多检索几轮，不要只搜一次
2. 挑出真正相关的论文，用 arxiv_abstract 读摘要，不要只看标题就下结论
3. 输出结构化的对比：方法核心 / 实验设置 / 主要结论 / 局限
4. 最后归纳共识、分歧、以及尚未被回答的问题
5. 检索失败的关键词要如实报告，说明覆盖有缺口，不要假装检索全了

需要保存时调用 save_note。回答用中文，不要编造 arXiv 编号。'''


# ---------------------------------------------------------------------------
# run_id 的 URL 形式
# ---------------------------------------------------------------------------

# orchify 生成的 run_id 形如 ``会话名#8位码``。``#`` 在 URL 里是 fragment 分隔符，
# 不编码会被截断（实测直接返回 404）。这里给一个 URL 安全的等价写法：
# ``#`` 换成 ``!``，前端拼路径时不用记着 encodeURIComponent。
_ID_SAFE = '!'
_ID_RAW = '#'


def _url_id(run_id: str) -> str:
    '''把 run_id 转成可安全放进 URL path 的形式。'''
    return str(run_id).replace(_ID_RAW, _ID_SAFE)


def _norm_id(raw: str) -> str:
    '''把 URL 里的 id 还原成原始 run_id。'''
    return raw.replace(_ID_SAFE, _ID_RAW)


def _with_url_id(record: Dict[str, Any]) -> Dict[str, Any]:
    '''给运行记录补一个 URL 安全的名字。'''
    record = dict(record)
    if record.get('run_id'):
        record['run_id_url'] = _url_id(record['run_id'])
    return record


# ---------------------------------------------------------------------------
# 会话池：session_id -> Agent
# ---------------------------------------------------------------------------


class AgentPool:
    '''按会话复用 Agent，使同一会话保留对话上下文。

    容量满时淘汰最久未使用的会话。淘汰只丢上下文，不影响正在跑的 run
    （run 已经由内核持有引用）。
    '''

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._items: 'OrderedDict[str, Agent]' = OrderedDict()
        self._lock = threading.Lock()
        self.created = 0

    def get(self, session_id: str) -> Agent:
        '''取出或构造该会话的 Agent。'''
        with self._lock:
            agent = self._items.get(session_id)
            if agent is not None:
                self._items.move_to_end(session_id)
                return agent
            agent = Agent.from_env(
                name=session_id,
                system_prompt=_SYSTEM_PROMPT,
                tools=list(_TOOLS),
            )
            self._items[session_id] = agent
            self.created += 1
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return agent

    def size(self) -> int:
        '''当前存活的会话数。'''
        with self._lock:
            return len(self._items)


# ---------------------------------------------------------------------------
# 事件收集：供前端展示执行过程
# ---------------------------------------------------------------------------


#: 只保留有信息量的事件。
#:
#: 被刻意排除的两类（实测一次 2 步运行里它们占了 35/43 条，全是噪音）：
#:
#: - ``agent:reason:step`` / ``agent:answer``：token 级流式碎片，一次回答会拆成
#:   二十多条。前者由 ``_accum_reason`` 累积后合并成一条，后者直接用
#:   ``run:finish`` 里的完整答案代替。
#: - ``tool:assembly:*``：工具参数的分片拼接过程，前端只关心 args 和结果。
_KEEP_TYPES = {
    'run:start': 'status',
    'run:queued': 'status',
    'run:dequeued': 'status',
    'run:next': 'status',
    'run:paused': 'status',
    'run:resumed': 'status',
    'run:finish': 'status',
    'run:approval:required': 'approval',
    'run:approval:resolved': 'approval',
    'run:budget:exceeded': 'budget',
    'run:call:blocked': 'budget',
    'agent:start': 'status',
    'agent:reason:finish': 'reason',
    'agent:finish': 'status',
    'agent:abort': 'error',
    'tool:call:start': 'tool',
    'tool:call:finish': 'tool',
    'group:start': 'status',
    'group:step': 'reason',
    'group:end': 'status',
    'group:error': 'error',
    'group:agent_call': 'tool',
    'group:call_agent': 'tool',
    'group:call_group': 'tool',
}

#: 从 payload 里按优先级找正文的键名。
_TEXT_KEYS = ('content', 'text', 'answer', 'message', 'reason', 'error', 'summary')


class EventLog:
    '''订阅 orchify 事件并环形缓存，按 run_id 提供增量拉取。

    用 ``background=True`` 挂到共享 dispatcher 上，慢回调不会拖慢 run。
    只保留 ``_KEEP_TYPES`` 里的事件，token 碎片按 run 合并成一条"思考"。
    '''

    def __init__(self, per_run: int = 300) -> None:
        self.per_run = per_run
        self._lock = threading.Lock()
        self._buf: Dict[str, Deque[Dict[str, Any]]] = {}
        self._reason: Dict[str, List[str]] = {}
        self._seq = 0
        self._binding: Any = None
        self.kept = 0
        self.dropped = 0

    def start(self) -> None:
        '''注册 hook。'''
        self._binding = api.on('*', self._on_event, owner='http-api', background=True)

    def stop(self) -> None:
        '''注销 hook。'''
        if self._binding is not None:
            api.off(self._binding)
            self._binding = None

    @staticmethod
    def _detail(event: Any, event_type: str) -> str:
        '''从事件里抽出一句话摘要。'''
        payload = getattr(event, 'payload', None) or {}

        if event_type == 'tool:call:start':
            name = getattr(event, 'tool_name', '') or payload.get('tool_name', '')
            args = getattr(event, 'args', None) or payload.get('args')
            if args:
                try:
                    shown = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    shown = str(args)
                return f'{name} {shown[:300]}'
            return name

        if event_type == 'tool:call:finish':
            name = getattr(event, 'tool_name', '')
            result = payload.get('result')
            if result is not None:
                return f'{name} → {str(result)[:300]}'
            return name

        if event_type == 'run:finish':
            parts: List[str] = []
            if payload.get('answer'):
                parts.append(' '.join(str(payload['answer']).split())[:300])
            if payload.get('total_steps'):
                parts.append(f"steps={payload['total_steps']}")
            if str(payload.get('stopped')).lower() == 'true':
                parts.append('stopped')
            return ' | '.join(parts)

        if event_type == 'run:start':
            return str(payload.get('user_input', ''))[:200]

        if event_type == 'agent:finish':
            total = payload.get('total_tokens')
            if total:
                return f"tokens={total}"

        for key in _TEXT_KEYS:
            value = payload.get(key) or getattr(event, key, None)
            if value:
                return ' '.join(str(value).split())[:400]
        return ''

    def _accum_reason(self, run_id: str, content: str) -> None:
        '''累积推理 token。'''
        if not content:
            return
        with self._lock:
            self._reason.setdefault(run_id, []).append(content)

    def _flush_reason(self, run_id: str) -> str:
        '''取出并清空该 run 累积的推理文本。'''
        with self._lock:
            buf = self._reason.pop(run_id, None)
        if not buf:
            return ''
        return ' '.join(''.join(buf).split())[:600]

    def _append(self, record: Dict[str, Any]) -> None:
        '''写入环形缓冲。调用方需持有锁。'''
        buf = self._buf.get(record['run_id'])
        if buf is None:
            buf = self._buf[record['run_id']] = deque(maxlen=self.per_run)
        buf.append(record)

    def _on_event(self, event: Any) -> None:
        event_type = getattr(event, 'event_type', '') or ''
        run_id = getattr(event, 'run_id', '') or ''

        # 推理 token 只累积，不发事件；等 agent:reason:finish 时合并成一条
        if event_type == 'agent:reason:step':
            self._accum_reason(run_id, str(getattr(event, 'content', '') or ''))
            return

        if event_type == 'agent:answer' or event_type.startswith('tool:assembly:'):
            with self._lock:
                self.dropped += 1
            return

        kind = _KEEP_TYPES.get(event_type)
        if kind is None or not run_id:
            with self._lock:
                self.dropped += 1
            return

        detail = self._detail(event, event_type)
        if event_type == 'agent:reason:finish':
            merged = self._flush_reason(run_id)
            if merged:
                detail = merged

        with self._lock:
            self._seq += 1
            self.kept += 1
            record = {
                'seq': self._seq,
                'run_id': run_id,
                'type': event_type,
                'kind': kind,
                'step': getattr(event, 'turn', None),
                'ts': getattr(event, 'created_at', None) or time.time(),
            }
            if detail:
                record['detail'] = detail
            self._append(record)
            if event_type in ('run:finish', 'agent:abort', 'group:error'):
                # 收尾时清掉可能残留的推理缓存
                self._reason.pop(run_id, None)

    def since(self, run_id: str, seq: int = 0) -> List[Dict[str, Any]]:
        '''取该 run 序号大于 seq 的事件。'''
        with self._lock:
            buf = self._buf.get(run_id)
            if not buf:
                return []
            return [r for r in buf if r['seq'] > seq]

    def stats(self) -> Dict[str, Any]:
        '''当前缓存概况。'''
        with self._lock:
            return {
                'runs_tracked': len(self._buf),
                'events_kept': self.kept,
                'events_dropped': self.dropped,
            }

    def size(self) -> int:
        '''已缓存的运行数。'''
        with self._lock:
            return len(self._buf)


_POOL = AgentPool()
_EVENTS = EventLog()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _EVENTS.start()
    try:
        yield
    finally:
        _EVENTS.stop()


app = FastAPI(title='Orchify HTTP API', version='0.1.0', lifespan=_lifespan)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatIn(BaseModel):
    '''提交一个任务。'''

    text: str = Field(..., min_length=1, description='用户输入')
    session_id: str = Field(
        'default',
        pattern=r'^[A-Za-z0-9_-]{1,32}$',
        description='会话标识，相同值共享上下文。只允许字母数字下划线短横线',
    )
    max_steps: int = Field(8, ge=1, le=30, description='步数上限')


class AnswerIn(BaseModel):
    '''对 ask_user 的回答。'''

    text: str = Field(..., min_length=1)


class ReasonIn(BaseModel):
    '''审批意见。'''

    reason: str = ''


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.get('/api/health')
def health() -> Dict[str, Any]:
    '''健康检查：进程活着、模型配置、会话池与事件缓存概况。'''
    return {
        'ok': True,
        'model': os.environ.get('MODEL_ID', ''),
        'base_url': os.environ.get('BASE_URL', ''),
        'sessions': _POOL.size(),
        'sessions_created': _POOL.created,
        'events': _EVENTS.stats(),
        'orchify': api.status(),
    }


@app.post('/api/chat')
def chat(body: ChatIn) -> Dict[str, Any]:
    '''提交任务，立即返回 run_id。

    ``agent.run`` 本身是非阻塞的（调度到内核 loop），所以这里不用起线程。
    '''
    agent = _POOL.get(body.session_id)
    try:
        run_id = agent.run(body.text, max_steps=body.max_steps)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if 'AlreadyActive' in name:
            raise HTTPException(
                status_code=409,
                detail='该会话已有任务在跑，等它结束再提交',
            ) from exc
        raise HTTPException(status_code=500, detail=f'{name}: {exc}') from exc
    return {
        'run_id': run_id,
        'run_id_url': _url_id(run_id),
        'session_id': body.session_id,
    }


@app.get('/api/runs')
def list_runs(status: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    '''列出运行记录，新的在前。

    注意 orchify 把「还在跑」和「已结束」分开放：``api.runs()`` 只给活跃的，
    结束的进了 ``api.history()``。前端要的是完整列表，所以两边合并。
    '''
    limit = max(1, min(limit, 100))
    live = api.runs(status=status)
    finished = api.history(limit=limit, status=status)
    rows = live + finished
    rows.sort(key=lambda r: r.get('started_at') or 0, reverse=True)
    rows = [_with_url_id(r) for r in rows[:limit]]
    return {'total': len(rows), 'active': len(live), 'items': rows}


@app.get('/api/runs/{run_id}')
def get_run(run_id: str) -> Dict[str, Any]:
    '''查单个运行的详情：状态、答案、进度、用量、预算。'''
    record = api.run(_norm_id(run_id))
    if record is None:
        raise HTTPException(status_code=404, detail=f'没有这个运行：{run_id}')
    return _with_url_id(record)


@app.get('/api/runs/{run_id}/events')
def get_events(run_id: str, since: int = 0) -> Dict[str, Any]:
    '''增量拉取执行事件。前端轮询用：记下返回的 last_seq，下次带上。'''
    real_id = _norm_id(run_id)
    items = _EVENTS.since(real_id, since)
    last = items[-1]['seq'] if items else since
    return {'run_id': real_id, 'items': items, 'last_seq': last}


@app.get('/api/approvals')
def list_approvals(run_id: Optional[str] = None) -> Dict[str, Any]:
    '''待人工确认的操作。``args`` 是 agent 想传的参数，前端要展示出来。'''
    items = api.pending_approvals(_norm_id(run_id) if run_id else None)
    items = [_with_url_id(i) for i in items]
    return {'total': len(items), 'items': items}


@app.post('/api/approvals/{approval_id}/approve')
def approve(approval_id: str, body: ReasonIn) -> Dict[str, Any]:
    '''批准。批准后 agent 才会真正执行那个工具。'''
    if not api.approve(approval_id, reason=body.reason):
        raise HTTPException(status_code=404, detail='审批项不存在或已处理')
    return {'ok': True, 'approval_id': approval_id, 'action': 'approve'}


@app.post('/api/approvals/{approval_id}/reject')
def reject(approval_id: str, body: ReasonIn) -> Dict[str, Any]:
    '''拒绝。拒绝理由会回传给 agent。'''
    if not api.reject(approval_id, reason=body.reason or 'denied'):
        raise HTTPException(status_code=404, detail='审批项不存在或已处理')
    return {'ok': True, 'approval_id': approval_id, 'action': 'reject'}


@app.post('/api/approvals/{approval_id}/answer')
def answer(approval_id: str, body: AnswerIn) -> Dict[str, Any]:
    '''回答 agent 主动提出的问题（ask_user 工具）。'''
    if not api.answer(approval_id, body.text):
        raise HTTPException(status_code=404, detail='提问不存在或已回答')
    return {'ok': True, 'approval_id': approval_id, 'action': 'answer'}


def main() -> None:
    '''启动服务。'''
    import uvicorn

    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8010'))
    print(f'orchify HTTP API -> http://{host}:{port}/docs')
    uvicorn.run(app, host=host, port=port, log_level='info')


if __name__ == '__main__':
    main()
