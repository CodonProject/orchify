'''HTTP 服务示例：把 orchify 包成一个文献调研助手。

这一层刻意做得很薄，生产可用的部分（会话池、事件缓冲、SSE、鉴权、配额、
``/metrics``）都在 :mod:`orchify.server` 里，这里只提供"这个 demo 用哪个
agent、装哪些工具"。

启动::

    python examples/serve_api.py                 # 默认 127.0.0.1:8010
    HOST=0.0.0.0 PORT=8010 python examples/serve_api.py
    ORCHIFY_API_KEYS=dev-key python examples/serve_api.py    # 打开鉴权

前端契约见 docs/api-contract.md。

依赖：fastapi + uvicorn（``pip install "orchify[web]"``）。orchify 本身仍只依赖 httpx。
'''

import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, Tool  # noqa: E402
from orchify.server import create_app, serve  # noqa: E402
from orchify.telemetry import Metrics  # noqa: E402

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
    Tool(func=arxiv_search, name='arxiv_search', description='在 arXiv 检索论文。',
         retry=2, backoff=0.5),
    Tool(func=arxiv_abstract, name='arxiv_abstract', description='读取某篇 arXiv 论文的摘要。',
         retry=2, backoff=0.5),
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


def build_agent(session_id: str) -> Agent:
    '''为某个会话构造 agent。

    Args:
        session_id (str): 会话标识，同时作为 agent 名字。

    Returns:
        Agent: 已装好工具与系统提示词的 agent。
    '''
    return Agent.from_env(name=session_id, system_prompt=_SYSTEM_PROMPT, tools=list(_TOOLS))


#: 每个部署一个 metrics 收集器；``/metrics`` 与事件钩子在 app lifespan 里挂载。
METRICS = Metrics(prefix='orchify_demo')

app = create_app(
    agent_factory=build_agent,
    metrics=METRICS,
    poll_per_run=300,
    title='Orchify 文献调研 API',
)


def main() -> None:
    '''启动服务。'''
    serve(app)


if __name__ == '__main__':
    main()
