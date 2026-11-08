"""用 Orchify 搭一个最小的文献综述 agent。

它证明一件事：不碰知网，只靠公开的 arXiv API，也能做出"检索 → 结构化抽取 → 对比表"
的闭环。老师那个平台真正值钱的部分在这里，不在抓取。

用法：
    python examples/lit_review.py "attention sink hybrid linear attention"
    python examples/lit_review.py "graph neural network drug discovery" --limit 10

产物：终端输出 + 当前目录下的 lit_review_<slug>.md
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

import httpx

from orchify import Agent, Tool

ARXIV_API = 'https://export.arxiv.org/api/query'
ATOM = {'a': 'http://www.w3.org/2005/Atom'}
UA = 'orchify-lit-review/0.1 (+https://github.com/CodonProject)'

#: 最近一次检索的原始条目，供第二个工具按 id 取摘要
_CACHE: dict[str, dict] = {}


def _query(search_query: str, max_results: int) -> list[dict]:
    """调 arXiv API 并解析成扁平结构。"""
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': str(max_results),
        'sortBy': 'relevance',
        'sortOrder': 'descending',
    }
    resp = httpx.get(
        ARXIV_API,
        params=params,
        headers={'User-Agent': UA},
        timeout=30.0,
        follow_redirects=True,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    out = []
    for entry in root.findall('a:entry', ATOM):
        raw_id = (entry.findtext('a:id', default='', namespaces=ATOM) or '').strip()
        arxiv_id = raw_id.rsplit('/abs/', 1)[-1]
        if not arxiv_id:
            continue
        title = ' '.join((entry.findtext('a:title', default='', namespaces=ATOM) or '').split())
        summary = ' '.join((entry.findtext('a:summary', default='', namespaces=ATOM) or '').split())
        published = (entry.findtext('a:published', default='', namespaces=ATOM) or '')[:10]
        authors = [
            (a.findtext('a:name', default='', namespaces=ATOM) or '').strip()
            for a in entry.findall('a:author', ATOM)
        ]
        out.append({
            'id': arxiv_id,
            'title': title,
            'year': published[:4],
            'published': published,
            'authors': authors,
            'abstract': summary,
        })
    return out


def arxiv_search(query: str, limit: int = 8) -> str:
    '''在 arXiv 上检索论文，返回标题、年份、作者和摘要片段。

    Args:
        query (str): 检索词。支持 arXiv 语法，例如 "all:attention sink" 或
            "ti:retrieval augmented generation"。留空则按 all: 匹配。
        limit (int): 返回条数，1~20。

    Returns:
        str: 每条一行，格式为 "[序号] 标题 | 年份 | arXiv id | 作者 | 摘要前 400 字"。
    '''
    limit = max(1, min(int(limit), 20))
    q = query.strip()
    if ':' not in q.split()[0]:
        q = f'all:{q}'
    items = _query(q, limit)
    if not items:
        return '没有检索到结果，换关键词再试。'

    for i, it in enumerate(items, 1):
        _CACHE[it['id']] = it
        _CACHE[str(i)] = it

    lines = []
    for i, it in enumerate(items, 1):
        authors = ', '.join(it['authors'][:3]) + (' 等' if len(it['authors']) > 3 else '')
        lines.append(
            f"[{i}] {it['title']} | {it['year']} | {it['id']} | {authors}\n"
            f"    摘要: {it['abstract'][:400]}"
        )
    lines.append(f'\n（共 {len(items)} 条。要读全文摘要，把 id 传给 arxiv_abstract）')
    return '\n'.join(lines)


def arxiv_abstract(arxiv_id: str) -> str:
    '''按 arXiv id 或检索序号读取某篇论文的完整摘要。

    Args:
        arxiv_id (str): arXiv 标识，如 2410.10781；也可直接传检索结果里的序号，如 "3"。

    Returns:
        str: 完整摘要与元信息。
    '''
    key = str(arxiv_id).strip()
    it = _CACHE.get(key)
    if it is None:
        items = _query(f'id:{key}', 1)
        it = items[0] if items else None
    if it is None:
        return f'找不到 {key}，先调用 arxiv_search 拿到有效 id。'
    return (
        f"标题: {it['title']}\n"
        f"年份: {it['year']}   arXiv: {it['id']}\n"
        f"作者: {', '.join(it['authors'])}\n"
        f"摘要: {it['abstract']}"
    )


SYSTEM = """你是文献综述助手，服务对象是要快速进入一个新方向的科研人员。

工作方式：
1. 先用 arxiv_search 检索，必要时换 2~4 组不同关键词（同义词、上位词、方法名）。
2. 对最相关的 3~5 篇，用 arxiv_abstract 读完整摘要。
3. 输出结构化综述。

硬约束：
- 只能使用检索到的内容。摘要里没有的信息写"未提及"，绝不推测。
- 每个论断后面标 arXiv id，例如 (2410.10781)。
- 不评价论文好坏到"创新性强"这种空话，只写可核验的方法、数据、结论。
"""

TASK = """研究主题：{topic}

请完成：
一、检索概况：用了哪些关键词、各命中多少条、最终留下的 N 篇及筛选理由（一句话）。
二、对比表（Markdown）：每篇一行，列为 | 文献 | 年份 | 方法核心 | 数据/实验设置 | 主要结论 | 局限 |
三、共同点与分歧：这批工作在什么问题上一致，在什么问题上结论相反。
四、三条研究空白：每条要具体到"缺什么实验/缺什么数据"，不要写"值得进一步研究"。
五、如果这批文献不足以支撑某个判断，直接说明证据不足。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('topic', help='研究主题')
    ap.add_argument('--limit', type=int, default=8, help='每次检索的条数')
    ap.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()

    agent = Agent.from_env(
        name='LitReview',
        system_prompt=SYSTEM,
        tools=[
            Tool(func=arxiv_search, name='arxiv_search',
                 description='在 arXiv 检索论文，返回标题/年份/作者/摘要片段。'),
            Tool(func=arxiv_abstract, name='arxiv_abstract',
                 description='按 arXiv id 或检索序号读取完整摘要。'),
        ],
    )
    # 让检索条数受命令行控制
    limit_hint = f'\n\n（调用 arxiv_search 时，limit 一律传 {args.limit}。）'
    answer = agent.ask(
        TASK.format(topic=args.topic) + limit_hint,
        max_steps=14,
        timeout=args.timeout,
    )

    print('\n' + '=' * 70)
    print(answer)

    slug = re.sub(r'[^\w\u4e00-\u9fff]+', '_', args.topic)[:60].strip('_') or 'review'
    path = os.path.join(os.getcwd(), f'lit_review_{slug}.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f'# 文献综述：{args.topic}\n\n')
        fh.write(f'> 由 orchify 自动生成，检索源 arXiv。共缓存 {len(_CACHE) // 2} 条。\n\n')
        fh.write(answer + '\n')
    print(f'\n已保存: {path}')


if __name__ == '__main__':
    main()
