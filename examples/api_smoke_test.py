"""HTTP 层的冒烟测试：提交 → 审批 → 拒绝/批准 → 事件流 → 会话上下文。

这些检查会**真实调用模型**（每次约 1500–2000 tokens），所以放在 examples
而不是 tests 里——它验证的是「这一层接线对不对」，不是业务逻辑。

用法::

    # 另开一个终端先把服务跑起来
    python examples/serve_api.py

    # 然后
    python examples/api_smoke_test.py
    BASE=http://127.0.0.1:8010 python examples/api_smoke_test.py

退出码 0 表示全部通过。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get('BASE', 'http://127.0.0.1:8010').rstrip('/')
_FAILS: list[str] = []


def call(method: str, path: str, body: dict | None = None,
         timeout: int = 40) -> tuple[int, dict]:
    '''发一个 JSON 请求，返回 (状态码, 响应体)。'''
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode('utf-8', 'replace')
        try:
            return exc.code, json.loads(raw or '{}')
        except json.JSONDecodeError:
            return exc.code, {'detail': raw}
    except urllib.error.URLError as exc:
        print(f'  连不上 {BASE}：{exc.reason}')
        print('  先把服务跑起来：python examples/serve_api.py')
        sys.exit(2)


def check(label: str, ok: bool, detail: str = '') -> None:
    '''记录一项检查。'''
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f' —— {detail}' if detail else ''))
    if not ok:
        _FAILS.append(label)


def wait_final(run_id_url: str, want: str = 'final', tries: int = 30) -> dict:
    '''等到指定状态。want='final' 表示等到任一终态。'''
    rec: dict = {}
    for _ in range(tries):
        time.sleep(2)
        code, rec = call('GET', f'/api/runs/{run_id_url}')
        if code != 200:
            return rec
        status = rec.get('status')
        if status == want or (want == 'final' and status in ('done', 'error', 'stopped')):
            return rec
    return rec


def main() -> int:
    '''跑完所有检查。'''
    print(f'目标：{BASE}')

    print('\n[1] 健康检查')
    code, health = call('GET', '/api/health')
    check('GET /api/health 返回 200', code == 200, str(health.get('model', '')))

    print('\n[2] 提交任务并触发审批')
    code, sub = call('POST', '/api/chat', {
        'text': '用 save_note 把 smoke test 写入 s1.md，只做这一件事',
        'session_id': 'smoke1',
        'max_steps': 3,
    })
    check('POST /api/chat 返回 200', code == 200)
    rid_url = sub.get('run_id_url', '')
    check('返回 run_id_url 且不含 #', rid_url != '' and '#' not in rid_url, rid_url)

    rec = wait_final(rid_url, 'awaiting_approval')
    check('进入 awaiting_approval', rec.get('status') == 'awaiting_approval',
          str(rec.get('status')))

    print('\n[3] 审批列表')
    code, approvals = call('GET', '/api/approvals')
    items = approvals.get('items') or []
    check('有待审批项', len(items) > 0, f'{len(items)} 项')
    first = items[0] if items else {}
    check('带 tool_name', bool(first.get('tool_name')), str(first.get('tool_name')))
    check('args 是完整 dict', isinstance(first.get('args'), dict),
          json.dumps(first.get('args'), ensure_ascii=False)[:60])

    print('\n[4] 拒绝后不应写盘')
    aid = first.get('approval_id', '')
    code, _ = call('POST', f'/api/approvals/{aid}/reject', {'reason': 'smoke 测试拒绝'})
    check('POST reject 返回 200', code == 200)
    rec = wait_final(rid_url)
    check('运行正常结束', rec.get('status') == 'done', str(rec.get('status')))
    check('answer 说明被拒绝', '拒绝' in str(rec.get('answer')) or 'denied' in str(rec.get('answer')))
    notes = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notes')
    check('s1.md 未生成', not os.path.exists(os.path.join(notes, 's1.md')))

    print('\n[5] 事件流已筛洗')
    code, events = call('GET', f'/api/runs/{rid_url}/events?since=0')
    items = events.get('items') or []
    check('事件条数在 5–20 之间（碎片已合并）', 5 <= len(items) <= 20, f'{len(items)} 条')
    kinds = {e.get('kind') for e in items}
    check('含 tool 与 approval 类事件', {'tool', 'approval'} <= kinds, str(sorted(kinds)))
    check('带 last_seq', isinstance(events.get('last_seq'), int))

    print('\n[6] 批准后应真的写盘')
    code, sub2 = call('POST', '/api/chat', {
        'text': '用 save_note 把 approved 写入 s2.md，只做这一件事',
        'session_id': 'smoke2',
        'max_steps': 3,
    })
    rid2 = sub2.get('run_id_url', '')
    wait_final(rid2, 'awaiting_approval')
    code, approvals = call('GET', '/api/approvals')
    items = [i for i in (approvals.get('items') or []) if i.get('run_id_url') == rid2]
    check('找到该运行的审批项', len(items) == 1)
    if items:
        code, _ = call('POST', f"/api/approvals/{items[0]['approval_id']}/approve",
                       {'reason': 'smoke 测试批准'})
        check('POST approve 返回 200', code == 200)
    rec2 = wait_final(rid2)
    check('运行结束', rec2.get('status') == 'done', str(rec2.get('status')))
    target = os.path.join(notes, 's2.md')
    exists = os.path.exists(target)
    check('s2.md 已生成', exists)
    if exists:
        with open(target, encoding='utf-8') as fh:
            content = fh.read()
        check('内容正确', 'approved' in content, repr(content[:40]))

    print('\n[7] 同一会话保留上下文')
    code, a = call('POST', '/api/chat', {'text': '记住数字 9183，只回复"好"',
                                         'session_id': 'smoke3', 'max_steps': 2})
    wait_final(a.get('run_id_url', ''))
    code, b = call('POST', '/api/chat', {'text': '刚才那个数字是多少？只回答数字',
                                         'session_id': 'smoke3', 'max_steps': 2})
    rec3 = wait_final(b.get('run_id_url', ''))
    check('记得上一轮的内容', '9183' in str(rec3.get('answer')), str(rec3.get('answer'))[:40])

    print('\n[8] 运行列表合并活跃与历史')
    code, listing = call('GET', '/api/runs?limit=10')
    check('返回 200 且有记录', code == 200 and listing.get('total', 0) > 0,
          f"total={listing.get('total')}")
    check('每条都有 run_id_url',
          all('run_id_url' in r for r in listing.get('items') or []))

    print('\n' + '=' * 46)
    if _FAILS:
        print(f'失败 {len(_FAILS)} 项：' + '、'.join(_FAILS))
        return 1
    print('全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
