import re

from orchify import Plugin

HEADER_RE = re.compile(r'(?m)^\*\*([^*]+)\*\*[ \t]*\n')


class CotBlocks(Plugin):
    name = 'cot_blocks'
    version = '1.0.0'
    description = ('Injects a default structured CoT format and splits the streamed reasoning '
                  'into **标题** blocks, emitting one `cot:block` event per completed block '
                  'and a `cot:finish` event when the reasoning ends.')
    tags = ['cot', 'stream']
    scope = '*'
    events = ['cot:block', 'cot:finish']
    auto_inject = True

    default_format = '''**定义核心概念**
我需要先锚定<主题>的准确定义。但要先厘清其数学基础，再提及物理应用；同时必须小心，TA 可能将<主题>与相近概念混淆，因此要先区分，再展开。

**构建解释框架**
我计划按以下顺序组织回答：定义核心概念 -> 关键公式 -> 直觉解释 -> 具体例子。这样既覆盖本质，又自然引出应用；如果直接跳入公式，TA 可能被符号淹没，所以会先用直觉语言铺垫，再给出关键公式。'''


    def on_load(self):
        if self.auto_inject:
            self.middleware(self.inject_format)

    # ---------- hooks ----------

    @Plugin.hook('run:start')
    def on_run_start(self, event):
        st = self.state.setdefault('_cot_blocks', {}).setdefault(event.turn_id, {})
        st.update({'raw': '', 'title': None, 'content_start': 0, 'flushed': False})

    @Plugin.hook('agent:reason:step')
    def on_reason_step(self, event):
        self._feed(event, event.content)

    @Plugin.hook('agent:reason:finish')
    def on_reason_finish(self, event):
        self._flush(event)

    @Plugin.hook('run:finish')
    def on_run_finish(self, event):
        self._flush(event)

    # ---------- block parsing ----------

    def _feed(self, event, text):
        st = self.state.get('_cot_blocks', {}).get(event.turn_id)
        if st is None or st['flushed']:
            return
        st['raw'] += text
        pos = st['content_start']
        while True:
            m = HEADER_RE.search(st['raw'], pos)
            if m is None:
                break
            if st['title'] is not None:
                content = st['raw'][st['content_start']:m.start()]
                self._emit_block(event, st['title'], content)
            st['title'] = m.group(1).strip()
            st['content_start'] = m.end()
            pos = m.end()

    def _flush(self, event):
        st = self.state.get('_cot_blocks', {}).get(event.turn_id)
        if st is None or st['flushed']:
            return
        if st['title'] is not None:
            content = st['raw'][st['content_start']:]
            self._emit_block(event, st['title'], content)
        st['flushed'] = True
        ev = self.event('cot:finish', payload={'title': st['title']},
                        turn_id=event.turn_id, run_id=event.run_id, turn=event.turn,
                        agent_name=event.agent_name, agent_code=event.agent_code)
        self.broker.emit(ev)

    def _emit_block(self, event, title, content):
        ev = self.event('cot:block', payload={'title': title, 'content': content},
                        turn_id=event.turn_id, run_id=event.run_id, turn=event.turn,
                        agent_name=event.agent_name, agent_code=event.agent_code)
        self.broker.emit(ev)

    # ---------- format injection middleware ----------

    async def inject_format(self, kwargs, next_request):
        msgs = list(kwargs.get('messages') or [])
        instruction = ('请把你的推理思考过程组织成以下格式。每个标题单独占一行，'
                       '用 **标题** 包裹。在标题下写出该部分内容：\n' + self.default_format)
        if msgs and msgs[0].get('role') == 'system':
            msgs.insert(1, {'role': 'system', 'content': instruction})
        else:
            msgs.insert(0, {'role': 'system', 'content': instruction})
        kwargs['messages'] = msgs
        async for resp in next_request(kwargs):
            yield resp


PLUGINS = [CotBlocks]