from orchify import Plugin
from orchify.llm import Response, Chunk


class CotSplitter(Plugin):
    name = 'cot_splitter'
    version = '1.0.0'
    description = ('Splits the answer stream at a separator token: everything before it '
                  'is turned into CoT reasoning, everything after it is streamed as the answer.')
    tags = ['stream', 'cot']
    scope = '*'
    require_sep = True

    def on_load(self):
        self.middleware(self.split)

    async def split(self, kwargs, next_request):
        extra = kwargs.get('extra_data') or {}
        sep = str(extra.get('sep') or 'sep:')

        buffer = ''
        started = False

        async for resp in next_request(kwargs):
            if resp.is_final:
                if not started:
                    if self.require_sep:
                        yield Response(Chunk(is_cot=True, content=buffer), is_final=False)
                        resp.final_status.content = ''
                    else:
                        resp.final_status.content = buffer
                else:
                    resp.final_status.content = buffer
                yield resp
                continue

            chunk = resp.current_chunk
            if chunk.is_cot or chunk.is_cot_end or chunk.is_assembly_tool:
                yield resp
                continue

            if started:
                if chunk.content:
                    yield Response(Chunk(is_cot=False, content=chunk.content), is_final=False)
                continue

            buffer += chunk.content
            idx = buffer.find(sep)
            if idx != -1:
                started = True
                cot = buffer[:idx]
                answer = buffer[idx + len(sep):]
                if cot:
                    yield Response(Chunk(is_cot=True, content=cot), is_final=False)
                if answer:
                    yield Response(Chunk(is_cot=False, content=answer), is_final=False)
                buffer = answer


PLUGINS = [CotSplitter]