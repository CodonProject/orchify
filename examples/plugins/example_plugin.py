from orchify import Plugin, Middleware


class ExamplePlugin(Plugin):
    name = 'example'
    version = '1.0.0'
    description = 'Demonstrates the declarative plugin API: convention hooks, tools, scoped middleware and custom events.'
    tags = ['example']
    events = ['example:event']
    scope = 'agent'

    def on_run_start(self, event):
        print(f"[{self.name}] run started: {event.turn_id} <- {event.payload.get('user_input', '')}")

    def on_agent_finish(self, event):
        print(f"[{self.name}] answer: {event.content}")

    def on_tool_call_start(self, event):
        print(f"[{self.name}] tool '{event.tool_name}' called with {event.args}")

    @Plugin.tool('reflect')
    def reflect(self, text: str) -> str:
        '''Echoes the input back to the agent.'''
        return text

    @Plugin.hook('example:event')
    def on_custom_event(self, event):
        print(f'[{self.name}] custom event payload: {event.payload}')

    def on_load(self):
        self.log('up')
        self.middleware(self.monitor_middleware)
        self.event('example:event', payload={'from': self.id})

    async def monitor_middleware(self, kwargs, next_request):
        self.log(f"request {kwargs.get('scope')} -> model {kwargs.get('model')}")
        async for resp in next_request(kwargs):
            yield resp

    def on_unload(self):
        self.log('down')


PLUGINS = [ExamplePlugin]