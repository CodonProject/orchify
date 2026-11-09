import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchify import Agent, Tool, orchify_broker
from orchify.backend import orchify_web_backend
from orchify.embeddings import HashingEmbeddings, OpenAIEmbeddings, cosine
from orchify.multimodal import (as_data_url, content, describe, detect_mime,
                                image_part, image_parts, text_of, text_part)
from orchify.persistence import EventStore
from orchify.rag import Document, VectorIndex, index_event_store, retrieve_tool
from orchify.testing import FakeLLM, Recorder, final, wait_finish, wait_until

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.data')

_PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16


def _directory():
    '''Create a fresh directory under the ignored tests/.data tree.'''
    path = os.path.join(_DATA, f'rag-{uuid.uuid4().hex[:8]}')
    os.makedirs(path, exist_ok=True)
    return path


def _drop(path):
    '''Remove a test directory.'''
    shutil.rmtree(path, ignore_errors=True)


def _corpus():
    '''A small index with four distinguishable documents.'''
    index = VectorIndex(HashingEmbeddings(256))
    index.add_texts(
        [
            'Refund policy: customers may return items within 30 days.',
            'Shipping takes three to five business days.',
            'The API rate limit is 60 requests per minute.',
            'Our office is in Berlin and opens at nine.',
        ],
        metadatas=[{'source': 'refunds.md'}, {'source': 'shipping.md'},
                   {'source': 'api.md'}, {'source': 'office.md'}],
    )
    return index


def test_hashing_embeddings_are_deterministic():
    '''The hashing backend is stable, normalised and self-similar.'''
    backend = HashingEmbeddings(256)
    first = backend.embed_one('refund policy returns')
    second = backend.embed_one('refund policy returns')
    assert first == second
    assert len(first) == 256 and backend.dimension == 256
    assert abs(cosine(first, second) - 1.0) < 1e-9
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
    assert backend.missing_credentials() is None
    assert backend.snapshot()['name'] == 'hash'


def test_vector_index_ranks_the_relevant_document_first():
    '''A keyword query finds the passage that talks about it.'''
    index = _corpus()
    assert len(index) == 4
    hits = index.search('refund policy days', k=2)
    assert hits, 'expected at least one hit'
    assert hits[0].document.metadata['source'] == 'refunds.md'
    assert hits[0].score > hits[-1].score
    assert index.search('') == []
    assert index.search('refund', k=0)[0].document.metadata['source'] == 'refunds.md'


def test_vector_index_save_and_load_round_trip():
    '''The index persists to JSONL and reloads with identical hits.'''
    directory = _directory()
    try:
        index = _corpus()
        path = os.path.join(directory, 'index.jsonl')
        assert index.save(path) == 4
        loaded = VectorIndex.load(path, HashingEmbeddings(256))
        assert len(loaded) == 4
        assert loaded.documents[1].metadata == {'source': 'shipping.md'}
        before = [hit.as_dict() for hit in index.search('rate limit', k=2)]
        after = [hit.as_dict() for hit in loaded.search('rate limit', k=2)]
        assert before == after
        assert before[0]['metadata']['source'] == 'api.md'
        assert loaded.snapshot()['documents'] == 4
    finally:
        _drop(directory)


def test_retrieve_tool_formats_hits_and_caps_length():
    '''The tool returns scored blocks, truncates text and handles an empty index.'''
    index = _corpus()
    tool = retrieve_tool(index, top_k=2, max_chars=20)
    assert isinstance(tool, Tool)
    assert tool.name == 'retrieve'
    text = tool.execute(query='refund policy')
    assert text.startswith('[0.'), text
    assert 'refunds.md' in text
    assert '…' in text, 'long passages must be truncated'
    assert 'No matching passages.' in retrieve_tool(VectorIndex()).execute(query='anything')

    schema = tool.info['function']['parameters']
    assert list(schema['properties']) == ['query', 'k']
    assert schema['required'] == ['query']


def test_index_event_store_indexes_finished_answers():
    '''Answers journalled by an EventStore become searchable passages.'''
    directory = _directory()
    store = EventStore(directory, flush_every=0)
    try:
        rec = Recorder()
        agent = Agent(name='indexed', llm=FakeLLM([[final('The launch date is March 14.')]]))
        store.attach(orchify_broker)
        try:
            assert agent.ask('when do we launch?', max_steps=1) == 'The launch date is March 14.'
            run_id = wait_finish(rec).run_id
            wait_until(lambda: bool(list(store.events(run_id))), timeout=5.0)
        finally:
            rec.close()
            store.detach()
            orchify_broker.drain_hooks(timeout=5.0)
            store.flush()

        index = VectorIndex(HashingEmbeddings(256))
        added = index_event_store(store, index)
        assert added >= 1, list(store.events(run_id))
        hits = index.search('launch date March', k=1)
        assert 'March 14' in hits[0].document.text
        assert hits[0].document.metadata['run_id'] == run_id
        ids = [document.doc_id for document in index.documents]
        assert index_event_store(store, index, run_id=run_id) == 0
        assert [document.doc_id for document in index.documents] == ids
        still = index.search('launch date March', k=1)
        assert 'March 14' in still[0].document.text
    finally:
        _drop(directory)


def test_openai_embeddings_orders_rows_and_requires_a_key(monkeypatch):
    '''Rows are reordered by index; a missing key explains the fix.'''
    seen = {}

    def fake_post(url, headers, payload, timeout=None):
        seen['url'] = url
        seen['payload'] = payload
        seen['headers'] = headers
        return {'data': [{'index': 1, 'embedding': [0.0, 1.0]},
                         {'index': 0, 'embedding': [1.0, 0.0]}]}

    monkeypatch.setattr(orchify_web_backend, 'post_json', fake_post)
    backend = OpenAIEmbeddings(api_key='k-1', base_url='https://example.test/v1')
    vectors = backend.embed(['first', 'second'])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert backend.dimension == 2
    assert seen['payload'] == {'model': backend.model, 'input': ['first', 'second']}
    assert seen['headers']['Authorization'] == 'Bearer k-1'
    assert seen['url'] == 'https://example.test/v1/embeddings'
    assert backend.snapshot()['endpoint'].endswith('/embeddings')

    empty = OpenAIEmbeddings(api_key='', base_url='https://example.test/v1')
    assert 'API_KEY' in empty.missing_credentials()
    try:
        empty.embed(['x'])
    except RuntimeError as exc:
        assert 'API_KEY' in str(exc)
    else:
        raise AssertionError('embedding without a key must fail')


def test_multimodal_parts_build_and_describe_content():
    '''Parts are OpenAI-shaped, data URLs round-trip and images are countable.'''
    assert detect_mime(_PNG) == 'image/png'
    assert detect_mime(b'hello') == 'application/octet-stream'

    part = image_part(_PNG)
    assert part['type'] == 'image_url'
    url = part['image_url']['url']
    assert url.startswith('data:image/png;base64,')
    assert as_data_url(url) == url
    assert as_data_url('https://example.test/a.png') == 'https://example.test/a.png'

    directory = _directory()
    try:
        path = os.path.join(directory, 'chart.png')
        with open(path, 'wb') as handle:
            handle.write(_PNG)
        assert as_data_url(path).startswith('data:image/png;base64,')
        assert image_part(path, detail='low')['image_url']['detail'] == 'low'
    finally:
        _drop(directory)

    parts = content('look here', image_part(_PNG), ['and', text_part('more')])
    assert [item['type'] for item in parts] == ['text', 'image_url', 'text', 'text']
    assert text_of(parts) == 'look here\nand\nmore'
    assert len(image_parts(parts)) == 1
    assert describe(parts) == {'texts': 3, 'images': 1}
    assert describe('plain') == {'texts': 1, 'images': 0}


def test_agent_accepts_content_parts_as_user_input():
    '''A multimodal content list reaches the LLM unchanged.'''
    llm = FakeLLM([[final('I see a chart.')]])
    agent = Agent(name='vision', llm=llm)
    parts = content('what is this?', image_part(_PNG))
    assert agent.ask(parts, max_steps=1) == 'I see a chart.'
    sent = llm.calls[-1]['messages']
    user = [message for message in sent if message.get('role') == 'user'][-1]
    assert user['content'] == parts


def test_document_and_hit_serialise():
    '''Documents and hits round-trip through dicts.'''
    document = Document(text='hello', metadata={'source': 'a'}, doc_id='d1')
    assert Document.from_dict(document.as_dict()) == document
    hit = VectorIndex(HashingEmbeddings(64))
    hit.add_documents([document])
    result = hit.search('hello', k=1)[0].as_dict()
    assert result['doc_id'] == 'd1' and result['text'] == 'hello'
    assert 0.0 < result['score'] <= 1.0
