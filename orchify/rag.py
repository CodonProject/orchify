"""Retrieval-augmented generation: an in-memory vector index plus a tool.

The index is deliberately small: documents, their embeddings, and a cosine
search. It persists as JSON Lines so a knowledge base can be checked in, shipped
with an app or rebuilt from a run journal::

    from orchify import Agent, EventStore
    from orchify.embeddings import HashingEmbeddings
    from orchify.rag import VectorIndex, retrieve_tool

    index = VectorIndex(HashingEmbeddings())
    index.add_texts(['Q3 revenue grew 12%', 'churn fell to 3.1%'],
                    metadatas=[{'source': 'report.pdf'}, {'source': 'deck.pptx'}])
    agent = Agent(tools=[retrieve_tool(index)])

Swap :class:`orchify.embeddings.HashingEmbeddings` for
:class:`orchify.embeddings.OpenAIEmbeddings` to get semantic retrieval; nothing else
changes.
"""

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from orchify.embeddings import Embeddings, HashingEmbeddings
from orchify.tool import Tool
from orchify.utils import get_logger

logger = get_logger(__name__)

#: Event types indexed by :func:`index_event_store`. ``run:finish`` carries the
#: authoritative answer, while ``tool:call:finish`` keeps tool results searchable.
DEFAULT_INDEXED_TYPES = ('agent:answer', 'agent:finish', 'run:finish', 'tool:call:finish')


@dataclass
class Document:
    '''One indexed passage.

    Attributes:
        text (str): The passage itself.
        metadata (Dict[str, Any]): Free-form labels (``source``, ``url``, ``ts``…).
        doc_id (str): Stable identifier; generated from the position when empty.
    '''

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ''

    def as_dict(self) -> Dict[str, Any]:
        '''Return the document without its vector.'''
        return {'doc_id': self.doc_id, 'text': self.text, 'metadata': dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        '''Rebuild a document from :meth:`as_dict` output.'''
        return cls(text=str(data.get('text') or ''),
                   metadata=dict(data.get('metadata') or {}),
                   doc_id=str(data.get('doc_id') or ''))


@dataclass
class Hit:
    '''One search result.

    Attributes:
        document (Document): The matched passage.
        score (float): Similarity, higher is better.
    '''

    document: Document
    score: float

    def as_dict(self) -> Dict[str, Any]:
        '''Return the hit as a JSON-serialisable dict.'''
        return {'doc_id': self.document.doc_id, 'text': self.document.text,
                'metadata': dict(self.document.metadata), 'score': round(self.score, 6)}


class VectorIndex:
    '''In-memory vector index with JSONL persistence.

    Args:
        embeddings (Optional[Embeddings]): Backend; hashing embeddings by default.
        min_score (float): Results below this cosine similarity are dropped.
    '''

    version = 1

    def __init__(self, embeddings: Optional[Embeddings] = None, *, min_score: float = 0.0):
        self.embeddings = embeddings or HashingEmbeddings()
        self.min_score = float(min_score)
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []
        # Cached L2 norms, parallel to ``vectors``; search would otherwise
        # recompute every stored norm on every query.
        self._norms: List[float] = []

    def __len__(self) -> int:
        '''Number of indexed documents.'''
        return len(self.documents)

    def clear(self) -> None:
        '''Drop every document and vector.'''
        self.documents.clear()
        self.vectors.clear()
        self._norms.clear()

    def _ensure_norms(self) -> None:
        '''Recompute the cached vector norms when the index changed outside add_texts.'''
        if len(self._norms) == len(self.vectors):
            return
        self._norms = [math.sqrt(sum(value * value for value in vector)) or 0.0
                       for vector in self.vectors]

    def add_texts(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None,
                  ids: Optional[Sequence[str]] = None) -> int:
        '''Embed and index texts.

        Args:
            texts (Sequence[str]): Passages to index.
            metadatas (Optional[Sequence[Dict[str, Any]]]): One metadata dict per text.
            ids (Optional[Sequence[str]]): One id per text; generated when omitted.

        Returns:
            int: How many documents were added; a document whose id is already
                indexed is updated in place and not counted.
        '''
        items = [str(text or '') for text in texts or []]
        if not items:
            return 0
        vectors = self.embeddings.embed(items)
        if len(vectors) != len(items):
            raise RuntimeError(
                f'embeddings backend returned {len(vectors)} vector(s) for {len(items)} text(s)'
            )
        metas = list(metadatas or [])
        supplied = list(ids or [])
        positions = {document.doc_id: position
                     for position, document in enumerate(self.documents)}
        added = 0
        for offset, (text, vector) in enumerate(zip(items, vectors)):
            meta = dict(metas[offset]) if offset < len(metas) else {}
            doc_id = (str(supplied[offset]) if offset < len(supplied)
                      else f'doc-{len(self.documents):06d}')
            norm = math.sqrt(sum(value * value for value in vector)) or 0.0
            position = positions.get(doc_id)
            if position is None:
                positions[doc_id] = len(self.documents)
                self.documents.append(Document(text=text, metadata=meta, doc_id=doc_id))
                self.vectors.append(list(vector))
                self._norms.append(norm)
                added += 1
            else:
                self.documents[position] = Document(text=text, metadata=meta, doc_id=doc_id)
                self.vectors[position] = list(vector)
                self._norms[position] = norm
        return added

    def add_documents(self, documents: Iterable[Document]) -> int:
        '''Index already-built documents.

        Args:
            documents (Iterable[Document]): Documents to add.

        Returns:
            int: How many documents were added.
        '''
        items = list(documents or [])
        if not items:
            return 0
        return self.add_texts([item.text for item in items],
                              [item.metadata for item in items],
                              [item.doc_id for item in items])

    def search(self, query: str, k: int = 4) -> List[Hit]:
        '''Return the ``k`` passages most similar to ``query``.

        Args:
            query (str): Natural-language query.
            k (int): Maximum number of hits.

        Returns:
            List[Hit]: Hits ordered by descending score.
        '''
        limit = max(1, int(k or 1))
        if not self.documents or not query:
            return []
        vector = self.embeddings.embed_one(query)
        self._ensure_norms()
        query_norm = math.sqrt(sum(value * value for value in vector)) or 0.0
        scored: List[Any] = []
        for index, stored in enumerate(self.vectors):
            stored_norm = self._norms[index]
            denominator = query_norm * stored_norm
            if denominator <= 0.0:
                continue
            score = sum(left * right for left, right in zip(vector, stored)) / denominator
            if score > self.min_score:
                scored.append((score, index))
        # Keep only the top ``k`` pairs; materialise Hit objects afterwards.
        best = heapq.nlargest(limit, scored)
        return [Hit(document=self.documents[index], score=score) for score, index in best]

    def snapshot(self) -> Dict[str, Any]:
        '''Return index metadata for reports.'''
        return {'documents': len(self.documents), 'dimension': self.embeddings.dimension,
                'embeddings': self.embeddings.name, 'min_score': self.min_score}

    def save(self, path: str) -> int:
        '''Write the index to ``path`` as JSON Lines.

        Args:
            path (str): Destination file; parent directories are created.

        Returns:
            int: Number of documents written.
        '''
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'version': self.version,
                'embeddings': self.embeddings.name,
                'dimension': len(self.vectors[0]) if self.vectors else 0,
                'documents': len(self.documents),
            }) + '\n')
            for document, vector in zip(self.documents, self.vectors):
                handle.write(json.dumps({'doc_id': document.doc_id, 'text': document.text,
                                         'metadata': document.metadata, 'vector': vector}) + '\n')
        return len(self.documents)

    @classmethod
    def load(cls, path: str, embeddings: Optional[Embeddings] = None,
             *, min_score: float = 0.0) -> 'VectorIndex':
        '''Read an index written by :meth:`save`.

        Args:
            path (str): JSONL file to read.
            embeddings (Optional[Embeddings]): Backend for future queries; hashing
                embeddings when omitted.
            min_score (float): Minimum cosine similarity for hits.

        Returns:
            VectorIndex: The loaded index.
        '''
        index = cls(embeddings, min_score=min_score)
        with open(path, 'r', encoding='utf-8') as handle:
            for number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    logger.warning('rag: skipping damaged line %d in %s', number + 1, path)
                    continue
                if number == 0 and 'vector' not in record:
                    continue
                index.documents.append(Document(text=str(record.get('text') or ''),
                                                metadata=dict(record.get('metadata') or {}),
                                                doc_id=str(record.get('doc_id') or '')))
                index.vectors.append([float(value) for value in record.get('vector') or []])
        if index.vectors and index.embeddings.dimension == 0:
            index.embeddings.dimension = len(index.vectors[0])
        return index


def _hit_source(document: Document) -> str:
    '''Best human label for a document: source, title, url or id.'''
    for key in ('source', 'title', 'url', 'run_id'):
        value = document.metadata.get(key)
        if value:
            return str(value)
    return document.doc_id or 'document'


def _format_hits(hits: Sequence[Hit], max_chars: int) -> str:
    '''Render hits as compact blocks for a tool result.'''
    blocks = []
    for hit in hits:
        text = hit.document.text
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + '…'
        blocks.append(f'[{hit.score:.3f}] {_hit_source(hit.document)}\n{text}')
    return '\n\n'.join(blocks)


def retrieve_tool(index: VectorIndex, *, name: str = 'retrieve',
                  description: Optional[str] = None, top_k: int = 4,
                  max_chars: int = 1200, **tool_kwargs: Any) -> Tool:
    '''Build a ``retrieve(query, k)`` tool over an index.

    The model calls it whenever it needs facts it does not have; the result lists
    the best passages with their score and source.

    Args:
        index (VectorIndex): Index to search.
        name (str): Tool name the model sees.
        description (Optional[str]): Tool description; a sensible default is used.
        top_k (int): Default number of passages per call.
        max_chars (int): Per-passage character cap in the result.
        **tool_kwargs (Any): Passed to :class:`orchify.tool.Tool` (cache, timeout…).

    Returns:
        Tool: The retrieval tool.
    '''

    def retrieve(query: str, k: int = top_k) -> str:
        '''Search the knowledge base for passages relevant to a query.

        Args:
            query (str): What to look for, in natural language.
            k (int): How many passages to return.

        Returns:
            str: The matching passages, each prefixed with its score and source.
        '''
        hits = index.search(query, k or top_k)
        if not hits:
            return 'No matching passages.'
        return _format_hits(hits, max_chars)

    return Tool(
        retrieve,
        name=name,
        description=description or (
            'Search the indexed knowledge base. Use it whenever the answer needs '
            'facts, quotes or numbers you do not already have.'
        ),
        **tool_kwargs,
    )


def _record_text(record: Dict[str, Any]) -> str:
    '''Extract the searchable text of one event journal record.'''
    payload = record.get('payload') or {}
    if not isinstance(payload, dict):
        return ''
    for key in ('answer', 'content', 'reasoning', 'result', 'text'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def index_event_store(store: Any, index: VectorIndex, *, run_id: Optional[str] = None,
                      event_types: Optional[Sequence[str]] = None) -> int:
    '''Index the interesting events of an :class:`orchify.EventStore`.

    Args:
        store (Any): A store exposing ``runs()`` and ``events(run_id)``.
        index (VectorIndex): Index to fill.
        run_id (Optional[str]): Only index one run; every run when omitted.
        event_types (Optional[Sequence[str]]): Journal types to index.

    Returns:
        int: How many passages were added. Indexing the same journal twice
        updates the existing passages instead of duplicating them.
    '''
    wanted = tuple(event_types or DEFAULT_INDEXED_TYPES)
    run_ids = [run_id] if run_id else list(store.runs())
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    for current in run_ids:
        position = 0
        for record in store.events(current):
            if record.get('type') not in wanted:
                continue
            text = _record_text(record)
            if not text:
                continue
            texts.append(text)
            metadatas.append({'run_id': current, 'type': record.get('type'),
                              'ts': record.get('ts')})
            # The journal is append-only, so the nth kept event of a run keeps
            # its identity across re-indexing.
            ids.append(f'{current}:{position}')
            position += 1
    return index.add_texts(texts, metadatas, ids)


__all__ = ['Document', 'Hit', 'VectorIndex', 'retrieve_tool', 'index_event_store',
           'DEFAULT_INDEXED_TYPES']
