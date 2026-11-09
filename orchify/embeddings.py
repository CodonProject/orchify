"""Text embeddings for retrieval-augmented tools.

Two implementations ship with Orchify:

* :class:`HashingEmbeddings` — deterministic, dependency-free, no network. Good
  enough for keyword-ish retrieval, tests and offline demos.
* :class:`OpenAIEmbeddings` — any OpenAI-compatible ``/embeddings`` endpoint.

Both are small objects with one method, ``embed(texts) -> list[list[float]]``, so a
custom backend only has to implement that::

    class MyEmbeddings(Embeddings):
        dimension = 1024
        def embed(self, texts):
            return [my_model.encode(t) for t in texts]
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from orchify.backend import orchify_web_backend
from orchify.env import req_key, req_url
from orchify.utils import get_logger

logger = get_logger(__name__)

#: Words, numbers and single CJK/kana/hangul characters.
_TOKEN_RE = re.compile(r'[A-Za-z0-9_]+|[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]')

#: Upper bound on the token -> bucket cache kept by :class:`HashingEmbeddings`.
_MAX_TOKEN_CACHE = 20000

DEFAULT_OPENAI_MODEL = 'text-embedding-3-small'


class Embeddings:
    '''Interface implemented by every embedding backend.

    Attributes:
        name (str): Backend name used in reports.
        dimension (int): Vector length; 0 until the backend knows it.
    '''

    name = 'embeddings'
    dimension = 0

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        '''Embed a batch of texts.

        Args:
            texts (Sequence[str]): Texts to embed.

        Returns:
            List[List[float]]: One vector per input, in order.
        '''
        raise NotImplementedError('Embedding backends must implement embed()')

    def embed_one(self, text: str) -> List[float]:
        '''Embed a single text.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: The embedding vector.
        '''
        vectors = self.embed([text])
        return vectors[0] if vectors else []

    def missing_credentials(self) -> Optional[str]:
        '''Explain how to authenticate, or None when the backend needs no key.'''
        return None

    def snapshot(self) -> Dict[str, Any]:
        '''Return backend metadata for reports.'''
        return {'name': self.name, 'dimension': self.dimension}


class HashingEmbeddings(Embeddings):
    '''Deterministic hashing embedding: no model, no network, no dependencies.

    Tokens are hashed into fixed buckets with a log-scaled frequency weight and the
    vector is L2-normalised, which makes cosine similarity a plain dot product and
    keeps every similarity non-negative: two passages that share a token can never
    be pushed apart by a collision. Retrieval is keyword based rather than semantic
    — use it for demos, tests and small knowledge bases, or as a fallback when no
    embedding key is configured.

    Args:
        dimension (int): Number of buckets; keep it well above the token count.
    '''

    name = 'hash'

    def __init__(self, dimension: int = 512):
        self.dimension = max(8, int(dimension))
        # token -> bucket, bounded: the same tokens recur across a corpus and
        # across queries, and blake2b per occurrence is pure overhead.
        self._buckets: Dict[str, int] = {}

    def _bucket(self, token: str) -> int:
        '''Return (and cache) the bucket a token hashes into.'''
        bucket = self._buckets.get(token)
        if bucket is None:
            if len(self._buckets) >= _MAX_TOKEN_CACHE:
                self._buckets.clear()
            digest = int.from_bytes(
                hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest(), 'big'
            )
            bucket = digest % self.dimension
            self._buckets[token] = bucket
        return bucket

    def _vector(self, text: str) -> List[float]:
        counts: Dict[str, int] = {}
        for token in _TOKEN_RE.findall((text or '').lower()):
            counts[token] = counts.get(token, 0) + 1
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            index = self._bucket(token)
            # No sign flipping: a shared token must always increase similarity, so
            # a bucket collision between unrelated tokens cannot make two related
            # passages look dissimilar.
            vector[index] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        '''Embed texts with token hashing.'''
        return [self._vector(text) for text in texts or []]


class OpenAIEmbeddings(Embeddings):
    '''Embeddings from any OpenAI-compatible ``/embeddings`` endpoint.

    Args:
        model (str): Embedding model id.
        api_key (Optional[str]): Key; defaults to ``API_KEY``/``OPENAI_API_KEY``.
        base_url (Optional[str]): API root; defaults to ``BASE_URL`` or OpenAI.
        batch_size (int): Texts per HTTP request.
        timeout (float): Per-request timeout in seconds.
        url (str): Endpoint path, for gateways that mount it elsewhere.
    '''

    name = 'openai'

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, batch_size: int = 64,
                 timeout: float = 60.0, url: str = '/embeddings'):
        self.model = model
        self.api_key = api_key if api_key is not None else req_key()
        self.base_url = (base_url or req_url('https://api.openai.com/v1')).rstrip('/')
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)
        self.url = url if url.startswith('/') else '/' + url
        self.dimension = 0

    @property
    def endpoint(self) -> str:
        '''The full embeddings URL.'''
        if self.url.startswith('http'):
            return self.url
        return f'{self.base_url}{self.url}'

    def missing_credentials(self) -> Optional[str]:
        '''Explain how to authenticate when no key is configured.'''
        if self.api_key:
            return None
        return ('Embeddings need a key: set API_KEY (or OPENAI_API_KEY) or pass '
                'api_key=... to OpenAIEmbeddings().')

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        '''Embed texts through the remote endpoint, batching as needed.'''
        items = [str(text or '') for text in texts or []]
        if not items:
            return []
        if not self.api_key:
            raise RuntimeError(self.missing_credentials())
        headers = {'Authorization': f'Bearer {self.api_key}', 'content-type': 'application/json'}
        vectors: List[List[float]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start + self.batch_size]
            payload = {'model': self.model, 'input': batch}
            data = orchify_web_backend.post_json(self.endpoint, headers, payload,
                                                 timeout=self.timeout)
            rows = sorted(data.get('data') or [], key=lambda row: row.get('index', 0))
            if len(rows) != len(batch):
                raise RuntimeError(
                    f'embeddings endpoint returned {len(rows)} vector(s) for {len(batch)} text(s)'
                )
            for row in rows:
                vectors.append([float(value) for value in row.get('embedding') or []])
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors

    def snapshot(self) -> Dict[str, Any]:
        '''Return backend metadata, including the model and endpoint.'''
        return {'name': self.name, 'dimension': self.dimension,
                'model': self.model, 'endpoint': self.endpoint}


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    '''Cosine similarity of two vectors (0.0 when either is empty).

    Args:
        left (Sequence[float]): First vector.
        right (Sequence[float]): Second vector.

    Returns:
        float: Similarity in ``[-1, 1]``.
    '''
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for index in range(size):
        a = left[index]
        b = right[index]
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if not norm_left or not norm_right:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


__all__ = ['Embeddings', 'HashingEmbeddings', 'OpenAIEmbeddings', 'cosine',
           'DEFAULT_OPENAI_MODEL']
