"""Multimodal message parts.

A message ``content`` may be a plain string or a list of parts. These helpers build
the OpenAI-style parts that every adapter understands — the OpenAI backend forwards
them, while the Anthropic and Gemini adapters translate ``image_url`` into a base64
image block / ``inlineData``::

    from orchify.multimodal import image_part, text_part

    agent.ask([text_part('What is in this chart?'), image_part('chart.png')])

Images can be a local path, raw bytes, an ``http(s)`` URL or an existing
``data:`` URL. Local files are inlined as base64 so the provider never has to
fetch them.
"""

import base64
import mimetypes
import os
from typing import Any, Dict, List, Optional, Sequence, Union

Source = Union[str, bytes, bytearray, os.PathLike]

#: Magic prefixes used to recognise common image formats.
_IMAGE_MAGIC = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
)


def detect_mime(data: bytes) -> str:
    '''Guess the MIME type of raw bytes.

    Args:
        data (bytes): File content.

    Returns:
        str: A MIME type, ``application/octet-stream`` when unknown.
    '''
    for prefix, mime in _IMAGE_MAGIC:
        if data.startswith(prefix):
            return mime
    if len(data) > 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return 'application/octet-stream'


def as_data_url(source: Source, *, mime: Optional[str] = None) -> str:
    '''Turn an image source into a ``data:`` URL.

    Args:
        source (Source): Path, bytes, ``data:`` URL or ``http(s)`` URL.
        mime (Optional[str]): Override the detected MIME type.

    Returns:
        str: A ``data:`` URL (URLs are returned unchanged).
    '''
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        kind = mime or detect_mime(data)
        return f'data:{kind};base64,' + base64.b64encode(data).decode('ascii')
    text = str(source or '')
    if text.startswith('data:'):
        return text
    if text.startswith(('http://', 'https://')):
        return text
    with open(text, 'rb') as handle:
        data = handle.read()
    kind = mime or mimetypes.guess_type(text)[0] or detect_mime(data)
    return f'data:{kind};base64,' + base64.b64encode(data).decode('ascii')


def text_part(text: str) -> Dict[str, Any]:
    '''Build a text content part.

    Args:
        text (str): The text.

    Returns:
        Dict[str, Any]: An OpenAI-style ``text`` part.
    '''
    return {'type': 'text', 'text': str(text or '')}


def image_part(source: Source, *, mime: Optional[str] = None,
               detail: Optional[str] = None) -> Dict[str, Any]:
    '''Build an image content part.

    Args:
        source (Source): Path, bytes, ``data:`` URL or ``http(s)`` URL.
        mime (Optional[str]): Override the detected MIME type for local data.
        detail (Optional[str]): Provider hint such as ``'low'`` or ``'high'``.

    Returns:
        Dict[str, Any]: An OpenAI-style ``image_url`` part.
    '''
    payload: Dict[str, Any] = {'url': as_data_url(source, mime=mime)}
    if detail:
        payload['detail'] = str(detail)
    return {'type': 'image_url', 'image_url': payload}


def content(*parts: Any) -> List[Dict[str, Any]]:
    '''Build a content list from strings and parts.

    Args:
        *parts (Any): Strings become text parts; dicts are kept as they are.

    Returns:
        List[Dict[str, Any]]: The content list.
    '''
    built: List[Dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            built.append(text_part(part))
        elif isinstance(part, dict):
            built.append(part)
        elif isinstance(part, (list, tuple)):
            built.extend(content(*part))
        else:
            raise TypeError(f'content() accepts strings, parts or lists, got {type(part)!r}')
    return built


def text_of(value: Any) -> str:
    '''Flatten message content into plain text, ignoring non-text parts.

    Args:
        value (Any): A string or a list of content parts.

    Returns:
        str: The concatenated text parts.
    '''
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return '\n'.join(
            str(part.get('text') or '') for part in value
            if isinstance(part, dict) and part.get('type') == 'text'
        )
    return '' if value is None else str(value)


def image_parts(value: Any) -> List[Dict[str, Any]]:
    '''Return only the image parts of a content value.

    Args:
        value (Any): A string or a list of content parts.

    Returns:
        List[Dict[str, Any]]: The ``image_url`` parts found.
    '''
    if not isinstance(value, (list, tuple)):
        return []
    return [part for part in value if isinstance(part, dict) and part.get('type') == 'image_url']


def describe(value: Any) -> Dict[str, int]:
    '''Count the text and image parts of a content value.

    Args:
        value (Any): A string or a list of content parts.

    Returns:
        Dict[str, int]: ``{'texts': n, 'images': m}``.
    '''
    if isinstance(value, str):
        return {'texts': 1, 'images': 0}
    texts = sum(1 for part in value or []
                if isinstance(part, dict) and part.get('type') == 'text')
    return {'texts': texts, 'images': len(image_parts(value))}


__all__ = ['as_data_url', 'content', 'describe', 'detect_mime', 'image_part',
           'image_parts', 'text_of', 'text_part']
