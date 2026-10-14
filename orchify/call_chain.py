from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class CallFrame:
    '''Represents a single frame in the call chain.'''

    caller: str
    callee: str
    call_type: str
    input_data: str = ''
    output_data: str = ''
    timestamp: float = field(default_factory=time.time)


@dataclass
class CallChain:
    '''Tracks the full call chain for a task execution.'''

    root_caller: str
    frames: List[CallFrame] = field(default_factory=list)

    def add_frame(self, caller: str, callee: str, call_type: str, input_data: str = '') -> CallFrame:
        '''Add a new frame to the call chain.'''
        frame = CallFrame(
            caller=caller,
            callee=callee,
            call_type=call_type,
            input_data=input_data
        )
        self.frames.append(frame)
        return frame

    def get_chain_str(self) -> str:
        '''Returns a human-readable string of the call chain.'''
        if not self.frames:
            return f'{self.root_caller} (no calls)'

        chain = [self.root_caller] + [frame.callee for frame in self.frames]
        return ' -> '.join(chain)

    def to_dict(self) -> Dict[str, Any]:
        '''Serialize to dictionary.'''
        return {
            'root_caller': self.root_caller,
            'frames': [
                {
                    'caller': f.caller,
                    'callee': f.callee,
                    'call_type': f.call_type,
                    'input_data': f.input_data,
                    'output_data': f.output_data,
                    'timestamp': f.timestamp
                }
                for f in self.frames
            ]
        }

    def get_last_frame(self) -> Optional[CallFrame]:
        '''Get the most recent call frame.'''
        return self.frames[-1] if self.frames else None

    def get_depth(self) -> int:
        '''Get the current depth of the call chain.'''
        return len(self.frames)

    def contains(self, entity_name: str) -> bool:
        '''Check if an entity is in the call chain.'''
        if self.root_caller == entity_name:
            return True
        return any(f.caller == entity_name or f.callee == entity_name for f in self.frames)


class CallBlocked(RuntimeError):
    '''Raised when an outgoing call would exceed the depth cap or form a cycle.

    The structured attributes are kept separate from the message so callers can
    emit an event before re-raising.

    Attributes:
        reason (str): Either ``'depth'`` or ``'cycle'``.
        caller (str): Label of the entity that tried to call out.
        callee (str): Label of the entity that was about to be called.
        path (List[str]): Call path at the moment the call was blocked.
        limit (Optional[int]): Depth cap in force, when the reason is ``'depth'``.
    '''

    def __init__(
        self,
        reason: str,
        caller: str,
        callee: str,
        path: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.reason = reason
        self.caller = caller
        self.callee = callee
        self.path = list(path or [])
        self.limit = limit
        if reason == 'depth':
            message = (
                f"Call blocked: depth cap reached ({limit}) calling '{callee}' "
                f"from '{caller}' (path: {' -> '.join(self.path)}). "
                f"Raise ORCHIFY_MAX_CALL_DEPTH (or max_call_depth=...) to allow more."
            )
        else:
            message = (
                f"Call blocked: '{callee}' is already on the call path "
                f"({' -> '.join(self.path)}), which would create a call cycle."
            )
        super().__init__(message)


def plan_call(
    owner_label: str,
    path: Optional[List[str]],
    callee_label: str,
    limit: Optional[int] = None,
) -> List[str]:
    '''Return the child call path for one outgoing call, or raise CallBlocked.

    Args:
        owner_label (str): Label of the calling entity, e.g. ``'Agent:writer'``.
        path (Optional[List[str]]): Call path of the calling entity; its own
            label is appended when missing.
        callee_label (str): Label of the entity about to be called.
        limit (Optional[int]): Maximum call depth; ``0`` or ``None`` disables the
            depth cap.

    Returns:
        List[str]: The call path inherited by the callee. The input path is
        never mutated, so a sequential repeat (A calls B, then calls B again)
        is not a cycle while a nested repeat (A -> B -> A) is.

    Raises:
        CallBlocked: If the call would exceed ``limit`` or close a cycle.
    '''
    base = list(path) if path else [owner_label]
    if base[-1] != owner_label:
        base.append(owner_label)
    if limit and len(base) > limit:
        raise CallBlocked('depth', owner_label, callee_label, base, limit)
    if callee_label in base:
        raise CallBlocked('cycle', owner_label, callee_label, base, limit)
    return base + [callee_label]
