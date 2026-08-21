from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class CallFrame:
    '''Represents a single frame in the call chain.'''
    caller: str           # e.g., 'Agent:A' or 'Group:G'
    callee: str           # e.g., 'Agent:B' or 'Group:G2'
    call_type: str        # 'agent', 'group', or 'plugin'
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
        
        chain = [self.root_caller]
        for frame in self.frames:
            chain.append(f'  -> {frame.callee}')
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