from dataclasses import dataclass, field
from typing import Callable, Dict, Union

import os
from RL.utils.MyRL_path import buffer_path, underline2camel
import importlib

@dataclass
class Spec:
    """Data class for buffer core info: name, entry class, kwargs."""
    buffer_name: str
    entry_point: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=buffer name (str), value=Spec object
registry: Dict[str, Spec] = {}

# Register buffer to global registry
def register(
    buffer_name: str, 
    entry_point: Callable, 
    **kwargs,
):
    global registry
    new_spec = Spec(
        buffer_name=buffer_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.buffer_name] = new_spec

# Register all valid buffer modules (RL/trainer/buffer/*.py) to registry
buffer_file_list = os.listdir(buffer_path)
for buffer_file in buffer_file_list:
    if buffer_file[-3:] == ".py" and buffer_file[0] != "_" and buffer_file != "base.py":
        buffer_name = buffer_file[:-3]
        mdl = importlib.import_module("RL.trainer.buffer." + buffer_name)
        register(
            buffer_name=buffer_name, 
            entry_point=getattr(mdl, underline2camel(buffer_name))
        )

def create_buffer(**kwargs) -> object:
    """Create buffer instance from registry (None for on-policy trainer)."""
    buffer_name = kwargs.get("buffer_name", None)
    buffer_spec = registry.get(buffer_name)

    if buffer_spec is None:
        raise KeyError(f"No registered buffer with id: {buffer_name}")
    
    buffer_spec_kwargs = buffer_spec.kwargs.copy()
    buffer_spec_kwargs.update(kwargs)

    if callable(buffer_spec.entry_point):
        buffer_creator = buffer_spec.entry_point
    else:
        raise RuntimeError(f"{buffer_spec.buffer_name} registered but entry_point is not specified")

    trainer_name = buffer_spec_kwargs.get("trainer", None)
    if trainer_name is None or trainer_name.startswith("on"):
        buffer = None
        print("No buffer for on-policy trainer! return None")
    else:
        buffer = buffer_creator(**buffer_spec_kwargs) 
        print(buffer_name, "created successfully")

    return buffer