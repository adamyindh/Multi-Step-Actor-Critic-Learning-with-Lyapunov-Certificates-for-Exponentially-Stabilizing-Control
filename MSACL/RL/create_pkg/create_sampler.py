# Register and create sampler (on/off sampler from RL/trainer/sampler/)
from dataclasses import dataclass, field
from typing import Callable, Dict, Union

import os
from RL.utils.MyRL_path import sampler_path, underline2camel
import importlib

@dataclass
class Spec:
    """Data class for sampler core info: name, entry class, kwargs."""
    sampler_name: str
    entry_point: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=sampler name (str), value=Spec object
registry: Dict[str, Spec] = {}

# Register sampler to global registry
def register(
    sampler_name: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        sampler_name=sampler_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.sampler_name] = new_spec

# Register all valid sampler modules (RL/trainer/sampler/*.py) to registry
sampler_file_list = os.listdir(sampler_path)
for sampler_file in sampler_file_list:
    if sampler_file[-3:] == ".py" and sampler_file[0] != "_" and sampler_file != "base.py":
        sampler_name = sampler_file[:-3]
        mdl = importlib.import_module("RL.trainer.sampler." + sampler_name)
        register(
            sampler_name=sampler_name, 
            entry_point=getattr(mdl, underline2camel(sampler_name))
        )

def create_sampler(**kwargs):
    """Create sampler instance from registry."""
    sampler_name = kwargs["sampler_name"]
    sampler_spec = registry.get(sampler_name)

    if sampler_spec is None:
        raise KeyError(f"No registered sampler with id: {sampler_name}")
    
    sampler_spec_kwargs = sampler_spec.kwargs.copy()
    sampler_spec_kwargs.update(kwargs)

    if callable(sampler_spec.entry_point):
        sampler_creator = sampler_spec.entry_point
        sampler = sampler_creator(**sampler_spec_kwargs)
    else:
        raise RuntimeError(f"{sampler_spec.sampler_name} registered but entry_point is not specified")
    
    print(sampler_name, "created successfully!")
    return sampler