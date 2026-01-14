# Register and create evaluator
import os
from typing import Callable, Dict, Union
from dataclasses import dataclass, field

from RL.trainer.evaluator import Evaluator

@dataclass
class Spec:
    """Data class for evaluator core info: name, entry class, kwargs."""
    evaluator_name: str
    entry_point: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=evaluator name (str), value=Spec object
registry: Dict[str, Spec] = {}

# Register evaluator to global registry
def register(
    evaluator_name: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        evaluator_name=evaluator_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.evaluator_name] = new_spec

# Register Evaluator class to registry
register(
    evaluator_name="evaluator", 
    entry_point=Evaluator
)

# Create evaluator instance (no ray)
def create_evaluator(evaluator_name: str, **kwargs) -> object:
    """Create evaluator instance from registry (no ray)."""
    evaluator_spec = registry.get(evaluator_name)

    if evaluator_spec is None:
        raise KeyError(f"No registered evaluator with id: {evaluator_name}")

    evaluator_spec_kwargs = evaluator_spec.kwargs.copy()
    evaluator_spec_kwargs.update(kwargs)
    
    if callable(evaluator_spec.entry_point):
        evaluator_creator = evaluator_spec.entry_point
        evaluator = evaluator_creator(**evaluator_spec_kwargs)
    else:
        raise RuntimeError(f"{evaluator_spec.evaluator_name} registered but entry_point is not specified")

    print(evaluator_name, "created successfully!")
    return evaluator