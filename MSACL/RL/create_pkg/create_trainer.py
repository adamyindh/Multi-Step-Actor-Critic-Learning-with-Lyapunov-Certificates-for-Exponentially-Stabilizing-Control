# Register and create trainer (on/off policy trainers from RL/trainer/)
import os
from typing import Callable, Dict, Union
from dataclasses import dataclass, field
import importlib

from RL.utils.MyRL_path import trainer_path, underline2camel

@dataclass
class Spec:
    """Data class for trainer core info: name, entry class, kwargs."""
    trainer: str
    entry_point: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=trainer name (str), value=Spec object
registry: Dict[str, Spec] = {}

# Register trainer to global registry
def register(
    trainer: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        trainer=trainer, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.trainer] = new_spec

# Register valid trainer modules (RL/trainer/*trainer.py) to registry
trainer_file_list = os.listdir(trainer_path)
for trainer_file in trainer_file_list:
    if trainer_file.endswith("trainer.py"):
        trainer_name = trainer_file[:-3]
        mdl = importlib.import_module("RL.trainer." + trainer_name)
        register(trainer=trainer_name, entry_point=getattr(mdl, underline2camel(trainer_name)))

def create_trainer(alg, sampler, buffer, evaluator, **kwargs,) -> object:
    """Create trainer instance (diff args for on/off policy)."""
    trainer_name = kwargs["trainer"]
    trainer_spec = registry.get(trainer_name)

    if trainer_spec is None:
        raise KeyError(f"No registered trainer with id: {trainer_name}")

    if callable(trainer_spec.entry_point):
        trainer_creator = trainer_spec.entry_point
    else:
        raise RuntimeError(f"{trainer_spec.trainer} registered but entry_point is not specified")

    if trainer_spec.trainer.startswith("off") or trainer_spec.trainer.startswith("nstep_off"):
        trainer = trainer_creator(alg, sampler, buffer, evaluator,** kwargs)
    elif trainer_spec.trainer.startswith("on"):
        trainer = trainer_creator(alg, sampler, evaluator, **kwargs)
    else:
        raise RuntimeError(f"trainer {trainer_spec.trainer} not recognized")
    
    print(trainer_name, "created successfully!")

    return trainer