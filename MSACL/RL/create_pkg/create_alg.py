import os
from typing import Callable, Dict
from dataclasses import dataclass, field
import importlib

from RL.utils.MyRL_path import algorithm_path

# Global registry for algorithm classes (under RL/algorithm/)
@dataclass
class Spec:
    """Data class for algorithm core info: name, entry class, approx container class, kwargs."""
    algorithm: str
    entry_point: Callable
    approx_container_cls: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=algorithm name (str), value=Spec object
registry: Dict[str, Spec] = {}

# Register algorithm to global registry
def register(
    algorithm: str, 
    entry_point: Callable, 
    approx_container_cls: Callable, 
    **kwargs,
):
    global registry
    new_spec = Spec(
        algorithm=algorithm, 
        entry_point=entry_point, 
        approx_container_cls=approx_container_cls, 
        kwargs=kwargs
        )
    registry[new_spec.algorithm] = new_spec


# Register all valid algorithm modules (RL/algorithm/*.py) to registry
alg_file_list = os.listdir(algorithm_path)
for alg_file in alg_file_list:
    if alg_file[-3:] == ".py" and alg_file[0] != "_" and alg_file != "base.py":
        alg_name = alg_file[:-3]
        mdl = importlib.import_module("RL.algorithm." + alg_name)
        register(
            algorithm=alg_name,
            entry_point=getattr(mdl, alg_name.upper()),
            approx_container_cls=getattr(mdl, "ApproxContainer"),
        )


def create_alg(**kwargs) -> object:
    """Create algorithm instance from registry."""
    algorithm = kwargs["algorithm"]
    algo_spec = registry.get(algorithm)

    if algo_spec is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")
    
    algo_spec_kwargs = algo_spec.kwargs.copy()
    algo_spec_kwargs.update(kwargs)

    if callable(algo_spec.entry_point):
        algorithm_creator = algo_spec.entry_point
    else:
        raise RuntimeError(f"{algo_spec.algorithm} registered but entry_point is not specified")
    
    trainer_name = algo_spec_kwargs.get("trainer", None)
    if (
        trainer_name is None
        or trainer_name.startswith("off_serial")
        or trainer_name.startswith("nstep_off_serial")
        or trainer_name.startswith("on_serial")
        or trainer_name.startswith("on_sync")
    ):
        algo = algorithm_creator(**algo_spec_kwargs)
    else:
        raise RuntimeError(f"trainer {trainer_name} can not recognized")
    
    print(algorithm, "algorithm created successfully!")
    return algo


def create_approx_contrainer(algorithm: str, **kwargs,) -> object:
    """Create approx container instance from registry."""
    algo_spec = registry.get(algorithm)

    if algo_spec is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")

    algo_spec_kwargs = algo_spec.kwargs.copy()
    algo_spec_kwargs.update(kwargs)

    if callable(algo_spec.approx_container_cls):
        approx_contrainer = algo_spec.approx_container_cls(**algo_spec_kwargs)
    else:
        raise RuntimeError(f"{algo_spec.algorithm} registered but approx_container_cls is not specified")
    
    return approx_contrainer
