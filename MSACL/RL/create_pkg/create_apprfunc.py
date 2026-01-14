# Registry for function approximator specifications
import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict

from RL.utils.MyRL_path import apprfunc_path

# Global registry for approximation function classes (under RL/apprfunc/)
@dataclass
class Spec:
    """Data class for apprfunc core info: type, class name, entry class, kwargs."""
    apprfunc: str
    name: str
    entry_point: Callable
    kwargs: dict = field(default_factory=dict)

# Registry: key=apprfunc_name+"_"+class_name (e.g., mlp_StochaPolicy), value=Spec object
registry: Dict[str, Spec] = {}

# Register apprfunc to global registry
def register(
    apprfunc: str, 
    name: str,
    entry_point: Callable, 
    **kwargs,
):
    global registry
    new_spec = Spec(
        apprfunc=apprfunc, 
        name=name, 
        entry_point=entry_point,
        kwargs=kwargs
        )
    registry[new_spec.apprfunc + "_" + new_spec.name] = new_spec


# Register all valid apprfunc modules (RL/apprfunc/*.py) to registry
apprfunc_file_list = os.listdir(apprfunc_path)
for apprfunc_file in apprfunc_file_list:
    if apprfunc_file[-3:] == ".py" and apprfunc_file[0] != "_" and apprfunc_file != "base.py":
        apprfunc_name = apprfunc_file[:-3]
        mdl = importlib.import_module("RL.apprfunc." + apprfunc_name)
        for name in mdl.__all__:
            register(
                apprfunc=apprfunc_name, 
                name=name, 
                entry_point=getattr(mdl, name)
            )


def create_apprfunc(**kwargs) -> object:
    """Create approximation function instance from registry."""
    apprfunc = kwargs["apprfunc"].lower()
    name = kwargs["name"]

    apprfunc_spec = registry.get(apprfunc + "_" + name)

    if apprfunc_spec is None:
        raise KeyError(f"No registered apprfunc with id: {apprfunc}_{name}")

    apprfunc_spec_kwargs = apprfunc_spec.kwargs.copy()
    apprfunc_spec_kwargs.update(kwargs)

    if callable(apprfunc_spec.entry_point):
        apprfunc_creator = apprfunc_spec.entry_point
    else:
        raise RuntimeError(f"{apprfunc_spec.apprfunc}-{apprfunc_spec.name} registered but entry_point is not specified")

    apprfunc = apprfunc_creator(**apprfunc_spec_kwargs)

    return apprfunc