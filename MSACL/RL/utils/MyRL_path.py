import os

# Get RL root directory path
RL_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to core modules
algorithm_path = os.path.join(RL_path, "algorithm")  # Algorithm files (e.g., sac.py)
apprfunc_path = os.path.join(RL_path, "apprfunc")    # Approximator network files (e.g., mlp.py)
trainer_path = os.path.join(RL_path, "trainer")      # Trainer module root
sampler_path = os.path.join(trainer_path, "sampler") # Sampler modules (on_sampler/off_sampler)
buffer_path = os.path.join(trainer_path, "buffer")   # Buffer modules

def underline2camel(s: str, first_upper: bool = False) -> str:
    """Convert snake_case to CamelCase (lower camel by default)"""
    arr = s.split("_")
    if first_upper:
        res = arr.pop(0).upper()
    else:
        res = ""
    for a in arr:
        res = res + a[0].upper() + a[1:]
    return res