# Import self-created envs
from RL.env.Pendulum import pendulum
from RL.env.DuctedFan import ductedfan
from RL.env.TwoLink import twolink
from RL.env.SingleTrackCar import singletrackcar
from RL.env.VanderPol import vanderpol
from RL.env.QuadTracking import quadtracking
import gymnasium as gym

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    env_id: Env name (e.g., pendulum)
    seed: Env seed (set action space seed)
    idx: Env index
    """
    def thunk():
        render_mode = None
        # Select custom env by env_id
        if env_id == "Pendulum":
            env = pendulum()
        elif env_id == "SingleTrackCar":
            env = singletrackcar()
        elif env_id == "DuctedFan":
            env = ductedfan()
        elif env_id == "TwoLink":
            env = twolink()
        elif env_id == "VanderPol":
            env = vanderpol()
        elif env_id == "QuadTracking":
            env = quadtracking()
        else:
            raise ValueError(f"Unknown custom env: {env_id}")
        
        # Add episode stats wrapper
        # Auto record episode stats (reward/length/time) to info when episode ends (terminated/truncated=True)
        # For SyncVectorEnv: sub-env info collected into array
        # Reset env for new episode at end, return initial state
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Set action space seed (for reproducibility)
        env.action_space.seed(seed) 
        return env
    return thunk