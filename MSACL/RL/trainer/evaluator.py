import torch
import numpy as np

from RL.create_pkg.create_envs import create_envs
from RL.create_pkg.create_alg import create_approx_contrainer

from RL.utils.rew_plus_cost import rew_plus_cost

class Evaluator:
    """
    Class for evaluating a trained policy, requiring corresponding environments 
    and trained policy networks.
    """
    def __init__(self, index=0, **kwargs):
        # Set random seed for evaluation environment (applied via env.reset())
        self.seed = kwargs["eval_env_seed"]

        # Environment ID
        self.env_id = kwargs["env_name"]
        self.reward_scale = kwargs["reward_scale"]
        self.cost_scale = kwargs["cost_scale"]

        # Target value for the current task
        self.target_value = kwargs["target_value"]

        # Create environments for policy evaluation
        print("Starting policy evaluation! Attempting to create evaluation environments...")
        self.num_eval_episode = kwargs["num_eval_episode"]  # Total evaluation episodes
        # Whether to use parallel evaluation (default: True)
        self.is_parallel_eval = kwargs.get("is_parallel_eval", True)
        if self.is_parallel_eval:
            # Parallel eval: create envs equal to number of evaluation episodes
            kwargs["env_num"] = self.num_eval_episode
        else:
            # Sequential eval: use only 1 environment
            kwargs["env_num"] = 1
        # Create environments
        self.envs = create_envs(** kwargs)

        # Create function approximators (Q-network, policy network, etc.) for evaluation
        # Will be loaded with pre-trained weights later
        print("Creating function approximators for evaluation")
        self.networks = create_approx_contrainer(** kwargs)
        self.render = kwargs["is_render"]

        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
  
        self.print_time = 0
        self.print_iteration = -1


    def load_state_dict(self, state_dict):
        """
        Load pre-trained model parameters (weights, biases, etc.) into the networks.
        """
        self.networks.load_state_dict(state_dict)


    def run_an_episode(self, iteration, render=False):
        # Only store rewards (cost can be derived from -reward)
        reward_list = []
        cost_list = []

        # Initialize environment (seed set to ensure consistent initial states for evaluation)
        obs, _ = self.envs.reset(seed=None)
        # Convert observation dtype
        obs = np.float32(obs)
        done = False

        # Interact with environment until termination/truncation
        while not done:
            # Environment interaction logic similar to off_sampler.base._step()
            obs_tensor = torch.from_numpy(obs)
            logits = self.networks.policy(obs_tensor)
            action_distribution = self.networks.create_action_distributions(logits)
            
            # Use mode (most probable) action for evaluation (no random sampling)
            action = action_distribution.mode()
            # Keep env dimension for action (consistent with obs shape)
            action = action.detach().numpy().astype("float32")

            next_obs, original_reward, termination, truncation, info = self.envs.step(action)
            # Convert dtype to float32
            next_obs = np.float32(next_obs)
            original_reward = np.float32(original_reward)

            # Update real_next_obs for terminated/truncated envs
            episode_finish = np.logical_or(termination, truncation)
            real_next_obs = next_obs.copy()
            for idx, finish in enumerate(episode_finish):
                if finish:
                    real_next_obs[idx] = info["final_observation"][idx]

            # Get rewards/costs via rew_plus_cost (supports multi-env evaluation)
            reward, cost = rew_plus_cost(
                self.env_id,
                obs,
                real_next_obs,
                self.target_value,
                self.cost_scale,
                self.reward_scale,
                original_reward,
                termination
            )

            # Remove env dimension when storing data (single env case)
            reward_list.append(reward[0])
            cost_list.append(cost[0])

            # Update observation
            obs = next_obs

            # Check if episode is finished
            done = np.logical_or(termination, truncation)[0]
            
        # Calculate total episode return and cost
        episode_return = sum(reward_list)
        episode_cost = sum(cost_list)
        return episode_return, episode_cost

    def run_n_episodes(self, n, iteration):
        # iteration: current training iteration number
        episode_return_list = []
        episode_cost_list = []

        for _ in range(n):
            # self.render is False by default for evaluation
            episode_return, episode_cost = self.run_an_episode(iteration, self.render)
            episode_return_list.append(episode_return)
            episode_cost_list.append(episode_cost)

        # Calculate mean and std of episode returns/costs
        episode_return_mean, episode_return_std = np.mean(episode_return_list), np.std(episode_return_list)
        episode_cost_mean, episode_cost_std = np.mean(episode_cost_list), np.std(episode_cost_list)

        return episode_return_mean, episode_return_std, episode_cost_mean, episode_cost_std
    

    def run_parallel_episodes(self,):
        # Initialization
        episode_return_list = [[] for _ in range(self.num_eval_episode)]
        episode_cost_list = [[] for _ in range(self.num_eval_episode)]
        episode_dones = [False for _ in range(self.num_eval_episode)]

        # Initialize environments
        obs, _ = self.envs.reset(seed=None)
        # Convert observation dtype
        obs = np.float32(obs)

        # Interact with envs until all episodes finish
        while not all(episode_dones):
            obs_tensor = torch.from_numpy(obs)
            logits = self.networks.policy(obs_tensor)
            action_distribution = self.networks.create_action_distributions(logits)

            # Use mode action for evaluation (plural 'actions' for parallel envs; no random sampling)
            actions = action_distribution.mode()
            actions = actions.detach().numpy().astype("float32")

            next_obs, original_rewards, terminations, truncations, infos = self.envs.step(actions)
            # Convert dtype to float32
            next_obs = np.float32(next_obs)
            original_rewards = np.float32(original_rewards)

            # Handle terminated/truncated sub-environments
            dones = np.logical_or(terminations, truncations)
            real_next_obs = next_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    real_next_obs[idx] = infos["final_observation"][idx]

            rewards, costs = rew_plus_cost(
                self.env_id,
                obs,
                real_next_obs,
                self.target_value,
                self.cost_scale,
                self.reward_scale,
                original_rewards,
                terminations
            )
            
            for i in range(self.num_eval_episode):
                # Add rewards/costs only for unfinished episodes
                if not episode_dones[i]:
                    episode_return_list[i].append(rewards[i])
                    episode_cost_list[i].append(costs[i])
                    # Mark episode as done when finished
                    episode_dones[i] = dones[i]  

            # Update observation (within while loop)
            obs = next_obs

        # Calculate per-step mean return/cost for each episode (fair metric for path quality)
        episode_return = [np.mean(sublist) for sublist in episode_return_list]
        episode_cost = [np.mean(sublist) for sublist in episode_cost_list]
        
        # Calculate mean and std of episode-level returns/costs
        episode_return_mean, episode_return_std = np.mean(episode_return), np.std(episode_return)
        episode_cost_mean, episode_cost_std = np.mean(episode_cost), np.std(episode_cost)

        return episode_return_mean, episode_return_std, episode_cost_mean, episode_cost_std
    

    def run_evaluation(self, iteration):
        if self.is_parallel_eval:
            return self.run_parallel_episodes()
        else:
            # Fallback to sequential evaluation (single env)
            return self.run_n_episodes(self.num_eval_episode, iteration)