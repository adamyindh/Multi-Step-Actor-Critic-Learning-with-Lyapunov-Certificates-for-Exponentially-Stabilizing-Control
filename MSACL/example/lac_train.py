"""
The training
Algorithm: LAC
"""
import argparse
import math

from RL.create_pkg.create_envs import create_envs
from RL.utils.init_args import init_args
from RL.create_pkg.create_alg import create_alg
from RL.create_pkg.create_sampler import create_sampler
from RL.create_pkg.create_buffer import create_buffer
from RL.create_pkg.create_evaluator import create_evaluator
from RL.create_pkg.create_trainer import create_trainer
from RL.utils.tensorboard_setup import save_tb_to_csv, open_tb_in_browser

if __name__ == "__main__":
    # Parameter configuration
    # parser: Instance of argparse.ArgumentParser for parsing command-line arguments
    parser = argparse.ArgumentParser()

    ################################################
    # Core settings: Environment + Algorithm + CUDA
    parser.add_argument("--env_name", type=str, default="Pendulum", help="ID of the environment")
    # Algorithm name (lowercase) must match the algorithm file (e.g., sac.py) for direct lookup
    parser.add_argument("--algorithm", type=str, default="lac", help="Reinforcement Learning algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA acceleration")
    
    ################################################
    # 1. Environment parameters
    parser.add_argument("--env_num", type=int, default=4, help="Number of parallel environments")
    # Random seed for action space (ensures reproducibility of action sampling)
    # Different from common_utils.seed_everything (sets seeds for random/CUDA/etc.)
    parser.add_argument("--env_seed", type=int, default=1, help="Random seed for action space")
    parser.add_argument("--capture_vedio", type=bool, default=False, help="Generate environment animation video")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Enable adversarial training")
    parser.add_argument("--is_render", type=bool, default=False, help="Render environment visualization")

    # Target value for environment (used in cost/reward function design)
    parser.add_argument("--target_value", type=float, default=0.0, help="Target value for environment")
    # Scaling factors for reward and cost
    parser.add_argument("--reward_scale", type=float, default=100.0, help="Reward scaling factor")
    parser.add_argument("--cost_scale", type=float, default=100.0, help="Cost scaling factor")

    ################################################
    # 2. Value function approximation parameters (state/action value)
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDistri",
    )
    parser.add_argument(
        "--value_func_type", 
        type=str, 
        default="MLP", 
        help="Options: MLP",
    )
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--value_hidden_activation", 
        type=str, 
        default="relu", 
        help="Hidden layer activation: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Output layer activation: linear/tanh")

    ################################################
    # 3. Policy function approximation parameters (action generation)
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: DetermPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", 
        type=str, 
        default="MLP", 
        help="Options: MLP"
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Action distribution: default/TanhGaussDistribution/GaussDistribution",
    )
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument(
            "--policy_hidden_activation",
            type=str,
            default="relu", 
            help="Hidden layer activation: relu/gelu/elu/selu/sigmoid/tanh"
        )
    parser.add_argument("--policy_min_log_std", type=float, default=-20)
    parser.add_argument("--policy_max_log_std", type=float, default=1)  # Original GOPS value: 0.5 (increased to 1 for more randomness)

    # Note: Policy output layer is defined in get_apprfunc_dict (common_utils.py) - MLP uses linear output layer

    ################################################
    # 4. RL algorithm hyperparameters
    parser.add_argument("--l_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta_learning_rate", type=float, default=1e-3)

    # Core hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    # auto_alpha: Controls automatic adjustment of both alpha and beta
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--bound", default=True)
    # alpha3: Coefficient for cost c in Lyapunov descent condition
    parser.add_argument("--alpha3", type=float, default=0.01)

    # Delayed update settings
    parser.add_argument("--policy_frequency", type=int, default=2)
    # Update frequency for Lagrange multiplier (target network)
    parser.add_argument("--target_network_frequency", type=int, default=1)

    ################################################
    # 5. Trainer parameters
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Trainer type: on_serial_trainer/off_serial_trainer",
    )
    parser.add_argument("--max_iteration", type=int, default=1000000, help="Maximum training iterations")
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None,
        help="Initial network weights directory"
    )
    trainer_type = parser.parse_known_args()[0].trainer

    ################################################
    # 6. Sampler parameters
    parser.add_argument(
        "--sampler_name", 
        type=str, 
        default="off_sampler", 
        help="Sampler type: on_sampler/off_sampler"
    )
    parser.add_argument("--sample_interval", type=int, default=1, help="Sampling interval")
    parser.add_argument("--sample_batch_size", type=int, default=20, help="Total interaction steps per sampling")
    # Action noise for exploration enhancement
    parser.add_argument("--noise_params", type=dict, default=None, help="Noise parameters for action exploration")

    ################################################
    # 7. Replay buffer parameters
    parser.add_argument(
        "--buffer_name", 
        type=str, 
        default="replay_buffer", 
        help="Buffer type: replay_buffer/prioritized_replay_buffer"
    )
    parser.add_argument("--buffer_warm_size", type=int, default=int(5e3), help="Sample size before training starts")
    parser.add_argument("--buffer_max_size", type=int, default=int(1e6), help="Maximum replay buffer capacity")
    parser.add_argument("--replay_batch_size", type=int, default=256, help="Batch size for buffer sampling")

    ################################################
    # 8. Evaluator parameters
    parser.add_argument("--eval_env_seed", type=int, default=2, help="Random seed for evaluation environment")
    parser.add_argument("--is_parallel_eval", type=bool, default=True, help="Enable parallel evaluation")
    parser.add_argument("--evaluator_name", type=str, default="evaluator", help="Evaluator type")
    parser.add_argument("--num_eval_episode", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval (iterations)")
    parser.add_argument("--eval_save", type=str, default=False, help="Save evaluation results")

    ################################################
    # 9. Data saving parameters
    parser.add_argument("--save_folder", type=str, default=None, help="Directory for saving training data")
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000, help="Network save interval (updates)")
    parser.add_argument("--log_save_interval", type=int, default=50000, help="Log save interval (updates)")

    ################################################
    # Convert parsed arguments to dictionary (for keyword argument passing)
    # vars(): Converts argparse Namespace to dictionary
    # **: Unpacks dictionary into keyword arguments
    args = vars(parser.parse_args())

    # Create environments (refer to CleanRL for implementation)
    envs = create_envs(**args)

    # Initialize additional arguments based on environment
    args = init_args(envs, **args)

    ################################################
    # Training pipeline: Create core components
    # Step 1: Create RL algorithm and approximation functions
    print("-------------------- Create algorithm and approximation function! -------------------- ")
    alg = create_alg(**args)

    # Step 2: Create sampler
    print("-------------------- Create sampler! -------------------- ")
    sampler = create_sampler(**args)

    # Step 3: Create replay buffer
    print("-------------------- Create buffer! -------------------- ")
    buffer = create_buffer(**args)

    # Step 4: Create evaluator
    print("-------------------- Create evaluator! -------------------- ")
    evaluator = create_evaluator(**args)

    # Step 5: Create trainer
    print("-------------------- Create trainer! -------------------- ")
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training
    print("Start training!")
    print("Training data save directory: {}".format(args["save_folder"]))
    trainer.train()
    print("Training is finished!")

    ################################################
    # Export TensorBoard logs to CSV
    save_tb_to_csv(args["save_folder"])

    ################################################
    # Open TensorBoard visualization in local browser
    open_tb_in_browser(args["save_folder"])