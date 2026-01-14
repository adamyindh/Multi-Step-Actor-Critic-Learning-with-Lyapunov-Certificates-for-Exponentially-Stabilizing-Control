"""
The training
Algorithm: SAC
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
    # Initialize argument parser for command line parameter configuration
    parser = argparse.ArgumentParser()

    ################################################
    # Core settings: environment, algorithm, CUDA enablement
    parser.add_argument("--env_name", type=str, default="TwoLink", help="id of environment")
    # Algorithm name (lowercase, matches algorithm file name e.g. sac.py)
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA")
    
    ################################################
    # 1. Environment parameters
    # Number of parallel environments for data collection
    parser.add_argument("--env_num", type=int, default=4, help="num of environment")
    # Random seed for environment action space (ensure reproducibility)
    parser.add_argument("--env_seed", type=int, default=1, help="seed of action space")
    parser.add_argument("--capture_vedio", type=bool, default=False, help="Draw environment animation")
    # Whether to perform adversarial training
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training") 
    # Whether to render the environment
    parser.add_argument("--is_render", type=bool, default=False, help="Env Render") 

    # Target value for custom environments (all components are 0)
    parser.add_argument("--target_value", type=float, default=0.0, help="Target value") 

    # Scaling factors for reward and cost
    parser.add_argument("--reward_scale", type=float, default=100.0, help="reward scale factor")
    parser.add_argument("--cost_scale", type=float, default=100.0, help="cost scale factor")

    ################################################
    # 2. Value function approximation parameters
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
        help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 3. Policy function approximation parameters
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
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    if policy_func_type == "MLP":
        # Hidden layer size [256,256] performs better than [64,64]
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument(
            "--policy_hidden_activation",
            type=str,
            default="relu", 
            help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
    parser.add_argument("--policy_min_log_std", type=float, default=-20)
    # Increased variance for better exploration (v0: 0.5)
    parser.add_argument("--policy_max_log_std", type=float, default=1)  

    ################################################
    # 4. RL algorithm parameters
    parser.add_argument("--q_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)

    # Core algorithm hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=1.0)
    # Auto adjust alpha and beta
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--bound", default=True)

    # Delayed update settings
    parser.add_argument("--policy_frequency", type=int, default=2)  # Test delayed update effect
    # Target network (Q-network) update frequency
    parser.add_argument("--target_network_frequency", type=int, default=1)

    ################################################
    # 5. Trainer parameters
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, off_serial_trainer",
    )
    # Maximum training iterations
    parser.add_argument("--max_iteration", type=int, default=1000000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer

    ################################################
    # 6. Sampler parameters
    parser.add_argument(
        "--sampler_name", 
        type=str, 
        default="off_sampler", 
        help="Options: on_sampler/off_sampler"
    )
    # Sampling interval
    parser.add_argument("--sample_interval", type=int, default=1)
    # Interaction steps = sample_batch_size (modified in base.py)
    parser.add_argument("--sample_batch_size", type=int, default=20)

    # Action noise for exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 7. Replay buffer parameters
    parser.add_argument(
        "--buffer_name", 
        type=str, 
        default="replay_buffer", 
        help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Initial sample size before training
    parser.add_argument("--buffer_warm_size", type=int, default=int(5e3))
    # Maximum buffer capacity
    parser.add_argument("--buffer_max_size", type=int, default=int(1e6))
    # Batch size for sampling from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)

    ################################################
    # 8. Evaluator parameters
    # Random seed for evaluation environment
    parser.add_argument("--eval_env_seed", type=int, default=2)
    # Whether to use parallel evaluation
    parser.add_argument("--is_parallel_eval", type=bool, default=True)

    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    # Evaluation interval
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 9. Data saving parameters
    parser.add_argument("--save_folder", type=str, default=None)
    # Network checkpoint save interval
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    # Training log save interval
    parser.add_argument("--log_save_interval", type=int, default=50000)

    ################################################
    # Convert parsed arguments to dictionary for keyword passing
    args = vars(parser.parse_args())

    # Create environments with parsed arguments
    envs = create_envs(**args)

    # Initialize additional arguments based on created environments
    args = init_args(envs, **args)

    ################################################
    # Training and evaluation pipeline

    # Step 1: Create RL algorithm and approximation functions
    print("-------------------- Create algorithm and approximate function!-------------------- ")
    alg = create_alg(**args)

    # Step 2: Create sampler
    print("-------------------- Create sampler in trainer!-------------------- ")
    sampler = create_sampler(**args)

    # Step 3: Create replay buffer
    print("-------------------- Create buffer in trainer!-------------------- ")
    buffer = create_buffer(**args)

    # Step 4: Create evaluator
    print("-------------------- Create evaluator in trainer!-------------------- ")
    evaluator = create_evaluator(**args)

    # Step 5: Create trainer and start training
    print("-------------------- Create trainer!-------------------- ")
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training process
    print("Start training!")
    print("Training data save folder：{}".format(args["save_folder"]))
    trainer.train()
    print("Training is finished!")

    ################################################
    # Export TensorBoard data to CSV
    save_tb_to_csv(args["save_folder"])

    ################################################
    # Open TensorBoard visualization in browser
    open_tb_in_browser(args["save_folder"])