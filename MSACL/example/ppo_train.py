"""
The training
Algorithm: PPO
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
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    ################################################
    # Env, algorithm & hardware settings
    parser.add_argument("--env_name", type=str, default="Pendulum", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm name (lowercase)")
    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA acceleration")
    
    ################################################
    # Environment parameters
    parser.add_argument("--env_num", type=int, default=1, help="Number of environments")
    parser.add_argument("--env_seed", type=int, default=1, help="Random seed for action space")
    parser.add_argument("--capture_vedio", type=bool, default=False, help="Capture environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Enable adversary training") 
    parser.add_argument("--is_render", type=bool, default=False, help="Render environment") 
    parser.add_argument("--target_value", type=float, default=0.0, help="Environment target value") 
    parser.add_argument("--reward_scale", type=float, default=100.0, help="Reward scaling factor")
    parser.add_argument("--cost_scale", type=float, default=100.0, help="Cost scaling factor")

    ################################################
    # Value function approximation parameters
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="StateValue",
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
    # Lyapunov function approximation parameters
    parser.add_argument(
        "--lyapunov_func_name",
        type=str,
        default="LyapunovValue",
        help="Options: StateValue/ActionValue/ActionValueDistri/LyapunovValue",
    )
    parser.add_argument(
        "--lyapunov_func_type", 
        type=str, 
        default="MLP", 
        help="Options: MLP",
    )
    lyapunov_func_type = parser.parse_known_args()[0].lyapunov_func_type
    parser.add_argument("-lyapunov_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--lyapunov_hidden_activation", 
        type=str, 
        default="tanh",
        help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--lyapunov_output_dim", type=int, default=256)
    parser.add_argument("--lyapunov_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--lyapunov_single_input_dim", type=bool, default=False)

    ################################################
    # Policy function approximation parameters
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
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument(
            "--policy_hidden_activation",
            type=str,
            default="relu", 
            help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
    parser.add_argument("--policy_min_log_std", type=float, default=-20)
    parser.add_argument("--policy_max_log_std", type=float, default=1)

    ################################################
    # RL algorithm parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--loss_coefficient_value", type=float, default=1.0)
    parser.add_argument("--loss_coefficient_entropy", type=float, default=0.01)
    parser.add_argument("--loss_coefficient_kl", type=float, default=0.0)
    parser.add_argument("--loss_value_clip", type=bool, default=False)
    parser.add_argument("--value_clip", type=float, default=10)
    parser.add_argument("--lya_diff_sacle", type=float, default=1.0)
    parser.add_argument("--lya_zero_sacle", type=float, default=10.0)
    parser.add_argument("--lya_positive_scale", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--schedule_adam", type=str, default="None")
    parser.add_argument("--schedule_clip", type=str, default="None")
    parser.add_argument("--clip", type=float, default=0.1)

    ################################################
    # Trainer parameters
    parser.add_argument(
        "--trainer",
        type=str,
        default="on_serial_trainer",
        help="Options: on_serial_trainer, off_serial_trainer",
    )
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--max_iteration", type=int, default=1000000)
    parser.add_argument("--num_repeat", type=int, default=2)
    parser.add_argument("--num_mini_batch", type=int, default=25)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=parser.parse_known_args()[0].num_repeat * parser.parse_known_args()[0].num_mini_batch,
        help="Total epochs = num_repeat * num_mini_batch",
    )

    ################################################
    # Sampler parameters
    parser.add_argument(
        "--sampler_name", 
        type=str, 
        default="on_sampler", 
        help="Options: on_sampler/off_sampler"
    )
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument(
        "--sample_batch_size", 
        type=int, 
        default=1600, 
        help="Batch size per sampling per env"
    )
    assert (
        parser.parse_known_args()[0].num_mini_batch * parser.parse_known_args()[0].mini_batch_size
        == parser.parse_known_args()[0].sample_batch_size
    ), "sample_batch_size error"
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # Buffer parameters (compatibility for on-policy alg)
    parser.add_argument(
        "--buffer_name", 
        type=str, 
        default="replay_buffer", 
        help="Options:replay_buffer/prioritized_replay_buffer"
    )
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    parser.add_argument("--buffer_max_size", type=int, default=50000)

    ################################################
    # Evaluator parameters
    parser.add_argument("--eval_env_seed", type=int, default=2)
    parser.add_argument("--is_parallel_eval", type=bool, default=True)
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="Save evaluation data")

    ################################################
    # Data saving parameters
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    parser.add_argument("--log_save_interval", type=int, default=50000)

    ################################################
    # Convert args to dictionary
    args = vars(parser.parse_args())

    # Create environments
    envs = create_envs(**args)

    # Initialize additional arguments
    args = init_args(envs, **args)

    ################################################
    # Build training components
    print("-------------------- Create algorithm and approximation functions! -------------------- ")
    alg = create_alg(**args)

    print("-------------------- Create sampler! -------------------- ")
    sampler = create_sampler(**args)

    print("-------------------- Create buffer! -------------------- ")
    buffer = create_buffer(**args)

    print("-------------------- Create evaluator! -------------------- ")
    evaluator = create_evaluator(**args)

    print("-------------------- Create trainer! -------------------- ")
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training
    print("Start training!")
    print("Training data save folder: {}".format(args["save_folder"]))
    trainer.train()
    print("Training is finished!")

    ################################################
    # Export tensorboard logs to CSV
    save_tb_to_csv(args["save_folder"])

    ################################################
    # Open tensorboard in local browser
    open_tb_in_browser(args["save_folder"])