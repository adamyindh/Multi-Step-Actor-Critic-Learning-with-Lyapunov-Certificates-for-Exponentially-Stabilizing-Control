"""
The training
Algorithm: MSACL
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
    # Parameter Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Env, Algorithm & CUDA
    parser.add_argument("--env_name", type=str, default="QuadTracking", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="msacl", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA acceleration")
    
    ################################################
    # 1. Environment Params
    parser.add_argument("--env_num", type=int, default=4, help="Number of environments")
    parser.add_argument("--env_seed", type=int, default=1, help="Action space seed")
    parser.add_argument("--capture_vedio", type=bool, default=False, help="Capture env animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training") 
    parser.add_argument("--is_render", type=bool, default=False, help="Render environment") 
    parser.add_argument("--target_value", type=float, default=0.0, help="Env target value") 
    parser.add_argument("--reward_scale", type=float, default=100.0, help="Reward scale factor")
    parser.add_argument("--cost_scale", type=float, default=100.0, help="Cost scale factor")

    ################################################
    # 2. Value Function Params
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--value_hidden_activation", type=str, default="relu", help="Activation func")
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Output activation")

    ################################################
    # 3. Lyapunov Function Params
    parser.add_argument(
        "--lyapunov_func_name",
        type=str,
        default="LyapunovValue",
        help="Options: StateValue/ActionValue/ActionValueDistri/LyapunovValue",
    )
    parser.add_argument("--lyapunov_func_type", type=str, default="MLP", help="Options: MLP")
    lyapunov_func_type = parser.parse_known_args()[0].lyapunov_func_type
    parser.add_argument("--lyapunov_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--lyapunov_hidden_activation", type=str, default="tanh", help="Activation func")
    parser.add_argument("--lyapunov_output_dim", type=int, default=256)
    parser.add_argument("--lyapunov_output_activation", type=str, default="linear", help="Output activation")
    parser.add_argument("--lyapunov_single_input_dim", type=bool, default=False)

    ################################################
    # 4. Policy Function Params
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: DetermPolicy/StochaPolicy",
    )
    parser.add_argument("--policy_func_type", type=str, default="MLP", help="Options: MLP")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument("--policy_hidden_activation", type=str, default="relu", help="Activation func")
    parser.add_argument("--policy_min_log_std", type=float, default=-20)
    parser.add_argument("--policy_max_log_std", type=float, default=1)

    ################################################
    # 5. RL Algorithm Params
    parser.add_argument("--q_learning_rate", type=float, default=1e-3)
    parser.add_argument("--lyapunov_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)

    parser.add_argument("--lya_diff_scale", type=float, default=10.0)
    parser.add_argument("--lya_zero_scale", type=float, default=1.0)
    parser.add_argument("--lya_positive_scale", type=float, default=1.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--retrace_lambda", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--disable_auto_alpha", action="store_true", help="Disable auto alpha tuning")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--set_alpha_bound", action="store_true", help="Set alpha bound")
    parser.add_argument("--alpha_bound", type=float, default=2.0)

    parser.add_argument("--n_step", type=int, default=20)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--anneal_lr", type=bool, default=False)

    parser.add_argument("--alpha1", type=float, default=1)
    parser.add_argument("--alpha2", type=float, default=2)
    parser.add_argument("--lya_eta", type=float, default=0.15)
    parser.add_argument("--clip_coef", type=float, default=0.1)

    ################################################
    # 6. Trainer Params
    parser.add_argument(
        "--trainer",
        type=str,
        default="nstep_off_serial_trainer",
        help="Options: on_serial_trainer, off_serial_trainer, nstep_off_serial_trainer",
    )
    parser.add_argument("--max_iteration", type=int, default=1000000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    trainer_type = parser.parse_known_args()[0].trainer

    ################################################
    # 7. Sampler Params
    parser.add_argument(
        "--sampler_name", 
        type=str, 
        default="nstep_off_sampler", 
        help="Options: on_sampler/off_sampler/nstep_off_sampler"
    )
    parser.add_argument("--sample_interval", type=int, default=1)
    parser.add_argument("--sample_batch_size", type=int, default=20, help="Sampling batch size")
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 8. Buffer Params
    parser.add_argument(
        "--buffer_name", 
        type=str, 
        default="nstep_replay_buffer", 
        help="Options:replay_buffer/prioritized_replay_buffer/nstep_replay_buffer"
    )
    parser.add_argument("--buffer_warm_size", type=int, default=int(5e3))
    parser.add_argument("--buffer_max_size", type=int, default=int(1e6))
    parser.add_argument("--replay_batch_size", type=int, default=256)

    ################################################
    # 9. Evaluator Params
    parser.add_argument("--eval_env_seed", type=int, default=2)
    parser.add_argument("--is_parallel_eval", type=bool, default=True)
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="Save evaluation data")

    ################################################
    # 10. Data Saving
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    parser.add_argument("--log_save_interval", type=int, default=50000)

    ################################################
    # Parse Params
    args = vars(parser.parse_args())
    envs = create_envs(**args)
    args = init_args(envs, **args)

    ################################################
    # Training Pipeline
    print("-------------------- Create algorithm -------------------- ")
    alg = create_alg(**args)

    print("-------------------- Create sampler -------------------- ")
    sampler = create_sampler(**args)

    print("-------------------- Create buffer -------------------- ")
    buffer = create_buffer(**args)

    print("-------------------- Create evaluator -------------------- ")
    evaluator = create_evaluator(**args)

    print("-------------------- Create trainer -------------------- ")
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start Training
    print("Start training!")
    print("Save dir: {}".format(args["save_folder"]))
    trainer.train()
    print("Training finished!")

    ################################################
    # Export & Visualize Logs
    save_tb_to_csv(args["save_folder"])
    open_tb_in_browser(args["save_folder"])