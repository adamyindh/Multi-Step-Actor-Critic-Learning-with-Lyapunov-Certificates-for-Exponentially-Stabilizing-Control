__all__ = ["OnSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from RL.utils.tensorboard_setup import tb_tags, add_scalars
from RL.utils.log_data import LogData
from RL.utils.common_utils import ModuleOnDevice


class OnSerialTrainer:
    def __init__(self, alg, sampler, evaluator,** kwargs):
        # Core components
        self.alg = alg
        self.sampler = sampler
        self.evaluator = evaluator

        # Network setup (shared across alg/sampler/evaluator)
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # Load initial network if provided
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        # Training hyperparameters
        self.max_iteration = kwargs["max_iteration"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]

        # Logging setup
        self.save_folder = kwargs["save_folder"]
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        add_scalars({tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0)
        self.writer.flush()

        # Sampling time tracking
        self.sampler_tb_dict = LogData()

        # Best reward tracking (initial: -inf)
        self.best_tar = -inf
        self.global_iteration = 0

        # GPU config
        self.use_gpu = kwargs["use_gpu"]

        # Training start time
        self.start_time = time.time()


    def step(self):
        # Sampling
        samples_with_replay_format, sampler_tb_dict = self.sampler.sample_with_replay_format()
        self.sampler_tb_dict.add_average(sampler_tb_dict)

        # Learning
        if self.use_gpu:
            for k, v in samples_with_replay_format.items():
                samples_with_replay_format[k] = v.cuda()
        with ModuleOnDevice(self.networks, "cuda" if self.use_gpu else "cpu"):
            self.networks.train()
            alg_tb_dict, self.global_iteration = self.alg.model_update(samples_with_replay_format)
            self.networks.eval()

        # Logging
        if self.global_iteration % self.log_save_interval == 0:
            print("Iter = ", self.global_iteration, "save training data and average sampling time!")
            add_scalars(alg_tb_dict, self.writer, step=self.global_iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.global_iteration)

        # Model saving
        if self.global_iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

        # Evaluation (no Ray, run at eval_interval)
        if self.global_iteration % self.eval_interval == 0 and self.global_iteration > 0:
            with ModuleOnDevice(self.networks, "cpu"):
                total_return_mean, total_return_std, total_cost_mean, total_cost_std = self.evaluator.run_evaluation(self.global_iteration)

            # Save best model (reward-based, training > 20% complete)
            if total_return_mean >= self.best_tar and self.global_iteration >= self.max_iteration / 5:
                self.best_tar = total_return_mean
                print("Eval_Iter: {}, Highest total average return = {}! Current total average cost = {}".format(
                    str(self.global_iteration), str(self.best_tar), str(total_cost_mean)))

                # Delete old optimal models
                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                # Save new best model
                torch.save(self.networks.state_dict(),  
                           self.save_folder + "/apprfunc/apprfunc_{}_opt.pkl".format(self.global_iteration))

            # Log return metrics
            self.writer.add_scalar(tb_tags["TRM of RL iteration"], total_return_mean, self.global_iteration)
            self.writer.add_scalar(tb_tags["TRS of RL iteration"], total_return_std, self.global_iteration)
            self.writer.add_scalar(tb_tags["TRM of total time"], total_return_mean, int(time.time() - self.start_time))

            # Log cost metrics
            self.writer.add_scalar(tb_tags["TCM of RL iteration"], total_cost_mean, self.global_iteration)
            self.writer.add_scalar(tb_tags["TCS of RL iteration"], total_cost_std, self.global_iteration)
            self.writer.add_scalar(tb_tags["TCM of total time"], total_cost_mean, int(time.time() - self.start_time))


    def train(self):
        # Main training loop
        while self.global_iteration < self.max_iteration:
            self.step()

        self.save_apprfunc()
        self.writer.flush()

    # Save network weights
    def save_apprfunc(self):
        torch.save(self.networks.state_dict(),
                   self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.global_iteration))