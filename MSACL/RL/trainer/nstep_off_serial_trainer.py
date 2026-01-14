"""
Derived from off_serial_trainer.py for n-step off-policy serial training
"""
"""
Integrates alg/sampler/buffer/evaluator for complete training+evaluation pipeline
"""

_all__ = ["NstepOffSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from RL.utils.tensorboard_setup import tb_tags, add_scalars
from RL.utils.log_data import LogData
from RL.utils.common_utils import ModuleOnDevice

class NstepOffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator,** kwargs):
        # Core components
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        # PER disabled by default
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"

        # Network initialization
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # Load initial network if provided
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        # Training hyperparameters
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.policy_frequency = kwargs["policy_frequency"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.save_folder = kwargs["save_folder"]
        self.eval_interval = kwargs["eval_interval"]

        # Training tracking
        self.best_tar = -inf
        self.iteration = 0

        # TensorBoard initialization
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        add_scalars({tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0)
        self.writer.flush()

        # Pre-sampling: fill buffer to warm-up size
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.sampler_tb_dict = LogData()

        # GPU setup
        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        self.start_time = time.time()


    def step(self):
        # Sampling
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.sampler_tb_dict.add_average(sampler_tb_dict)

        # Replay buffer sampling
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # Move data to GPU
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        # Model training
        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.model_update(replay_samples, self.iteration)
            self.buffer.update_batch(idx, new_priority)
        else:
            if self.iteration % self.policy_frequency == 0:
                alg_tb_dict = self.alg.model_update(replay_samples, self.iteration)
                if self.iteration % self.log_save_interval == 0:
                    print("Iter = ", self.iteration, "save training data!")
                    add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            else:
                self.alg.model_update(replay_samples, self.iteration)
        self.networks.eval()

        # Log sampling time
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration, "save average sampling time!")
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

        # Save model
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

        # Evaluation
        if self.iteration % self.eval_interval == 0 and self.iteration > 0:
            with ModuleOnDevice(self.networks, "cpu"):
                total_return_mean, total_return_std, total_cost_mean, total_cost_std = self.evaluator.run_evaluation(self.iteration)

            # Save best model
            if total_return_mean >= self.best_tar and self.iteration >= self.max_iteration / 5:
                self.best_tar = total_return_mean
                print("Eval_Iter: {}, Highest total average return = {}! Current total average cost = {}".format(
                    str(self.iteration), str(self.best_tar), str(total_cost_mean)))

                # Delete old optimal models
                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                # Save new best model
                torch.save(self.networks.state_dict(),  
                           self.save_folder + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration))

            # Log buffer RAM
            self.writer.add_scalar(tb_tags["Buffer RAM of RL iteration"], self.buffer.__get_RAM__(), self.iteration)

            # Log return metrics
            self.writer.add_scalar(tb_tags["TRM of RL iteration"], total_return_mean, self.iteration)
            self.writer.add_scalar(tb_tags["TRS of RL iteration"], total_return_std, self.iteration)
            self.writer.add_scalar(tb_tags["TRM of total time"], total_return_mean, int(time.time() - self.start_time))

            
            # Log cost metrics
            self.writer.add_scalar(tb_tags["TCM of RL iteration"], total_cost_mean, self.iteration)
            self.writer.add_scalar(tb_tags["TCS of RL iteration"], total_cost_std, self.iteration)
            self.writer.add_scalar(tb_tags["TCM of total time"], total_cost_mean, int(time.time() - self.start_time))



    def train(self):
        # Main training loop
        while self.iteration <= self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()
    
    # Save network weights
    def save_apprfunc(self):
        torch.save(self.networks.state_dict(),
                   self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration))