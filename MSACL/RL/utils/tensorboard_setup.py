"""TensorBoard logging config (TRM: total return mean)"""
import os
import numpy as np
import pandas as pd

# TensorBoard browser setup
import socket
import subprocess
import webbrowser
import random

# TensorBoard tag mappings (for organized logging)
tb_tags = {
    # Evaluation - Return metrics
    "TRM of RL iteration": "Evaluation/1-1. TRM-RL iter",
    "TRS of RL iteration": "Evaluation/1-1. TRS-RL iter",  # Return std
    "TRM of total time": "Evaluation/2-1. TRM-Total time [s]",
    "TRM of collected samples": "Evaluation/3-1. TRM-Collected samples",
    "TRM of replay samples": "Evaluation/4-1. TRM-Replay samples",

    # Evaluation - Cost metrics (TCM: total cost mean)
    "TCM of RL iteration": "Evaluation/1-2. TCM-RL iter",
    "TCS of RL iteration": "Evaluation/1-2. TCS-RL iter",  # Cost std
    "TCM of total time": "Evaluation/2-2. TCM-Total time [s]",
    "TCM of collected samples": "Evaluation/3-2. TCM-Collected samples",
    "TCM of replay samples": "Evaluation/4-2. TCM-Replay samples",

    # Buffer RAM usage
    "Buffer RAM of RL iteration": "RAM/RAM [MB]-RL iter",

    # Training losses
    "loss_actor": "Loss/Actor loss-RL iter",
    "loss_critic": "Loss/Critic loss-RL iter",
    "loss_entropy": "Loss/Entropy loss-RL iter",
    "loss_lyapunov": "Loss/Lyapunov loss-RL iter",

    # Timing metrics
    "alg_time": "Time/Algorithm time [ms]-RL iter",
    "sampler_time": "Time/Sampler time [ms]-RL iter",
}

def add_scalars(tb_info, writer, step):
    """Add scalars to TensorBoard writer"""
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)

def read_tensorboard(path):
    """Parse TensorBoard event files to extract scalar data"""
    import logging
    import tensorboard
    from tensorboard.backend.event_processing import event_accumulator

    logging.getLogger("tensorboard").setLevel(logging.ERROR)
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    valid_key_list = ea.scalars.Keys()

    output_dict = dict()
    for key in valid_key_list:
        event_list = ea.scalars.Items(key)
        x, y = [], []
        for e in event_list:
            x.append(e.step)
            y.append(e.value)
        data_dict = {"x": np.array(x), "y": np.array(y)}
        output_dict[key] = data_dict
    return output_dict

def save_csv(path, step, value):
    """Save step/value data to CSV (2 columns)"""
    df = pd.DataFrame({"Step": step, "Value": value})
    df.to_csv(path, index=False, sep=",")

def save_tb_to_csv(path):
    """Convert TensorBoard scalar data to CSV files"""
    data_dict = read_tensorboard(path)
    for data_name in data_dict.keys():
        data_name_format = data_name.replace("\\", "/").replace("/", "_")
        csv_dir = os.path.join(path, "data")
        os.makedirs(csv_dir, exist_ok=True)
        save_csv(
            os.path.join(csv_dir, "{}.csv".format(data_name_format)),
            step=data_dict[data_name]["x"],
            value=data_dict[data_name]["y"],
        )

def find_free_port():
    """Find free port in 6006-7006 range (max 20 attempts)"""
    for _ in range(20):
        port = random.randint(6006, 7006)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                s.listen(1)
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found (6006-7006)")

def open_tb_in_browser(logdir):
    """Launch TensorBoard and show access URL (VS Code SSH compatible)"""
    port = find_free_port()
    host = "0.0.0.0"  # Listen on all interfaces for port forwarding

    # Start TensorBoard process (silent output)
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", host],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    local_url = f"http://localhost:{port}"
    print("\n TensorBoard started!")
    print(f"  Log directory: {logdir}")
    print(f"  Listening port: {port}")
    print(f"  Local access URL (VS Code port forwarding): {local_url}")