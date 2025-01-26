import os
import pickle
from collections import defaultdict

import absl.app
import absl.flags
import numpy as np
import gym
import transformers
from flax.training.early_stopping import EarlyStopping
from flaxmodels.flaxmodels.lstm.lstm import LSTMRewardModel
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

from typing import Any, Dict, Optional, Union

import h5py
import torch
from .sampler import TrajSampler
from .jax_utils import batch_to_jax
from .replay_buffer import index_batch
import JaxPref.reward_transform as r_tf
from .model import FullyConnectedQFunction
from viskit.logging import logger, setup_logger
from .MR import MR
from .NMR import NMR
from .PrefTransformer import PrefTransformer
from .utils import (
    Timer,
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    prefix_metrics,
    WandBLogger,
    save_pickle,
)

# Jax memory settings
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

FLAGS_DEF = define_flags_with_default(
    env="assembly",
    model_type="MLP",
    max_traj_length=1000,
    seed=42,
    data_seed=42,
    save_model=True,
    batch_size=64,
    early_stop=False,
    min_delta=1e-3,
    patience=10,
    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,
    reward_arch="256-256",
    orthogonal_init=False,
    activations="relu",
    activation_final="none",
    training=True,
    n_epochs=2000,
    eval_period=5,
    data_dir="./human_label",
    num_query=500,
    query_len=25,
    skip_flag=0,
    balance=False,
    topk=10,
    window=2,
    use_human_label=False,
    feedback_random=False,
    feedback_uniform=False,
    enable_bootstrap=False,
    comment="",
    robosuite=False,
    robosuite_dataset_type="ph",
    robosuite_dataset_path="./data",
    robosuite_max_episode_steps=500,
    reward=MR.get_default_config(),
    transformer=PrefTransformer.get_default_config(),
    lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def nest_dict(d: Dict, separator: str = ".") -> Dict:
    nested_d = dict()
    for key in d.keys():
        key_parts = key.split(separator)
        current_d = nested_d
        while len(key_parts) > 1:
            if key_parts[0] not in current_d:
                current_d[key_parts[0]] = dict()
            current_d = current_d[key_parts[0]]
            key_parts.pop(0)
        current_d[key_parts[0]] = d[key]  # Set the value
    return nested_d


def get_from_batch(
    batch: Any, start: Union[int, np.ndarray, torch.Tensor], end: Optional[int] = None
) -> Any:
    if isinstance(batch, (dict, h5py.Group)):
        return {k: get_from_batch(v, start, end=end) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [get_from_batch(v, start, end=end) for v in batch]
    elif isinstance(batch, (np.ndarray, torch.Tensor, h5py.Dataset)):
        if end is None:
            return batch[start]
        else:
            return batch[start:end]
    else:
        raise ValueError("Unsupported type passed to `get_from_batch`")


# def remove_float64(batch: Any):
#     if isinstance(batch, dict):
#         return {k: remove_float64(v) for k, v in batch.items()}
#     elif isinstance(batch, (list, tuple)):
#         return [remove_float64(v) for v in batch]
#     elif isinstance(batch, np.ndarray):
#         if batch.dtype == np.float64:
#             return batch.astype(np.float32)
#     elif isinstance(batch, torch.Tensor):
#         if batch.dtype == torch.double:
#             return batch.float()
#     else:
#         raise ValueError("Unsupported type passed to `remove_float64`")
#     return batch


def load_metaworld_dataset(file_path):
    """Load and preprocess MetaWorld dataset from npz file."""
    data = np.load(file_path)
    N1, T1 = data["obs_1"].shape[:2]  # N and T from obs_1
    N2, T2 = data["obs_2"].shape[:2]  # N and T from obs_2

    # Create timestamps with shape (N, T)
    timestep_1 = np.tile(np.arange(T1), (N1, 1))  # Shape: (N1, T1)
    timestep_2 = np.tile(np.arange(T2), (N2, 1))  # Shape: (N2, T2)

    # Apply one-hot encoding to labels
    one_hot_labels = np.array([to_one_hot(label) for label in data["label"]])

    dataset = {
        "observations": data["obs_1"],  # Shape: (N, T, obs_dim)
        "observations_2": data["obs_2"],  # Shape: (N, T, obs_dim)
        "timestep_1": timestep_1,
        "timestep_2": timestep_2,
        "actions": data["action_1"],  # Shape: (N, T, action_dim)
        "actions_2": data["action_2"],  # Shape: (N, T, action_dim)
        "labels": one_hot_labels,  # Shape: (N,)
    }
    total_size = dataset["observations"].shape[0]
    random_indices = np.random.choice(total_size, size=500, replace=False)

    dataset = {
        k: v[random_indices] if isinstance(v, np.ndarray) else v
        for k, v in dataset.items()
    }

    data = nest_dict(dataset)
    # dataset = remove_float64(data)
    # data = nest_dict(dataset)
    # data = get_from_batch(data, 0, 500)
    # dataset = remove_float64(data)
    lim = 1 - 1e-3
    dataset["actions"] = np.clip(dataset["actions"], a_min=-lim, a_max=lim)
    dataset["actions_2"] = np.clip(dataset["actions_2"], a_min=-lim, a_max=lim)

    traj_len = np.asarray([o.shape[0] for o in dataset["observations"]])
    data_size = len(traj_len)
    print(data_size)
    return dataset


def to_one_hot(label):
    """Convert binary label to one-hot encoded format."""
    if label == 0:
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def main(_):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    # Setup logging
    save_dir = os.path.join(
        FLAGS.logging.output_dir,
        "metaworld",
        FLAGS.model_type,
        FLAGS.comment,
        f"s{FLAGS.seed}",
    )
    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False,
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    # Load MetaWorld dataset
    dataset = load_metaworld_dataset("./human_label/drawer-open_10000.npz")
    pref_dataset = {}
    pref_eval_dataset = {}

    for key, array in dataset.items():
        # Shuffle the array indices
        total_samples = array.shape[0]
        indices = np.arange(total_samples)
        # np.random.shuffle(indices)

        # Compute split point
        split_point = int(total_samples * 0.9)

        # Split the data
        train_indices = indices[:split_point]
        eval_indices = indices[split_point:]

        pref_dataset[key] = array[train_indices]
        pref_eval_dataset[key] = array[eval_indices]

    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    # Setup training parameters
    data_size = pref_dataset["observations_2"].shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1

    eval_data_size = pref_eval_dataset["observations_2"].shape[0]
    eval_interval = int(eval_data_size / FLAGS.batch_size) + 1

    early_stop = EarlyStopping(min_delta=FLAGS.min_delta, patience=FLAGS.patience)

    # Initialize reward model based on model type
    if FLAGS.model_type == "MR":
        rf = FullyConnectedQFunction(
            observation_dim,
            action_dim,
            FLAGS.reward_arch,
            FLAGS.orthogonal_init,
            FLAGS.activations,
            FLAGS.activation_final,
        )
        reward_model = MR(FLAGS.reward, rf)

    elif FLAGS.model_type == "PrefTransformer":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(**FLAGS.transformer)
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        trans = TransRewardModel(
            config=config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=FLAGS.activations,
            activation_final=FLAGS.activation_final,
        )
        reward_model = PrefTransformer(config, trans)

    elif FLAGS.model_type == "NMR":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(**FLAGS.lstm)
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        lstm = LSTMRewardModel(
            config=config,
            observation_dim=observation_dim,
            action_dim=action_dim,
            activation=FLAGS.activations,
            activation_final=FLAGS.activation_final,
        )
        reward_model = NMR(config, lstm)
    # Training loop
    criteria_key = None

    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        if epoch:
            # Training phase
            shuffled_idx = np.random.permutation(pref_dataset["observations"].shape[0])
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min(
                    (i + 1) * FLAGS.batch_size, pref_dataset["observations"].shape[0]
                )
                with Timer() as train_timer:
                    # Training
                    batch = batch_to_jax(
                        index_batch(pref_dataset, shuffled_idx[start_pt:end_pt])
                    )
                    for key, val in prefix_metrics(
                        reward_model.train(batch, FLAGS.env), "reward"
                    ).items():
                        metrics[key].append(val)
            metrics["train_time"] = train_timer()
        else:
            # For early stopping initialization
            metrics["train_loss"] = [float(FLAGS.query_len)]

        # Evaluation phase
        if epoch % FLAGS.eval_period == 0:
            for j in range(eval_interval):
                eval_start_pt, eval_end_pt = j * FLAGS.batch_size, min(
                    (j + 1) * FLAGS.batch_size,
                    pref_eval_dataset["observations"].shape[0],
                )
                batch_eval = batch_to_jax(
                    index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt))
                )
                for key, val in prefix_metrics(
                    reward_model.evaluation(batch_eval, FLAGS.env), "reward"
                ).items():
                    metrics[key].append(val)

            # Select criteria_key for early stopping
            if not criteria_key:
                if "antmaze" in FLAGS.env and not "dense" in FLAGS.env and not True:
                    criteria_key = (
                        "train_loss"  # Use train loss for antmaze sparse environments
                    )
                else:
                    criteria_key = key  # Default to eval loss

            # Evaluate early stopping and model improvement
            criteria = np.mean(metrics[criteria_key])
            has_improved, should_stop = early_stop.update(criteria)

            if should_stop and FLAGS.early_stop:
                print("Met early stopping criteria, breaking...")
                break
            elif epoch > 0 and has_improved:
                # print(f"Improved at epoch {epoch}: {criteria_key} = {criteria}")
                metrics["best_epoch"] = epoch
                metrics[f"{criteria_key}_best"] = criteria

                # Save the best model
                save_data = {
                    "reward_model": reward_model,
                    "variant": variant,
                    "epoch": epoch,
                }
                save_pickle(save_data, "best_model.pkl", save_dir)

        # Aggregate metrics for logging
        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)

        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics)

    # Save the final model
    if FLAGS.save_model:
        save_data = {"reward_model": reward_model, "variant": variant, "epoch": epoch}
        save_pickle(save_data, "model.pkl", save_dir)
    # # Training loop
    # for epoch in range(FLAGS.n_epochs + 1):
    #     metrics = defaultdict(list)
    #     metrics["epoch"] = epoch

    #     if epoch:
    #         # Training phase
    #         shuffled_idx = np.random.permutation(data_size)
    #         for i in range(interval):
    #             start_pt = i * FLAGS.batch_size
    #             end_pt = min((i + 1) * FLAGS.batch_size, data_size)

    #             with Timer() as train_timer:
    #                 batch_idx = shuffled_idx[start_pt:end_pt]
    #                 batch = {
    #                     "observations": pref_dataset["observations"][batch_idx],
    #                     "observations_2": pref_dataset["observations_2"][batch_idx],
    #                     "actions": pref_dataset["actions"][batch_idx],
    #                     "actions_2": pref_dataset["actions_2"][batch_idx],
    #                     "labels": pref_dataset["labels"][batch_idx],
    #                 }
    #                 batch = batch_to_jax(batch)

    #                 for key, val in prefix_metrics(
    #                     reward_model.train(batch), "reward"
    #                 ).items():
    #                     metrics[key].append(val)

    #         metrics["train_time"] = train_timer()

    #     # Evaluation phase
    #     if epoch % FLAGS.eval_period == 0:
    #         for j in range(eval_interval):
    #             eval_start_pt = j * FLAGS.batch_size
    #             eval_end_pt = min((j + 1) * FLAGS.batch_size, eval_data_size)

    #             batch_eval = {
    #                 "observations": pref_eval_dataset["observations"][
    #                     eval_start_pt:eval_end_pt
    #                 ],
    #                 "observations_2": pref_eval_dataset["observations_2"][
    #                     eval_start_pt:eval_end_pt
    #                 ],
    #                 "actions": pref_eval_dataset["actions"][
    #                     eval_start_pt:eval_end_pt
    #                 ],
    #                 "actions_2": pref_eval_dataset["actions_2"][
    #                     eval_start_pt:eval_end_pt
    #                 ],
    #                 "labels": pref_eval_dataset["labels"][eval_start_pt:eval_end_pt],
    #             }
    #             batch_eval = batch_to_jax(batch_eval)

    #             for key, val in prefix_metrics(
    #                 reward_model.evaluation(batch_eval), "reward"
    #             ).items():
    #                 metrics[key].append(val)

    #     # Log metrics and save model
    #     for key, val in metrics.items():
    #         if isinstance(val, list):
    #             metrics[key] = np.mean(val)

    #     logger.record_dict(metrics)
    #     logger.dump_tabular(with_prefix=False, with_timestamp=False)
    #     wb_logger.log(metrics)

    #     if FLAGS.save_model and epoch % FLAGS.eval_period == 0:
    #         save_data = {
    #             "reward_model": reward_model,
    #             "variant": variant,
    #             "epoch": epoch,
    #         }
    #         save_pickle(save_data, f"model_epoch_{epoch}.pkl", save_dir)


if __name__ == "__main__":
    absl.app.run(main)
