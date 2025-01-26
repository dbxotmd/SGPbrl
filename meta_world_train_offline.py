import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import metaworld
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

from typing import Any, Dict, Optional, Union

import h5py
import torch

from dataset_utils import (
    RelabeledDataset,
    MetaworldDataset,
    metaworld_reward_from_preference,
    metaworld_reward_from_preference_transformer,
    split_into_trajectories,
)
from metaworld_evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "button-press-v2", "Metaworld environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 500, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 128, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "use_reward_model", True, "Use reward model for relabeling reward."
)
flags.DEFINE_string("model_type", "MLP", "type of reward model.")
flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
flags.DEFINE_string("comment", "metaworld", "comment for distinguishing experiments.")
flags.DEFINE_integer(
    "seq_len", 25, "sequence length for relabeling reward in Transformer."
)
flags.DEFINE_boolean(
    "use_diff",
    False,
    "boolean whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with tranformer.")
flags.DEFINE_string(
    "pref_attn_type", "max", "mode for preference attention with tranformer."
)
flags.DEFINE_integer("max_episode_steps", 500, "max_episode_steps for rollout.")
flags.DEFINE_string("dataset_path", "./data", "dataset path for demonstrations")

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def normalize(dataset, bs, max_episode_steps=500):

    data_size = dataset.observations.shape[0]
    for i_batch in range((data_size - 1) // bs + 1):
        idx = np.arange(i_batch * bs, min((i_batch + 1) * bs, data_size))
        dataset.rewards[idx] = dataset.rewards[idx] * dataset.masks[idx]

    print(dataset.rewards.shape)
    return_ = dataset.rewards.sum(1)
    print(abs(return_.max()))
    print(abs(return_.min()))
    print(return_.max() - return_.min())
    max_return = max(
        abs(return_.max()), abs(return_.min()), return_.max() - return_.min(), 1.0
    )
    norm = 500.0 / max_return
    dataset.rewards *= norm
    return_ = dataset.rewards.sum(1)
    print(
        f"[MetaworldOfflineDataset]: return range: [{return_.min()}, {return_.max()}], multiplying norm factor {norm}."
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


def load_metaworld_dataset(file_path):
    data = np.load(file_path)
    # data = nest_dict(data)
    # data = get_from_batch(data, 0, 5000)
    # data = remove_float64(data)
    total_size = data["obs_1"].shape[0]
    random_indices = np.random.choice(total_size, size=5000, replace=False)
    dataset = {
        k: v[random_indices] if isinstance(v, np.ndarray) else v
        for k, v in data.items()
    }

    data = nest_dict(dataset)
    # dataset = remove_float64(data)
    N, L = data["obs_1"].shape[:2]

    data = {
        "observations": np.stack([data["obs_1"], data["obs_2"]], axis=0).reshape(
            2 * N, L, -1
        )[:, :-1],
        "next_observations": np.stack([data["obs_1"], data["obs_2"]], axis=0).reshape(
            2 * N, L, -1
        )[:, 1:],
        "actions": np.stack([data["action_1"], data["action_2"]], axis=0).reshape(
            2 * N, L, -1
        )[:, :-1],
    }
    data["terminals"] = np.zeros([2 * N, L - 1], dtype=np.bool_)
    data["rewards"] = np.zeros([2 * N, L - 1], dtype=np.float32)
    data["timestep"] = np.tile(np.arange(L), (2 * N, 1))
    data["masks"] = np.ones([2 * N, L - 1], dtype=np.float32)

    traj_len = np.asarray([o.shape[0] for o in data["observations"]])
    data_size = len(traj_len)
    print(data_size)
    print(data["observations"].shape)

    if 5000 > data_size:
        print(f"[Warning]: capacity 5000 exceeds dataset size {data_size}")
    # data_size = min(data_size, 5000)
    # data = {k: data[k][:data_size] for k in data}
    # traj_len = traj_len[:data_size]
    return data


def make_env_and_dataset(
    env_name: str, seed: int, dataset_path: str, max_episode_steps: int = 500
) -> Tuple[gym.Env, RelabeledDataset]:

    # Create Metaworld environment
    env_cls = ALL_V2_ENVIRONMENTS[env_name]
    env = env_cls()
    env._max_episode_steps = env.max_path_length

    # Set random seeds
    env.seed(seed)
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Load offline dataset
    dataset = load_metaworld_dataset("./human_label/drawer-open_10000.npz")
    dataset = MetaworldDataset(dataset)

    # Apply reward relabeling if using reward model
    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        if FLAGS.model_type == "MR":
            dataset = metaworld_reward_from_preference(
                FLAGS.env_name, dataset, reward_model, batch_size=FLAGS.batch_size
            )
        else:
            dataset, _ = metaworld_reward_from_preference_transformer(
                FLAGS.env_name,
                dataset,
                reward_model,
                batch_size=FLAGS.batch_size,
                seq_len=FLAGS.seq_len,
                use_diff=FLAGS.use_diff,
                label_mode=FLAGS.label_mode,
            )
        del reward_model

        normalize(dataset, FLAGS.batch_size, max_episode_steps=max_episode_steps)
    return env, dataset


def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt["reward_model"]
    if FLAGS.model_type == "PrefTransformer":
        reward_model.trans.config.pref_attn_type = FLAGS.pref_attn_type
    return reward_model


def main(_):
    save_dir = os.path.join(
        FLAGS.save_dir,
        "tb",
        FLAGS.env_name,
        (
            f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}"
            if FLAGS.use_reward_model
            else "original"
        ),
        FLAGS.comment,
        str(FLAGS.seed),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        FLAGS.dataset_path,
        max_episode_steps=FLAGS.max_episode_steps,
    )

    kwargs = dict(FLAGS.config)
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        **kwargs,
    )

    eval_returns = []
    for i in tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", v, i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.seed, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f"evaluation/average_{k}s", v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats["return"]))
            np.savetxt(
                os.path.join(save_dir, "progress.txt"), eval_returns, fmt=["%d", "%.1f"]
            )

    # Save final model checkpoints
    checkpoints.save_checkpoint(
        os.path.join(save_dir, "actor"), target=agent.actor, step=FLAGS.max_steps
    )
    checkpoints.save_checkpoint(
        os.path.join(save_dir, "critic"), target=agent.critic, step=FLAGS.max_steps
    )
    checkpoints.save_checkpoint(
        os.path.join(save_dir, "value"), target=agent.value, step=FLAGS.max_steps
    )
    checkpoints.save_checkpoint(
        os.path.join(save_dir, "target_critic"),
        target=agent.target_critic,
        step=FLAGS.max_steps,
    )


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)
