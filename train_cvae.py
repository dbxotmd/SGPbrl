import datetime
import os
import pickle
from typing import Tuple
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from JaxPref.subgoal_cvae_model import SubgoalCVAE
from dataset_utils import reward_from_preference_transformer

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer(
    "seq_len", 25, "sequence length for relabeling reward in Transformer."
)
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("eval_interval", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "use_reward_model", False, "Use reward model for relabeling reward."
)
flags.DEFINE_string("model_type", "MLP", "type of reward model.")
flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
flags.DEFINE_bool(
    "use_diff",
    False,
    "boolean whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with tranformer.")
flags.DEFINE_string("comment", "cvae", "comment for distinguishing experiments.")

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


class SubgoalCVAETrainer:
    def __init__(self, observation_dim, action_dim, config):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config

        # CVAE parameters
        self.latent_dim = 16
        self.hidden_dims = [32, 64, 32]
        self.learning_rate = 1e-4
        self.batch_size = FLAGS.batch_size

        self.cvae = SubgoalCVAE(
            self.latent_dim, self.observation_dim, self.action_dim, self.hidden_dims
        )
        self.train_state = self.create_train_state()

    def create_train_state(self):
        params = self.cvae.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, self.observation_dim)),
            jnp.ones((1, self.action_dim)),
            jnp.ones((1, self.observation_dim)),
        )
        tx = optax.adam(self.learning_rate)
        return TrainState.create(apply_fn=self.cvae.apply, params=params, tx=tx)

    # @partial(jax.jit, static_argnames=["self"])
    def train_step(self, state, batch):
        def loss_fn(params):
            recon_subgoal, post_mu, post_log_var, prior_mu, prior_log_var = (
                state.apply_fn(
                    params,
                    batch["state"],
                    batch["action"],
                    batch["subgoal_state"],
                    training=True,
                )
            )

            # Reconstruction loss (cosine similarity)
            similarity = self.cosine_similarity(recon_subgoal, batch["subgoal_state"])
            recon_loss = -jnp.mean(
                similarity
            )  # Negative because we want to maximize similarity

            # KL divergence loss
            kl_loss = 0.5 * jnp.mean(
                jnp.exp(post_log_var - prior_log_var)
                + ((post_mu - prior_mu) ** 2) / jnp.exp(prior_log_var)
                - 1
                + prior_log_var
                - post_log_var
            )

            # Total loss
            loss = recon_loss + 0.1 * kl_loss  # Beta-VAE style weighting
            return loss, (recon_loss, kl_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (recon_loss, kl_loss)), grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss, recon_loss, kl_loss

    def cosine_similarity(self, a, b):
        return jnp.sum(a * b, axis=-1) / (
            jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
        )


def process_dataset(dataset, attn_weight):
    """Process dataset to include subgoal states based on attention weights."""
    print(attn_weight)
    processed_data = []
    attn_threshold = 0.1

    # Assume dataset contains attention weights for each state
    for i in range(dataset.size):
        state = dataset.observations[i]
        action = dataset.actions[i]
        next_state = dataset.next_observations[i]
        attn_weight = dataset.attention_weights[i]  # Assuming this exists

        # Mark states with high attention weights as subgoals
        is_subgoal = attn_weight > attn_threshold
        subgoal_state = next_state if is_subgoal else state

        processed_data.append(
            {
                "state": state,
                "action": action,
                "subgoal_state": subgoal_state,
                "is_valid": is_subgoal,
            }
        )

    return processed_data


def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt["reward_model"]
    return reward_model


def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )
    trj_mapper = []
    for trj_idx, traj in tqdm(
        enumerate(trajs), total=len(trajs), desc="chunk trajectories"
    ):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(
        sorted_trajs[-1]
    )

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = dataset.rewards[i]
        if "antmaze" in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset.rewards = np.array(normalized_rewards)


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        dataset, (attn_weights, pts) = reward_from_preference_transformer(
            FLAGS.env_name,
            dataset,
            reward_model,
            batch_size=FLAGS.batch_size,
            seq_len=FLAGS.seq_len,
            use_diff=FLAGS.use_diff,
            label_mode=FLAGS.label_mode,
            with_attn_weights=True,
        )
        del reward_model

    if FLAGS.use_reward_model:
        normalize(
            dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps
        )
        if "antmaze" in FLAGS.env_name:
            dataset.rewards -= 1.0
        if (
            "halfcheetah" in FLAGS.env_name
            or "walker2d" in FLAGS.env_name
            or "hopper" in FLAGS.env_name
        ):
            dataset.rewards += 0.5
    else:
        if "antmaze" in FLAGS.env_name:
            dataset.rewards -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # but I found no difference between (x - 0.5) * 4 and x - 1.0
        elif (
            "halfcheetah" in FLAGS.env_name
            or "walker2d" in FLAGS.env_name
            or "hopper" in FLAGS.env_name
        ):
            normalize(
                dataset,
                FLAGS.env_name,
                max_episode_steps=env.env.env._max_episode_steps,
            )

    dataset = process_dataset(dataset, attn_weights)

    return (env,)


def main(_):
    save_dir = os.path.join(
        FLAGS.save_dir,
        "cvae",
        FLAGS.env_name,
        FLAGS.comment,
        str(FLAGS.seed),
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    trainer = SubgoalCVAETrainer(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=FLAGS.config,
    )

    # Training loop
    for step in tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        # Sample batch
        batch_indices = np.random.choice(len(dataset), FLAGS.batch_size, replace=False)
        batch = {
            "state": jnp.array([dataset[i]["state"] for i in batch_indices]),
            "action": jnp.array([dataset[i]["action"] for i in batch_indices]),
            "subgoal_state": jnp.array(
                [dataset[i]["subgoal_state"] for i in batch_indices]
            ),
            "is_valid": jnp.array([dataset[i]["is_valid"] for i in batch_indices]),
        }

        # Training step
        trainer.train_state, loss, recon_loss, kl_loss = trainer.train_step(
            trainer.train_state, batch
        )

        # Logging
        if step % FLAGS.log_interval == 0:
            metrics = {
                "loss": loss,
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
            }

            for k, v in metrics.items():
                summary_writer.add_scalar(f"training/{k}", v, step)
            summary_writer.flush()

        # Save model periodically
        if step % (FLAGS.max_steps // 10) == 0:
            model_path = os.path.join(save_dir, f"cvae_model_step_{step}.npz")
            jnp.savez(model_path, **trainer.train_state.params)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)
