import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import (
    D4RLDataset,
    reward_from_preference,
    reward_from_preference_transformer,
    split_into_trajectories,
)
from evaluation import evaluate
from learner import Learner
from JaxPref.subgoal_cvae_model import SubgoalCVAE

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax
from flax import linen as nn

# 추가로 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 필요하다면 PCA도 불러올 수 있음
from sklearn.decomposition import PCA

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "use_reward_model", False, "Use reward model for relabeling reward."
)
flags.DEFINE_string("model_type", "MLP", "type of reward model.")
flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
flags.DEFINE_string("comment", "base", "comment for distinguishing experiments.")
flags.DEFINE_integer(
    "seq_len", 25, "sequence length for relabeling reward in Transformer."
)
flags.DEFINE_bool(
    "use_diff",
    False,
    "boolean whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with tranformer.")

config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


class EmbeddingModel(nn.Module):
    target_dim: int  # Target dimension for embedding

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.target_dim)(x)  # Linear layer to map to target_dim
        return x


def load_vae_model(filename, observation_dim, action_dim):
    with open(filename, "rb") as f:
        loaded_params = pickle.load(f)

    vae_latent_dim = 16
    vae_hidden_dim = [32, 64, 32]

    vae = SubgoalCVAE(vae_latent_dim, observation_dim, action_dim, vae_hidden_dim)
    rng = jax.random.PRNGKey(0)

    dummy_state = jnp.ones((1, observation_dim))
    dummy_action = jnp.ones((1, action_dim))
    dummy_subgoal_state = jnp.ones((1, observation_dim))
    params = vae.init(rng, dummy_state, dummy_action, dummy_subgoal_state)

    # Create a dummy optimizer
    tx = optax.adam(learning_rate=1e-4)

    # Create a dummy train state
    vae_state = train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=tx)

    vae_state = vae_state.replace(params=loaded_params)
    print(f"VAE model parameters loaded from {filename}")

    # vae_state = vae_state.replace(params=load_params["params"])
    return vae_state


# def load_vae_model(filename, observation_dim, action_dim):
#     params = jax.numpy.load(filename)
#     vae_latent_dim = 16
#     vae_hidden_dim = [32, 64, 32]
#     # Initialize a fresh VAE model to get its parameter structure
#     vae = SubgoalCVAE(vae_latent_dim, observation_dim, action_dim, vae_hidden_dim)
#     # Create a dummy input to initialize the model parameters
#     dummy_state = jnp.ones((1, observation_dim))
#     dummy_action = jnp.ones((1, action_dim))
#     subgoal_state = jnp.ones((1, observation_dim))
#     rng = jax.random.PRNGKey(0)
#     params = vae.init(rng, dummy_state, dummy_action, subgoal_state)

#     # Create a dummy optimizer
#     tx = optax.adam(learning_rate=1e-4)

#     # Create a dummy train state
#     vae_state = train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=tx)
#     vae_state = vae_state.replace(params=params)
#     return vae_state


def normalize(dataset, env_name, max_episode_steps=1000):
    """d4rl dataset의 reward를 (min, max)로 정규화 + antmaze 보정 등."""
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
        return sum([step[2] for step in traj])

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
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    dataset.rewards = np.array(normalized_rewards)


def make_env_and_dataset(
    env_name: str, seed: int
) -> Tuple[gym.Env, D4RLDataset, np.ndarray]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)
    ground_truth = dataset.rewards.copy()  # 원본 reward 백업

    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        if FLAGS.model_type == "MR":
            dataset = reward_from_preference(
                FLAGS.env_name, dataset, reward_model, batch_size=FLAGS.batch_size
            )
        else:
            dataset = reward_from_preference_transformer(
                FLAGS.env_name,
                dataset,
                reward_model,
                batch_size=FLAGS.batch_size,
                seq_len=FLAGS.seq_len,
                use_diff=FLAGS.use_diff,
                label_mode=FLAGS.label_mode,
            )
        del reward_model

    # 보통 reward를 정규화 (antmaze, mujoco 계열 등등)
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

    return env, dataset, ground_truth


def initialize_model():
    """
    reward model(혹은 preference transformer)의 파라미터를 로드.
    """
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt["reward_model"]
    return reward_model


def cosine_similarity(a, b):
    return jnp.sum(a * b, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
    )

def visualize_reward_distributions(reward_dict, save_path=None):
    """
    Visualize reward distributions for GT, Relabeled, and Shaped rewards.

    Args:
        reward_dict (dict): Dictionary with keys 'GT', 'Relabeled', and 'Shaped'.
        save_path (str, optional): Path to save the visualization.
    """
    plt.figure(figsize=(8, 6))
    for key, rewards in reward_dict.items():
        plt.hist(
            rewards,
            bins=50,
            alpha=0.5,
            label=f"{key} Distribution",
            histtype="stepfilled",
        )
    plt.title("Reward Distributions")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reward distributions saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_reward_correlation_scatter(reward_dict, save_path=None):
    """
    Visualizes scatter plots of ground truth rewards against other reward signals after normalizing them to [0, 1].

    Args:
        reward_dict (dict): Dictionary containing reward signals with keys like 'GT', 'Relabeled', 'Shaped', etc.
                            Example: {"GT": [...], "Relabeled": [...], "Shaped": [...]}
        save_path (str, optional): Path to save the scatter plot image. If None, the plot is displayed.

    Raises:
        ValueError: If 'GT' (Ground Truth) key is not present in the reward_dict.
    """
    if "GT" not in reward_dict:
        raise ValueError("reward_dict에 'GT' (Ground Truth)가 존재해야 합니다.")

    normalized_reward_dict = {}
    for key, rewards in reward_dict.items():
        rewards = np.array(rewards)
        min_val = rewards.min()
        max_val = rewards.max()
        if min_val == max_val:
            normalized = np.zeros_like(rewards)
        else:
            normalized = (rewards - min_val) / (max_val - min_val)
        normalized_reward_dict[key] = normalized

    keys = list(normalized_reward_dict.keys())
    other_keys = [k for k in keys if k != "GT"]

    if not other_keys:
        raise ValueError("reward_dict에 'GT' 외에 비교할 다른 reward 키가 없습니다.")

    fig, axes = plt.subplots(1, len(other_keys), figsize=(5 * len(other_keys), 5), squeeze=False)

    correlation_results = {}

    for idx, k in enumerate(other_keys):
        ax = axes[0, idx]
        x = normalized_reward_dict["GT"]
        y = normalized_reward_dict[k]

        scatter = ax.scatter(x, y, alpha=0.7, color="steelblue", edgecolors="k", s=40)

        corr = np.corrcoef(x, y)[0, 1]
        correlation_results[k] = corr

        ax.set_title(f"{k} vs GT\nCorrelation: {corr:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Ground Truth (normalized)")
        ax.set_ylabel(f"{k} (normalized)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Correlation scatter plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

    print("Correlation Results:", correlation_results)


def visualize_cosine_similarity(obs, recon, save_path=None):
    """
    Visualize the distribution of cosine similarity between observations and reconstructed subgoals.

    Args:
        obs (numpy.ndarray or jax.numpy.ndarray): Original observations, shape (n_samples, obs_dim).
        recon (numpy.ndarray or jax.numpy.ndarray): Reconstructed subgoals, shape (n_samples, obs_dim).
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
    """
    if obs.ndim != 2 or recon.ndim != 2:
        raise ValueError(
            "Both obs and recon must be 2D arrays with shape (n_samples, obs_dim)."
        )
    if obs.shape != recon.shape:
        raise ValueError("obs and recon must have the same shape.")

    # Compute cosine similarity
    obs_norm = np.linalg.norm(obs, axis=1, keepdims=True)
    recon_norm = np.linalg.norm(recon, axis=1, keepdims=True)
    cosine_similarity = np.sum(obs * recon, axis=1) / (obs_norm * recon_norm + 1e-8)

    # Plot cosine similarity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(
        cosine_similarity,
        bins=50,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Cosine Similarity",
    )
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Cosine similarity plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def embed_observation(obs, embedding_model):
    """
    Embed the observations to match the dimensions of reconstructed subgoals.
    
    Args:
        obs (jax.numpy.ndarray): Original observations, shape (n_samples, obs_dim).
        embedding_model (flax.linen.Module): Embedding model to transform observations.
    
    Returns:
        jax.numpy.ndarray: Embedded observations, shape (n_samples, target_dim).
    """
    return embedding_model.apply({}, obs)  # Embedding the observations

def visualize_obs_recon_tsne_with_embedding(obs, recon, embedding_model, save_path=None):
    """
    Visualize t-SNE for observations and reconstructed subgoals after embedding observations.

    Args:
        obs (numpy.ndarray or jax.numpy.ndarray): Original observations, shape (n_samples, obs_dim).
        recon (numpy.ndarray or jax.numpy.ndarray): Reconstructed subgoals, shape (n_samples, recon_dim).
        embedding_model (flax.linen.Module): Model to embed observations to match recon dimensions.
        save_path (str, optional): Path to save the visualization. If None, shows the plot.
    """
    # Embed observations to match reconstruction dimensions
    embedded_obs = embed_observation(obs, embedding_model)

    # Combine embedded observations and reconstructions for t-SNE
    combined = np.concatenate([embedded_obs, recon], axis=0)
    labels = np.array([0] * len(embedded_obs) + [1] * len(recon))  # 0: obs, 1: recon

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(combined)

    # Separate embedded observations and reconstructions
    tsne_obs = tsne_result[labels == 0]
    tsne_recon = tsne_result[labels == 1]

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_obs[:, 0], tsne_obs[:, 1], s=20, c='blue', label='Embedded Observations', alpha=0.5)
    plt.scatter(tsne_recon[:, 0], tsne_recon[:, 1], s=20, c='red', label='Reconstructed Subgoals', alpha=0.5)
    plt.title('t-SNE: Embedded Observations vs Reconstructed Subgoals')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


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
        f"{FLAGS.comment}",
        str(FLAGS.seed),
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # 1) 환경, 데이터셋 로드
    env, dataset, ground_truth = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # ------------------------------
    # 2) CVAE 로드 (TrainState)
    cvae_state = load_vae_model(
        filename="subgoal_vae_hopper-medium-expert-v2_cos.pkl",  # 실제 파일 경로
        observation_dim=obs_dim,
        action_dim=act_dim,
    )

    # ------------------------------
    # 3) 배치로 샘플 뽑기
    n_samples = 1000
    n_samples = min(
        n_samples, dataset.size
    )  # 혹은 env.observation_space.shape[0] 과는 별개
    indices = np.random.choice(dataset.size, n_samples, replace=False)

    obs = jnp.array(dataset.observations[indices])  # (n_samples, obs_dim)
    acts = jnp.array(dataset.actions[indices])  # (n_samples, act_dim)
    next_obs = jnp.array(dataset.next_observations[indices])
    r_batch_gt = ground_truth[indices]  # (n_samples,)
    r_batch_relabeled = dataset.rewards[indices]  # (n_samples,)

    # ------------------------------
    # 4) Reconstructed Subgoal 얻기
    #    - 크기가 큰 경우 batch단위로 처리 -> 전체 concat
    batch_size = FLAGS.batch_size
    reconstructed_list = []
    # parameters = {"params": cvae_state.params}
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_obs = obs[start_idx:end_idx]
        batch_acts = acts[start_idx:end_idx]

        # subgoal_state=None, training=False -> prior 기반
        # cvae_state.apply_fn(params, state, action, subgoal_state=None, training=False)

        recon_subgoal = cvae_state.apply_fn(
            cvae_state.params, batch_obs, batch_acts, training=False
        )
        reconstructed_list.append(recon_subgoal)

    reconstructed_subgoals = jnp.concatenate(
        reconstructed_list, axis=0
    )  # (n_samples, obs_dim)

    # ------------------------------
    # 5) Reward shaping
    #    - 예시: shaped_reward = r_relabeled - 0.1 * shaping_term
    #    - 여기서는 cosine_similarity를 전체 샘플에 대해 계산
    similarity = cosine_similarity(reconstructed_subgoals, next_obs)  # (n_samples,)

    shaped_rewards = r_batch_relabeled + 1 * (jnp.mean((similarity + 1) / 2))
    # MSE
    # shaped_rewards = r_batch_relabeled + 0.1 * (
    #     jnp.mean((recon_subgoal - next_obs) ** 2, axis=-1)
    # )
    # (similarity + 1.0) / 2.0  # hyperparameter?

    reward_dict = {
        "GT": r_batch_gt,
        "Relabeled": r_batch_relabeled,
        "Shaped": shaped_rewards,
    }

    # ------------------------------
    # 6) Correlation 시각화 -> 파일 저장
    corr_save_path = os.path.join(save_dir, "reward_correlation.png")
    visualize_reward_correlation_scatter(reward_dict, save_path=corr_save_path)

    # Reward distribution plot
    dist_save_path = os.path.join(save_dir, "reward_distributions.png")
    visualize_reward_distributions(reward_dict, save_path=dist_save_path)

    # ------------------------------
    # 7) t-SNE 시각화 (원본 obs vs recon_subgoals) -> 파일 저장
    #    - GPU 메모리가 부족하면 n_samples를 더 줄이거나, perplexity 조정
    obs_np = np.array(obs)  # CPU상의 numpy
    recon_np = np.array(reconstructed_subgoals)
    tsne_save_path = os.path.join(save_dir, "tsne_obs_recon.png")
    visualize_cosine_similarity(obs_np, recon_np, save_path=tsne_save_path)

    # Define embedding model
    print(reconstructed_subgoals.shape)
    embedding_model = EmbeddingModel(target_dim=reconstructed_subgoals.shape[-1])

    # t-SNE visualization
    tsne_save_path = os.path.join(save_dir, "tsne_obs_recon_embedded.png")
    visualize_obs_recon_tsne_with_embedding(
    obs_np, recon_np, embedding_model, save_path=tsne_save_path
    )


    print("Done visualization & correlation checks.")
    print(f"All images saved to {save_dir}")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)
