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
    RelabeledDataset,
)
from evaluation import evaluate
from learner import Learner
from JaxPref.subgoal_cvae_model import SubgoalCVAE

import robosuite as suite
from robosuite.wrappers import GymWrapper
import robomimic.utils.env_utils as EnvUtils

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
from scipy.stats import kendalltau
import umap
from JaxPref.reward_transform import qlearning_robosuite_dataset
import JaxPref.reward_transform as r_tf
from JaxPref.jax_utils import batch_to_jax
from JaxPref.sampler import TrajSampler
from JaxPref.replay_buffer import get_d4rl_dataset, index_batch
from JaxPref.utils import set_random_seed
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import entropy, wasserstein_distance
import imageio
import os, tqdm

# 1a) Force OSMesa before mujoco_py ever loads
os.environ["MUJOCO_GL"] = "egl"

# 1b) Disable tqdm’s monitor thread entirely
tqdm.tqdm.monitor_interval = 0


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"
os.environ["MUJOCO_GL"] = "osmesa"
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_traj_length", 1000, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_float("clip_action", 0.999, "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean(
    "use_reward_model", False, "Use reward model for relabeling reward."
)
flags.DEFINE_string("model_type", "MLP", "type of reward model.")
flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
flags.DEFINE_string("comment", "base", "comment for distinguishing experiments.")
flags.DEFINE_integer(
    "num_query", 100, "number of query for relabeling reward in Transformer."
)
flags.DEFINE_integer(
    "seq_len", 25, "sequence length for relabeling reward in Transformer."
)
flags.DEFINE_bool(
    "use_diff",
    False,
    "boolean whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_bool(
    "balance",
    False,
    "balance whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_bool(
    "use_human_label",
    True,
    "balance whether use difference in sequence for reward relabeling.",
)
flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with tranformer.")
flags.DEFINE_string("data_dir", "./human_label/", "label directory")
flags.DEFINE_integer('max_episode_steps', 500, 'max_episode_steps for rollout.')
flags.DEFINE_string('robosuite_dataset_path', './data', 'hdf5 dataset path for demonstrations')
flags.DEFINE_string('robosuite_dataset_type', 'ph', 'dataset type for robosuite')
config_flags.DEFINE_config_file(
    "config",
    "default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


class EmbeddingModel(nn.Module):
    hidden_dims: int  # Target dimension for embedding
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(self.target_dim)(x)  # Linear layer to map to target_dim
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        x = nn.Dense(self.latent_dim)(x)

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

    # dataset.rewards = np.array(normalized_rewards)

    return np.array(normalized_rewards), trj_mapper

def normalize_gt(dataset, gt, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    gt, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        traj_len = len(traj)

        for _ in range(traj_len):
            trj_mapper.append((trj_idx, traj_len))

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = gt[i]
        if 'antmaze' in env_name:
            _, len_trj = trj_mapper[i]
            _reward -= min_return / len_trj
        _reward /= max_return - min_return
        # if ('halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name):
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)

    return normalized_rewards

def make_env_and_dataset(
    env_name: str, seed: int, dataset_path: str, max_episode_steps: int =500
) -> Tuple[gym.Env, D4RLDataset, np.ndarray]:
    ds = qlearning_robosuite_dataset(dataset_path)
    dataset = RelabeledDataset(ds['observations'], ds['actions'], ds['rewards'], ds['terminals'], ds['next_observations'])

    ds['env_meta']['env_kwargs']['horizon'] = max_episode_steps
    env = EnvUtils.create_env_from_metadata(
        env_meta=ds['env_meta'],
        render=False,            # no on-screen rendering
        render_offscreen=True,   # off-screen rendering to support rendering video frames
    ).env
    env.ignore_done = False

    env._max_episode_steps = env.horizon
    env = GymWrapper(env)
    env = wrappers.RobosuiteWrapper(env)
    env = wrappers.EpisodeMonitor(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

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
        # normalize(
        #     dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps
        # )
    normalized_rewards, trj_mapper = normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
    dataset.rewards = normalized_rewards
    if "antmaze" in FLAGS.env_name:
        dataset.rewards -= 1.0
    if (
        "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        dataset.rewards += 0.5
    if "antmaze" in FLAGS.env_name:
        ground_truth -= 1.0
    elif (
        "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        ground_truth = normalize_gt(dataset, ground_truth, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
    return env, dataset, ground_truth,trj_mapper

def make_env_and_dataset_d4rl(
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
        # normalize(
        #     dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps
        # )
    normalized_rewards, trj_mapper = normalize(dataset, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
    dataset.rewards = normalized_rewards
    if "antmaze" in FLAGS.env_name:
        dataset.rewards -= 1.0
    if (
        "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        dataset.rewards += 0.5
    if "antmaze" in FLAGS.env_name:
        ground_truth -= 1.0
    elif (
        "halfcheetah" in FLAGS.env_name
        or "walker2d" in FLAGS.env_name
        or "hopper" in FLAGS.env_name
    ):
        ground_truth = normalize_gt(dataset, ground_truth, FLAGS.env_name, max_episode_steps=env.env.env._max_episode_steps)
    return env, dataset, ground_truth,trj_mapper

from scipy.stats import kendalltau

def visualize_rank_shift(gt, shaped, save_path=None):
    # gt_rank = np.argsort(np.argsort(gt))
    # shaped_rank = np.argsort(np.argsort(shaped))
    gt_rank = (gt)
    shaped_rank = (shaped)
    rank_diff = shaped_rank - gt_rank

    plt.figure(figsize=(8, 4))
    plt.hist(rank_diff, bins=50, color='coral', alpha=0.8, edgecolor='k')
    plt.title("Rank Shift Histogram (Shaped - GT)")
    plt.xlabel("Rank Difference")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rank shift histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()
def visualize_rank_shift_pt(gt, shaped, save_path=None):
    # gt_rank = np.argsort(np.argsort(gt))
    # shaped_rank = np.argsort(np.argsort(shaped))
    gt_rank = (gt)
    shaped_rank = (shaped)
    rank_diff = shaped_rank - gt_rank

    plt.figure(figsize=(8, 4))
    plt.hist(rank_diff, bins=50, color='coral', alpha=0.8, edgecolor='k')
    plt.title("Rank Shift Histogram (PT - GT)")
    plt.xlabel("Rank Difference")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rank shift histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()

# 추가 시각화 함수 2: Subgoal deviation

def visualize_subgoal_deviation(obs, recon, save_path=None):
    deviation = np.abs(obs - recon)  # (n_samples, obs_dim)
    mean_deviation = np.mean(deviation, axis=0)  # (obs_dim,)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(mean_deviation)), mean_deviation, color='slateblue', edgecolor='k')
    plt.title("Average Absolute Deviation per Feature (obs - recon)")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Mean Absolute Deviation")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subgoal deviation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# 추가 시각화 함수 3: Subgoal diversity

def visualize_subgoal_diversity(recon, save_path=None):
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(recon)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_similarities = sim_matrix[upper_tri_indices]

    plt.figure(figsize=(8, 4))
    plt.hist(pairwise_similarities, bins=50, color='teal', alpha=0.7, edgecolor='k')
    plt.title("Subgoal Diversity (Pairwise Cosine Similarity)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subgoal diversity histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_reward_difference_heatmap(gt_rewards, spot_rewards, save_path=None):
    reward_diff = spot_rewards - gt_rewards
    plt.figure(figsize=(12, 4))
    plt.plot(reward_diff, color='purple')
    plt.title("Reward Difference per Sample (Shaped - GT)")
    plt.xlabel("Sample Index")
    plt.ylabel("Reward Difference")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reward difference plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_rank_correlation(gt, shaped, save_path=None):
    gt_rank = (gt)
    shaped_rank = (shaped)
    # gt_rank = gt
    # shaped_rank = shaped
    tau, _ = kendalltau(gt_rank, shaped_rank)

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_rank, shaped_rank, alpha=0.6, color="green", edgecolors="k")
    plt.title(f"Rank Correlation (Kendall's tau): {tau:.2f}")
    plt.xlabel("GT Rank")
    plt.ylabel("Shaped Reward Rank")
    plt.plot([0, len(gt)-1], [0, len(gt)-1], 'k--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Rank correlation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_rank_correlation_pt(gt, relabeled_rewards, save_path=None):
    # gt_rank = np.argsort(np.argsort(gt))
    # shaped_rank = np.argsort(np.argsort(relabeled_rewards))
    gt_rank = (gt)
    shaped_rank = (relabeled_rewards)
    tau, _ = kendalltau(gt_rank, shaped_rank)

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_rank, shaped_rank, alpha=0.6, color="green", edgecolors="k")
    plt.title(f"Rank Correlation (Kendall's tau): {tau:.2f}")
    plt.xlabel("GT Rank")
    plt.ylabel("PT Reward Rank")
    plt.plot([0, len(gt)-1], [0, len(gt)-1], 'k--')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Rank correlation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_reward_variance_comparison(reward_dict, save_path=None):
    keys = [k for k in reward_dict.keys()]
    variances = [float(np.var(np.array(reward_dict[k]))) for k in keys]

    # --- 추가: reward-level variance 통계 출력 ---
    print("\n[Reward-level Variance Statistics]")
    for k in keys:
        arr = np.array(reward_dict[k])
        print(f"{k}: var={np.var(arr):.4f}, mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, "
              f"IQR=({np.percentile(arr,25):.4f}-{np.percentile(arr,75):.4f}), min={np.min(arr):.4f}, max={np.max(arr):.4f}")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=keys, y=variances, palette="Set2")
    plt.title("Reward Variance Comparison")
    plt.ylabel("Variance")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reward variance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


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

def visualize_latent_projection(obs, recon,embedding_model, embedding_params, method="pca", save_path=None):
    """
    Visualize t-SNE for observations and reconstructed subgoals after embedding observations.
    """
    # Embed observations using both model and its parameters
    embedded_obs = embedding_model.apply(embedding_params, obs)
    embedded_recon = embedding_model.apply(embedding_params, recon)
    
    X = np.concatenate([embedded_obs, embedded_recon], axis=0)
    labels = np.array(["obs"] * len(obs) + ["recon"] * len(recon))

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Unknown method for projection.")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], label=label, alpha=0.5, s=5)
    plt.title(f"{method.upper()} Projection: obs vs recon")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{method.upper()} plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

# trajectory-level correlation

def visualize_trajectory_level_scatter_comparison(gt_rewards, pt_rewards, spot_rewards, trajectory_ids, save_path=None, normalize_rewards=True):
    from collections import defaultdict
    from scipy.stats import kendalltau

    def compute_sums(rewards, ids):
        traj_dict = defaultdict(list)
        for r, i in zip(rewards, ids):
            traj_dict[i].append(r)
        sums = np.array([np.mean(traj_dict[i]) for i in sorted(traj_dict)])
        return sums

    # 1. Compute trajectory-level average rewards
    gt_means = compute_sums(gt_rewards, trajectory_ids)
    pt_means = compute_sums(pt_rewards, trajectory_ids)
    spot_means = compute_sums(spot_rewards, trajectory_ids)


    gt_plot = gt_means
    pt_plot = pt_means
    spot_plot = spot_means

    # 3. Compute statistics
    pt_corr = np.corrcoef(gt_plot, pt_plot)[0, 1]
    pt_mse = np.mean((gt_plot - pt_plot) ** 2)
    pt_tau, _ = kendalltau(gt_plot, pt_plot)

    spot_corr = np.corrcoef(gt_plot, spot_plot)[0, 1]
    spot_mse = np.mean((gt_plot - spot_plot) ** 2)
    spot_tau, _ = kendalltau(gt_plot, spot_plot)

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(gt_plot, pt_plot, alpha=0.7, color='royalblue', edgecolors='k')
    axes[0].plot([gt_plot.min(), gt_plot.max()], [gt_plot.min(), gt_plot.max()], 'k--', lw=1)
    axes[0].set_title(f"PT vs GT\nCorr: {pt_corr:.2f}  MSE: {pt_mse:.2f}  τ: {pt_tau:.2f}")
    axes[0].set_xlabel("GT Trajectory Return")
    axes[0].set_ylabel("PT Trajectory Return")

    axes[1].scatter(gt_plot, spot_plot, alpha=0.7, color='orange', edgecolors='k')
    axes[1].plot([gt_plot.min(), gt_plot.max()], [gt_plot.min(), gt_plot.max()], 'k--', lw=1)
    axes[1].set_title(f"GUIDER vs GT\nCorr: {spot_corr:.2f}  MSE: {spot_mse:.2f}  τ: {spot_tau:.2f}")
    axes[1].set_xlabel("GT Trajectory Return")
    axes[1].set_ylabel("GUIDER Trajectory Return")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory-level scatter plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
# def visualize_reward_correlation_scatter(reward_dict, save_path=None, include_kendall=True):
#     """
#     Visualize scatter plots of GT vs other rewards (on raw values),
#     computing raw Pearson correlation, MSE, and optionally Kendall's tau.
#     """
#     if "GT" not in reward_dict:
#         raise ValueError("'GT' key must be present in reward_dict.")

#     gt = np.asarray(reward_dict["GT"])
#     other_keys = [k for k in reward_dict if k != "GT"]
    
#     fig, axes = plt.subplots(1, len(other_keys), figsize=(5 * len(other_keys), 5), squeeze=False)
#     pearson_results, mse_results, kendall_results = {}, {}, {}

#     for i, key in enumerate(other_keys):
#         r = np.asarray(reward_dict[key])

#         # Compute metrics
#         pearson_corr = np.corrcoef(gt, r)[0, 1]
#         mse = np.mean((gt - r) ** 2)
#         tau, _ = kendalltau(rankdata(gt), rankdata(r)) if include_kendall else (None, None)

#         # Save stats
#         pearson_results[key] = pearson_corr
#         mse_results[key] = mse
#         if include_kendall:
#             kendall_results[key] = tau

#         # Plot
#         ax = axes[0, i]
#         ax.scatter(gt, r, alpha=0.6, edgecolors='k')
#         min_v, max_v = min(gt.min(), r.min()), max(gt.max(), r.max())
#         ax.plot([min_v, max_v], [min_v, max_v], 'k--')

#         title = f"{key} vs GT\nCorr: {pearson_corr:.2f}  MSE: {mse:.2f}"
#         if include_kendall:
#             title += f"  τ: {tau:.2f}"
#         ax.set_title(title)
#         ax.set_xlabel("GT")
#         ax.set_ylabel(key)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print(f"Saved to {save_path}")
#     else:
#         plt.show()
#     plt.close()

#     print("\n--- Raw Pearson Correlation ---")
#     for k, v in pearson_results.items():
#         print(f"{k}: {v:.4f}")
#     print("\n--- Raw MSE ---")
#     for k, v in mse_results.items():
#         print(f"{k}: {v:.4f}")
#     if include_kendall:
#         print("\n--- Kendall's Tau (Rank Correlation) ---")
#         for k, v in kendall_results.items():
#             print(f"{k}: {v:.4f}")
def visualize_reward_correlation_scatter(reward_dict, save_path=None, include_kendall=True):
    """
    Visualize scatter plots of GT vs other rewards (on raw values),
    computing Pearson correlation, MSE, Kendall's tau, KL divergence, and Wasserstein distance.
    """
    if "GT" not in reward_dict:
        raise ValueError("'GT' key must be present in reward_dict.")

    gt = np.asarray(reward_dict["GT"])
    other_keys = [k for k in reward_dict if k != "GT"]
    
    fig, axes = plt.subplots(1, len(other_keys), figsize=(5 * len(other_keys), 5), squeeze=False)
    pearson_results, mse_results, kendall_results = {}, {}, {}
    kl_results, wass_results = {}, {}

    for i, key in enumerate(other_keys):
        r = np.asarray(reward_dict[key])

        # Compute standard metrics
        pearson_corr = np.corrcoef(gt, r)[0, 1]
        mse = np.mean((gt - r) ** 2)
        tau, _ = kendalltau((gt), (r)) if include_kendall else (None, None)

        # Compute histogram-based metrics (KL and Wasserstein)
        hist_gt, bin_edges = np.histogram(gt, bins=100, density=True)
        hist_r, _ = np.histogram(r, bins=bin_edges, density=True)
        kl_div = entropy(hist_gt + 1e-10, hist_r + 1e-10)  # add small value for numerical stability
        wass_dist = wasserstein_distance(gt, r)

        # Save results
        pearson_results[key] = pearson_corr
        mse_results[key] = mse
        kl_results[key] = kl_div
        wass_results[key] = wass_dist
        if include_kendall:
            kendall_results[key] = tau

        # Plot
        ax = axes[0, i]
        ax.scatter(gt, r, alpha=0.6, edgecolors='k')
        min_v, max_v = min(gt.min(), r.min()), max(gt.max(), r.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--')

        title = f"{key} vs GT\nCorr: {pearson_corr:.2f}  MSE: {mse:.2f}"
        if include_kendall:
            title += f"  τ: {tau:.2f}"
        ax.set_title(title)
        ax.set_xlabel("GT")
        ax.set_ylabel(key)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()

    # Print results
    print("\n--- Raw Pearson Correlation ---")
    for k, v in pearson_results.items():
        print(f"{k}: {v:.4f}")
    print("\n--- Raw MSE ---")
    for k, v in mse_results.items():
        print(f"{k}: {v:.4f}")
    if include_kendall:
        print("\n--- Kendall's Tau (Rank Correlation) ---")
        for k, v in kendall_results.items():
            print(f"{k}: {v:.4f}")
    print("\n--- KL Divergence (GT || Pred) ---")
    for k, v in kl_results.items():
        print(f"{k}: {v:.4f}")
    print("\n--- Wasserstein Distance ---")
    for k, v in wass_results.items():
        print(f"{k}: {v:.4f}")

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
    # keys = [k for k in reward_dict.keys() if k != "GT"]
    # plt.figure(figsize=(8, 6))
    # for key in keys:
    #     plt.hist(
    #         reward_dict[key],
    #         bins=50,
    #         alpha=0.5,
    #         label=f"{key} Distribution",
    #         histtype="stepfilled",
    #     )
    # keys = [k for k in reward_dict.keys() if k != "GT"]
    plt.figure(figsize=(8, 6))
    for key,reward in reward_dict.items():
        # print(key,reward.shape)
        plt.hist(
            reward,
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


def visualize_cosine_similarity(obs, recon, save_path=None):
    """
    Visualize the distribution of cosine similarity between observations and reconstructed subgoals (scaled to [0, 1]).

    Args:
        obs (numpy.ndarray or jax.numpy.ndarray): Original observations, shape (n_samples, obs_dim).
        recon (numpy.ndarray or jax.numpy.ndarray): Reconstructed subgoals, shape (n_samples, obs_dim).
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
    """
    if obs.ndim != 2 or recon.ndim != 2:
        raise ValueError("Both obs and recon must be 2D arrays with shape (n_samples, obs_dim).")
    if obs.shape != recon.shape:
        raise ValueError("obs and recon must have the same shape.")

    # Compute cosine similarity using defined function
    cosine_sim = cosine_similarity(obs, recon)  # jnp array of shape (n_samples,)
    cosine_sim = jax.device_get(cosine_sim)  # move to CPU numpy if needed

    # Normalize to [0, 1]
    similarity_scores = (cosine_sim + 1.0) / 2.0
    similarity_scores = similarity_scores.flatten()

    # Plot cosine similarity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(
        similarity_scores,
        bins=50,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Cosine Similarity (0–1 scaled)",
    )
    plt.title("Cosine Similarity Distribution (Scaled to [0, 1])")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Cosine similarity plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_obs_recon_tsne_with_embedding(obs, recon, embedding_model, embedding_params, save_path=None):
    """
    Visualize t-SNE for observations and reconstructed subgoals after embedding observations.
    """
    # Embed observations using both model and its parameters
    embedded_obs = embedding_model.apply(embedding_params, obs)
    embedded_recon = embedding_model.apply(embedding_params, recon)

    # Combine for t-SNE
    combined = np.concatenate([embedded_obs, embedded_recon], axis=0)
    labels = np.array([0] * len(embedded_obs) + [1] * len(recon))

    # tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    perplexity = min(30, len(obs) // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(combined)

    tsne_obs = tsne_result[labels == 0]
    tsne_recon = tsne_result[labels == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_obs[:, 0], tsne_obs[:, 1], s=5, c='blue', label='Embedded Observations', alpha=0.5)
    plt.scatter(tsne_recon[:, 0], tsne_recon[:, 1], s=5, c='red', label='Reconstructed Subgoals', alpha=0.5)
    plt.title('t-SNE: Embedded Observations vs Reconstructed Subgoals')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def run_attention_reward_analysis(env_name, FLAGS, save_dir):
    gym_env = gym.make(FLAGS.env_name)
    eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    reward_model = initialize_model()
    label_type = 0

    dataset["actions"] = np.clip(
        dataset["actions"], -FLAGS.clip_action, FLAGS.clip_action
    )
    set_random_seed(FLAGS.seed)

    print("load saved indices.")
    if "dense" in FLAGS.env_name:
        env = "-".join(FLAGS.env.split("-")[:-2] + [FLAGS.env.split("-")[-1]])
    else:
        env = FLAGS.env_name

    base_path = os.path.join(FLAGS.data_dir, env)
    if os.path.exists(base_path):
        human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(base_path))
        with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:
            human_indices = pickle.load(fp)
        with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:
            human_indices_2 = pickle.load(fp)
        with open(os.path.join(base_path, human_labels_file), "rb") as fp:
            human_labels = pickle.load(fp)

        pref_dataset = r_tf.load_queries_with_indices(
            gym_env,
            dataset,
            FLAGS.num_query,
            FLAGS.seq_len,
            label_type=label_type,
            saved_indices=[human_indices, human_indices_2],
            saved_labels=human_labels,
            balance=FLAGS.balance,
            scripted_teacher=not FLAGS.use_human_label,
        )

    def analyze_segment(obs, actions, gt_rewards, segment_name="preferred"):
        top_rewards_gt, non_top_rewards_gt = [], []
        top_rewards_rm, non_top_rewards_rm = [], []

        for start in range(0, len(obs), FLAGS.batch_size):
            end = min(start + FLAGS.batch_size, len(obs))
            batch_len = end - start

            batch = {
                "observations": obs[start:end],
                "actions": actions[start:end],
                "timestep": np.tile(np.arange(1, FLAGS.seq_len + 1), (batch_len, 1)),
                "attn_mask": np.ones((batch_len, FLAGS.seq_len), dtype=np.float32),
            }

            jax_input = batch_to_jax(batch)
            reward, attn = reward_model.get_reward(jax_input)
            reward = np.array(reward)
            attn = np.array(attn)

            obs_attn = attn[:, 0, FLAGS.seq_len : 2 * FLAGS.seq_len, FLAGS.seq_len : 2 * FLAGS.seq_len]
            diag_attn = np.einsum("bii->bi", obs_attn)

            for i in range(batch_len):
                topk_mask = diag_attn[i] >= np.percentile(diag_attn[i], 90)
                gt = gt_rewards[start + i].reshape(-1)
                rm = reward[i].reshape(-1)

                # Optional RM normalization
                # rm = (rm - rm.mean()) / (rm.std() + 1e-6)

                if gt.shape[0] != topk_mask.shape[0]:
                    continue

                top_rewards_gt.append(gt[topk_mask].mean() if np.any(topk_mask) else 0)
                non_top_rewards_gt.append(gt[~topk_mask].mean() if np.any(~topk_mask) else 0)
                top_rewards_rm.append(rm[topk_mask].mean() if np.any(topk_mask) else 0)
                non_top_rewards_rm.append(rm[~topk_mask].mean() if np.any(~topk_mask) else 0)

        print(f"[{segment_name}] Avg GT Top10%: {np.mean(top_rewards_gt):.4f}, Non-Top: {np.mean(non_top_rewards_gt):.4f}")
        print(f"[{segment_name}] Avg RM Top10%: {np.mean(top_rewards_rm):.4f}, Non-Top: {np.mean(non_top_rewards_rm):.4f}")

        return top_rewards_gt, non_top_rewards_gt, top_rewards_rm, non_top_rewards_rm

    preferred_mask = (pref_dataset["labels"][:, 1] > pref_dataset["labels"][:, 0])

    top_gt1, non_top_gt1, top_rm1, non_rm1 = analyze_segment(
        obs=pref_dataset["observations"][preferred_mask],
        actions=pref_dataset["actions"][preferred_mask],
        gt_rewards=pref_dataset["gt_rewards"][preferred_mask],
        segment_name="Preferred"
    )

    top_gt2, non_top_gt2, top_rm2, non_rm2 = analyze_segment(
        obs=pref_dataset["observations_2"][~preferred_mask],
        actions=pref_dataset["actions_2"][~preferred_mask],
        gt_rewards=pref_dataset["gt_rewards_2"][~preferred_mask],
        segment_name="Unpreferred"
    )

    # Prepare data for seaborn violin+strip plot
    data = [
        top_gt1, non_top_gt1, top_gt2, non_top_gt2,
        top_rm1, non_rm1, top_rm2, non_rm2
    ]
    labels = [
        "GT Top Pref", "GT Non-Top Pref", "GT Top Unpref", "GT Non-Top Unpref",
        "RM Top Pref", "RM Non-Top Pref", "RM Top Unpref", "RM Non-Top Unpref"
    ]

    flat_rewards, group_names = [], []
    for i, group in enumerate(labels):
        flat_rewards.extend(data[i])
        group_names.extend([group] * len(data[i]))

    df = pd.DataFrame({"Reward": flat_rewards, "Group": group_names})

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Group", y="Reward", data=df, inner="quartile", cut=0)
    sns.stripplot(x="Group", y="Reward", data=df, color='k', size=1.5, alpha=0.2)
    plt.xticks(rotation=30)
    plt.title("GT vs RM Reward by Attention Weight")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "attention_weight_reward_violinplot.png"))
    plt.show()

    # --- 차이 분포 시각화 ---
    diff_data = [
        np.array(top_gt1) - np.array(non_top_gt1),
        np.array(top_gt2) - np.array(non_top_gt2),
        np.array(top_rm1) - np.array(non_rm1),
        np.array(top_rm2) - np.array(non_rm2),
    ]
    diff_labels = [
        "GT Pref (Top - NonTop)",
        "GT Unpref (Top - NonTop)",
        "RM Pref (Top - NonTop)",
        "RM Unpref (Top - NonTop)",
    ]

    diff_df = pd.DataFrame({
        "Reward Difference": np.concatenate(diff_data),
        "Group": np.concatenate([[l] * len(d) for l, d in zip(diff_labels, diff_data)])
    })

    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Group", y="Reward Difference", data=diff_df, inner="quartile", cut=0)
    sns.stripplot(x="Group", y="Reward Difference", data=diff_df, color='k', size=1.5, alpha=0.2)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Reward Difference (Top10% - NonTop90%)")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "attention_top_vs_non_top_diff.png"))
    plt.show()   
    from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

    def print_statistical_tests(group1, group2, label1, label2):
        print(f"\n--- Statistical Comparison: {label1} vs {label2} ---")

        group1 = np.array(group1)
        group2 = np.array(group2)

        # Paired t-test (assuming same trajectory)
        try:
            t_stat, p_t = ttest_rel(group1, group2)
            print(f"Paired t-test: p = {p_t:.4e}")
        except Exception as e:
            print(f"Paired t-test failed: {e}")

        # Wilcoxon signed-rank test (non-parametric paired)
        try:
            w_stat, p_w = wilcoxon(group1, group2)
            print(f"Wilcoxon test: p = {p_w:.4e}")
        except Exception as e:
            print(f"Wilcoxon test failed: {e}")

        # Mann-Whitney U test (unpaired)
        try:
            u_stat, p_u = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"Mann-Whitney U test (unpaired): p = {p_u:.4e}")
        except Exception as e:
            print(f"Mann-Whitney test failed: {e}")

    # ---------------------
    # Run statistical tests
    # ---------------------

    # Top vs Non-Top (same trajectory) – GT
    print_statistical_tests(top_gt1, non_top_gt1, "GT Top Pref", "GT Non-Top Pref")
    print_statistical_tests(top_gt2, non_top_gt2, "GT Top Unpref", "GT Non-Top Unpref")

    # Top vs Non-Top – RM
    print_statistical_tests(top_rm1, non_rm1, "RM Top Pref", "RM Non-Top Pref")
    print_statistical_tests(top_rm2, non_rm2, "RM Top Unpref", "RM Non-Top Unpref")

    # Preferred vs Unpreferred – Top GT
    print_statistical_tests(top_gt1, top_gt2, "GT Top Pref", "GT Top Unpref")
    # Preferred vs Unpreferred – Top RM
    print_statistical_tests(top_rm1, top_rm2, "RM Top Pref", "RM Top Unpref")


def extract_sample_by_trajectories(dataset, ground_truth, trj_mapper, num_traj=20):
    """
    Select full trajectories and return the flattened timestep-level samples within those.
    """
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    num_total_traj = len(trajs)
    print(num_total_traj) 
    selected_traj_ids = np.random.choice(num_total_traj, size=min(num_traj, num_total_traj), replace=False)

    selected_indices = []
    traj_id_counter = 0
    for i, traj in enumerate(trajs):
        if i in selected_traj_ids:
            selected_indices.extend(range(traj_id_counter, traj_id_counter + len(traj)))
        traj_id_counter += len(traj)

    selected_indices = np.array(sorted(selected_indices))
    print(len(selected_indices))
    obs = jnp.array(dataset.observations[selected_indices])
    acts = jnp.array(dataset.actions[selected_indices])
    next_obs = jnp.array(dataset.next_observations[selected_indices])
    gt_rewards = np.array(ground_truth)[selected_indices]
    relabeled_rewards = dataset.rewards[selected_indices]
    trajectory_ids = [trj_mapper[i][0] for i in selected_indices]

    return obs, acts, next_obs, gt_rewards, relabeled_rewards, trajectory_ids, selected_indices

def visualize_trajectory_internal_variance(gt_rewards, relabeled, spot, trajectory_ids, save_path=None):
    from collections import defaultdict

    def compute_variance(rewards, ids):
        traj_dict = defaultdict(list)
        for r, i in zip(rewards, ids):
            traj_dict[i].append(r)
        return [np.var(traj_dict[i]) for i in sorted(traj_dict)]

    gt_vars = compute_variance(gt_rewards, trajectory_ids)
    pt_vars = compute_variance(relabeled, trajectory_ids)
    spot_vars = compute_variance(spot, trajectory_ids)

    # --- 추가: trajectory-level variance 통계 출력 ---
    print("\n[Trajectory-level Variance Statistics]")
    for name, arr in zip(["GT", "PT", "GUIDER"], [gt_vars, pt_vars, spot_vars]):
        arr = np.array(arr)
        print(f"{name}: mean={np.mean(arr):.4f}, median={np.median(arr):.4f}, "
              f"IQR=({np.percentile(arr,25):.4f}-{np.percentile(arr,75):.4f}), min={np.min(arr):.4f}, max={np.max(arr):.4f}")

    df = pd.DataFrame({
        "Variance": gt_vars + pt_vars + spot_vars,
        "Type": ["GT"] * len(gt_vars) + ["PT"] * len(pt_vars) + ["GUIDER"] * len(spot_vars)
    })

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Type", y="Variance", data=df)
    plt.title("Trajectory-wise Reward Variance")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trajectory variance plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def subgoal_predictiveness_check(reconstructed_subgoals, next_obs, gt_rewards):
    similarity = cosine_similarity(reconstructed_subgoals, next_obs)
    similarity = jax.device_get(similarity)
    similarity_scaled = (similarity + 1) / 2

    corr = np.corrcoef(similarity_scaled, gt_rewards)[0, 1]
    print(f"Correlation between subgoal similarity and GT reward: {corr:.4f}")

def visualize_return_distribution(gt_rewards, pt_rewards, spot_rewards, trajectory_ids, save_path=None):
    from collections import defaultdict

    def compute_sum(rewards, ids):
        traj_dict = defaultdict(list)
        for r, i in zip(rewards, ids):
            traj_dict[i].append(r)
        return [np.sum(traj_dict[i]) for i in sorted(traj_dict)]

    gt = compute_sum(gt_rewards, trajectory_ids)
    pt = compute_sum(pt_rewards, trajectory_ids)
    spot = compute_sum(spot_rewards, trajectory_ids)

    plt.figure(figsize=(8, 5))
    for r, label in zip([gt, pt, spot], ["GT", "PT", "GUIDER"]):
        sns.kdeplot(r, label=label, fill=True, alpha=0.3)
    plt.title("Trajectory Return Distribution")
    plt.xlabel("Return")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Return distribution saved to {save_path}")
    else:
        plt.show()
    plt.close()

def get_base_env(env):
    """
    Robosuite 환경에서 가장 안쪽 시뮬레이터까지 unwrap
    """
    while hasattr(env, "env"):
        env = env.env
    return env

def print_all_body_positions(env):
    """
    모든 body 이름과 현재 위치를 출력합니다.
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    print("\n🔍 All body positions in sim:")
    for i in range(base_env.sim.model.nbody):
        name = base_env.sim.model.body_id2name(i)
        pos = base_env.sim.data.body_xpos[i]
        print(f"{i:2d} | {name:<30} → {np.round(pos, 4)}")

def print_can_related_bodies(env):
    """
    "can"이라는 이름이 들어간 모든 body를 출력합니다.
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    print("\n🔍 Can-related body positions:")
    for i in range(base_env.sim.model.nbody):
        name = base_env.sim.model.body_id2name(i)
        if "can" in name.lower():
            pos = base_env.sim.data.body_xpos[i]
            print(f"{i:2d} | {name:<30} → {np.round(pos, 4)}")


def set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=None):
    """
    Set MuJoCo environment state from flat observation for D4RL environments.
    Handles different observation structures for different environments.
    """
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    
    if not hasattr(base_env, "sim"):
        print(f"Warning: Environment {env_name} does not have MuJoCo simulator")
        return False
    
    obs_dim = flat_obs.shape[0]
    
    # Different D4RL environments have different observation structures
    if "halfcheetah" in env_name.lower():
        # HalfCheetah: 17D observation (8 joint pos + 9 joint vel)
        qpos_dim = 8
        qvel_dim = 9
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    elif "walker2d" in env_name.lower():
        # Walker2d: 17D observation (8 joint pos + 9 joint vel)
        qpos_dim = 8
        qvel_dim = 9
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    elif "hopper" in env_name.lower():
        # Hopper: 11D observation (5 joint pos + 6 joint vel)
        qpos_dim = 5
        qvel_dim = 6
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    elif "ant" in env_name.lower():
        # Ant: 111D observation (15 joint pos + 14 joint vel + 84 contact forces)
        qpos_dim = 15
        qvel_dim = 14
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    else:
        # Default: assume first half is qpos, second half is qvel
        mid_point = obs_dim // 2
        qpos = flat_obs[:mid_point]
        qvel = flat_obs[mid_point:]
    
    # Debug info for first few images
    if debug_idx is not None and debug_idx < 3:
        print(f"Image {debug_idx}: qpos shape={qpos.shape}, qvel shape={qvel.shape}")
        print(f"qpos range: [{qpos.min():.3f}, {qpos.max():.3f}]")
        print(f"qvel range: [{qvel.min():.3f}, {qvel.max():.3f}]")
    
    try:
        # Set the environment state with bounds checking
        qpos_len = min(len(qpos), len(base_env.sim.data.qpos))
        qvel_len = min(len(qvel), len(base_env.sim.data.qvel))
        
        base_env.sim.data.qpos[:qpos_len] = qpos[:qpos_len]
        base_env.sim.data.qvel[:qvel_len] = qvel[:qvel_len]
        base_env.sim.forward()
        return True
        
    except Exception as e:
        print(f"Error setting environment state: {e}")
        return False


def set_env_state_from_flat_obs(env, flat_obs, debug_idx=None):
    """
    Set Mujoco environment state from a 53D flat observation vector.
    Includes robot joints, gripper, and object (Can) pose.
    Camera is positioned above object and looks downward.
    """
    # Unpack from flat_obs
    object_qpos = flat_obs[0:6]          # [x, y, z, qx, qy, qz]
    object_xyz = object_qpos[:3]         # just position
    object_vel = flat_obs[6:12]          # linear + angular vel
    joint_pos = flat_obs[14:21]
    joint_vel = flat_obs[35:42]
    gripper_qpos = flat_obs[49:51]
    gripper_qvel = flat_obs[51:53]

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # Set qpos and qvel
    qpos = base_env.sim.data.qpos.copy()
    qvel = base_env.sim.data.qvel.copy()

    qpos[0:7] = joint_pos
    qpos[7:9] = gripper_qpos
    qpos[9:15] = object_qpos  # Can_main
    qvel[0:7] = joint_vel
    qvel[7:9] = gripper_qvel
    qvel[9:15] = object_vel

    base_env.sim.data.qpos[:] = qpos
    base_env.sim.data.qvel[:] = qvel

    base_env.sim.forward()



def save_recon_subgoal_images(env, recon_subgoals, save_dir,
                              camera_name='frontview', width=256, height=256, max_images=10):
    os.makedirs(save_dir, exist_ok=True)
    base_env = get_base_env(env)
    recon_subgoals = np.array(recon_subgoals)  # Ensure numpy array

    for i, flat_obs in enumerate(recon_subgoals):
        set_env_state_from_flat_obs(env, flat_obs, debug_idx=i)
        img = base_env.sim.render(width=width, height=height, camera_name=camera_name, depth=False)
        img = np.flipud(img)
        path = os.path.join(save_dir, f"recon_{i:04d}.png")
        imageio.imwrite(path, img)
        print(f"Saved {path}")
    print(f"\n✅ All done! First {max_images} images saved in {save_dir}")

# def save_recon_subgoal_images(env, recon_subgoals, save_dir,
#                               camera_name='birdview', width=256, height=256):
#     os.makedirs(save_dir, exist_ok=True)

#     for i, flat_obs in enumerate(recon_subgoals):
#         # restore
#         print(env)
#         print(env.env)
#         print(env.env.env)
#         env.env.env.get_observation(flat_obs)
#         # render & save
#         img = env.render(width=width, height=height,mode="rgb_array" ,camera_name=camera_name)
#         path = os.path.join(save_dir, f"recon_{i:04d}.png")
#         imageio.imwrite(path, img)
#         print(f"Saved {path}")

#         if i >= 9:
#             break

#     print(f"All done!  First 10 subgoal images in {save_dir}")


# def obs_to_full_qpos_vel(obs, env):
#     """
#     Convert flat observation → full qpos, qvel.
#     Handles both 'Can' and 'Lift' robosuite tasks.
#     """
#     # ensure numpy
#     obs = np.asarray(obs, dtype=np.float32)

#     # find the raw robosuite env with .sim
#     robosuite_env = env
#     while hasattr(robosuite_env, 'env') and not hasattr(robosuite_env, 'sim'):
#         robosuite_env = robosuite_env.env
#     sim = robosuite_env.sim

#     # start from the simulator's current qpos/qvel
#     qpos = sim.data.qpos.copy()
#     qvel = sim.data.qvel.copy()

#     print(f"[obs_to_full_qpos_vel] obs.shape={obs.shape}, qpos.shape={qpos.shape}")

#     # detect environment type:
#     model_name = getattr(robosuite_env.unwrapped, 'model', None)
#     is_lift = False
#     if model_name is not None and hasattr(model_name, 'name'):
#         is_lift = 'Lift' in model_name.name  # e.g. 'Lift' in xml name
#     # fallback: horizon heuristic
#     if hasattr(env, 'horizon') and env.horizon >= 100:
#         # assuming Lift tasks usually have longer horizons...
#         is_lift = True

#     if is_lift:
#         print("Using Lift environment mapping")
#         # [0:7] robot joints, [7:14] robot joint vel
#         qpos[0:7] = obs[0:7]
#         qvel[0:7] = obs[7:14]
#         # object pos (3), object quat (4)
#         qpos[7:10]  = obs[14:17]
#         qpos[10:14] = obs[17:21]
#         # object linear vel, angular vel
#         qvel[7:10]  = obs[21:24]
#         qvel[10:13] = obs[24:27]
#         # gripper (if present at tail)
#         if obs.shape[0] >= 31:
#             qpos[14:16] = obs[27:29]
#             qvel[13:15] = obs[29:31]

#     else:
#         print("Using Can environment mapping")
#         # same indices as above for Can
#         qpos[0:7]   = obs[0:7]
#         qvel[0:7]   = obs[7:14]
#         qpos[7:10]  = obs[14:17]
#         qpos[10:14] = obs[17:21]
#         qvel[7:10]  = obs[21:24]
#         qvel[10:13] = obs[24:27]
#         if obs.shape[0] >= 31:
#             qpos[14:16] = obs[27:29]
#             qvel[13:15] = obs[29:31]

#     print(f"Mapped qpos[:8]={qpos[:8]}, qvel[:8]={qvel[:8]}")
#     return qpos, qvel


def print_robot_info(env):
    """
    Debugging: print robot-related body and joint info
    """
    print("--- robot joints ---")
    for name in env.sim.model.joint_names:
        print(name)

    print("--- robot bodies ---")
    for i, name in enumerate(env.sim.model.body_names):
        if "robot" in name or "gripper" in name:
            print(f"{i:2d} | {name:25s} -> {env.sim.data.body_xpos[i]}")

def save_state_images(env, state_array, save_dir, env_name=None,
                     camera_name='frontview', width=256, height=256, max_images=10, prefix="recon"):
    os.makedirs(save_dir, exist_ok=True)
    base_env = get_base_env(env)
    state_array = np.array(state_array)  # Ensure numpy array

    for i, flat_obs in enumerate(state_array):
        # Use D4RL-specific function if environment name is provided
        if env_name and any(name in env_name.lower() for name in ['hopper', 'walker2d', 'halfcheetah', 'ant']):
            success = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
        else:
            # Use original Robosuite function
            set_env_state_from_flat_obs(env, flat_obs, debug_idx=i)
            print(flat_obs)
            success = True
        
        if not success:
            print(f"Warning: Failed to set environment state for image {i}")
            continue
            
        try:
            img = base_env.sim.render(width=width, height=height, camera_name=camera_name, depth=False)
            img = np.flipud(img)
            path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
            imageio.imwrite(path, img)
            print(f"Saved {path}")
        except Exception as e:
            print(f"Error rendering image {i}: {e}")
            continue

    print(f"\n✅ All done! First {max_images} images saved in {save_dir} (prefix: {prefix})")

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

    # summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # 1) 환경, 데이터셋 로드
    dataset_path = os.path.join(FLAGS.robosuite_dataset_path, FLAGS.env_name.lower(), FLAGS.robosuite_dataset_type, "low_dim.hdf5")

    env, dataset, ground_truth,trj_mapper = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, dataset_path, max_episode_steps=FLAGS.max_episode_steps)
    # run_attention_reward_analysis(FLAGS.env_name,FLAGS,save_dir)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # ------------------------------
    # 2) CVAE 로드 (TrainState)
    print("obs_dim : ", obs_dim)
    cvae_state = load_vae_model(
        filename="subgoal_vae_Can_mh_0_16_32_10.pkl",  # 실제 파일 경로
        observation_dim=obs_dim,
        action_dim=act_dim,
    )

    # ------------------------------
    # 3) 배치로 샘플 뽑기
    # n_samples = 2500
    # n_samples = min(
    #     n_samples, dataset.size
    # )  # 혹은 env.observation_space.shape[0] 과는 별개
    # indices = np.random.choice(dataset.size, n_samples, replace=False)

    # obs = jnp.array(dataset.observations[indices])  # (n_samples, obs_dim)
    # acts = jnp.array(dataset.actions[indices])  # (n_samples, act_dim)
    # next_obs = jnp.array(dataset.next_observations[indices])
    # r_batch_gt = ground_truth[indices]  # (n_samples,)
    # r_batch_relabeled = dataset.rewards[indices]  # (n_samples,)
    obs, acts, next_obs, r_batch_gt, r_batch_relabeled, trajectory_ids,indices = extract_sample_by_trajectories(
    dataset, ground_truth, trj_mapper, num_traj=20
    )

    # ------------------------------
    # 4) Reconstructed Subgoal 얻기
    #    - 크기가 큰 경우 batch단위로 처리 -> 전체 concat
    n_samples = obs.shape[0]
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
        print("recon_subgoal : ", recon_subgoal)
        reconstructed_list.append(recon_subgoal)

    reconstructed_subgoals = jnp.concatenate(
        reconstructed_list, axis=0
    )  # (n_samples, obs_dim)

    # ------------------------------
    # 5) Reward shaping
    #    - 예시: shaped_reward = r_relabeled - 0.1 * shaping_term
    #    - 여기서는 cosine_similarity를 전체 샘플에 대해 계산
    similarity = cosine_similarity(reconstructed_subgoals, next_obs)  # (n_samples,)
    shaping_term = jnp.mean((similarity + 1) / 2)  # (n_samples,)
    shaped_rewards = r_batch_relabeled + shaping_term  # per-sample shaping

    # ------------------------------
    # [NEW] Normalize all rewards using GT min/max for fair comparison
    min_gt = np.min(r_batch_gt)
    max_gt = np.max(r_batch_gt)
    def norm_with_gt(x):
        return (x - min(x)) / (max(x) - min(x))
    # r_batch_gt_norm = norm_with_gt(r_batch_gt)
    # r_batch_relabeled_norm = norm_with_gt(r_batch_relabeled)
    # shaped_rewards_norm = norm_with_gt(shaped_rewards)

    r_batch_gt_norm = (r_batch_gt)
    r_batch_relabeled_norm = (r_batch_relabeled)
    shaped_rewards_norm = (shaped_rewards)

    reward_dict = {
        "GT": r_batch_gt_norm,
        "PT": r_batch_relabeled_norm,
        "GUIDER": shaped_rewards_norm,
    }
    print("[GT norm] min : ", min(reward_dict["GT"])," max : ", max(reward_dict["GT"]))
    print("[PT norm] min : ", min(reward_dict["PT"])," max : ", max(reward_dict["PT"]))
    print("[GUIDER norm] min : ", min(reward_dict["GUIDER"])," max : ", max(reward_dict["GUIDER"]))
    # print(reward_dict)
    # ------------------------------

    # Save reconstructed subgoal images (최대 10개, 필요시 max_images 조정)
    save_state_images(
        env,
        np.array(reconstructed_subgoals),
        os.path.join(save_dir, "recon_subgoal_images"),
        # camera_name='birdview',
        camera_name='frontview',
        width=256,
        height=256,
        max_images=10,
        prefix="recon"
    )
    # Save original observation images (최대 10개, 필요시 max_images 조정)
    save_state_images(
        env,
        np.array(obs),
        os.path.join(save_dir, "obs_images"),
        # camera_name='birdview',
        camera_name='frontview',
        width=256,
        height=256,
        max_images=10,
        prefix="obs"
    )

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
    # print(reconstructed_subgoals.shape)
    # embedding_model = EmbeddingModel(target_dim=reconstructed_subgoals.shape[-1])

    # # t-SNE visualization
    # tsne_save_path = os.path.join(save_dir, "tsne_obs_recon_embedded.png")
    # visualize_obs_recon_tsne_with_embedding(
    # obs_np, recon_np, embedding_model, save_path=tsne_save_path
    # )
    embedding_model = EmbeddingModel(latent_dim = 16,hidden_dims = [32, 64, 32])
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones_like(obs)
    embedding_params = embedding_model.init(rng, dummy_input)

    # t-SNE visualization with embedded obs
    tsne_save_path = os.path.join(save_dir, "tsne_obs_recon_embedded.png")
    visualize_obs_recon_tsne_with_embedding(
        obs_np, recon_np, embedding_model, embedding_params, save_path=tsne_save_path
    )
    # ------------------------------
    # [추가 시각화 1] Reward difference heatmap
    diff_save_path = os.path.join(save_dir, "reward_difference.png")
    visualize_reward_difference_heatmap(r_batch_gt_norm, shaped_rewards_norm, save_path=diff_save_path)

    # [추가 시각화 2] Rank correlation between GT and shaped reward
    rank_corr_save_path = os.path.join(save_dir, "rank_correlation.png")
    visualize_rank_correlation(r_batch_gt_norm, shaped_rewards_norm, save_path=rank_corr_save_path)

    rank_corr_save_path = os.path.join(save_dir, "PT_rank_correlation.png")
    visualize_rank_correlation_pt(r_batch_gt_norm, r_batch_relabeled_norm, save_path=rank_corr_save_path)

    # [추가 시각화 3] Reward variance comparison
    variance_save_path = os.path.join(save_dir, "reward_variance_comparison.png")
    visualize_reward_variance_comparison(reward_dict, save_path=variance_save_path)

    # [추가 시각화 4] Rank shift 시각화
    rank_shift_path = os.path.join(save_dir, "rank_shift_hist.png")
    visualize_rank_shift(r_batch_gt_norm, shaped_rewards_norm, save_path=rank_shift_path)

    rank_shift_path = os.path.join(save_dir, "rank_shift_hist_pt.png")
    visualize_rank_shift_pt(r_batch_gt_norm, r_batch_relabeled_norm, save_path=rank_shift_path)

    # [추가 시각화 5] Subgoal deviation
    deviation_path = os.path.join(save_dir, "subgoal_deviation.png")
    visualize_subgoal_deviation(obs_np, recon_np, save_path=deviation_path)

    # [추가 시각화 6] Subgoal diversity
    diversity_path = os.path.join(save_dir, "subgoal_diversity.png")
    visualize_subgoal_diversity(recon_np, save_path=diversity_path)

    # [추가 시각화 7] PCA / UMAP projection
    pca_path = os.path.join(save_dir, "pca_projection.png")
    visualize_latent_projection(obs_np, recon_np, embedding_model, embedding_params,method="pca", save_path=pca_path)

    umap_path = os.path.join(save_dir, "umap_projection.png")
    visualize_latent_projection(obs_np,recon_np,embedding_model, embedding_params, method="umap", save_path=umap_path)

    traj_path = os.path.join(save_dir, "trajectory_level_scatter_comparison.png")
    # trajectory_ids = [tid for (tid, _) in trj_mapper if tid in indices]  # 추출된 index에 대해 mapping
    trajectory_ids = [trj_mapper[i][0] for i in indices]
    # visualize_trajectory_level_scatter_comparison(r_batch_gt,r_batch_relabeled, shaped_rewards, trajectory_ids,traj_path)
    visualize_trajectory_level_scatter_comparison(
    r_batch_gt_norm, r_batch_relabeled_norm, shaped_rewards_norm, trajectory_ids,
    save_path=traj_path, normalize_rewards=False
    )

    traj_var_path = os.path.join(save_dir, "trajectory_variance_boxplot.png")
    visualize_trajectory_internal_variance(
        r_batch_gt_norm, r_batch_relabeled_norm, shaped_rewards_norm, trajectory_ids, save_path=traj_var_path
    )

    # --- 추가 분석: Subgoal Predictiveness ---
    subgoal_predictiveness_check(reconstructed_subgoals, next_obs, r_batch_gt_norm)

    # --- 추가 시각화: Trajectory Return Distribution ---
    return_dist_path = os.path.join(save_dir, "trajectory_return_distribution.png")
    visualize_return_distribution(
        r_batch_gt_norm, r_batch_relabeled_norm, shaped_rewards_norm, trajectory_ids, save_path=return_dist_path
    )

    print("Done visualization & correlation checks.")
    print(f"All images saved to {save_dir}")


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)