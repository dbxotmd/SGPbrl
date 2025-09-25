import datetime
import os
import pickle
from typing import Tuple
import json
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
from sklearn.neighbors import NearestNeighbors


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"
# Disable GPU rendering for MuJoCo to avoid segmentation fault
# Completely disable GPU rendering and force software rendering
os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)   # GLFW가 X 찾지 못하게

import matplotlib
matplotlib.use("Agg")             # Matplotlib이 X11 안 찾게
# Force single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"
os.environ["LAPACK_NUM_THREADS"] = "1"

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
flags.DEFINE_string('method', 'negative_distance', 'Reward shaping method')
flags.DEFINE_float('shaping_weight', 1.0, 'Logging interval.')
flags.DEFINE_integer('latent_dim', 16, 'latent dimension for CVAE')
flags.DEFINE_integer('hidden_dim', 32, 'hidden dimension for CVAE')
flags.DEFINE_integer('topkp', 10, 'top k percenetage for CVAE')
flags.DEFINE_boolean('state_action', True, 'Use only state or state-action pair for reward shaping')
flags.DEFINE_boolean('save_images', False, 'Whether to save reconstructed subgoal images (can cause segfault)')
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


def load_vae_model(filename, observation_dim, action_dim, vae_latent_dim, vae_hidden_dim,state_action):
    with open(filename, "rb") as f:
        loaded_params = pickle.load(f)

    vae_latent_dim = vae_latent_dim
    if vae_hidden_dim == 32:
        vae_hidden_dim = [32, 64, 32]
    elif vae_hidden_dim == 64:
        vae_hidden_dim = [64, 128, 64]
    elif vae_hidden_dim == 128:
        vae_hidden_dim = [128, 256, 128]
    elif vae_hidden_dim == 750:
        vae_hidden_dim = [750,750]
    else:
        raise ValueError(f"Invalid vae_hidden_dim: {vae_hidden_dim}")

    if state_action:
        from JaxPref.subgoal_cvae_model_state_action import SubgoalCVAE
        # Initialize a fresh VAE model to get its parameter structure
        from JaxPref.subgoal_cvae_model_state_action import SubgoalCVAE
        vae = SubgoalCVAE(
            vae_latent_dim, observation_dim, action_dim, vae_hidden_dim
        )
        # Create a dummy input to initialize the model parameters
        dummy_state = jnp.ones((1, observation_dim))
        dummy_action = jnp.ones((1, action_dim))
        subgoal_state = jnp.ones((1, observation_dim))
        subgoal_action = jnp.ones((1, observation_dim))
        rng = jax.random.PRNGKey(0)
        params = vae.init(rng, dummy_state, dummy_action, subgoal_state,subgoal_action)

    else:
        from JaxPref.subgoal_cvae_model import SubgoalCVAE
        # Initialize a fresh VAE model to get its parameter structure
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

def plot_extrapolation_error_vs_shaping(
    shaping_term,
    gt_rewards,
    pt_rewards,
    save_dir,
    method_name="unknown",
    shaping_weight=None,
    use_abs_x=True,
    nbins=30,
    x_clip_pct=99.0,     # x 상위 퍼센타일 클리핑
    y_clip_pct=99.0,     # y 상위 퍼센타일 클리핑
    use_hexbin=True,     # 산점도 대신 밀도(hexbin) 사용
    log_x=False,         # x -> log1p(x)
    n_boot=300,          # 부트스트랩 CI 샘플 수
    seed=0,
):
    """
    목적: distance(=shaping_term) 증가에 따라 PT 오차 |PT - GT|가 증가하는지 '명확히' 보여줌.
    - 배경: PT 오차 분포(산점/hexbin)
    - 요약: 분위수-bin 평균 ± 95% 부트스트랩 CI
    - 상관: Spearman rho, p-value (제목에 표기)
    - CSV: bin 통계 저장
    """
    import os, csv
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)

    # --- 1) 데이터 준비
    x = np.asarray(shaping_term).reshape(-1)
    if use_abs_x:
        x = np.abs(x)
    if log_x:
        x = np.log1p(x)

    gt = np.asarray(gt_rewards).reshape(-1)
    pt = np.asarray(pt_rewards).reshape(-1)
    err_pt = np.abs(pt - gt)

    # --- 2) 로버스트 클리핑(극단값이 축을 지배하지 않도록)
    def _clip(arr, pct):
        if pct is None: return arr
        hi = np.nanpercentile(arr, pct)
        return np.clip(arr, None, hi)

    x = _clip(x, x_clip_pct)
    err_pt = _clip(err_pt, y_clip_pct)

    # --- 3) 단조성: Spearman
    rho_pt, p_pt = spearmanr(x, err_pt)

    # --- 4) 분위수 bin 구간
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:  # 분산 거의 없을 때 보호
        edges = np.linspace(np.nanmin(x), np.nanmax(x), num=min(nbins + 1, 5))

    def _bootstrap_ci(vals, n_boot=300):
        vals = np.asarray(vals, float)
        n = vals.size
        if n <= 1:
            m = float(np.mean(vals))
            return m, 0.0
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(np.mean(vals[idx]))
        boots = np.sort(np.asarray(boots))
        lo = np.percentile(boots, 2.5)
        hi = np.percentile(boots, 97.5)
        return float(np.mean(vals)), float(hi - lo) / 2.0

    centers, mean_pt, ci_pt = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (x >= lo) & (x < hi) if i < len(edges) - 2 else (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        centers.append(0.5 * (lo + hi))
        m, c = _bootstrap_ci(err_pt[mask], n_boot=n_boot)
        mean_pt.append(m); ci_pt.append(c)

    centers = np.asarray(centers)
    mean_pt = np.asarray(mean_pt)
    ci_pt = np.asarray(ci_pt)

    # --- 5) 시각화
    plt.figure(figsize=(8, 6))

    # 배경 밀도(PT 오차만)
    if use_hexbin:
        plt.hexbin(x, err_pt, gridsize=60, mincnt=5, alpha=0.35)  # 기본 colormap/색상
    else:
        # 산점도: 많은 데이터면 과밀 주의
        N = x.shape[0]
        take = min(5000, N)
        idx = rng.choice(N, size=take, replace=False)
        plt.scatter(x[idx], err_pt[idx], s=6, alpha=0.15, edgecolors='none', label="PT |err| (scatter)")

    # 구간 평균 ± 95% CI
    # (의미를 강조하기 위해 굵은 선/마커 + CI 밴드)
    plt.errorbar(centers, mean_pt, yerr=ci_pt, linewidth=2, marker='o',
                 label="PT mean |err|", capsize=3)
    plt.fill_between(centers, mean_pt - ci_pt, mean_pt + ci_pt, alpha=0.2)

    # 제목/레이블
    abs_tag = "|x|" if use_abs_x else "x"
    sw_tag = "" if shaping_weight is None else f", w={shaping_weight}"
    log_tag = ", log1p(x)" if log_x else ""
    plt.title(f"PT Extrapolation Error vs Distance ({method_name}{sw_tag}, {abs_tag}{log_tag})\n"
              f"Spearman ρ (PT) = {rho_pt:.3f} (p={p_pt:.1e})")
    plt.xlabel("Shaping distance" + (" (log1p)" if log_x else ""))
    plt.ylabel("|PT − GT|")
    plt.grid(True)
    plt.legend()

    # --- 6) 저장 (PNG + CSV)
    tag = ("abs" if use_abs_x else "signed") + ("_logx" if log_x else "")
    if shaping_weight is not None:
        tag += f"_w{str(shaping_weight).replace('.', '_')}"
    fname = f"pt_error_vs_distance_{method_name}_{tag}"

    out_png = os.path.join(save_dir, fname + ".png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    out_csv = os.path.join(save_dir, fname + ".csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center_x", "pt_mean", "pt_ci"])
        for i in range(len(centers)):
            w.writerow([centers[i], mean_pt[i], ci_pt[i]])

    print(f"[PT Error] Saved figure: {out_png}")
    print(f"[PT Error] Saved bins CSV: {out_csv}")
    print(f"  Spearman rho (PT): {rho_pt:.4f} (p={p_pt:.2e})")



# def plot_extrapolation_error_vs_shaping(
#     shaping_term,
#     gt_rewards,
#     pt_rewards,
#     guider_rewards,
#     save_dir,
#     method_name="unknown",
#     shaping_weight=None,
#     use_abs_x=True,
#     nbins=30,
#     max_scatter=4000,
#     x_clip_pct=99.0,      # clip extreme x outliers
#     y_clip_pct=99.0,      # clip extreme y outliers
#     use_hexbin=True,      # better density visualization
#     log_x=False,          # use log1p(x) scale
#     plot_delta=False,     # plot Δ|err| = GUIDER - PT instead of two series
#     n_boot=300,           # bootstrap for CI
#     seed=0,
# ):
#     import os
#     import csv
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import spearmanr

#     rng = np.random.default_rng(seed)

#     x = np.asarray(shaping_term).reshape(-1)
#     if use_abs_x:
#         x = np.abs(x)
#     if log_x:
#         x = np.log1p(x)

#     gt = np.asarray(gt_rewards).reshape(-1)
#     pt = np.asarray(pt_rewards).reshape(-1)
#     gd = np.asarray(guider_rewards).reshape(-1)

#     err_pt = np.abs(pt - gt)
#     err_gd = np.abs(gd - gt)
#     d_err = err_gd - err_pt

#     # robust clipping (avoids a few extremes dominating scales)
#     def _clip(arr, pct):
#         if pct is None:
#             return arr
#         hi = np.nanpercentile(arr, pct)
#         return np.clip(arr, None, hi)

#     x = _clip(x, x_clip_pct)
#     err_pt = _clip(err_pt, y_clip_pct)
#     err_gd = _clip(err_gd, y_clip_pct)
#     d_err = _clip(d_err, y_clip_pct)

#     # correlations (report on the unclipped relation between x and errors in this view)
#     rho_pt, p_pt = spearmanr(x, err_pt)
#     rho_gd, p_gd = spearmanr(x, err_gd)

#     # quantile bins with unique edges
#     qs = np.linspace(0.0, 1.0, nbins + 1)
#     edges = np.unique(np.quantile(x, qs))
#     if len(edges) < 3:
#         edges = np.linspace(np.nanmin(x), np.nanmax(x), num=min(nbins + 1, 5))

#     def _bootstrap_ci(vals):
#         vals = np.asarray(vals, dtype=float)
#         n = vals.shape[0]
#         if n <= 1:
#             m = float(np.mean(vals))
#             return m, 0.0
#         boots = []
#         for _ in range(n_boot):
#             idx = rng.integers(0, n, size=n)
#             boots.append(np.mean(vals[idx]))
#         boots = np.sort(np.asarray(boots))
#         lo = np.percentile(boots, 2.5)
#         hi = np.percentile(boots, 97.5)
#         return float(np.mean(vals)), float(hi - lo) / 2.0

#     centers = []
#     mean_pt = []; ci_pt = []
#     mean_gd = []; ci_gd = []
#     mean_de = []; ci_de = []

#     for i in range(len(edges) - 1):
#         lo, hi = edges[i], edges[i + 1]
#         mask = (x >= lo) & (x < hi) if i < len(edges) - 2 else (x >= lo) & (x <= hi)
#         if not np.any(mask):
#             continue
#         centers.append(0.5 * (lo + hi))
#         m, c = _bootstrap_ci(err_pt[mask]);    mean_pt.append(m); ci_pt.append(c)
#         m, c = _bootstrap_ci(err_gd[mask]);    mean_gd.append(m); ci_gd.append(c)
#         m, c = _bootstrap_ci(d_err[mask]);     mean_de.append(m); ci_de.append(c)

#     centers = np.asarray(centers)
#     mean_pt = np.asarray(mean_pt); ci_pt = np.asarray(ci_pt)
#     mean_gd = np.asarray(mean_gd); ci_gd = np.asarray(ci_gd)
#     mean_de = np.asarray(mean_de); ci_de = np.asarray(ci_de)

#     # base figure
#     import matplotlib as mpl
#     plt.figure(figsize=(8, 6))

#     # background density
#     if use_hexbin:
#         # both clouds overlaid; set low alpha via default colormap
#         hb1 = plt.hexbin(x, err_pt, gridsize=60, mincnt=5, alpha=0.35)
#         hb2 = plt.hexbin(x, err_gd, gridsize=60, mincnt=5, alpha=0.35)
#     else:
#         N = x.shape[0]
#         take = min(max_scatter, N)
#         idx = rng.choice(N, size=take, replace=False)
#         plt.scatter(x[idx], err_pt[idx], s=6, alpha=0.15, edgecolors='none', label="PT |err| (scatter)")
#         plt.scatter(x[idx], err_gd[idx], s=6, alpha=0.15, edgecolors='none', label="GUIDER |err| (scatter)")

#     # mean +/- CI
#     if plot_delta:
#         plt.plot(centers, mean_de, linewidth=2, marker='o', label="Δ|err| (GUIDER−PT) mean")
#         plt.fill_between(centers, mean_de - ci_de, mean_de + ci_de, alpha=0.2)
#         plt.axhline(0.0, linestyle='--', linewidth=1)
#     else:
#         plt.errorbar(centers, mean_pt, yerr=ci_pt, linewidth=2, marker='o', label="PT mean |err|", capsize=3)
#         plt.errorbar(centers, mean_gd, yerr=ci_gd, linewidth=2, marker='o', label="GUIDER mean |err|", capsize=3)

#     abs_tag = "|x|" if use_abs_x else "x"
#     sw_tag = "" if shaping_weight is None else f", w={shaping_weight}"
#     log_tag = ", log1p(x)" if log_x else ""
#     title_stats = (f"Spearman ρ: PT={rho_pt:.3f} (p={p_pt:.1e}), "
#                    f"GUIDER={rho_gd:.3f} (p={p_gd:.1e})")

#     plt.title(f"Extrapolation Error vs Shaping ({method_name}{sw_tag}, {abs_tag}{log_tag})\n{title_stats}")
#     plt.xlabel("Shaping term" + (" (log1p)" if log_x else ""))
#     plt.ylabel("|predicted − true|" if not plot_delta else "Δ|err| (GUIDER−PT)")
#     plt.grid(True)
#     plt.legend()

#     # save figure + CSV of binned stats
#     tag = ("delta_" if plot_delta else "") + ("abs" if use_abs_x else "signed") + ("_logx" if log_x else "")
#     if shaping_weight is not None:
#         tag += f"_w{str(shaping_weight).replace('.', '_')}"
#     fname = f"extrapolation_error_vs_shaping_{method_name}_{tag}"

#     out_png = os.path.join(save_dir, fname + ".png")
#     plt.savefig(out_png, dpi=300, bbox_inches="tight")
#     plt.close()

#     out_csv = os.path.join(save_dir, fname + ".csv")
#     with open(out_csv, "w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["center_x", "pt_mean", "pt_ci", "gd_mean", "gd_ci", "delta_mean", "delta_ci"])
#         for i in range(len(centers)):
#             w.writerow([centers[i], mean_pt[i], ci_pt[i], mean_gd[i], ci_gd[i], mean_de[i], ci_de[i]])

#     print(f"[Extrapolation] Saved figure: {out_png}")
#     print(f"[Extrapolation] Saved bins CSV: {out_csv}")
#     print(f"  Spearman rho  ->  PT: {rho_pt:.4f} (p={p_pt:.2e}), GUIDER: {rho_gd:.4f} (p={p_gd:.2e})")


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

# def save_recon_subgoal_images_d4rl(env, recon_subgoals, save_dir, env_name=None, width=256, height=256):
#     print(f"save_recon_subgoal_images_d4rl called with {len(recon_subgoals)} subgoals")
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Limit the number of images to prevent memory issues
#     max_images = 5  # Increased back to 5 since we fixed the rendering
#     recon_subgoals = recon_subgoals[:max_images]
    
#     base_env = env
#     while hasattr(base_env, "env"):
#         base_env = base_env.env
    
#     if not hasattr(base_env, "sim"):
#         print("Warning: Environment does not have MuJoCo simulator. Creating placeholder images.")
#         for i, flat_obs in enumerate(recon_subgoals):
#             placeholder = np.zeros((128, 128, 3), dtype=np.uint8)
#             obs_norm = np.clip((flat_obs - flat_obs.min()) / (flat_obs.max() - flat_obs.min() + 1e-8), 0, 1)
#             for j in range(min(len(obs_norm), 64)):
#                 x = j % 8
#                 y = j // 8
#                 intensity = int(obs_norm[j] * 255)
#                 placeholder[y*16:(y+1)*16, x*16:(x+1)*16] = [intensity, intensity//2, 255-intensity]
#             path = os.path.join(save_dir, f"recon_{i:04d}_placeholder.png")
#             imageio.imwrite(path, placeholder)
#             print(f"Saved placeholder image {path}")
#         return
    
#     for i, flat_obs in enumerate(recon_subgoals):
#         print(f"Processing subgoal {i}")
#         try:
#             # Set environment state based on environment type
#             if env_name and any(name in env_name.lower() for name in ['hopper', 'walker2d', 'halfcheetah', 'ant']):
#                 success = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
#             else:
#                 # Fallback to original function for other environments
#                 set_env_state_from_flat_obs(env, flat_obs, debug_idx=i)
#                 success = True
            
#             if not success:
#                 print(f"Warning: Failed to set environment state for image {i}")
#                 continue
            
#             # Try to render with safer error handling
#             try:
#                 # Use only offscreen rendering with smaller size to reduce memory usage
#                 img = base_env.sim.render(width=128, height=128, depth=False, mode='offscreen')
                
#                 if img is not None:
#                     img = np.flipud(img)
#                     path = os.path.join(save_dir, f"recon_{i:04d}.png")
#                     imageio.imwrite(path, img)
#                     print(f"Saved {path}")
#                 else:
#                     print(f"Warning: Render returned None for image {i}")
                    
#             except Exception as render_error:
#                 print(f"Render error for image {i}: {render_error}")
#                 # Create a simple placeholder image instead of failing
#                 try:
#                     placeholder = np.zeros((128, 128, 3), dtype=np.uint8)
#                     placeholder[:, :, 0] = 255  # Red channel
#                     path = os.path.join(save_dir, f"recon_{i:04d}_placeholder.png")
#                     imageio.imwrite(path, placeholder)
#                     print(f"Saved placeholder image {path}")
#                 except:
#                     print(f"Failed to save placeholder for image {i}")
#                 continue
            
#         except Exception as e:
#             print(f"Error saving image {i}: {e}")
#             continue
    
#     print(f"\n✅ All done! First {max_images} images saved in {save_dir}")

# def set_env_state_from_flat_obs(env, flat_obs, debug_idx=None):
#     """
#     Set Mujoco environment state from a 53D flat observation vector.
#     Includes robot joints, gripper, and object (Can) pose.
#     Camera is positioned above object and looks downward.
#     """
#     # Unpack from flat_obs
#     object_qpos = flat_obs[0:6]          # [x, y, z, qx, qy, qz]
#     object_xyz = object_qpos[:3]         # just position
#     object_vel = flat_obs[6:12]          # linear + angular vel
#     joint_pos = flat_obs[14:21]
#     joint_vel = flat_obs[35:42]
#     gripper_qpos = flat_obs[49:51]
#     gripper_qvel = flat_obs[51:53]

#     base_env = env
#     while hasattr(base_env, "env"):
#         base_env = base_env.env

#     # Set qpos and qvel
#     qpos = base_env.sim.data.qpos.copy()
#     qvel = base_env.sim.data.qvel.copy()

#     qpos[0:7] = joint_pos
#     qpos[7:9] = gripper_qpos
#     qpos[9:15] = object_qpos  # Can_main
#     qvel[0:7] = joint_vel
#     qvel[7:9] = gripper_qvel
#     qvel[9:15] = object_vel

#     base_env.sim.data.qpos[:] = qpos
#     base_env.sim.data.qvel[:] = qvel

#     base_env.sim.forward()

import gc
from mujoco_py import MjRenderContextOffscreen

def get_base_env(env):
    """Get the base environment from wrapped environment"""
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return base_env

def set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=None):
    """
    Set MuJoCo environment state from flat observation for D4RL environments.
    Handles different observation structures for different environments.
    """
    base_env = get_base_env(env)
    
    if not hasattr(base_env, "sim"):
        print(f"Warning: Environment {env_name} does not have MuJoCo simulator")
        return False
    
    obs_dim = flat_obs.shape[0]
    if debug_idx is not None and debug_idx < 3:
        print(f"obs_dim: {obs_dim}")
    state = base_env.sim.get_state()
    new_qpos = state.qpos.copy()
    new_qvel = state.qvel.copy()

    # Different D4RL environments have different observation structures
    if "halfcheetah" in env_name.lower() or "walker2d" in env_name.lower():
        # qpos from obs has 8 elements, but sim.data.qpos has 9.
        # The missing element is the root x-position.
        qpos_dim = 8
        qvel_dim = 9
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
        # Reconstruct the full qpos
        new_qpos[0] = 0.0  # Set root x-position to a default value (e.g., 0)
        new_qpos[1:] = qpos # Assign the rest of the values
        new_qvel[:] = qvel
            
    elif "hopper" in env_name.lower():
        # qpos from obs has 5 elements, but sim.data.qpos has 6.
        # The missing element is the root x-position.
        qpos_dim = 5
        qvel_dim = 6
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
        # Reconstruct the full qpos
        new_qpos[0] = 0.0  # Set root x-position to a default value (e.g., 0)
        new_qpos[1:] = qpos # Assign the rest of the values
        new_qvel[:] = qvel
            
    elif "ant" in env_name.lower():
        # For Ant, the observation usually contains the full qpos (excluding root xy) and qvel.
        # However, the D4RL ant obs includes the full qpos.
        # Your original logic is likely correct here, but let's be explicit.
        qpos_dim = 15
        qvel_dim = 14
        qpos = flat_obs[:qpos_dim]
        qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
        new_qpos[:len(qpos)] = qpos
        new_qvel[:len(qvel)] = qvel

    # ... (기존 try-except 블록 내 state 설정 로직으로 이어짐) ...
    # Set the modified state
    try:
        # state = state._replace(qpos=new_qpos, qvel=new_qvel) # 이 부분은 이미 위에서 처리됨
        base_env.set_state(new_qpos, new_qvel) # 많은 D4RL 환경에는 set_state 헬퍼 함수가 있습니다.
                                            # 없다면 기존 방식을 사용하세요.
        # 또는 기존 방식
        state = base_env.sim.get_state()._replace(qpos=new_qpos, qvel=new_qvel)
        base_env.sim.set_state(state)
        base_env.sim.forward()
        return True

    except Exception as e:
        print(f"Error setting environment state: {e}")
        return False
    # # Different D4RL environments have different observation structures
    # if "halfcheetah" in env_name.lower():
    #     # HalfCheetah: 17D observation (8 joint pos + 9 joint vel)
    #     qpos_dim = 8
    #     qvel_dim = 9
    #     qpos = flat_obs[:qpos_dim]
    #     qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    # elif "walker2d" in env_name.lower():
    #     # Walker2d: 17D observation (8 joint pos + 9 joint vel)
    #     qpos_dim = 8
    #     qvel_dim = 9
    #     qpos = flat_obs[:qpos_dim]
    #     qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    # elif "hopper" in env_name.lower():
    #     # Hopper: 11D observation (5 joint pos + 6 joint vel)
    #     qpos_dim = 5
    #     qvel_dim = 6
    #     qpos = flat_obs[:qpos_dim]
    #     qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    # elif "ant" in env_name.lower():
    #     # Ant: 111D observation (15 joint pos + 14 joint vel + 84 contact forces)
    #     qpos_dim = 15
    #     qvel_dim = 14
    #     qpos = flat_obs[:qpos_dim]
    #     qvel = flat_obs[qpos_dim:qpos_dim + qvel_dim]
        
    # else:
    #     # Default: assume first half is qpos, second half is qvel
    #     mid_point = obs_dim // 2
    #     qpos = flat_obs[:mid_point]
    #     qvel = flat_obs[mid_point:]
    
    # # Debug info for first few images
    # if debug_idx is not None and debug_idx < 3:
    #     print(f"Image {debug_idx}: qpos shape={qpos.shape}, qvel shape={qvel.shape}")
    #     print(f"qpos range: [{qpos.min():.3f}, {qpos.max():.3f}]")
    #     print(f"qvel range: [{qvel.min():.3f}, {qvel.max():.3f}]")
    
    # try:
    #     # Get current state and preserve structure
    #     state = base_env.sim.get_state()
        
    #     # Create copies to modify
    #     new_qpos = state.qpos.copy()
    #     new_qvel = state.qvel.copy()
        
    #     # Update only the overlapping dimensions
    #     min_qpos_len = min(len(qpos), len(new_qpos))
    #     min_qvel_len = min(len(qvel), len(new_qvel))
        
    #     new_qpos[:min_qpos_len] = qpos[:min_qpos_len]
    #     new_qvel[:min_qvel_len] = qvel[:min_qvel_len]
        
    #     # Set the modified state
    #     state = state._replace(qpos=new_qpos, qvel=new_qvel)
    #     base_env.sim.set_state(state)
    #     base_env.sim.forward()
        
    #     if debug_idx is not None and debug_idx < 3:
    #         print(f"Environment state set successfully for image {debug_idx}")
        
    #     return True
        
    # except Exception as e:
    #     print(f"Error setting environment state: {e}")
    #     return False

def save_recon_subgoal_images_d4rl(
    env, state_array, save_dir, env_name=None,
    camera_name='track', width=256, height=256,
    max_images=10, prefix="recon"
):
    """
    Save reconstructed subgoal images with proper memory management and error handling
    """
    os.makedirs(save_dir, exist_ok=True)
    base_env = get_base_env(env)
    sim = base_env.sim

    # Clip to what we actually want to save
    state_array = np.array(state_array)
    print(f"Processing {len(state_array)} states")

    # Check camera names
    available_cameras = list(sim.model.camera_names)
    print("Available cameras:", available_cameras)
    
    if camera_name not in available_cameras:
        print(f"Warning: Camera '{camera_name}' not found. Using first available camera.")
        camera_name = available_cameras[0] if available_cameras else None
    
    if camera_name is None:
        print("Error: No cameras available in the environment")
        return

    # Create renderer with proper context management
    saved_count = 0
    
    try:
        for i, flat_obs in enumerate(state_array):
            try:
                # Set environment state
                set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
                print("working")

                # Ensure physics is updated
                sim.forward()
                print("forward")
                print(sim)
                print("sim")
                # Render with error handling
                try:
                    print("before rendering")
                    # Use environment's render method instead of MuJoCo GPU rendering
                    if hasattr(env, 'render'):
                        img = sim.render(width=width, height=height, camera_name=camera_name, depth=False)
                        if img is None:
                            raise Exception("Environment render returned None")
                        print("rendering completed using environment render")
                    else:
                        # Skip rendering if environment doesn't support it
                        print("Environment does not support rendering, skipping...")
                        continue
                    
                    img = np.flipud(img)
                    path = os.path.join(save_dir, f"recon_{i:04d}.png")
                    imageio.imwrite(path, img)
                    print(f"Saved {path}")
                    saved_count += 1
                    
                except Exception as render_error:
                    print(f"⚠️  Rendering failed for idx={i}: {render_error}")
                    continue

                # Force garbage collection every few images to prevent memory buildup
                if i % 5 == 0:
                    gc.collect()

            except Exception as state_error:
                print(f"⚠️  Error processing state {i}: {state_error}")
                continue

    except Exception as e:
        print(f"❌ Critical error in rendering setup: {e}")
        return

    finally:
        # Clean up renderer
        if renderer is not None:
            try:
                del renderer
                gc.collect()
                print("Renderer cleaned up")
            except:
                pass

    print(f"\n🏁 Done! Saved {saved_count}/{len(state_array)} images in {save_dir}")



# def save_recon_subgoal_images_d4rl(env, state_array, save_dir, env_name=None,
#                      camera_name='track', width=256, height=256, max_images=10, prefix="recon"):
#     os.makedirs(save_dir, exist_ok=True)
#     base_env = get_base_env(env)
#     state_array = np.array(state_array)  # Ensure numpy array

#     for i, flat_obs in enumerate(state_array):
#         # Use D4RL-specific function if environment name is provided
#         if env_name and any(name in env_name.lower() for name in ['hopper', 'walker2d', 'halfcheetah', 'ant']):
#             success = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
#         else:
#             # Use original Robosuite function
#             set_env_state_from_flat_obs(env, flat_obs, debug_idx=i)
#             success = True
        
#         if not success:
#             print(f"Warning: Failed to set environment state for image {i}")
#             continue
            
#         try:
#             img = base_env.sim.render(width=width, height=height, camera_name=camera_name, depth=False)
#             img = np.flipud(img)
#             path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
#             imageio.imwrite(path, img)
#             print(f"Saved {path}")
#         except Exception as e:
#             print(f"Error rendering image {i}: {e}")
#             continue
            
#     print(f"\n✅ All done! First {max_images} images saved in {save_dir} (prefix: {prefix})")
# def save_recon_subgoal_images_d4rl(env, recon_subgoals, save_dir, env_name=None, width=128, height=128):
#     os.makedirs(save_dir, exist_ok=True)
#     max_images = min(len(recon_subgoals), 5)

#     for i, flat_obs in enumerate(recon_subgoals[:max_images]):
#         path = os.path.join(save_dir, f"recon_{i:04d}.png")
#         try:
#             # 환경 상태 설정
#             env = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
            
#             # 렌더링 전 잠시 대기
#             import time
#             time.sleep(0.1)
            
#             # 안전한 렌더링
#             try:
#                 img = env.sim.render(width=width, height=height, mode='offscreen')
#                 img = img[::-1]  # Flip vertically if needed
#             except Exception as render_error:
#                 print(f"Direct render failed: {render_error}, trying alternative method")
#                 # 대안적 렌더링 방법
#                 img = env.sim.render(width=width, height=height, mode='offscreen')
#                 if img is not None:
#                     img = img[::-1]  # MuJoCo는 이미지를 뒤집어서 반환
            
#             if img is None:
#                 raise RuntimeError("Both rendering methods returned None")
                
#             imageio.imwrite(path, img)
#             print(f"Saved {path}")
            
#         except Exception as e:
#             print(f"Render failed for {i}: {e}, generating placeholder.")
#             placeholder = np.zeros((height, width, 3), dtype=np.uint8)
#             imageio.imwrite(path, placeholder)

# def save_recon_subgoal_images_d4rl(env, recon_subgoals, save_dir, env_name=None, width=128, height=128):
#     os.makedirs(save_dir, exist_ok=True)
#     max_images = min(len(recon_subgoals), 5)


#     for i, flat_obs in enumerate(recon_subgoals[:max_images]):
#         path = os.path.join(save_dir, f"recon_{i:04d}.png")
#         try:
#             # set the env state...
#             env = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
#             # use gym.render for stability
#             img = env.render(mode='rgb_array', width=width, height=height)
#             if img is None:
#                 raise RuntimeError("env.render returned None")
#             imageio.imwrite(path, img)
#             print(f"Saved {path}")
#         except Exception as e:
#             print(f"Render failed for {i}: {e}, generating placeholder.")
#             placeholder = np.zeros((height, width, 3), dtype=np.uint8)
#             imageio.imwrite(path, placeholder)

# def save_recon_subgoal_images_d4rl_ultra_safe(
#     env, state_array, save_dir, env_name=None,
#     camera_name='track', width=256, height=256,
#     max_images=1000, prefix="recon"
# ):
#     """
#     Ultra-safe implementation that completely avoids GPU rendering
#     """
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Clip to what we actually want to save
#     state_array = np.array(state_array)[:max_images]
#     print(f"Processing {len(state_array)} states using ultra-safe method (no GPU rendering)")

#     saved_count = 0
    
#     for i, flat_obs in enumerate(state_array):
#         try:
#             # Set environment state
#             success = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
#             if not success:
#                 print(f"⚠️  Failed to set state for idx={i}")
#                 continue

#             # Use only environment's render method - no MuJoCo GPU rendering
#             try:
#                 if hasattr(env, 'render'):
#                     img = env.render(mode='rgb_array', width=width, height=height)
#                     if img is not None and img.size > 0:
#                         path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
#                         imageio.imwrite(path, img)
#                         print(f"✅ Saved {path}")
#                         saved_count += 1
#                     else:
#                         print(f"⚠️  Render returned empty image for idx={i}")
#                 else:
#                     print("⚠️  Environment does not support rendering")
#                     break
                    
#             except Exception as render_error:
#                 print(f"⚠️  Rendering failed for idx={i}: {render_error}")
#                 continue

#             # Clean up memory periodically
#             if i % 5 == 0:
#                 gc.collect()

#         except Exception as e:
#             print(f"⚠️  Error processing state {i}: {e}")
#             continue

#     print(f"\n🏁 Ultra-safe method done! Saved {saved_count}/{len(state_array)} images in {save_dir}")


def save_recon_subgoal_images_d4rl_ultra_safe(
    env, state_array, save_dir, env_name=None,
    camera_name='track', width=256, height=256,
    max_images=100, prefix="recon"
):
    import gc, imageio, numpy as np
    os.makedirs(save_dir, exist_ok=True)

    base_env = get_base_env(env)
    sim = base_env.sim

    state_array = np.array(state_array)[:max_images]
    print(f"Processing {len(state_array)} states (EGL offscreen)")

    # 카메라 확인
    cam_names = list(sim.model.camera_names)
    if camera_name not in cam_names:
        print(f"Warning: camera '{camera_name}' not found. Using '{cam_names[0] if cam_names else None}'")
        camera_name = cam_names[0] if cam_names else None
    if camera_name is None:
        print("Error: no cameras in this model"); return

    saved = 0
    for i, flat_obs in enumerate(state_array):
        try:
            ok = set_env_state_from_flat_obs_d4rl(env, flat_obs, env_name, debug_idx=i)
            if not ok:
                print(f"⚠️  Failed to set state for idx={i}")
                continue

            sim.forward()
            # --- 오프스크린 렌더 (GLFW 불필요) ---
            img = sim.render(width=width, height=height, camera_name=camera_name, depth=False)
            if img is None:
                raise RuntimeError("sim.render returned None")
            img = np.flipud(img)  # mujoco_py는 상하 반전 필요

            path = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
            imageio.imwrite(path, img)
            print(f"✅ Saved {path}")
            saved += 1

            if i % 5 == 0:
                gc.collect()

        except Exception as e:
            print(f"⚠️  Rendering failed for idx={i}: {e}")
            continue

    print(f"\n🏁 Done! Saved {saved}/{len(state_array)} images in {save_dir}")

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

def save_prd_results( env_name, seed, latent_dim, hidden_dim, topkp, method, weight,
    k, precision, recall
):
    output_path=f"./logs/tb/{env_name}/reward_True_PrefTransformer/all_prd_results.json"

    result = {
        "env_name": env_name,
        "seed": seed,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "topkp": topkp,
        "method": method,
        "shaping_weight": weight,
        "k": k,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
    }

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Appended PRD result to: {output_path}")

def main(_):
    cvae_path = "subgoal_vae_"+FLAGS.env_name+"_"+str(FLAGS.seed)+"_"+str(FLAGS.latent_dim)+"_"+str(FLAGS.hidden_dim)+"_"+str(FLAGS.topkp)+"_"+str(FLAGS.state_action)+".pkl"
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
        f"{cvae_path}",
        f"{FLAGS.method}",
        str(FLAGS.shaping_weight),
        f"{FLAGS.state_action}",
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # 1) 환경, 데이터셋 로드
    # dataset_path = os.path.join(FLAGS.robosuite_dataset_path, FLAGS.env_name.lower(), FLAGS.robosuite_dataset_type, "low_dim.hdf5")
    env, dataset, ground_truth,trj_mapper = make_env_and_dataset_d4rl(FLAGS.env_name, FLAGS.seed)
    # env, dataset, ground_truth,trj_mapper = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, dataset_path, max_episode_steps=FLAGS.max_episode_steps)
    # run_attention_reward_analysis(FLAGS.env_name,FLAGS,save_dir)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    # ------------------------------
    # 2) CVAE 로드 (TrainState)
    print("obs_dim : ", obs_dim)
    cvae_state = load_vae_model(
        filename=cvae_path,  # 실제 파일 경로
        # filename="subgoal_vae_walker2d-medium-replay-v2_0.pkl",  # 실제 파일 경로
        observation_dim=obs_dim,
        action_dim=act_dim,
        vae_latent_dim=FLAGS.latent_dim,
        vae_hidden_dim=FLAGS.hidden_dim,
        state_action=FLAGS.state_action,
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
    dataset, ground_truth, trj_mapper, num_traj=10
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
        # print("recon_subgoal.shape : ", recon_subgoal.shape[-1])
        print(recon_subgoal)
        reconstructed_list.append(recon_subgoal)

    reconstructed_subgoals = jnp.concatenate(
        reconstructed_list, axis=0
    )  # (n_samples, obs_dim)

       # Save reconstructed subgoal images (optional - can cause segmentation fault)
    if FLAGS.save_images:
        try:
            print("Attempting to save reconstructed subgoal images...")
            # save_recon_subgoal_images_d4rl(env, reconstructed_subgoals, os.path.join(save_dir, "recon_subgoal_images"), env_name=FLAGS.env_name)
             # Save reconstructed subgoal images (최대 10개, 필요시 max_images 조정)
            save_recon_subgoal_images_d4rl_ultra_safe(
                env,
                np.array(reconstructed_subgoals),
                os.path.join(save_dir, "recon_subgoal_images"),
                env_name = FLAGS.env_name,
                camera_name='track',
                width=256,
                height=256,
                max_images=10,
                prefix="recon"
            )
            # Save original observation images (최대 10개, 필요시 max_images 조정)
            save_recon_subgoal_images_d4rl_ultra_safe(
                env,
                np.array(obs),
                os.path.join(save_dir, "obs_images"),
                env_name = FLAGS.env_name,
                camera_name='track',
                width=256,
                height=256,
                max_images=10,
                prefix="obs"
            )
        except Exception as e:
            print(f"Warning: Failed to save subgoal images due to error: {e}")
            print("Continuing with other visualizations...")
    else:
        print("Skipping subgoal image generation (use --save_images=True --disable_rendering=False to enable)")

    print("Done visualization & correlation checks.")
    print(f"All images saved to {save_dir}")
    # ------------------------------
    # [NEW] Normalize all rewards using GT min/max for fair comparison
    min_gt = np.min(r_batch_gt)
    max_gt = np.max(r_batch_gt)
    def norm_with_gt(x):
        return (x - min(x)) / (max(x) - min(x))
    r_batch_gt_norm = (r_batch_gt)
    r_batch_relabeled_norm = (r_batch_relabeled)
    # shaped_rewards_norm = norm_with_gt(shaped_rewards)
    # shaped_rewards_norm = r_batch_relabeled_norm + shaping_term  # per-sample shaping

    # r_batch_gt_norm = (r_batch_gt)
    # r_batch_relabeled_norm = (r_batch_relabeled)
    method = FLAGS.method
    discount =0.99
    eps = 1e-8
    # ------------------------------
    # 5) Reward shaping
    if FLAGS.state_action:
        recon_subgoals_state  = reconstructed_subgoals[..., :obs_dim]    # shape = (batch, seq_len, state_dim)
        recon_subgoals_action = reconstructed_subgoals[..., obs_dim:]    # shape = (batch, seq_len, action_dim)
        if method == "negative_distance":
            # 방법 1: (state, action) 거리의 음수 (가까울수록 높은 보상)
            state_dist = jnp.linalg.norm(
                obs - recon_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                acts - recon_subgoals_action, axis=-1
            )
            shaping_term = -(state_dist + action_dist)

        elif method == "gaussian_kernel":
            # 방법 3: (state, action) 가우시안 커널 (sigma로 조절)
            sigma = 1.0  # 하이퍼파라미터
            state_dist = jnp.linalg.norm(
                obs - recon_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                acts - recon_subgoals_action, axis=-1
            )
            shaping_term = (
                jnp.exp(-state_dist**2 / (2 * sigma**2)) +
                jnp.exp(-action_dist**2 / (2 * sigma**2))
            )

        elif method == "cosine_similarity":
            # 방법 4: (state, action) 코사인 유사도 (-1~1 범위)
            def cosine_sim(a, b):
                return jnp.sum(a * b, axis=-1) / (
                    jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8
                )
            state_cos = (1+cosine_sim(obs, recon_subgoals_state))/2
            action_cos = (1+cosine_sim(acts, recon_subgoals_action))/2
            shaping_term = (state_cos + action_cos)

        elif method == "normalized_distance":
            # 방법 5: (state, action) 정규화된 거리 (0~1 범위로 스케일)
            state_dist = jnp.linalg.norm(
                obs - recon_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                acts - recon_subgoals_action, axis=-1
            )
            state_dist_norm = (state_dist / (jnp.max(state_dist) + 1e-8))
            action_dist_norm =(action_dist / (jnp.max(action_dist) + 1e-8))
            shaping_term = -(state_dist_norm + action_dist_norm)

        elif method == "potential_based":
            # 방법 6: (state, action) Potential-based shaping
            # Φ(s, a) = -distance_to_subgoal(s, a)
            current_potential = jnp.linalg.norm(
                obs - recon_subgoals_state, axis=-1
            )
            next_potential = jnp.linalg.norm(
                next_obs - recon_subgoals_state, axis=-1
            )
            shaping_term = discount * next_potential - current_potential

        else:
            # 디폴트: 거리의 음수
            state_dist = jnp.linalg.norm(
                obs - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                acts - recon_subgoals_action, axis=-1
            )
            shaping_term = -(state_dist + action_dist)

    else:

        if method == "negative_distance":
            # 방법 1: 거리의 음수 (가까울수록 높은 보상)
            distances = jnp.linalg.norm(
                next_obs - reconstructed_subgoals, axis=-1
            )
            shaping_term = -distances
            
        elif method == "gaussian_kernel":
            # 방법 3: 가우시안 커널 (sigma로 조절)
            sigma = 1.0  # 하이퍼파라미터
            distances = jnp.linalg.norm(
                next_obs - reconstructed_subgoals, axis=-1
            )
            shaping_term = jnp.exp(-distances**2 / (2 * sigma**2))
            
        elif method == "cosine_similarity":
            # 방법 4: 코사인 유사도 (-1~1 범위)
            def cosine_sim(a, b):
                return jnp.sum(a * b, axis=-1) / (
                    jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8
                )
            shaping_term = (1+cosine_sim(next_obs, reconstructed_subgoals))/2
            
        elif method == "normalized_distance":
            # 방법 6: 정규화된 거리 (0~1 범위로 스케일)
            distances = jnp.linalg.norm(
                next_obs - reconstructed_subgoals, axis=-1
            )
            max_distance = jnp.max(distances)
            shaping_term = - (distances / (max_distance + 1e-8))
            
        elif method == "potential_based":
            # 방법 7: Potential-based shaping (이론적으로 보장된 방법)
            # Φ(s) = -distance_to_subgoal(s)
            current_potential = jnp.linalg.norm(
                obs - reconstructed_subgoals, axis=-1
            )
            next_potential = jnp.linalg.norm(
                next_obs - reconstructed_subgoals, axis=-1
            )
            shaping_term = discount * next_potential - current_potential
        else:
            distances = jnp.linalg.norm(
               next_obs - reconstructed_subgoals, axis=-1
            )
            shaping_term = -distances
    
    
    shaped_rewards = r_batch_relabeled + FLAGS.shaping_weight * shaping_term  # per-sample shaping

    shaped_rewards_norm = (shaped_rewards)
    print(shaping_term)

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

    # Save reconstructed subgoal images
    # save_recon_subgoal_images(env, reconstructed_subgoals, os.path.join(save_dir, "recon_subgoal_images"))

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
    if FLAGS.state_action:
        recon_np = np.array(recon_subgoals_state)
    else:
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
    if FLAGS.hidden_dim == 32:
        hidden_dims = [32, 64, 32]
    elif FLAGS.hidden_dim == 64:
        hidden_dims = [64, 128, 64]
    elif FLAGS.hidden_dim == 128:
        hidden_dims = [128, 256, 128]
    elif FLAGS.hidden_dim == 750:
        hidden_dims = [750,750]
    else:
        raise ValueError(f"Invalid hidden_dim: {FLAGS.hidden_dim}")
    embedding_model = EmbeddingModel(latent_dim = FLAGS.latent_dim ,hidden_dims = hidden_dims)
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
    if FLAGS.state_action:
        subgoal_predictiveness_check(recon_subgoals_state, next_obs, r_batch_gt_norm)
    else:
        subgoal_predictiveness_check(reconstructed_subgoals, next_obs, r_batch_gt_norm)

    # --- 추가 시각화: Trajectory Return Distribution ---
    return_dist_path = os.path.join(save_dir, "trajectory_return_distribution.png")
    visualize_return_distribution(
        r_batch_gt_norm, r_batch_relabeled_norm, shaped_rewards_norm, trajectory_ids, save_path=return_dist_path
    )

    # --- Precision/Recall (PRD) 계산 및 출력 ---
    k = 3  # 논문 기본값
    precision, recall = compute_prd(obs_np, recon_np, k=k)
    print(f"[PRD] Precision (recon in obs manifold, k={k}): {precision:.4f}")
    print(f"[PRD] Recall    (obs in recon manifold, k={k}): {recall:.4f}")

    save_prd_results(
        env_name=FLAGS.env_name,
        seed=FLAGS.seed,
        latent_dim=FLAGS.latent_dim,
        hidden_dim=FLAGS.hidden_dim,
        topkp=FLAGS.topkp,
        method=FLAGS.method,
        weight=FLAGS.shaping_weight,
        k=k,
        precision=precision,
        recall=recall
    )
    plot_extrapolation_error_vs_shaping(
        shaping_term=shaping_term,
        gt_rewards=r_batch_gt_norm,
        pt_rewards=r_batch_relabeled_norm,       # 보상모델(PT) 예측값
        guider_rewards=shaped_rewards_norm,      # shaping 반영된 예측값
        save_dir=save_dir,
        method_name=FLAGS.method,
        shaping_weight=FLAGS.shaping_weight,
        use_abs_x=False,   # signed x
        nbins=30,
    )
    plot_extrapolation_error_vs_shaping(
        shaping_term=shaping_term,
        gt_rewards=r_batch_gt_norm,
        pt_rewards=r_batch_relabeled_norm,
        guider_rewards=shaped_rewards_norm,
        save_dir=save_dir,
        method_name=FLAGS.method,
        shaping_weight=FLAGS.shaping_weight,
        use_abs_x=True,    # |x|: 순수 '거리' 해석에 가까운 보기
        nbins=30,
    )

def compute_prd(obs, recon, k=3):
    """
    Precision-Recall for Distributions (NeurIPS 2019) 방식 구현.
    obs: (N, D) 실제 데이터
    recon: (N, D) 생성 데이터
    k: k-NN
    """
    # 1. 각 obs/recon에 대해 k-NN 거리 계산 (자기 자신 제외)
    nbrs_obs = NearestNeighbors(n_neighbors=k+1).fit(obs)
    obs_dists, _ = nbrs_obs.kneighbors(obs)
    obs_radii = obs_dists[:, -1]  # k+1번째(자기 자신 포함) -> k번째 이웃까지 거리

    nbrs_recon = NearestNeighbors(n_neighbors=k+1).fit(recon)
    recon_dists, _ = nbrs_recon.kneighbors(recon)
    recon_radii = recon_dists[:, -1]

    # 2. Precision: recon이 obs 매니폴드 안에 있는 비율
    dists = np.linalg.norm(recon[:, None, :] - obs[None, :, :], axis=-1)  # (n_recon, n_obs)
    min_dists = dists.min(axis=1)
    # 각 recon 샘플이 obs 매니폴드(해당 obs의 k-NN 반경) 안에 있는지
    # precision = np.mean(min_dists <= obs_radii[np.argmin(dists, axis=1)])
    # 각 recon에 대해 obs 중 하나라도 포함되는지 체크
    in_obs = np.any(dists <= obs_radii[None, :] , axis=1)  # shape (n_recon,)
    precision = np.mean(in_obs)


    # 3. Recall: obs가 recon 매니폴드 안에 있는 비율
    dists = np.linalg.norm(obs[:, None, :] - recon[None, :, :], axis=-1)  # (n_obs, n_recon)
    min_dists = dists.min(axis=1)
    # recall = np.mean(min_dists <= recon_radii[np.argmin(dists, axis=1)])
    # 각 obs에 대해 recon 중 하나라도 포함되는지 체크
    in_recon = np.any(dists <= recon_radii[None, :] , axis=1)
    recall = np.mean(in_recon)


    return precision, recall

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)

