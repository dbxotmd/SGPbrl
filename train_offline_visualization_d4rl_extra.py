import datetime
import os
import pickle
from typing import Tuple, Dict
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

# viz & stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from scipy.stats import kendalltau, spearmanr, entropy, wasserstein_distance
import pandas as pd
import imageio
from sklearn.neighbors import NearestNeighbors

# JaxPref utils
from JaxPref.reward_transform import qlearning_robosuite_dataset
import JaxPref.reward_transform as r_tf
from JaxPref.jax_utils import batch_to_jax
from JaxPref.sampler import TrajSampler
from JaxPref.replay_buffer import get_d4rl_dataset
from JaxPref.utils import set_random_seed

# env / threads
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"
os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"
os.environ["LAPACK_NUM_THREADS"] = "1"

FLAGS = flags.FLAGS

# ===== Existing flags =====
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("use_reward_model", False, "Use reward model for relabeling reward.")
flags.DEFINE_string("model_type", "MLP", "type of reward model.")
flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
flags.DEFINE_string("comment", "base", "comment for distinguishing experiments.")
flags.DEFINE_integer("seq_len", 25, "sequence length for relabeling reward in Transformer.")
flags.DEFINE_bool("use_diff", False, "use difference in sequence for reward relabeling.")
flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with transformer.")
flags.DEFINE_string('method', 'negative_distance', 'Reward shaping method')
flags.DEFINE_float('shaping_weight', 1.0, 'Shaping weight.')
flags.DEFINE_integer('latent_dim', 16, 'latent dimension for CVAE')
flags.DEFINE_integer('hidden_dim', 32, 'hidden dimension for CVAE')
flags.DEFINE_integer('topkp', 10, 'top k percentage for CVAE')
flags.DEFINE_boolean('state_action', True, 'Use state-action CVAE')
flags.DEFINE_boolean('save_images', False, 'Save recon images (mujoco render)')
config_flags.DEFINE_config_file("config","default.py","HP config file",lock_config=False)

# ===== New flags for preference index loading (from your training script) =====
flags.DEFINE_string("data_dir", "./logs/pref_indices", "Directory that stores saved human_indices(_2) and labels")
flags.DEFINE_integer("num_query", 2000, "Number of preference queries used in training")
flags.DEFINE_integer("query_len", 50, "Segment length used for preference queries")
flags.DEFINE_boolean("balance", True, "Balance positive/negative when forming queries")
flags.DEFINE_boolean("use_human_label", True, "Whether human preferences were used")
flags.DEFINE_integer("data_seed", 0, "Seed used when collecting preference segments")
flags.DEFINE_integer("max_traj_length", 1000, "Max traj length (for sampler)")
flags.DEFINE_float("clip_action", 1.0, "Action clip abs value")
flags.DEFINE_boolean("robosuite", False, "Use robosuite dataset")
flags.DEFINE_string("robosuite_dataset_type", "mh", "robosuite dataset type (mh/ph)")
flags.DEFINE_string("robosuite_dataset_path", "~/.robomimic/datasets", "robomimic dataset path")

# ========= Models =========
class EmbeddingModel(nn.Module):
    hidden_dims: list
    latent_dim: int
    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x); x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

def load_vae_model(filename, observation_dim, action_dim, vae_latent_dim, vae_hidden_dim, state_action):
    with open(filename, "rb") as f:
        loaded_params = pickle.load(f)

    if vae_hidden_dim == 32:   vae_hidden_dim = [32, 64, 32]
    elif vae_hidden_dim == 64: vae_hidden_dim = [64, 128, 64]
    elif vae_hidden_dim == 128: vae_hidden_dim = [128, 256, 128]
    elif vae_hidden_dim == 750: vae_hidden_dim = [750, 750]
    else: raise ValueError(f"Invalid vae_hidden_dim: {vae_hidden_dim}")

    if state_action:
        from JaxPref.subgoal_cvae_model_state_action import SubgoalCVAE
        vae = SubgoalCVAE(vae_latent_dim, observation_dim, action_dim, vae_hidden_dim)
        dummy_state = jnp.ones((1, observation_dim))
        dummy_action = jnp.ones((1, action_dim))
        subgoal_state = jnp.ones((1, observation_dim))
        subgoal_action = jnp.ones((1, action_dim))
        rng = jax.random.PRNGKey(0)
        params = vae.init(rng, dummy_state, dummy_action, subgoal_state, subgoal_action)
    else:
        from JaxPref.subgoal_cvae_model import SubgoalCVAE
        vae = SubgoalCVAE(vae_latent_dim, observation_dim, action_dim, vae_hidden_dim)
        rng = jax.random.PRNGKey(0)
        dummy_state = jnp.ones((1, observation_dim))
        dummy_action = jnp.ones((1, action_dim))
        dummy_subgoal_state = jnp.ones((1, observation_dim))
        params = vae.init(rng, dummy_state, dummy_action, dummy_subgoal_state)

    tx = optax.adam(1e-4)
    vae_state = train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=tx)
    vae_state = vae_state.replace(params=loaded_params)
    print(f"VAE model parameters loaded from {filename}")
    return vae_state

def initialize_model():
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt["reward_model"]
    return reward_model

# ========= Dataset utils & normalization =========
def normalize(dataset, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(
        dataset.observations, dataset.actions, dataset.rewards, dataset.masks,
        dataset.dones_float, dataset.next_observations
    )

    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        for _ in range(len(traj)):
            trj_mapper.append((trj_idx, len(traj)))

    def compute_returns(traj): return sum([step[2] for step in traj])
    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = dataset.rewards[i]
        if "antmaze" in env_name:
            _, len_trj = trj_mapper[i]; _reward -= min_return / len_trj
        _reward /= max_return - min_return
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)
    return np.array(normalized_rewards), trj_mapper

def normalize_gt(dataset, gt, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions, gt, dataset.masks, dataset.dones_float, dataset.next_observations)
    trj_mapper = []
    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
        for _ in range(len(traj)):
            trj_mapper.append((trj_idx, len(traj)))
    def compute_returns(traj): return sum([step[2] for step in traj])
    sorted_trajs = sorted(trajs, key=compute_returns)
    min_return, max_return = compute_returns(sorted_trajs[0]), compute_returns(sorted_trajs[-1])

    normalized_rewards = []
    for i in range(dataset.size):
        _reward = gt[i]
        if 'antmaze' in env_name:
            _, len_trj = trj_mapper[i]; _reward -= min_return / len_trj
        _reward /= max_return - min_return
        _reward *= max_episode_steps
        normalized_rewards.append(_reward)
    return normalized_rewards

def make_env_and_dataset_d4rl(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset, np.ndarray, list]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    env.seed(seed); env.action_space.seed(seed); env.observation_space.seed(seed)
    dataset = D4RLDataset(env)
    ground_truth = dataset.rewards.copy()

    if FLAGS.use_reward_model:
        reward_model = initialize_model()
        if FLAGS.model_type == "MR":
            dataset = reward_from_preference(env_name, dataset, reward_model, batch_size=FLAGS.batch_size)
        else:
            dataset = reward_from_preference_transformer(
                env_name, dataset, reward_model, batch_size=FLAGS.batch_size,
                seq_len=FLAGS.seq_len, use_diff=FLAGS.use_diff, label_mode=FLAGS.label_mode
            )
        del reward_model

    normalized_rewards, trj_mapper = normalize(dataset, env_name, max_episode_steps=env.env.env._max_episode_steps)
    dataset.rewards = normalized_rewards
    if "antmaze" in env_name:
        dataset.rewards -= 1.0
    if ("halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name):
        ground_truth = normalize_gt(dataset, ground_truth, env_name, max_episode_steps=env.env.env._max_episode_steps)
    elif "antmaze" in env_name:
        ground_truth = ground_truth - 1.0
    return env, dataset, ground_truth, trj_mapper

# ========= Preference indices → masks =========
def resolve_env_key_for_indices(env_name: str, robosuite: bool, robosuite_dataset_type: str) -> str:
    if "dense" in env_name:
        return "-".join(env_name.split("-")[:-2] + [env_name.split("-")[-1]])
    if robosuite:
        return f"{env_name}_{robosuite_dataset_type}"
    return env_name

def build_pref_mask_from_saved_indices(
    saved_indices,         # [indices_1, indices_2]
    saved_labels,          # labels used during training (or None)
    num_query: int,        # number of queries used
    len_query: int,        # segment length
    dataset_size: int,     # total timesteps
) -> np.ndarray:
    """
    Construct a boolean mask where True means the timestep was part of ANY preference segment
    actually used for training, i.e., for each start index, cover [start, start+len_query).
    Follows load_queries_with_indices query_range rule.
    """
    in_pref_mask = np.zeros((dataset_size,), dtype=bool)

    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        # load_queries_with_indices: last num_query are the ones used
        query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))

    def _as1d(a):
        a = np.asarray(a)
        return a.reshape(-1)

    idx1 = _as1d(saved_indices[0])
    idx2 = _as1d(saved_indices[1])

    idx1 = idx1[query_range]
    idx2 = idx2[query_range]

    for start in np.concatenate([idx1, idx2], axis=0):
        if np.isnan(start):
            continue
        s = int(start)
        if s < 0 or s >= dataset_size:
            continue
        e = min(s + len_query, dataset_size)
        in_pref_mask[s:e] = True

    return in_pref_mask

def load_pref_indices_and_mask(gym_env, dataset, env_key: str) -> Dict[str, np.ndarray]:
    """
    Load saved human_indices(_2) & labels from FLAGS.data_dir/env_key
    Build boolean masks over the entire dataset timesteps:
        in_pref_mask: timesteps that appeared in ANY preference segment (start..start+len_query)
        out_pref_mask: complement
    """
    base_path = os.path.join(FLAGS.data_dir, env_key)
    assert os.path.exists(base_path), f"Preference index path not found: {base_path}"

    human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(base_path))
    with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:
        human_indices = pickle.load(fp)
    with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:
        human_indices_2 = pickle.load(fp)
    with open(os.path.join(base_path, human_labels_file), "rb") as fp:
        human_labels = pickle.load(fp)

    in_pref_mask = build_pref_mask_from_saved_indices(
        saved_indices=[human_indices, human_indices_2],
        saved_labels=human_labels,
        num_query=FLAGS.num_query,
        len_query=FLAGS.query_len,
        dataset_size=dataset.size,
    )
    out_pref_mask = ~in_pref_mask

    print(f"[PrefIndices] total dataset timesteps = {dataset.size}")
    print(f"[PrefIndices] in-pref timesteps      = {in_pref_mask.sum()}")
    print(f"[PrefIndices] out-of-pref timesteps  = {out_pref_mask.sum()}")

    return {
        "in_pref_mask": in_pref_mask,
        "out_pref_mask": out_pref_mask,
        "human_indices": human_indices,
        "human_indices_2": human_indices_2,
        "human_labels": human_labels,
    }

# ========= Viz helpers =========
def cosine_similarity(a, b):
    return jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1))

def visualize_reward_correlation_scatter(reward_dict, save_path=None, include_kendall=True):
    if "GT" not in reward_dict: raise ValueError("'GT' key must be present in reward_dict.")
    gt = np.asarray(reward_dict["GT"])
    other_keys = [k for k in reward_dict if k != "GT"]
    fig, axes = plt.subplots(1, len(other_keys), figsize=(5 * len(other_keys), 5), squeeze=False)
    pearson_results, mse_results, kendall_results = {}, {}, {}
    kl_results, wass_results = {}, {}
    for i, key in enumerate(other_keys):
        r = np.asarray(reward_dict[key])
        pearson_corr = np.corrcoef(gt, r)[0, 1]
        mse = np.mean((gt - r) ** 2)
        tau, _ = kendalltau(gt, r) if include_kendall else (None, None)
        hist_gt, bin_edges = np.histogram(gt, bins=100, density=True)
        hist_r, _ = np.histogram(r, bins=bin_edges, density=True)
        kl_div = entropy(hist_gt + 1e-10, hist_r + 1e-10)
        wass_dist = wasserstein_distance(gt, r)
        pearson_results[key] = pearson_corr; mse_results[key] = mse; kl_results[key] = kl_div; wass_results[key] = wass_dist
        if include_kendall: kendall_results[key] = tau
        ax = axes[0, i]
        ax.scatter(gt, r, alpha=0.6, edgecolors='k')
        min_v, max_v = min(gt.min(), r.min()), max(gt.max(), r.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--')
        title = f"{key} vs GT\nCorr: {pearson_corr:.2f}  MSE: {mse:.2f}"
        if include_kendall: title += f"  τ: {tau:.2f}"
        ax.set_title(title); ax.set_xlabel("GT"); ax.set_ylabel(key)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("\n--- Raw Pearson Correlation ---"); [print(f"{k}: {v:.4f}") for k, v in pearson_results.items()]
    print("\n--- Raw MSE ---"); [print(f"{k}: {v:.4f}") for k, v in mse_results.items()]
    if include_kendall:
        print("\n--- Kendall's Tau (Rank) ---"); [print(f"{k}: {v:.4f}") for k, v in kendall_results.items()]
    print("\n--- KL Divergence (GT || Pred) ---"); [print(f"{k}: {v:.4f}") for k, v in kl_results.items()]
    print("\n--- Wasserstein Distance ---"); [print(f"{k}: {v:.4f}") for k, v in wass_results.items()]

def visualize_reward_distributions(reward_dict, save_path=None):
    plt.figure(figsize=(8,6))
    for key, reward in reward_dict.items():
        plt.hist(reward, bins=50, alpha=0.5, label=f"{key} Distribution", histtype="stepfilled")
    plt.title("Reward Distributions"); plt.xlabel("Reward Value"); plt.ylabel("Frequency"); plt.legend()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_cosine_similarity(obs, recon, save_path=None):
    cosine_sim = cosine_similarity(obs, recon)
    cosine_sim = jax.device_get(cosine_sim)
    similarity_scores = ((cosine_sim + 1.0) / 2.0).flatten()
    plt.figure(figsize=(8,6))
    plt.hist(similarity_scores, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Cosine Similarity Distribution (Scaled to [0, 1])")
    plt.xlabel("Cosine Similarity"); plt.ylabel("Frequency")
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_obs_recon_tsne_with_embedding(obs, recon, embedding_model, embedding_params, save_path=None):
    embedded_obs = embedding_model.apply(embedding_params, obs)
    embedded_recon = embedding_model.apply(embedding_params, recon)
    combined = np.concatenate([embedded_obs, embedded_recon], axis=0)
    labels = np.array([0] * len(embedded_obs) + [1] * len(recon))
    perplexity = max(5, min(30, len(obs) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_result = tsne.fit_transform(combined)
    tsne_obs = tsne_result[labels == 0]; tsne_recon = tsne_result[labels == 1]
    plt.figure(figsize=(8,6))
    plt.scatter(tsne_obs[:,0], tsne_obs[:,1], s=5, label='Embedded Observations', alpha=0.5)
    plt.scatter(tsne_recon[:,0], tsne_recon[:,1], s=5, label='Reconstructed Subgoals', alpha=0.5)
    plt.title('t-SNE: Embedded Observations vs Reconstructed Subgoals'); plt.legend()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ========= NEW: group comparison plot (ID vs OOD) =========
def plot_extrapolation_error_vs_shaping_groups(
    shaping_term, gt_rewards, pt_rewards, save_dir, method_name="unknown",
    shaping_weight=None, use_abs_x=True, nbins=30, x_clip_pct=99.0, y_clip_pct=99.0,
    log_x=False, group_masks: Dict[str, np.ndarray] = None, seed=0
):
    """
    Plot |PT−GT| vs distance with multiple groups (e.g., In-Preference vs Out-of-Preference).
    Each group has binned mean ± 95% bootstrap CI curves on the same axes.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(shaping_term).reshape(-1)
    if use_abs_x: x = np.abs(x)
    if log_x: x = np.log1p(x)
    gt = np.asarray(gt_rewards).reshape(-1)
    pt = np.asarray(pt_rewards).reshape(-1)
    err = np.abs(pt - gt)

    def _clip(arr, pct):
        if pct is None: return arr
        hi = np.nanpercentile(arr, pct); return np.clip(arr, None, hi)

    x = _clip(x, x_clip_pct)
    err = _clip(err, y_clip_pct)

    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(x), np.nanmax(x), num=min(nbins + 1, 5))

    def _bootstrap_ci(vals, n_boot=300):
        vals = np.asarray(vals, float); n = vals.size
        if n <= 1: return float(np.mean(vals)), 0.0
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(np.mean(vals[idx]))
        boots = np.sort(np.asarray(boots))
        lo = np.percentile(boots, 2.5); hi = np.percentile(boots, 97.5)
        return float(np.mean(vals)), float(hi - lo) / 2.0

    centers = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        centers.append(0.5 * (lo + hi))
    centers = np.asarray(centers)

    colors = ["C0", "C1", "C2", "C3", "C4"]
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, err, gridsize=60, mincnt=5, alpha=0.25)

    csv_rows = [["group", "center_x", "mean_abs_err", "half_CI", "count", "spearman_rho", "spearman_p"]]

    if group_masks is None:
        group_masks = {"ALL": np.ones_like(x, dtype=bool)}

    for gi, (gname, gmask) in enumerate(group_masks.items()):
        gmask = np.asarray(gmask, dtype=bool)
        gx = x[gmask]; ge = err[gmask]
        rho, pval = spearmanr(gx, ge)
        g_means, g_cis = [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            m = (gx >= lo) & (gx < hi) if i < len(edges) - 2 else (gx >= lo) & (gx <= hi)
            if not np.any(m): g_means.append(np.nan); g_cis.append(0.0); continue
            mval, mci = _bootstrap_ci(ge[m], n_boot=300)
            g_means.append(mval); g_cis.append(mci)
            csv_rows.append([gname, centers[i], mval, mci, int(m.sum()), rho, pval])
        g_means = np.asarray(g_means); g_cis = np.asarray(g_cis)
        plt.errorbar(centers, g_means, yerr=g_cis, marker='o', linewidth=2, capsize=3,
                     label=f"{gname} (ρ={rho:.2f}, p={pval:.1e})", color=colors[gi % len(colors)])
        plt.fill_between(centers, g_means - g_cis, g_means + g_cis, alpha=0.15, color=colors[gi % len(colors)])

    abs_tag = "|x|" if use_abs_x else "x"; sw_tag = "" if shaping_weight is None else f", w={shaping_weight}"
    log_tag = ", log1p(x)" if log_x else ""
    plt.title(f"PT Extrapolation Error vs Distance ({method_name}{sw_tag}, {abs_tag}{log_tag})")
    plt.xlabel("Shaping distance" + (" (log1p)" if log_x else ""))
    plt.ylabel("|PT − GT|"); plt.grid(True); plt.legend()

    tag = ("abs" if use_abs_x else "signed") + ("_logx" if log_x else "")
    if shaping_weight is not None: tag += f"_w{str(shaping_weight).replace('.', '_')}"
    fname = f"pt_error_vs_distance_groups_{method_name}_{tag}"
    out_png = os.path.join(save_dir, fname + ".png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

    out_csv = os.path.join(save_dir, fname + ".csv")
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); [w.writerow(r) for r in csv_rows]
    print(f"[Group PT Error] Saved figure: {out_png}")
    print(f"[Group PT Error] Saved CSV   : {out_csv}")

# ========= PRD =========
def compute_prd(obs, recon, k=3):
    nbrs_obs = NearestNeighbors(n_neighbors=k+1).fit(obs)
    obs_dists, _ = nbrs_obs.kneighbors(obs)
    obs_radii = obs_dists[:, -1]

    nbrs_recon = NearestNeighbors(n_neighbors=k+1).fit(recon)
    recon_dists, _ = nbrs_recon.kneighbors(recon)
    recon_radii = recon_dists[:, -1]

    dists = np.linalg.norm(recon[:, None, :] - obs[None, :, :], axis=-1)
    in_obs = np.any(dists <= obs_radii[None, :], axis=1)
    precision = np.mean(in_obs)

    dists2 = np.linalg.norm(obs[:, None, :] - recon[None, :, :], axis=-1)
    in_recon = np.any(dists2 <= recon_radii[None, :], axis=1)
    recall = np.mean(in_recon)
    return precision, recall

# ========= Main =========
def main(_):
    # save dir
    cvae_path = "subgoal_vae_"+FLAGS.env_name+"_"+str(FLAGS.seed)+"_"+str(FLAGS.latent_dim)+"_"+str(FLAGS.hidden_dim)+"_"+str(FLAGS.topkp)+"_"+str(FLAGS.state_action)+".pkl"
    save_dir = os.path.join(
        FLAGS.save_dir, "tb", FLAGS.env_name,
        (f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}" if FLAGS.use_reward_model else "original"),
        f"{FLAGS.comment}", str(FLAGS.seed),
        f"{cvae_path}", f"{FLAGS.method}", str(FLAGS.shaping_weight), f"{FLAGS.state_action}",
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    summary_writer = SummaryWriter(save_dir, write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    # 1) env & dataset
    env, dataset, ground_truth, trj_mapper = make_env_and_dataset_d4rl(FLAGS.env_name, FLAGS.seed)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    size = dataset.size

    # 2) load CVAE
    print("obs_dim : ", obs_dim)
    cvae_state = load_vae_model(
        filename=cvae_path, observation_dim=obs_dim, action_dim=act_dim,
        vae_latent_dim=FLAGS.latent_dim, vae_hidden_dim=FLAGS.hidden_dim, state_action=FLAGS.state_action,
    )

    # 3) Build full arrays (NO sampling): use the entire dataset
    obs = jnp.array(dataset.observations)
    acts = jnp.array(dataset.actions)
    next_obs = jnp.array(dataset.next_observations)
    r_gt = np.asarray(ground_truth)
    r_pt = np.asarray(dataset.rewards)

    # 4) Load preference indices & masks (ID vs OOD)
    gym_env = gym.make(FLAGS.env_name)
    eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
    _ = get_d4rl_dataset(eval_sampler.env)
    env_key = resolve_env_key_for_indices(FLAGS.env_name, FLAGS.robosuite, FLAGS.robosuite_dataset_type)
    masks = load_pref_indices_and_mask(gym_env, dataset, env_key)
    in_mask = masks["in_pref_mask"]
    out_mask = masks["out_pref_mask"]

    # 5) Reconstruct subgoals for the entire dataset (batched)
    n_samples = obs.shape[0]
    batch_size = FLAGS.batch_size
    reconstructed_list = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_obs = obs[start_idx:end_idx]
        batch_acts = acts[start_idx:end_idx]
        recon = cvae_state.apply_fn(cvae_state.params, batch_obs, batch_acts, training=False)
        reconstructed_list.append(recon)
    reconstructed_subgoals = jnp.concatenate(reconstructed_list, axis=0)

    # 6) Compute shaping term
    discount = 0.99
    if FLAGS.state_action:
        recon_subgoals_state  = reconstructed_subgoals[..., :obs_dim]
        recon_subgoals_action = reconstructed_subgoals[..., obs_dim:]
        method = FLAGS.method
        if method == "negative_distance":
            state_dist = jnp.linalg.norm(obs - recon_subgoals_state, axis=-1)
            action_dist = jnp.linalg.norm(acts - recon_subgoals_action, axis=-1)
            shaping_term = -(state_dist + action_dist)
        elif method == "gaussian_kernel":
            sigma = 1.0
            state_dist = jnp.linalg.norm(obs - recon_subgoals_state, axis=-1)
            action_dist = jnp.linalg.norm(acts - recon_subgoals_action, axis=-1)
            shaping_term = jnp.exp(-state_dist**2/(2*sigma**2)) + jnp.exp(-action_dist**2/(2*sigma**2))
        elif method == "cosine_similarity":
            def cosine_sim(a, b): return jnp.sum(a*b, axis=-1) / (jnp.linalg.norm(a, axis=-1)*jnp.linalg.norm(b, axis=-1))
            state_cos = (1+cosine_sim(obs, recon_subgoals_state))/2
            action_cos = (1+cosine_sim(acts, recon_subgoals_action))/2
            shaping_term = (state_cos + action_cos)
        elif method == "normalized_distance":
            state_dist = jnp.linalg.norm(obs - recon_subgoals_state, axis=-1)
            action_dist = jnp.linalg.norm(acts - recon_subgoals_action, axis=-1)
            state_dist_norm = state_dist / (jnp.max(state_dist) + 1e-8)
            action_dist_norm = action_dist / (jnp.max(action_dist) + 1e-8)
            shaping_term = -(state_dist_norm + action_dist_norm)
        elif method == "potential_based":
            current_potential = jnp.linalg.norm(obs - recon_subgoals_state, axis=-1)
            next_potential = jnp.linalg.norm(next_obs - recon_subgoals_state, axis=-1)
            shaping_term = discount * next_potential - current_potential
        else:
            state_dist = jnp.linalg.norm(obs - recon_subgoals_state, axis=-1)
            action_dist = jnp.linalg.norm(acts - recon_subgoals_action, axis=-1)
            shaping_term = -(state_dist + action_dist)
    else:
        method = FLAGS.method
        if method == "negative_distance":
            distances = jnp.linalg.norm(next_obs - reconstructed_subgoals, axis=-1)
            shaping_term = -distances
        elif method == "gaussian_kernel":
            sigma = 1.0
            distances = jnp.linalg.norm(next_obs - reconstructed_subgoals, axis=-1)
            shaping_term = jnp.exp(-distances**2/(2*sigma**2))
        elif method == "cosine_similarity":
            def cosine_sim(a, b): return jnp.sum(a*b, axis=-1) / (jnp.linalg.norm(a, axis=-1)*jnp.linalg.norm(b, axis=-1))
            shaping_term = (1+cosine_sim(next_obs, reconstructed_subgoals))/2
        elif method == "normalized_distance":
            distances = jnp.linalg.norm(next_obs - reconstructed_subgoals, axis=-1)
            max_distance = jnp.max(distances)
            shaping_term = -(distances / (max_distance + 1e-8))
        elif method == "potential_based":
            current_potential = jnp.linalg.norm(obs - reconstructed_subgoals, axis=-1)
            next_potential = jnp.linalg.norm(next_obs - reconstructed_subgoals, axis=-1)
            shaping_term = discount * next_potential - current_potential
        else:
            distances = jnp.linalg.norm(next_obs - reconstructed_subgoals, axis=-1)
            shaping_term = -distances

    shaping_term = np.asarray(shaping_term)
    shaped_rewards = r_pt + FLAGS.shaping_weight * shaping_term

    # 7) Basic visualizations
    reward_dict = {"GT": r_gt, "PT": r_pt, "GUIDER": shaped_rewards}
    print("[GT] min/max:", np.min(r_gt), np.max(r_gt))
    print("[PT] min/max:", np.min(r_pt), np.max(r_pt))
    print("[GUIDER] min/max:", np.min(shaped_rewards), np.max(shaped_rewards))

    corr_save_path = os.path.join(save_dir, "reward_correlation.png")
    visualize_reward_correlation_scatter(reward_dict, save_path=corr_save_path)

    dist_save_path = os.path.join(save_dir, "reward_distributions.png")
    visualize_reward_distributions(reward_dict, save_path=dist_save_path)

    obs_np = np.array(obs)
    recon_np = np.array(reconstructed_subgoals[..., :obs_dim] if FLAGS.state_action else reconstructed_subgoals)
    tsne_save_path = os.path.join(save_dir, "cosine_similarity_obs_recon.png")
    visualize_cosine_similarity(obs_np, recon_np, save_path=tsne_save_path)

    if FLAGS.hidden_dim == 32:   hidden_dims = [32,64,32]
    elif FLAGS.hidden_dim == 64: hidden_dims = [64,128,64]
    elif FLAGS.hidden_dim == 128: hidden_dims = [128,256,128]
    elif FLAGS.hidden_dim == 750: hidden_dims = [750,750]
    else: raise ValueError(f"Invalid hidden_dim: {FLAGS.hidden_dim}")

    embedding_model = EmbeddingModel(latent_dim=FLAGS.latent_dim, hidden_dims=hidden_dims)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones_like(obs)
    embedding_params = embedding_model.init(rng, dummy_input)

    tsne_save_path = os.path.join(save_dir, "tsne_obs_recon_embedded.png")
    visualize_obs_recon_tsne_with_embedding(obs_np, recon_np, embedding_model, embedding_params, save_path=tsne_save_path)

    # 8) Extrapolation error vs distance with ID vs OOD groups
    group_masks = {
        "In-Preference": in_mask,
        "Out-of-Preference": out_mask,
    }
    plot_extrapolation_error_vs_shaping_groups(
        shaping_term=shaping_term, gt_rewards=r_gt, pt_rewards=r_pt, save_dir=save_dir,
        method_name=FLAGS.method, shaping_weight=FLAGS.shaping_weight,
        use_abs_x=True, nbins=30, group_masks=group_masks
    )
    plot_extrapolation_error_vs_shaping_groups(
        shaping_term=shaping_term, gt_rewards=r_gt, pt_rewards=r_pt, save_dir=save_dir,
        method_name=FLAGS.method, shaping_weight=FLAGS.shaping_weight,
        use_abs_x=False, nbins=30, group_masks=group_masks
    )

    # 9) PRD on embeddings (obs vs recon)
    k = 3
    precision, recall = compute_prd(obs_np, recon_np, k=k)
    print(f"[PRD] Precision (recon in obs manifold, k={k}): {precision:.4f}")
    print(f"[PRD] Recall    (obs in recon manifold, k={k}): {recall:.4f}")

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    app.run(main)
