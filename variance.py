# import datetime
# import os
# import pickle
# from typing import Tuple

# import gym
# import numpy as np
# from tqdm import tqdm
# from absl import app, flags
# from ml_collections import config_flags
# from tensorboardX import SummaryWriter

# import wrappers
# from dataset_utils import (
#     D4RLDataset,
#     reward_from_preference,
#     reward_from_preference_transformer,
#     split_into_trajectories,
#     RelabeledDataset,
# )
# from evaluation import evaluate
# from learner import Learner
# from JaxPref.subgoal_cvae_model import SubgoalCVAE

# import robosuite as suite
# from robosuite.wrappers import GymWrapper
# import robomimic.utils.env_utils as EnvUtils

# import jax
# import jax.numpy as jnp
# import optax
# from flax.training import train_state
# import flax
# from flax import linen as nn

# # 추가로 시각화를 위한 라이브러리
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from scipy.stats import kendalltau, ttest_rel, wilcoxon, mannwhitneyu, rankdata, entropy, wasserstein_distance
# import umap
# import pandas as pd
# import imageio

# # 내부 normalization, utils
# from JaxPref.utils import set_random_seed

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

# FLAGS = flags.FLAGS

# flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
# flags.DEFINE_string("save_dir", "./logs/", "Tensorboard logging dir.")
# flags.DEFINE_integer("seed", 42, "Random seed.")
# flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
# flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
# flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
# flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
# flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
# flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
# flags.DEFINE_boolean(
#     "use_reward_model", False, "Use reward model for relabeling reward."
# )
# flags.DEFINE_string("model_type", "MLP", "type of reward model.")
# flags.DEFINE_string("ckpt_dir", "./logs/pref_reward", "ckpt path for reward model.")
# flags.DEFINE_string("comment", "base", "comment for distinguishing experiments.")
# flags.DEFINE_integer(
#     "seq_len", 25, "sequence length for relabeling reward in Transformer."
# )
# flags.DEFINE_bool(
#     "use_diff",
#     False,
#     "boolean whether use difference in sequence for reward relabeling.",
# )
# flags.DEFINE_string("label_mode", "last", "mode for relabeling reward with tranformer.")

# config_flags.DEFINE_config_file(
#     "config",
#     "default.py",
#     "File path to the training hyperparameter configuration.",
#     lock_config=False,
# )

# # -----------------------------------------------------------------------------
# # 스텝 1: 데이터 로드 및 전처리 함수
# # -----------------------------------------------------------------------------

# def normalize(dataset, env_name, max_episode_steps=1000):
#     """d4rl dataset의 reward를 정규화 + antmaze 보정"""
#     trajs = split_into_trajectories(
#         dataset.observations,
#         dataset.actions,
#         dataset.rewards,
#         dataset.masks,
#         dataset.dones_float,
#         dataset.next_observations,
#     )
#     trj_mapper = [(i, len(t)) for i,t in enumerate(trajs) for _ in t]
#     returns = [sum(step[2] for step in t) for t in trajs]
#     min_r, max_r = min(returns), max(returns)
#     norm = []
#     for i, rew in enumerate(dataset.rewards):
#         if 'antmaze' in env_name:
#             _, L = trj_mapper[i]; rew -= min_r/L
#         rew = (rew - min_r)/(max_r - min_r) * max_episode_steps
#         norm.append(rew)
#     return np.array(norm), trj_mapper


# def normalize_gt(dataset, gt, env_name, max_episode_steps=1000):
#     trajs = split_into_trajectories(dataset.observations, dataset.actions, gt,
#                                     dataset.masks, dataset.dones_float,
#                                     dataset.next_observations)
#     trj_mapper = [(i,len(t)) for i,t in enumerate(trajs) for _ in t]
#     returns = [sum(step[2] for step in t) for t in trajs]
#     min_r, max_r = min(returns), max(returns)
#     norm = []
#     for i, rew in enumerate(gt):
#         if 'antmaze' in env_name:
#             _, L = trj_mapper[i]; rew -= min_r/L
#         rew = (rew - min_r)/(max_r - min_r) * max_episode_steps
#         norm.append(rew)
#     return np.array(norm)


# def load_vae_model(filename, observation_dim, action_dim):
#     with open(filename,'rb') as f: params = pickle.load(f)
#     vae = SubgoalCVAE(16, observation_dim, action_dim, [32,64,32])
#     rng = jax.random.PRNGKey(0)
#     dummy = jnp.ones((1,observation_dim)); da = jnp.ones((1,action_dim)); ds = jnp.ones((1,observation_dim))
#     init_params = vae.init(rng, dummy, da, ds)
#     tx = optax.adam(1e-4)
#     state = train_state.TrainState.create(apply_fn=vae.apply, params=init_params, tx=tx)
#     return state.replace(params=params)


# def initialize_model():
#     path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl" if os.path.exists(os.path.join(FLAGS.ckpt_dir,"best_model.pkl")) else "model.pkl")
#     ckpt = pickle.load(open(path,'rb'))
#     return ckpt['reward_model']

# # -----------------------------------------------------------------------------
# # 스텝 2: 동일 trajectory 샘플링 & 분산 계산 함수
# # -----------------------------------------------------------------------------

# def extract_samples_from_same_trajectory(dataset, ground_truth, trj_mapper,
#                                          traj_id=0, stride=1, window_len=1):
#     """하나의 trajectory에서 timestep별로 샘플 추출"""
#     trajs = split_into_trajectories(
#         dataset.observations, dataset.actions, dataset.rewards,
#         dataset.masks, dataset.dones_float, dataset.next_observations)
#     t = trajs[traj_id]
#     obs = np.stack([s[0] for s in t]); acts = np.stack([s[1] for s in t])
#     next_obs = np.stack([s[5] for s in t]); gt = np.array([s[2] for s in t])
#     start = sum(len(x) for x in trajs[:traj_id]); rel = dataset.rewards[start:start+len(t)]
#     return jnp.array(obs), jnp.array(acts), jnp.array(next_obs), gt, rel


# def cosine_similarity(a,b):
#     return jnp.sum(a*b,axis=-1)/(jnp.linalg.norm(a,axis=-1)*jnp.linalg.norm(b,axis=-1))


# def evaluate_reward_model_variance_on_single_trajectory(cvae_state, dataset, ground_truth, trj_mapper, FLAGS, save_dir, num_repeat=10, traj_id=0):
#     """
#     동일 trajectory에서 reward relabel 및 shaping을 반복하여
#     timestep별 분산을 계산, 값 출력 및 이미지 저장
#     """
#     # trajectory 샘플링
#     obs, acts, next_obs, _, rel = extract_samples_from_same_trajectory(
#         dataset, ground_truth, trj_mapper, traj_id)

#     pt_list, gd_list = [], []
#     for _ in range(num_repeat):
#         recon_batches = []
#         for i in range(0, len(obs), FLAGS.batch_size):
#             batch_obs = obs[i:i+FLAGS.batch_size]
#             batch_acts = acts[i:i+FLAGS.batch_size]
#             recon = cvae_state.apply_fn(
#                 cvae_state.params, batch_obs, batch_acts, training=False
#             )
#             recon_batches.append(recon)
#         recon = jnp.concatenate(recon_batches, axis=0)

#         sim = cosine_similarity(recon, next_obs)
#         shaping_term = (sim + 1) / 2
#         shaped = rel + shaping_term

#         pt_list.append(np.array(rel))
#         gd_list.append(np.array(shaped))

#     pt_arr = np.stack(pt_list)  # (repeats, T)
#     gd_arr = np.stack(gd_list)
#     pt_var = np.var(pt_arr, axis=0)
#     gd_var = np.var(gd_arr, axis=0)

#     # 값 출력
#     print("Timestep별 PT 분산:", pt_var)
#     print("Timestep별 GUIDER 분산:", gd_var)

#     # 이미지 저장
#     os.makedirs(save_dir, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(pt_var, marker='o', label='PT Variance')
#     ax.plot(gd_var, marker='x', label='GUIDER Variance')
#     ax.set_title(f"Variance Across Timesteps (traj={traj_id}, repeats={num_repeat})")
#     ax.set_xlabel("Timestep")
#     ax.set_ylabel("Variance")
#     ax.legend()
#     ax.grid(True)
#     fig.tight_layout()
#     img_path = os.path.join(save_dir, f"variance_traj{traj_id}.png")
#     fig.savefig(img_path, dpi=300)
#     print(f"분산 플롯 저장됨: {img_path}")
#     plt.close(fig)

#     # 값 저장 (CSV)
#     csv_path = os.path.join(save_dir, f"variance_values_traj{traj_id}.csv")
#     np.savetxt(
#         csv_path,
#         np.vstack([pt_var, gd_var]).T,
#         header="pt_variance,guider_variance",
#         delimiter=",",
#         comments=''
#     )
#     print(f"분산 값 CSV 저장됨: {csv_path}")

#     return pt_arr, gd_arr, pt_var, gd_var

# # -----------------------------------------------------------------------------
# # 스텝 3: main()
# # -----------------------------------------------------------------------------

# def make_env_and_dataset_d4rl(env_name: str, seed: int):
#     env = gym.make(env_name); env=wrappers.EpisodeMonitor(env); env=wrappers.SinglePrecision(env)
#     env.seed(seed); env.action_space.seed(seed); env.observation_space.seed(seed)
#     ds = D4RLDataset(env); gt=ds.rewards.copy()
#     if FLAGS.use_reward_model:
#         rm=initialize_model()
#         ds = reward_from_preference(ds) if FLAGS.model_type=='MR' else reward_from_preference_transformer(
#             FLAGS.env_name, ds, rm, batch_size=FLAGS.batch_size,
#             seq_len=FLAGS.seq_len, use_diff=FLAGS.use_diff, label_mode=FLAGS.label_mode)
#     norm, mapper = normalize(ds, FLAGS.env_name, env.env.env._max_episode_steps)
#     ds.rewards = norm
#     if 'antmaze' in FLAGS.env_name: ds.rewards-=1; gt-=1
#     elif any(x in FLAGS.env_name for x in ['halfcheetah','walker2d','hopper']):
#         ds.rewards+=0.5; gt=normalize_gt(ds,gt,FLAGS.env_name,env.env.env._max_episode_steps)
#     return env, ds, gt, mapper


# def main(_):
#     save_dir = os.path.join(
#         FLAGS.save_dir,
#         "tb",
#         FLAGS.env_name,
#         (
#             f"reward_{FLAGS.use_reward_model}_{FLAGS.model_type}"
#             if FLAGS.use_reward_model
#             else "original"
#         ),
#         f"{FLAGS.comment}",
#         str(FLAGS.seed),
#         f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
#     )
#     SummaryWriter(save_dir,write_to_disk=True)
#     os.makedirs(FLAGS.save_dir,exist_ok=True)
#     env, ds, gt, mapper = make_env_and_dataset_d4rl(FLAGS.env_name,FLAGS.seed)
#     cvae=load_vae_model("subgoal_vae_hopper-medium-expert-v2_0.pkl", env.observation_space.shape[-1], env.action_space.shape[-1])
#     evaluate_reward_model_variance_on_single_trajectory(cvae, ds, gt, mapper, FLAGS, num_repeat=10, traj_id=0)

# if __name__=='__main__': app.run(main)
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
from sklearn.decomposition import PCA
from scipy.stats import kendalltau, ttest_rel, wilcoxon, mannwhitneyu, rankdata, entropy, wasserstein_distance
import umap
import pandas as pd
import imageio

# 내부 normalization, utils
from JaxPref.utils import set_random_seed
from JaxPref.jax_utils import batch_to_jax


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

# -----------------------------------------------------------------------------
# 스텝 1: 데이터 로드 및 전처리 함수
# -----------------------------------------------------------------------------

def normalize(dataset, env_name, max_episode_steps=1000):
    """d4rl dataset의 reward를 정규화 + antmaze 보정"""
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )
    trj_mapper = [(i, len(t)) for i,t in enumerate(trajs) for _ in t]
    returns = [sum(step[2] for step in t) for t in trajs]
    min_r, max_r = min(returns), max(returns)
    norm = []
    for i, rew in enumerate(dataset.rewards):
        if 'antmaze' in env_name:
            _, L = trj_mapper[i]; rew -= min_r/L
        rew = (rew - min_r)/(max_r - min_r) * max_episode_steps
        norm.append(rew)
    return np.array(norm), trj_mapper


def normalize_gt(dataset, gt, env_name, max_episode_steps=1000):
    trajs = split_into_trajectories(dataset.observations, dataset.actions, gt,
                                    dataset.masks, dataset.dones_float,
                                    dataset.next_observations)
    trj_mapper = [(i,len(t)) for i,t in enumerate(trajs) for _ in t]
    returns = [sum(step[2] for step in t) for t in trajs]
    min_r, max_r = min(returns), max(returns)
    norm = []
    for i, rew in enumerate(gt):
        if 'antmaze' in env_name:
            _, L = trj_mapper[i]; rew -= min_r/L
        rew = (rew - min_r)/(max_r - min_r) * max_episode_steps
        norm.append(rew)
    return np.array(norm)


def load_vae_model(filename, observation_dim, action_dim):
    with open(filename,'rb') as f: params = pickle.load(f)
    vae = SubgoalCVAE(16, observation_dim, action_dim, [32,64,32])
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1,observation_dim)); da = jnp.ones((1,action_dim)); ds = jnp.ones((1,observation_dim))
    init_params = vae.init(rng, dummy, da, ds)
    tx = optax.adam(1e-4)
    state = train_state.TrainState.create(apply_fn=vae.apply, params=init_params, tx=tx)
    return state.replace(params=params)


def initialize_model():
    path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl" if os.path.exists(os.path.join(FLAGS.ckpt_dir,"best_model.pkl")) else "model.pkl")
    ckpt = pickle.load(open(path,'rb'))
    return ckpt['reward_model']

# -----------------------------------------------------------------------------
# 스텝 2: 동일 trajectory 샘플링 & 분산 계산 함수
# -----------------------------------------------------------------------------

def extract_samples_from_same_trajectory(dataset, ground_truth, trj_mapper,
                                         traj_id=0, stride=1, window_len=1):
    """하나의 trajectory에서 timestep별로 샘플 추출"""
    trajs = split_into_trajectories(
        dataset.observations, dataset.actions, dataset.rewards,
        dataset.masks, dataset.dones_float, dataset.next_observations)
    t = trajs[traj_id]
    obs = np.stack([s[0] for s in t]); acts = np.stack([s[1] for s in t])
    next_obs = np.stack([s[5] for s in t]); gt = np.array([s[2] for s in t])
    start = sum(len(x) for x in trajs[:traj_id]); rel = dataset.rewards[start:start+len(t)]
    return jnp.array(obs), jnp.array(acts), jnp.array(next_obs), gt, rel


def cosine_similarity(a,b):
    return jnp.sum(a*b,axis=-1)/(jnp.linalg.norm(a,axis=-1)*jnp.linalg.norm(b,axis=-1))


def evaluate_reward_model_variance_on_single_trajectory(cvae_state, reward_model, dataset, ground_truth, trj_mapper, FLAGS, save_dir, num_repeat=30, traj_id=0):
    """
    동일 trajectory에서 reward relabel 및 shaping을 반복하여
    timestep별 분산을 계산, 값 출력 및 이미지 저장
    """
    # trajectory 샘플링
    obs, acts, next_obs, _, _ = extract_samples_from_same_trajectory(
        dataset, ground_truth, trj_mapper, traj_id)

    pt_list, gd_list = [], []
    for rep in range(num_repeat):
        # Step 1: relabel reward via reward_model
        relabel_batches = []
        
        # sequence length만큼 sliding window로 처리
        for i in range(0, len(obs) - FLAGS.seq_len + 1, FLAGS.batch_size):
            end_idx = min(i + FLAGS.batch_size, len(obs) - FLAGS.seq_len + 1)
            batch_size = end_idx - i
            
            # 각 배치에서 연속된 seq_len개의 timestep을 sequence로 만들기
            obs_seq = []
            acts_seq = []
            for j in range(i, end_idx):
                # j부터 j+seq_len까지의 연속된 데이터
                obs_window = obs[j:j+FLAGS.seq_len]  # (seq_len, obs_dim)
                acts_window = acts[j:j+FLAGS.seq_len]  # (seq_len, act_dim)
                obs_seq.append(obs_window)
                acts_seq.append(acts_window)
            
            obs_seq = np.array(obs_seq)  # (batch_size, seq_len, obs_dim)
            acts_seq = np.array(acts_seq)  # (batch_size, seq_len, act_dim)
            
            # timestep을 (batch_size, seq_len) 형태로 생성
            timesteps = np.arange(1, FLAGS.seq_len + 1)
            batch_timesteps = np.tile(timesteps, (batch_size, 1))  # (batch_size, seq_len)
            
            batch = {
                'observations': obs_seq,
                'actions': acts_seq,
                'timestep': batch_timesteps,
                'attn_mask': np.ones((batch_size, FLAGS.seq_len), dtype=np.float32)
            }
            jax_input = batch_to_jax(batch)
            relabel, _ = reward_model.get_reward(jax_input)
            relabel = np.array(relabel).reshape(-1)
            relabel_batches.append(relabel)
        
        if relabel_batches:
            rel = np.concatenate(relabel_batches, axis=0)
        else:
            rel = np.array([])

        # Step 2: subgoal reconstruction and shaping
        recon_batches = []
        for i in range(0, len(obs), FLAGS.batch_size):
            end_idx = min(i + FLAGS.batch_size, len(obs))
            batch_obs = obs[i:end_idx]
            batch_acts = acts[i:end_idx]
            recon = cvae_state.apply_fn(
                cvae_state.params, batch_obs, batch_acts, training=False
            )
            recon_batches.append(recon)
        recon = jnp.concatenate(recon_batches, axis=0)

        # rel과 recon의 길이가 맞지 않을 수 있으므로 조정
        if len(rel) > len(recon):
            rel = rel[:len(recon)]
        elif len(rel) < len(recon):
            recon = recon[:len(rel)]

        sim = cosine_similarity(recon, next_obs[:len(recon)])
        shaping_term = (sim + 1) / 2
        shaped = rel + shaping_term

        pt_list.append(np.array(rel))
        gd_list.append(np.array(shaped))

    pt_arr = np.stack(pt_list)  # (repeats, T)
    gd_arr = np.stack(gd_list)
    pt_var = np.var(pt_arr, axis=0)
    gd_var = np.var(gd_arr, axis=0)

    # Trajectory-level statistics (전체 trajectory에 대한 통계)
    pt_traj_mean = np.mean(pt_arr)
    pt_traj_std = np.std(pt_arr)
    gd_traj_mean = np.mean(gd_arr)
    gd_traj_std = np.std(gd_arr)
    
    # Step-level statistics (각 timestep별 variance의 통계)
    pt_step_mean = np.mean(pt_var)
    pt_step_std = np.std(pt_var)
    gd_step_mean = np.mean(gd_var)
    gd_step_std = np.std(gd_var)

    # 값 출력
    print(f"Repeat {num_repeat} relabel/shaping on traj {traj_id}")
    print(f"Trajectory length: {len(obs)}, Sequence length: {FLAGS.seq_len}")
    print(f"Number of sequences: {len(obs) - FLAGS.seq_len + 1}")
    print("\n=== Trajectory-level Statistics ===")
    print(f"Relabel - Mean: {pt_traj_mean:.4f}, Std: {pt_traj_std:.4f}")
    print(f"GUIDER - Mean: {gd_traj_mean:.4f}, Std: {gd_traj_std:.4f}")
    print("\n=== Step-level Statistics ===")
    print(f"Relabel Variance - Mean: {pt_step_mean:.4f}, Std: {pt_step_std:.4f}")
    print(f"GUIDER Variance - Mean: {gd_step_mean:.4f}, Std: {gd_step_std:.4f}")
    print("\n=== Timestep별 Variance ===")
    print("Timestep별 Relabeled reward variance:", pt_var)
    print("Timestep별 GUIDER reward variance:", gd_var)

    # 이미지 저장
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Timestep별 variance plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pt_var, marker='o', label='Relabel Variance')
    ax.plot(gd_var, marker='x', label='GUIDER Variance')
    ax.set_title(f"Variance Across Timesteps (traj={traj_id}, repeats={num_repeat})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    img_path = os.path.join(save_dir, f"variance_traj{traj_id}.png")
    fig.savefig(img_path, dpi=300)
    print(f"분산 플롯 저장됨: {img_path}")
    plt.close(fig)

    # 2. Trajectory-level statistics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Relabel vs GUIDER comparison
    labels = ['Relabel', 'GUIDER']
    means = [pt_traj_mean, gd_traj_mean]
    stds = [pt_traj_std, gd_traj_std]
    
    ax1.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_title('Trajectory-level Mean ± Std')
    ax1.set_ylabel('Reward Value')
    ax1.grid(True, alpha=0.3)
    
    # Variance comparison
    var_means = [pt_step_mean, gd_step_mean]
    var_stds = [pt_step_std, gd_step_std]
    
    ax2.bar(labels, var_means, yerr=var_stds, capsize=5, alpha=0.7)
    ax2.set_title('Step-level Variance Mean ± Std')
    ax2.set_ylabel('Variance Value')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    stats_img_path = os.path.join(save_dir, f"statistics_traj{traj_id}.png")
    fig.savefig(stats_img_path, dpi=300)
    print(f"통계 플롯 저장됨: {stats_img_path}")
    plt.close(fig)

    # 값 저장 (CSV)
    csv_path = os.path.join(save_dir, f"variance_values_traj{traj_id}.csv")
    np.savetxt(
        csv_path,
        np.vstack([pt_var, gd_var]).T,
        header="relabel_variance,guider_variance",
        delimiter=",",
        comments=''
    )
    print(f"분산 값 CSV 저장됨: {csv_path}")

    # 통계 값 저장 (CSV)
    stats_csv_path = os.path.join(save_dir, f"statistics_traj{traj_id}.csv")
    stats_data = np.array([
        [pt_traj_mean, pt_traj_std, gd_traj_mean, gd_traj_std],
        [pt_step_mean, pt_step_std, gd_step_mean, gd_step_std]
    ])
    np.savetxt(
        stats_csv_path,
        stats_data,
        header="pt_traj_mean,pt_traj_std,gd_traj_mean,gd_traj_std",
        delimiter=",",
        comments=''
    )
    print(f"통계 값 CSV 저장됨: {stats_csv_path}")

    return pt_arr, gd_arr, pt_var, gd_var

# -----------------------------------------------------------------------------
# 스텝 3: main()
# -----------------------------------------------------------------------------

def make_env_and_dataset_d4rl(env_name: str, seed: int):
    env = gym.make(env_name); env=wrappers.EpisodeMonitor(env); env=wrappers.SinglePrecision(env)
    env.seed(seed); env.action_space.seed(seed); env.observation_space.seed(seed)
    ds = D4RLDataset(env); gt=ds.rewards.copy()
    rm = None
    if FLAGS.use_reward_model:
        rm=initialize_model()
        ds = reward_from_preference(ds) if FLAGS.model_type=='MR' else reward_from_preference_transformer(
            FLAGS.env_name, ds, rm, batch_size=FLAGS.batch_size,
            seq_len=FLAGS.seq_len, use_diff=FLAGS.use_diff, label_mode=FLAGS.label_mode)
    norm, mapper = normalize(ds, FLAGS.env_name, env.env.env._max_episode_steps)
    ds.rewards = norm
    if 'antmaze' in FLAGS.env_name: ds.rewards-=1; gt-=1
    elif any(x in FLAGS.env_name for x in ['halfcheetah','walker2d','hopper']):
        ds.rewards+=0.5; gt=normalize_gt(ds,gt,FLAGS.env_name,env.env.env._max_episode_steps)
    return env, ds, gt, mapper, rm


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
    SummaryWriter(save_dir,write_to_disk=True)
    os.makedirs(FLAGS.save_dir,exist_ok=True)
    env, ds, gt, mapper, rm = make_env_and_dataset_d4rl(FLAGS.env_name,FLAGS.seed)
    
    if rm is None:
        print("Warning: No reward model available. Please set use_reward_model=True")
        return
        
    cvae=load_vae_model("subgoal_vae_hopper-medium-expert-v2_0.pkl", env.observation_space.shape[-1], env.action_space.shape[-1])
    evaluate_reward_model_variance_on_single_trajectory(cvae, rm, ds, gt, mapper, FLAGS, save_dir, num_repeat=10, traj_id=0)

if __name__=='__main__': app.run(main)
