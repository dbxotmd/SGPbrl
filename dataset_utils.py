import collections
from typing import Optional

import jax
import d4rl
import gym
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm, trange

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations)), desc="split"):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
        )


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-5
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


class RelabeledDataset(Dataset):
    def __init__(
        self,
        observations,
        actions,
        rewards,
        terminals,
        next_observations,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        if clip_to_eps:
            lim = 1 - eps
            actions = np.clip(actions, -lim, lim)

        dones_float = np.zeros_like(rewards)
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(observations[i + 1] - next_observations[i]) > 1e-6
                or terminals[i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=1.0 - terminals,
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
            size=len(observations),
        )


class ReplayBuffer(Dataset):
    def __init__(
        self, observation_space: gym.spaces.Box, action_dim: int, capacity: int
    ):

        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset, num_samples: Optional[int]):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


def reward_from_preference(
    env_name: str,
    dataset: D4RLDataset,
    reward_model,
    batch_size: int = 256,
):
    data_size = dataset.rewards.shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset.rewards)
    for i in trange(interval):
        start_pt = i * batch_size
        end_pt = (i + 1) * batch_size

        input = dict(
            observations=dataset.observations[start_pt:end_pt],
            actions=dataset.actions[start_pt:end_pt],
            next_observations=dataset.next_observations[start_pt:end_pt],
        )

        jax_input = batch_to_jax(input)
        new_reward = reward_model.get_reward(jax_input)
        new_reward = np.asarray(list(new_reward))
        new_r[start_pt:end_pt] = new_reward

    dataset.rewards = new_r.copy()
    return dataset


def reward_from_preference_transformer(
    env_name: str,
    dataset: D4RLDataset,
    reward_model,
    seq_len: int,
    batch_size: int = 256,
    use_diff: bool = False,
    label_mode: str = "last",
    with_attn_weights: bool = True,
    percentile: float = 90.0,
):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )
    trajectories = []
    trj_mapper = []
    observation_dim = dataset.observations.shape[-1]
    action_dim = dataset.actions.shape[-1]

    for trj_idx, traj in tqdm(
        enumerate(trajs), total=len(trajs), desc="chunk trajectories"
    ):
        _obs, _act, _reward, _mask, _done, _next_obs = [], [], [], [], [], []
        for _o, _a, _r, _m, _d, _no in traj:
            _obs.append(_o)
            _act.append(_a)
            _reward.append(_r)
            _mask.append(_m)
            _done.append(_d)
            _next_obs.append(_no)

        traj_len = len(traj)
        _obs, _act = np.asarray(_obs), np.asarray(_act)
        trajectories.append((_obs, _act))

        for seg_idx in range(traj_len):
            trj_mapper.append((trj_idx, seg_idx))

    data_size = dataset.rewards.shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset.rewards)
    pts = []
    attn_weights = []
    for i in trange(interval, desc="relabel reward"):
        start_pt = i * batch_size
        end_pt = min((i + 1) * batch_size, data_size)

        _input_obs, _input_act, _input_timestep, _input_attn_mask, _input_pt = (
            [],
            [],
            [],
            [],
            [],
        )
        for pt in range(start_pt, end_pt):
            _trj_idx, _seg_idx = trj_mapper[pt]
            if _seg_idx < seq_len - 1:
                __input_obs = np.concatenate(
                    [
                        np.zeros((seq_len - 1 - _seg_idx, observation_dim)),
                        trajectories[_trj_idx][0][: _seg_idx + 1, :],
                    ],
                    axis=0,
                )
                __input_act = np.concatenate(
                    [
                        np.zeros((seq_len - 1 - _seg_idx, action_dim)),
                        trajectories[_trj_idx][1][: _seg_idx + 1, :],
                    ],
                    axis=0,
                )
                __input_timestep = np.concatenate(
                    [
                        np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32),
                        np.arange(1, _seg_idx + 2, dtype=np.int32),
                    ],
                    axis=0,
                )
                __input_attn_mask = np.concatenate(
                    [
                        np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32),
                        np.ones(_seg_idx + 1, dtype=np.float32),
                    ],
                    axis=0,
                )
                __input_pt = np.concatenate(
                    [
                        np.zeros(seq_len - 1 - _seg_idx),
                        np.arange(pt - _seg_idx, pt + 1),
                    ],
                    axis=0,
                )
            else:
                __input_obs = trajectories[_trj_idx][0][
                    _seg_idx - seq_len + 1 : _seg_idx + 1, :
                ]
                __input_act = trajectories[_trj_idx][1][
                    _seg_idx - seq_len + 1 : _seg_idx + 1, :
                ]
                __input_timestep = np.arange(1, seq_len + 1, dtype=np.int32)
                __input_attn_mask = np.ones((seq_len), dtype=np.float32)
                __input_pt = np.arange(pt - seq_len + 1, pt + 1)

            _input_obs.append(__input_obs)
            _input_act.append(__input_act)
            _input_timestep.append(__input_timestep)
            _input_attn_mask.append(__input_attn_mask)
            _input_pt.append(__input_pt)

        _input_obs = np.asarray(_input_obs)
        _input_act = np.asarray(_input_act)
        _input_timestep = np.asarray(_input_timestep)
        _input_attn_mask = np.asarray(_input_attn_mask)
        _input_pt = np.asarray(_input_pt)

        input = dict(
            observations=_input_obs,
            actions=_input_act,
            timestep=_input_timestep,
            attn_mask=_input_attn_mask,
        )

        jax_input = batch_to_jax(input)
        if with_attn_weights:
            new_reward, attn_weight = reward_model.get_reward(jax_input)
            attn_weights.append(np.array(attn_weight))
            pts.append(_input_pt)
        else:
            new_reward, _ = reward_model.get_reward(jax_input)
        new_reward = new_reward.reshape(end_pt - start_pt, seq_len) * _input_attn_mask

        if use_diff:
            prev_input = dict(
                observations=_input_obs[:, : seq_len - 1, :],
                actions=_input_act[:, : seq_len - 1, :],
                timestep=_input_timestep[:, : seq_len - 1],
                attn_mask=_input_attn_mask[:, : seq_len - 1],
            )
            jax_prev_input = batch_to_jax(prev_input)
            prev_reward, _ = reward_model.get_reward(jax_prev_input)
            prev_reward = (
                prev_reward.reshape(end_pt - start_pt, seq_len - 1)
                * prev_input["attn_mask"]
            )
            if label_mode == "mean":
                new_reward = jnp.sum(new_reward, axis=1).reshape(-1, 1)
                prev_reward = jnp.sum(prev_reward, axis=1).reshape(-1, 1)
            elif label_mode == "last":
                new_reward = new_reward[:, -1].reshape(-1, 1)
                prev_reward = prev_reward[:, -1].reshape(-1, 1)
            new_reward -= prev_reward
        else:
            if label_mode == "mean":
                new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(
                    _input_attn_mask, axis=1
                )
                new_reward = new_reward.reshape(-1, 1)
            elif label_mode == "last":
                new_reward = new_reward[:, -1].reshape(-1, 1)

        new_reward = np.asarray(list(new_reward))
        new_r[start_pt:end_pt, ...] = new_reward.squeeze(-1)

    dataset.rewards = new_r.copy()

    # # === Generate Guidance if attention is used ===
    # if with_attn_weights:
    #     guidance = np.zeros_like(dataset.observations)
    #     filled_mask = np.zeros((data_size,), dtype=bool)

    #     trj_start = 0
    #     for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="guidance/trajectory"):
    #         traj_len = len(traj)
    #         if traj_len == 0:
    #             trj_start += traj_len
    #             continue

    #         for pt_seq in np.array_split(np.arange(traj_len), max(1, traj_len // seq_len)):
    #             # 10% 개수 계산 (최소 1개)
    #             num_samples = max(1, int(len(pt_seq) * 0.1))

    #             # 10% 개수만큼 uniform random sampling
    #             sampled_indices = np.sort(
    #                 np.random.choice(pt_seq, size=num_samples, replace=False)
    #             )

    #             # 마지막 timestep(종료 state) 반드시 포함
    #             if pt_seq[-1] not in sampled_indices:
    #                 sampled_indices = np.append(sampled_indices, pt_seq[-1])

    #             prev = pt_seq[0]
    #             for idx in sampled_indices:
    #                 src_id = trj_start + int(idx)
    #                 # prev ~ idx까지 guidance를 src_id state로 채움
    #                 for t in pt_seq[pt_seq >= prev][pt_seq[pt_seq >= prev] <= idx]:
    #                     t_global = trj_start + int(t)
    #                     guidance[t_global] = dataset.observations[src_id]
    #                     filled_mask[t_global] = True
    #                 prev = idx + 1

    #         trj_start += traj_len

    #     dataset.guidance = guidance
    #     dataset.guidance_mask = filled_mask
        # 전체 데이터 길이만큼 guidance와 mask를 준비
    #     guidance = np.zeros_like(dataset.observations)
    #     filled_mask = np.zeros((data_size,), dtype=bool)

    #     # trajectory별로 attention weight를 모아서 처리
    #     trj_start = 0
    #     for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="guidance/trajectory"):
    #         traj_len = len(traj)
    #         # 해당 trajectory에 대한 attention weight 모으기
    #         trj_attn_weights = []
    #         trj_pts = []
    #         for batch_attn, batch_pts in zip(attn_weights, pts):
    #             for aw, pt in zip(batch_attn, batch_pts):
    #                 if trj_start <= pt[-1] < trj_start + traj_len:
    #                     trj_attn_weights.append(aw)
    #                     trj_pts.append(pt)
    #         if not trj_attn_weights:
    #             trj_start += traj_len
    #             continue
    #         trj_attn_weights = np.array(trj_attn_weights).reshape(-1, seq_len)
    #         trj_pts = np.array(trj_pts).reshape(-1, seq_len)

    #         # trajectory 내에서 top 10% timestep 구하기 (빈틈없이 구간별로 guidance 채우기)
    #         for aw_seq, pt_seq in zip(trj_attn_weights, trj_pts):
    #             threshold = np.percentile(aw_seq, percentile)
    #             top_indices = np.where(aw_seq >= threshold)[0]
    #             if len(top_indices) == 0:
    #                 continue
    #             top_indices = np.sort(top_indices)
    #             # 마지막까지 포함
    #             top_indices = np.append(top_indices, len(pt_seq) - 1)
    #             prev = 0
    #             for idx in top_indices:
    #                 src_id = int(pt_seq[idx])
    #                 # prev ~ idx까지 guidance를 src_id state로 채움
    #                 for t in pt_seq[prev:idx+1]:
    #                     t = int(t)
    #                     if trj_start <= t < trj_start + traj_len:
    #                         guidance[t] = dataset.observations[src_id]
    #                         filled_mask[t] = True
    #                 prev = idx + 1
    #         trj_start += traj_len

    #     dataset.guidance = guidance
    #     dataset.guidance_mask = filled_mask
    # print(dataset.guidance)
    return dataset
