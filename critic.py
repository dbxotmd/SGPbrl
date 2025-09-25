from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(
    critic: Model, value: Model, batch: Batch, expectile: float
) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def cosine_similarity(a, b):
    return jnp.sum(a * b, axis=-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
    )


def huber_distance(x, y, delta=1.0):
    """
    Calculate the Huber distance between two vectors
    Returns a value between 0 and 1, where 0 indicates high similarity
    """
    abs_diff = jnp.abs(x - y)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_dist = jnp.mean(0.5 * quadratic**2 + delta * linear, axis=-1)

    similarity = jnp.exp(-huber_dist)
    return similarity


def combined_similarity(x, y, alpha=0.5, delta=1.0):
    """
    Calculate combined similarity using both cosine similarity and Huber distance
    Returns a value between 0 and 1, where 1 indicates high similarity
    """
    cos_sim = cosine_similarity(x, y)
    cos_sim = (cos_sim + 1) / 2

    abs_diff = jnp.abs(x - y)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_dist = jnp.mean(0.5 * quadratic**2 + delta * linear, axis=-1)
    huber_sim = jnp.exp(
        -huber_dist
    )  
    combined_sim = alpha * cos_sim + (1 - alpha) * huber_sim
    return combined_sim


def update_q(
    critic: Model, target_value: Model, batch: Batch, discount: float,method:str,shaping_weight: float, state_action: bool, vae_state
) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    obs_dim = batch.observations.shape[-1]

    if state_action:
        reconstructed_subgoals = vae_state.apply_fn(
            vae_state.params, batch.observations, batch.actions, training=False
        )
        reconstructed_subgoals_state  = reconstructed_subgoals[..., :obs_dim]    # shape = (batch, seq_len, state_dim)
        reconstructed_subgoals_action = reconstructed_subgoals[..., obs_dim:]    # shape = (batch, seq_len, action_dim)
        if method == "negative_distance":
            # 방법 1: (state, action) 거리의 음수 (가까울수록 높은 보상)
            state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            shaping_term = -(state_dist + action_dist)

        elif method == "exponential_decay":
            # 방법 2: (state, action) 거리의 지수적 감소 (0~1 범위)
            state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            shaping_term = jnp.exp(-state_dist) + jnp.exp(-action_dist)

        elif method == "gaussian_kernel":
            # 방법 3: (state, action) 가우시안 커널 (sigma로 조절)
            sigma = 1.0  # 하이퍼파라미터
            state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            shaping_term = (
                jnp.exp(-state_dist**2 / (2 * sigma**2)) +
                jnp.exp(-action_dist**2 / (2 * sigma**2))
            )

        elif method == "cosine_similarity":
            # 방법 4: (state, action) 코사인 유사도 (-1~1 범위)
            # def cosine_sim(a, b):
            #     return jnp.sum(a * b, axis=-1) / (
            #         jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
            #     )
            def cosine_sim(a, b):
                return jnp.sum(a * b, axis=-1) / (
                    jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-8
                )
            state_cos = cosine_sim(batch.observations, reconstructed_subgoals_state)
            action_cos = cosine_sim(batch.actions, reconstructed_subgoals_action)

            state_cos_norm  = (state_cos  + 1.0) / 2.0
            action_cos_norm = (action_cos + 1.0) / 2.0

            # shaping term (평균을 쓰거나 합을 쓰는 건 선택)
            shaping_term = (state_cos_norm + action_cos_norm)

        elif method == "normalized_distance":
            # 방법 5: (state, action) 정규화된 거리 (0~1 범위로 스케일)
            state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            state_dist_norm =  (state_dist / (jnp.max(state_dist) + 1e-8)) - 1.0
            action_dist_norm = (action_dist / (jnp.max(action_dist) + 1e-8)) -1.0
            shaping_term = state_dist_norm + action_dist_norm

        elif method == "potential_based":
            # 방법 6: (state, action) Potential-based shaping
            # Φ(s, a) = -distance_to_subgoal(s, a)
            current_state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            current_action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            current_potential = current_state_dist + current_action_dist
            # (next potential은 없는 경우가 많으니, 필요시 추가)
            shaping_term = current_potential

        else:
            # 디폴트: 거리의 음수
            state_dist = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals_state, axis=-1
            )
            action_dist = jnp.linalg.norm(
                batch.actions - reconstructed_subgoals_action, axis=-1
            )
            shaping_term = -(state_dist + action_dist)

    else:
        reconstructed_subgoals = vae_state.apply_fn(
            vae_state.params, batch.observations, batch.actions, training=False
        )

        if method == "negative_distance":
            # 방법 1: 거리의 음수 (가까울수록 높은 보상)
            distances = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            shaping_term = -distances
            
        elif method == "exponential_decay":
            # 방법 2: 지수적 감소 (0~1 범위)
            distances = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            shaping_term = jnp.exp(-distances)
            
        elif method == "gaussian_kernel":
            # 방법 3: 가우시안 커널 (sigma로 조절)
            distances = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            sigma = 1.0  # 하이퍼파라미터
            shaping_term = jnp.exp(-distances**2 / (2 * sigma**2))
            
        elif method == "cosine_similarity":
            # 방법 4: 코사인 유사도 (-1~1 범위)
            def cosine_sim(a, b):
                return jnp.sum(a * b, axis=-1) / (
                    jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
                )
            shaping_term = (1 +cosine_sim(batch.next_observations, reconstructed_subgoals))/2
            
        elif method == "normalized_distance":
            # 방법 6: 정규화된 거리 (0~1 범위로 스케일)
            distances = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            max_distance = jnp.max(distances)
            shaping_term =  (distances / (max_distance + 1e-8)) - 1.0
            
        elif method == "potential_based":
            # 방법 7: Potential-based shaping (이론적으로 보장된 방법)
            # Φ(s) = -distance_to_subgoal(s)
            current_potential = jnp.linalg.norm(
                batch.observations - reconstructed_subgoals, axis=-1
            )
            next_potential = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            shaping_term = discount * next_potential - current_potential
        else:
            distances = jnp.linalg.norm(
                batch.next_observations - reconstructed_subgoals, axis=-1
            )
            shaping_term = -distances

    shaped_rewards = (
        batch.rewards + shaping_weight * shaping_term
    )

    target_q = shaped_rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(
            {"params": critic_params}, batch.observations, batch.actions
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
