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

    # Normalize the Huber distance to [0, 1] range
    # We use exp(-huber_dist) to convert distance to similarity
    similarity = jnp.exp(-huber_dist)
    return similarity


def combined_similarity(x, y, alpha=0.5, delta=1.0):
    """
    Calculate combined similarity using both cosine similarity and Huber distance
    Returns a value between 0 and 1, where 1 indicates high similarity
    """
    # 코사인 유사도 계산 (1에 가까울수록 유사)
    cos_sim = cosine_similarity(x, y)
    # [-1, 1] 범위를 [0, 1] 범위로 변환
    cos_sim = (cos_sim + 1) / 2

    # Huber 거리 기반 유사도 계산
    abs_diff = jnp.abs(x - y)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_dist = jnp.mean(0.5 * quadratic**2 + delta * linear, axis=-1)
    huber_sim = jnp.exp(
        -huber_dist
    )  # 거리를 유사도로 변환 (0에 가까운 거리가 1에 가까운 유사도가 됨)

    # 두 유사도 측정값을 결합
    combined_sim = alpha * cos_sim + (1 - alpha) * huber_sim
    return combined_sim


def update_q(
    critic: Model, target_value: Model, batch: Batch, discount: float, vae_state
) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    reconstructed_subgoals = vae_state.apply_fn(
        vae_state.params, batch.observations, batch.actions, training=False
    )

    # # # # Calculate the cosine similarity
    similarity = cosine_similarity(reconstructed_subgoals, batch.next_observations)

    # # Convert similarity to a shaping term (1 for most similar, 0 for least similar)
    # change this one to -1~1
    shaping_term = jnp.mean((similarity + 1) / 2)
    # cos_term = jnp.mean((similarity + 1) / 2)
    # shaping_term = ((similarity + 1) / 2)

    # MSE
    # shaping_term = jnp.mean((reconstructed_subgoals - batch.next_observations) ** 2)
    # mse_term = jnp.mean((reconstructed_subgoals - batch.next_observations) ** 2)

    # shaping_term = 0.5 * cos_term + 0.5 * mse_term

    # Add the shaping term to the original reward
    shaped_rewards = (
        batch.rewards
        + 1 * shaping_term
        # d batch.rewards
    )  # You can adjust the coefficient (0.1) as needed

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
