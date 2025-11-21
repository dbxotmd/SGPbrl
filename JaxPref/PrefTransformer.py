from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState
import flax.training.checkpoints as checkpoints


from .jax_utils import (
    next_rng,
    value_and_multi_grad,
    mse_loss,
    cross_ent_loss,
    kld_loss,
)
import imageio
import os
import cv2
import gym
import matplotlib.pyplot as plt
import pickle
import random
import jax.random as jrandom
from .subgoal_vae_model import SubgoalVAE
from flax import serialization


class PrefTransformer(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = "adamw"
        config.scheduler_type = "CosineDecay"
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.train_type = "mean"

        # Weighted Sum option
        config.use_weighted_sum = True

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim
        self._total_steps = 0
        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "adamw": optax.adamw,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            "CosineDecay": optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr,
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(value=self.config.trans_lr),
                ],
                [self.config.warmup_steps],
            ),
            "none": None,
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)

        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32),
        )
        self._train_states["trans"] = TrainState.create(
            params=trans_params, tx=tx, apply_fn=None
        )

        model_keys = ["trans"]
        self._model_keys = tuple(model_keys)
        self._total_steps = 0
        self.accumulated_dataset = []
        self.samples_per_segment = 1
        self.save_interval = 1000  # 1000 스텝마다 데이터셋 저장
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        if config.hidden_dim == 32:
            self.hidden_dims = [32, 64, 32]
        elif config.hidden_dim == 64:
            self.hidden_dims = [64, 128, 64]
        elif config.hidden_dim == 128:
            self.hidden_dims = [128, 256, 128]
        elif config.hidden_dim == 750:
            self.hidden_dims = [750, 750]
        self.vae_learning_rate = 1e-4
        self.vae_batch_size = 256
        self.seed = config.seed
        self.topkp = config.topkp
        self.state_action = config.state_action

        # CVAE
        if self.state_action:
            from .subgoal_cvae_model_state_action import SubgoalCVAE

            def create_vae_train_state(model, learning_rate):
                params = model.init(
                    jax.random.PRNGKey(0),
                    jnp.ones((1, model.state_dim)),
                    jnp.ones((1, model.action_dim)),
                    # CVAE
                    jnp.ones((1, model.state_dim)),
                    jnp.ones((1, model.action_dim)),
                )
                tx = optax.adam(learning_rate)
                return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            self.vae = SubgoalCVAE(
                self.latent_dim, self.observation_dim, self.action_dim, self.hidden_dims
            )
            self.vae_state = create_vae_train_state(self.vae, self.vae_learning_rate)

        else:
            from .subgoal_cvae_model import SubgoalCVAE

            def create_vae_train_state(model, learning_rate):
                params = model.init(
                    jax.random.PRNGKey(0),
                    jnp.ones((1, model.state_dim)),
                    jnp.ones((1, model.action_dim)),
                    # CVAE
                    jnp.ones((1, model.state_dim)),
                )
                tx = optax.adam(learning_rate)
                return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

            self.vae = SubgoalCVAE(
                self.latent_dim, self.observation_dim, self.action_dim, self.hidden_dims
            )
            self.vae_state = create_vae_train_state(self.vae, self.vae_learning_rate)

    def cosine_similarity(self, a, b):
        return jnp.sum(a * b, axis=-1) / (
            jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
        )

    def huber_loss(self, recon, target, delta=1.0):
        abs_diff = jnp.abs(recon - target)
        quadratic = jnp.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return jnp.mean(0.5 * quadratic**2 + delta * linear)

    def combined_loss(self, recon, target, alpha=0.5):
        # 코사인 유사도와 Huber Loss 결합
        cos_loss = self.cosine_similarity(recon, target)
        hub_loss = self.huber_loss(recon, target)
        return alpha * cos_loss + (1 - alpha) * hub_loss

    @partial(jax.jit, static_argnames=["self"])
    def vae_train_step(self, state, batch):
        def loss_fn(params):
            # CVAE
            if self.state_action:
                recon_subgoal, post_mu, post_log_var, prior_mu, prior_log_var = (
                    state.apply_fn(
                        params,
                        batch["state"],
                        batch["action"],
                        batch["subgoal_state"],
                        batch["subgoal_action"],
                        training=True,
                    )
                )
                obs_dim = batch["state"].shape[-1]
                recon_subgoal_state = recon_subgoal[
                    ..., :obs_dim
                ]  # shape = (batch, seq_len, state_dim)
                recon_subgoal_action = recon_subgoal[
                    ..., obs_dim:
                ]  # shape = (batch, seq_len, action_dim)
                # CVAE Reconstruction Loss - MSE is more appropriate for subgoal generation
                state_mse = jnp.mean(
                    (recon_subgoal_state - batch["subgoal_state"]) ** 2
                )
                action_mse = jnp.mean(
                    (recon_subgoal_action - batch["subgoal_action"]) ** 2
                )
                recon_loss = state_mse + action_mse
            else:
                recon_subgoal, post_mu, post_log_var, prior_mu, prior_log_var = (
                    state.apply_fn(
                        params,
                        batch["state"],
                        batch["action"],
                        batch["subgoal_state"],
                        training=True,
                    )
                )

                # CVAE Reconstruction Loss - MSE is more appropriate for subgoal generation
                mse_loss = jnp.mean((recon_subgoal - batch["subgoal_state"]) ** 2)

            similarity = self.cosine_similarity(recon_subgoal, batch["subgoal_state"])
            similarities_loss = jnp.mean((similarity + 1) / 2)
            variance_penalty = jnp.var(similarity)
            cos_loss = similarities_loss + 0.1 * variance_penalty
            recon_loss = mse_loss - cos_loss

            # KL divergence loss
            # CVAE
            kl_loss = 0.5 * jnp.mean(
                jnp.exp(post_log_var - prior_log_var)
                + ((post_mu - prior_mu) ** 2) / jnp.exp(prior_log_var)
                - 1
                + prior_log_var
                - post_log_var
            )

            # Total loss - ELBO: reconstruction - KL divergence
            loss = recon_loss + kl_loss
            return loss, (recon_loss, kl_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (recon_loss, kl_loss)), grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss, recon_loss, kl_loss

    def get_vae_batch(self):
        # Sample a batch from accumulated_dataset for VAE training
        indices = np.random.choice(
            len(self.accumulated_dataset), self.vae_batch_size, replace=False
        )
        batch = [self.accumulated_dataset[i] for i in indices]
        if self.state_action:
            return {
                "state": jnp.array([item[0]["state"] for item in batch]),
                "action": jnp.array([item[0]["action"] for item in batch]),
                "subgoal_state": jnp.array(
                    [item[0]["subgoal_state"] for item in batch]
                ),
                "subgoal_action": jnp.array(
                    [item[0]["subgoal_action"] for item in batch]
                ),
            }
        else:
            return {
                "state": jnp.array([item[0]["state"] for item in batch]),
                "action": jnp.array([item[0]["action"] for item in batch]),
                "subgoal_state": jnp.array(
                    [item[0]["subgoal_state"] for item in batch]
                ),
            }

    def save_vae_model(self, env_name):
        filename = f"subgoal_vae_{env_name}_{self.seed}_{self.latent_dim}_{self.hidden_dim}_{self.topkp}_{self.state_action}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.vae_state.params, f)
        print(f"VAE params (PyTree) saved to {filename}")

    # B. 로드 (pickle.load)
    def load_vae_model(self, filename="vae_params.pkl"):
        with open(filename, "rb") as f:
            loaded_params = pickle.load(f)
        self.vae_state = self.vae_state.replace(params=loaded_params)
        print(f"VAE params loaded from {filename}")
        return self.vae_state

    def evaluation(self, batch, env_name):
        metrics, learned_rewards, importance_weights = self._eval_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=("self"))
    def _get_reward_step(self, train_states, batch):
        obs = batch["observations"]
        act = batch["actions"]
        timestep = batch["timestep"]
        attn_mask = batch["attn_mask"]

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights, _ = self.trans.apply(
            train_params["trans"],
            obs,
            act,
            timestep,
            attn_mask=attn_mask,
            reverse=False,
        )
        return trans_pred["value"], attn_weights[-1]

    @partial(jax.jit, static_argnames=("self"))
    def _eval_pref_step(self, train_states, rng, batch):

        def loss_fn(flag, train_params, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep_1"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape
            B, T, _ = batch["actions"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, attn_weights_list_1, reverse = self.trans.apply(
                train_params["trans"],
                obs_1,
                act_1,
                timestep_1,
                training=False,
                attn_mask=None,
                rngs={"dropout": rng},
            )
            trans_pred_2, attn_weights_list_2, reverse = self.trans.apply(
                train_params["trans"],
                obs_2,
                act_2,
                timestep_2,
                training=False,
                attn_mask=None,
                rngs={"dropout": rng},
            )
            if flag == False:
                attention_1 = attn_weights_list_1[-1].primal
                attention_2 = attn_weights_list_2[-1].primal
            else:
                attention_1 = attn_weights_list_1[-1]
                attention_2 = attn_weights_list_2[-1]

            if self.config.use_weighted_sum:
                attention_1_mean = jnp.mean(attention_1, axis=1)
                attention_2_mean = jnp.mean(attention_2, axis=1)
                attention_1 = attention_1_mean
                attention_2 = attention_2_mean
            else:
                # attention_1의 shape: [B, num_heads, T, T] -> [B, T, T]로 평균
                attention_1_mean = jnp.mean(attention_1, axis=1)
                attention_2_mean = jnp.mean(attention_2, axis=1)

                # [B, T, T] -> [B, 2, T//2, 2, T//2]로 reshape (T가 짝수여야 함)
                attention_1_squeezed = attention_1_mean.reshape(B, 2, T // 2, 2, T // 2)
                attention_2_squeezed = attention_2_mean.reshape(B, 2, T // 2, 2, T // 2)

                if reverse:
                    attention_1 = attention_1_squeezed[:, 0, :, 0, :]  # action->action
                    attention_2 = attention_2_squeezed[:, 0, :, 0, :]
                else:
                    attention_1 = attention_1_squeezed[:, 1, :, 1, :]  # state->state
                    attention_2 = attention_2_squeezed[:, 1, :, 1, :]

                attention_1 = attention_1.reshape(B, T // 2, T // 2)
                attention_2 = attention_2.reshape(B, T // 2, T // 2)

            importance_weight_1 = jnp.diagonal(attention_1, axis1=1, axis2=2)
            importance_weight_2 = jnp.diagonal(attention_2, axis1=1, axis2=2)

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            learned_rewards = jnp.concatenate([trans_pred_1, trans_pred_2], axis=1)
            importance_weight = jnp.concatenate(
                [importance_weight_1, importance_weight_2], axis=1
            )
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss
            loss_collection["trans"] = trans_loss
            return (
                tuple(loss_collection[key] for key in self.model_keys),
                locals(),
                learned_rewards,
                importance_weight,
                _,
                _,
            )

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _, learned_rewards, importance_weight, _, _ = (
            value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(
                train_params, rng
            )
        )
        metrics = dict(
            eval_cse_loss=aux_values["cse_loss"],
            eval_trans_loss=aux_values["trans_loss"],
        )
        return metrics, learned_rewards, importance_weight

    @partial(jax.jit, static_argnames=["self", "threshold_percentile", "topkp"])
    def extract_and_accumulate_subgoals(
        self,
        batch,
        rng,
        importance_weight_1,
        importance_weight_2,
        trans_pred_1,
        trans_pred_2,
        topkp,
        threshold_percentile=90,
    ):
        obs_1, act_1, obs_2, act_2, labels = (
            batch["observations"],
            batch["actions"],
            batch["observations_2"],
            batch["actions_2"],
            batch["labels"],
        )
        B, T, _ = obs_1.shape
        if T % 2 != 0:
            T += 1

        def process_single_trajectory(b, rng):
            label_0, label_1 = labels[b]

            def select_trajectory(label_0, label_1):
                cond_1 = jnp.allclose(
                    jnp.array([label_0, label_1]), jnp.array([1.0, 0.0])
                )
                cond_2 = jnp.allclose(
                    jnp.array([label_0, label_1]), jnp.array([0.0, 1.0])
                )

                return jax.lax.cond(
                    cond_1,
                    lambda _: (
                        obs_1[b],
                        act_1[b],
                        importance_weight_1[0][b],
                        trans_pred_1[0][b],
                    ),
                    lambda _: jax.lax.cond(
                        cond_2,
                        lambda _: (
                            obs_2[b],
                            act_2[b],
                            importance_weight_2[0][b],
                            trans_pred_2[0][b],
                        ),
                        lambda _: (
                            jnp.full_like(obs_1[b], jnp.nan),
                            jnp.full_like(act_1[b], jnp.nan),
                            jnp.full_like(importance_weight_1[0][b], jnp.nan),
                            jnp.full_like(trans_pred_1[0][b], jnp.nan),
                        ),
                        None,
                    ),
                    None,
                )

            obs, act, importance_weight, trans_pred = select_trajectory(
                label_0, label_1
            )
            # Squeeze trans_pred to match importance_weight shape
            trans_pred = jnp.squeeze(trans_pred, axis=-1)

            is_valid = jnp.logical_not(jnp.any(jnp.isnan(obs)))

            def process_valid_trajectory():
                # From the top 10% result
                if topkp == 10:
                    threshold = jnp.sort(importance_weight)[
                        int(len(importance_weight) * threshold_percentile / 100)
                    ]
                    trans_mean = jnp.mean(trans_pred)  # reward 평균
                    sizing = len(importance_weight) - int(
                        len(importance_weight) * threshold_percentile / 100
                    )
                    subgoal_indices = jnp.sort(
                        jnp.concatenate(
                            [
                                jnp.where(
                                    (importance_weight >= threshold)
                                    & (trans_pred >= trans_mean),
                                    size=sizing,
                                )[0],
                                jnp.array([T - 1]),
                            ]
                        )
                    )
                elif topkp == 20:
                    threshold_upper = jnp.sort(importance_weight)[
                        int(len(importance_weight) * (threshold_percentile - 10) / 100)
                    ]
                    # Top 10% threshold
                    threshold_lower = jnp.sort(importance_weight)[
                        int(len(importance_weight) * threshold_percentile / 100)
                    ]
                    # 10~20% 구간에 해당하는 인덱스 추출
                    subgoal_indices = jnp.sort(
                        jnp.concatenate(
                            [
                                jnp.where(
                                    (importance_weight >= threshold_upper)
                                    & (importance_weight < threshold_lower),
                                    size=int(len(importance_weight) * 0.1),
                                )[0],
                                jnp.array([T - 1]),
                            ]
                        )
                    )
                elif topkp == (-10):
                    threshold = jnp.sort(importance_weight)[
                        int(len(importance_weight) * 10 / 100)
                    ]
                    sizing = int(len(importance_weight) * 0.1)  # 10%
                    subgoal_indices = jnp.sort(
                        jnp.concatenate(
                            [
                                jnp.where(importance_weight <= threshold, size=sizing)[
                                    0
                                ],
                                jnp.array([T - 1]),
                            ]
                        )
                    )
                elif topkp == (-20):
                    threshold_upper = jnp.sort(importance_weight)[
                        int(len(importance_weight) * 20 / 100)
                    ]
                    threshold_lower = jnp.sort(importance_weight)[
                        int(len(importance_weight) * 10 / 100)
                    ]

                    subgoal_indices = jnp.sort(
                        jnp.concatenate(
                            [
                                jnp.where(
                                    (importance_weight <= threshold_upper)
                                    & (importance_weight > threshold_lower),
                                    size=int(len(importance_weight) * 0.1),
                                )[0],
                                jnp.array([T - 1]),
                            ]
                        )
                    )

                rngs = jrandom.split(rng, len(subgoal_indices) - 1)

                def subgoal(i, rng):
                    start, end = subgoal_indices[i], subgoal_indices[i + 1]
                    sample_index = jrandom.choice(
                        rng,
                        jnp.arange(T),
                        p=jnp.where(
                            (jnp.arange(T) >= start) & (jnp.arange(T) < end), 1.0, 0.0
                        ),
                    )
                    if self.state_action:
                        return {
                            "state": obs[sample_index],
                            "action": act[sample_index],
                            "subgoal_state": obs[end],
                            "subgoal_action": act[end],
                            "is_valid": jnp.array(True),
                        }
                    else:
                        return {
                            "state": obs[sample_index],
                            "action": act[sample_index],
                            "subgoal_state": obs[end],
                            # "subgoal_state": act[end],
                            "is_valid": jnp.array(True),
                        }

                return jax.vmap(subgoal)(jnp.arange(len(subgoal_indices) - 1), rngs)

            def process_invalid_trajectory():
                sizing = len(importance_weight) - int(
                    len(importance_weight) * threshold_percentile / 100
                )
                dummy_state = jnp.zeros_like(
                    obs[:sizing]
                )  # Same shape as the valid case
                dummy_action = jnp.zeros_like(act[:sizing])
                dummy_subgoal_state = jnp.zeros_like(obs[:sizing])
                dummy_is_valid = jnp.zeros((sizing,), dtype=bool)

                if self.state_action:
                    dummy_subgoal_action = jnp.zeros_like(act[:sizing])
                    return {
                        "state": dummy_state,
                        "action": dummy_action,
                        "subgoal_state": dummy_subgoal_state,
                        "subgoal_action": dummy_subgoal_action,
                        "is_valid": dummy_is_valid,
                    }
                else:
                    return {
                        "state": dummy_state,
                        "action": dummy_action,
                        "subgoal_state": dummy_subgoal_state,
                        "is_valid": dummy_is_valid,
                    }

            return (
                jax.lax.cond(
                    is_valid, process_valid_trajectory, process_invalid_trajectory
                ),
                is_valid,
            )

        rngs = jrandom.split(rng, B)
        all_subgoals, is_valid = jax.vmap(process_single_trajectory)(
            jnp.arange(B), rngs
        )

        return all_subgoals, is_valid

    def save_accumulated_dataset(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self.accumulated_dataset, f)
        print(
            f"Accumulated dataset with {len(self.accumulated_dataset)} samples saved to {self.file_path}"
        )

    def train(self, batch, env_name):
        self._total_steps += 1
        (
            self._train_states,
            metrics,
            importance_weight_1,
            importance_weight_2,
            trans_pred_1,
            trans_pred_2,
        ) = self._train_pref_step(self._train_states, next_rng(), batch)

        all_subgoals, is_valid = self.extract_and_accumulate_subgoals(
            batch,
            next_rng(),
            importance_weight_1,
            importance_weight_2,
            trans_pred_1,
            trans_pred_2,
            self.topkp,
        )

        def filter_valid_batches(subgoals, valid_mask):
            def concatenate_valid(x):
                return x[valid_mask]

            filtered_subgoals = jax.tree_map(concatenate_valid, subgoals)
            num_valid = jnp.sum(valid_mask)

            return filtered_subgoals, num_valid

        all_subgoals, num_valid = filter_valid_batches(all_subgoals, is_valid)

        self.vae_state, vae_loss, recon_loss, kl_loss = self.vae_train_step(
            self.vae_state, all_subgoals
        )
        metrics["vae_loss"] = vae_loss
        metrics["vae_recon_loss"] = recon_loss
        metrics["vae_kl_loss"] = kl_loss

        if self._total_steps % self.save_interval == 0:
            self.save_vae_model(env_name)

        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn(flag, train_params, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep_1"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape
            B, T, _ = batch["actions"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, attn_weights_list_1, reverse = self.trans.apply(
                train_params["trans"],
                obs_1,
                act_1,
                timestep_1,
                training=True,
                attn_mask=None,
                rngs={"dropout": rng},
            )
            trans_pred_2, attn_weights_list_2, reverse = self.trans.apply(
                train_params["trans"],
                obs_2,
                act_2,
                timestep_2,
                training=True,
                attn_mask=None,
                rngs={"dropout": rng},
            )

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if flag == False:
                attention_1 = attn_weights_list_1[-1].primal
                attention_2 = attn_weights_list_2[-1].primal
            else:
                attention_1 = attn_weights_list_1[-1]
                attention_2 = attn_weights_list_2[-1]

            if self.config.use_weighted_sum:
                attention_1_mean = jnp.mean(attention_1, axis=1)
                attention_2_mean = jnp.mean(attention_2, axis=1)
                attention_1 = attention_1_mean
                attention_2 = attention_2_mean
            else:
                # attention_1의 shape: [B, num_heads, T, T] -> [B, T, T]로 평균
                attention_1_mean = jnp.mean(attention_1, axis=1)
                attention_2_mean = jnp.mean(attention_2, axis=1)

                # [B, T, T] -> [B, 2, T//2, 2, T//2]로 reshape (T가 짝수여야 함)
                attention_1_squeezed = attention_1_mean.reshape(B, 2, T // 2, 2, T // 2)
                attention_2_squeezed = attention_2_mean.reshape(B, 2, T // 2, 2, T // 2)

                if reverse:
                    attention_1 = attention_1_squeezed[:, 0, :, 0, :]  # action->action
                    attention_2 = attention_2_squeezed[:, 0, :, 0, :]
                else:
                    attention_1 = attention_1_squeezed[:, 1, :, 1, :]  # state->state
                    attention_2 = attention_2_squeezed[:, 1, :, 1, :]

                attention_1 = attention_1.reshape(B, T // 2, T // 2)
                attention_2 = attention_2.reshape(B, T // 2, T // 2)

            # Attention weight 계산
            importance_weight_1 = jnp.diagonal(attention_1, axis1=1, axis2=2)
            importance_weight_2 = jnp.diagonal(attention_2, axis1=1, axis2=2)

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)

            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss

            loss_collection["trans"] = trans_loss

            return (
                tuple(loss_collection[key] for key in self.model_keys),
                locals(),
                importance_weight_1,
                importance_weight_2,
                trans_pred_1,
                trans_pred_2,
            )

        train_params = {key: train_states[key].params for key in self.model_keys}
        (
            (_, aux_values),
            grads,
            importance_weight_1,
            importance_weight_2,
            trans_pred_1,
            trans_pred_2,
        ) = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(
            train_params, rng
        )

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values["cse_loss"],
            trans_loss=aux_values["trans_loss"],
        )

        return (
            new_train_states,
            metrics,
            importance_weight_1,
            importance_weight_2,
            trans_pred_1,
            trans_pred_2,
        )

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_semi_pref_step(
        self, train_states, labeled_batch, unlabeled_batch, lmd, tau, rng
    ):
        def compute_logits(train_params, batch, rng):
            obs_1 = batch["observations"]
            act_1 = batch["actions"]
            obs_2 = batch["observations_2"]
            act_2 = batch["actions_2"]
            timestep_1 = batch["timestep_1"]
            timestep_2 = batch["timestep_2"]
            labels = batch["labels"]

            B, T, _ = batch["observations"].shape
            B, T, _ = batch["actions"].shape

            rng, _ = jax.random.split(rng)

            trans_pred_1, _ = self.trans.apply(
                train_params["trans"],
                obs_1,
                act_1,
                timestep_1,
                training=True,
                attn_mask=None,
                rngs={"dropout": rng},
            )
            trans_pred_2, _ = self.trans.apply(
                train_params["trans"],
                obs_2,
                act_2,
                timestep_2,
                training=True,
                attn_mask=None,
                rngs={"dropout": rng},
            )

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)

            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            rng, _ = jax.random.split(rng)
            logits, labels = compute_logits(train_params, labeled_batch, rng)
            u_logits, _ = compute_logits(train_params, unlabeled_batch, rng)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)

            loss_ = optax.softmax_cross_entropy(
                logits=u_logits,
                labels=jax.nn.one_hot(pseudo_label_target, num_classes=2),
            )
            u_trans_loss = jnp.sum(jnp.where(u_confidence > tau, loss_, 0)) / (
                jnp.count_nonzero(u_confidence > tau) + 1e-4
            )
            u_trans_ratio = (
                jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100
            )

            # labeling neutral cases.
            binarized_idx = jnp.where(unlabeled_batch["labels"][:, 0] != 0.5, 1.0, 0.0)
            real_label = jnp.argmax(unlabeled_batch["labels"], axis=-1)
            u_trans_acc = (
                jnp.sum(
                    jnp.where(pseudo_label_target == real_label, 1.0, 0.0)
                    * binarized_idx
                )
                / jnp.sum(binarized_idx)
                * 100
            )

            loss_collection["trans"] = last_loss = trans_loss + lmd * u_trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, lmd, tau, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values["trans_loss"],
            u_trans_loss=aux_values["u_trans_loss"],
            last_loss=aux_values["last_loss"],
            u_trans_ratio=aux_values["u_trans_ratio"],
            u_train_acc=aux_values["u_trans_acc"],
        )

        return new_train_states, metrics

    def train_regression(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_regression_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=("self"))
    def _train_regression_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch["observations"]
            next_observations = batch["next_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]

            in_obs = jnp.concatenate([observations, next_observations], axis=-1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            rf_pred = self.rf.apply(train_params["rf"], observations, actions)
            reward_target = jax.lax.stop_gradient(rewards)
            rf_loss = mse_loss(rf_pred, reward_target)

            loss_collection["rf"] = rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            rf_loss=aux_values["rf_loss"],
            average_rf=aux_values["rf_pred"].mean(),
        )

        return new_train_states, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
