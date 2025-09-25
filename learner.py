"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v
# from JaxPref.subgoal_vae_model import SubgoalVAE
# from JaxPref.subgoal_cvae_model import SubgoalCVAE
from flax.training import train_state
import pickle
from functools import partial


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


# @jax.jit
@partial(jax.jit, static_argnames=["method", "state_action"])
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float,
    method:str,
    shaping_weight: float,
    state_action: bool,
    vae_state,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(
        key, actor, target_critic, new_value, batch, temperature
    )

    new_critic, critic_info = update_q(critic, new_value, batch,discount,method,shaping_weight,state_action,vae_state)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return (
        rng,
        new_actor,
        new_critic,
        new_value,
        new_target_critic,
        {**critic_info, **value_info, **actor_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        cvae_path:str = None,
        method:str = "negative_distance",
        shaping_weight: float =1.0,
        vae_latent_dim:int = 16,
        vae_hidden_dim: int = 32,
        state_action: bool = True,
        opt_decay_schedule: str = "cosine",
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        def load_vae_model(filename, observation_dim, action_dim):
            with open(filename, "rb") as f:
                loaded_params = pickle.load(f)
            if self.state_action:
                # Initialize a fresh VAE model to get its parameter structure
                from JaxPref.subgoal_cvae_model_state_action import SubgoalCVAE
                vae = SubgoalCVAE(
                    self.vae_latent_dim, observation_dim, action_dim, self.vae_hidden_dim
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
                vae = SubgoalCVAE(
                    self.vae_latent_dim, observation_dim, action_dim, self.vae_hidden_dim
                )
                # Create a dummy input to initialize the model parameters
                dummy_state = jnp.ones((1, observation_dim))
                dummy_action = jnp.ones((1, action_dim))
                subgoal_state = jnp.ones((1, observation_dim))
                rng = jax.random.PRNGKey(0)
                params = vae.init(rng, dummy_state, dummy_action, subgoal_state)

            # Create a dummy optimizer
            tx = optax.adam(learning_rate=1e-4)

            # Create a dummy train state
            vae_state = train_state.TrainState.create(
                apply_fn=vae.apply, params=params, tx=tx
            )
            vae_state = vae_state.replace(params=loaded_params)
            return vae_state

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.method = method
        self.shaping_weight = shaping_weight
        self.state_action = state_action

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        observation_dim = observations.shape[-1]
        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng
        self.vae_latent_dim = vae_latent_dim
        if vae_hidden_dim == 32:
            self.vae_hidden_dim = [32, 64, 32]
        elif vae_hidden_dim == 64:
            self.vae_hidden_dim = [64, 128, 64]
        elif vae_hidden_dim == 128:
            self.vae_hidden_dim = [128, 256, 128]
        elif vae_hidden_dim == 750:
            self.vae_hidden_dim = [750,750]   
        self.vae_state = load_vae_model(
            cvae_path,
            observation_dim,
            action_dim,
        )

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = (
            _update_jit(
                self.rng,
                self.actor,
                self.critic,
                self.value,
                self.target_critic,
                batch,
                self.discount,
                self.tau,
                self.expectile,
                self.temperature,
                self.method,
                self.shaping_weight,
                self.state_action,
                self.vae_state,
            )
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info
