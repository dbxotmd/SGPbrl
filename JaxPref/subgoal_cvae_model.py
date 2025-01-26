import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Sequence


class PriorNetwork(nn.Module):
    """Prior network to predict latent distribution from state and action"""

    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)

        return mu, log_var


class ConditionalEncoder(nn.Module):
    """Encoder network for training"""

    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, state, action, subgoal):
        x = jnp.concatenate([subgoal, state, action], axis=-1)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)

        return mu, log_var


class Decoder(nn.Module):
    """Decoder network"""

    output_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, z, state, action):
        x = jnp.concatenate([z, state, action], axis=-1)

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        return nn.Dense(self.output_dim)(x)


class SubgoalCVAE(nn.Module):
    latent_dim: int
    state_dim: int
    action_dim: int
    hidden_dims: Sequence[int]

    def setup(self):
        self.prior = PriorNetwork(self.latent_dim, self.hidden_dims)
        self.encoder = ConditionalEncoder(self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.state_dim, self.hidden_dims[::-1])

    def __call__(self, state, action, subgoal_state=None, training=True):
        # Get prior distribution
        prior_mu, prior_log_var = self.prior(state, action)

        if training and subgoal_state is not None:
            # During training, use both prior and encoder
            post_mu, post_log_var = self.encoder(state, action, subgoal_state)
            z = self.reparameterize(post_mu, post_log_var)
            predicted_subgoal = self.decoder(z, state, action)
            return predicted_subgoal, post_mu, post_log_var, prior_mu, prior_log_var
        else:
            # During inference, use only prior
            z = self.reparameterize(prior_mu, prior_log_var)
            predicted_subgoal = self.decoder(z, state, action)
            return predicted_subgoal

    def reparameterize(self, mu, log_var):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(jax.random.PRNGKey(0), shape=mu.shape)
        return mu + eps * std
