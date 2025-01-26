import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Sequence

class Encoder(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        return mu, log_var

class Decoder(nn.Module):
    output_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, z):
        for hidden_dim in self.hidden_dims:
            z = nn.Dense(hidden_dim)(z)
            z = nn.relu(z)
        x = nn.Dense(self.output_dim)(z)
        return x

class SubgoalVAE(nn.Module):
    latent_dim: int
    state_dim: int
    action_dim: int
    hidden_dims: Sequence[int]

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.hidden_dims)
        self.decoder = Decoder(self.state_dim, self.hidden_dims[::-1])

    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_subgoal = self.decoder(z)
        return reconstructed_subgoal, mu, log_var

    def reparameterize(self, mu, log_var):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(jax.random.PRNGKey(0), shape=mu.shape)
        return mu + eps * std