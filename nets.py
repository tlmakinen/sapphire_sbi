import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import numpy as np

import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt

from tqdm import tqdm
import optax

import cloudpickle as pickle

from typing import Sequence, Any
Array = Any


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)



class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.swish(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


# fishnet code
class Fishnet(nn.Module):
    theta_fid: jnp.array
    n_hidden_score: list
    n_hidden_fisher: list
    n_inputs: int=1
    n_parameters: int=2
    is_iid: bool=True
    priorCinv: jnp.array = jnp.eye(2)
    priormu: jnp.array = jnp.zeros((2,))

    def setup(self):

        self.model_score = MLP(self.n_hidden_score + (self.n_parameters,))
        self.model_fisher = MLP(self.n_hidden_fisher + (int(self.n_parameters * (self.n_parameters + 1)) // 2,))

    def __call__(self, x, scale=False):
        
        score = self.model_score(x)
        fisher_cholesky = self.model_fisher(x)
        
        t = jnp.sum(score, axis=0) + jnp.einsum('ij,j->i', self.priorCinv, (self.theta_fid - self.priormu))
        
        F = self.construct_fisher_matrix_multiple(fisher_cholesky)
        F = jnp.sum(F, axis=0) + self.priorCinv
        
        mle = self.theta_fid + jnp.einsum('jk,k->j', jnp.linalg.inv(F), t)

        return mle, t, F
    

    def construct_fisher_matrix_multiple(self, outputs):
        Q = (tfp.math.fill_triangular(outputs))
        # vmap the jnp.diag function for the batch
        _diag = jax.vmap(jnp.diag)
        middle = _diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
        padding = jnp.zeros(Q.shape)

        L = Q - fill_diagonal(padding, middle)

        return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (0, 2, 1)))
    
    def construct_fisher_matrix_single(self, outputs):
        Q = tfp.math.fill_triangular(outputs)
        middle = jnp.diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
        padding = jnp.zeros(Q.shape)

        L = Q - fill_diagonal(padding, middle)

        return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))

    
    
# fishnet-deepset code
class FishnetDeepset(nn.Module):
    theta_fid: jnp.array
    n_hidden_score: list
    n_hidden_fisher: list
    n_hidden_globals: list
    n_inputs: int=1
    n_parameters: int=2
    is_iid: bool=True
    priorCinv: jnp.array = jnp.eye(2)
    priormu: jnp.array = jnp.zeros((2,))

    def setup(self):
        
        self.n_upper_tri = int(self.n_parameters * (self.n_parameters + 1)) // 2

        self.model_score = MLP(self.n_hidden_score)
        self.model_fisher = MLP(self.n_hidden_fisher)
        self.model_globals = MLP(self.n_hidden_globals + (int(self.n_parameters \
                                        + int(self.n_parameters * (self.n_parameters + 1)) // 2),)
                                            )


    def __call__(self, x, scaling=1.0):
        
        score = self.model_score(x)
        fisher_cholesky = self.model_fisher(x)

        t = jnp.mean(score, axis=0)
        fisher_cholesky = jnp.mean(fisher_cholesky, axis=0)

        outputs = self.model_globals(jnp.concatenate([t, fisher_cholesky], axis=-1))
        
        t = outputs[:self.n_parameters]
        fisher_cholesky = outputs[self.n_parameters:]

        F = self.construct_fisher_matrix_single(fisher_cholesky) + self.priorCinv
        mle = t
        score = t
        
        return mle, score, F*scaling
    

    def construct_fisher_matrix_multiple(self, outputs):
        Q = jnp.squeeze(tfp.math.fill_triangular(outputs))
        # vmap the jnp.diag function for the batch
        _diag = jax.vmap(jnp.diag)
        middle = _diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
        padding = jnp.zeros(Q.shape)

        L = Q - fill_diagonal(padding, middle)

        return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (0, 2, 1)))
    
    def construct_fisher_matrix_single(self, outputs):
        Q = tfp.math.fill_triangular(outputs)
        middle = jnp.diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
        padding = jnp.zeros(Q.shape)

        L = Q - fill_diagonal(padding, middle)

        return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))

    
class Deepset(nn.Module):
    theta_fid: jnp.array
    n_hidden_score: list
    n_hidden_fisher: list
    n_hidden_globals: list
    n_inputs: int=1
    n_parameters: int=2
    is_iid: bool=True
    priorCinv: jnp.array = jnp.eye(2)
    priormu: jnp.array = jnp.zeros((2,))

    def setup(self):
        
        self.n_upper_tri = int(self.n_parameters * (self.n_parameters + 1)) // 2

        self.model_score = MLP(self.n_hidden_score)
        self.model_fisher = MLP(self.n_hidden_fisher)
        self.model_globals = MLP(self.n_hidden_globals + (int(self.n_parameters),))

    def __call__(self, x, scaling=1.0):
        
        # two networks here just to match complexity
        score = self.model_score(x)
        score2 = self.model_fisher(x)

        t = jnp.mean(jnp.concatenate([score, score2], -1), axis=0)
        t = self.model_globals(t)

        return t

