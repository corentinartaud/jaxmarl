"""End-to-End Jax implementation of UPDeT.

This implementation closesly follows the original one: https://github.com/Theohhhu/UPDeT
"""
import sys, pathlib
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
import chex

import flax.linen as nn

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print

# NOTE: add root folder to path to `import baselines`
sys.path.append(pathlib.Path(__file__).parents[2].as_posix())

import jaxmarl
from jaxmarl.wrappers.baselines import CTRolloutManager
from baselines.qlearning.qmix import MixingNetwork


class UPDeTCTRolloutManager(CTRolloutManager):
  @partial(jax.jit, static_argnums=0)
  def _preprocess_obs(self, arr, _):
    if "smax" in self._env.name.lower():
      # extract the relative features of others and self
      other_feats = arr[:self._env.obs_size - len(self._env.own_features)]
      other_feats = other_feats.reshape(-1, len(self._env.unit_features))
      own_feats = arr[-len(self._env.own_features):]
      # pad the own features to match the length of other features
      pad_width = [(0, max(0, other_feats.shape[-1] - own_feats.shape[-1]))]
      own_feats = jnp.pad(own_feats, pad_width, mode="constant", constant_values=0)
      arr = jnp.concatenate((other_feats, own_feats[jnp.newaxis, :]), axis=0)
    else:
      raise NotImplementedError(f"Preprocessing for {self._env.name} not implemented")
    return arr
  

class Embedder(nn.Module):
  hidden_dim: int
  init_scale: int
  scale_inputs: bool = True
  activation: bool = False

  @nn.compact
  def __call__(self, x, train: bool):
    x = nn.BatchNorm(use_running_average=not train)(x) if self.scale_inputs else x
    x = nn.Dense(
      self.hidden_dim,
      kernel_init=nn.initializers.orthogonal(self.scale_inputs),
      bias_init=nn.initializers.constant(0.),
    )(x)
    x = nn.relu(x) if self.activation else x
    x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class EncoderBlock(nn.Module):
  hidden_dim: int 
  num_heads: int
  feedforward_dim: int
  init_scale: float
  dropout_rate: float = 0.
  use_fast_attention: bool = False

  @nn.compact
  def __call__(self, x, mask: bool = None, deterministic: bool = True):
    attention = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dropout_rate=self.dropout_rate,
      kernel_init=nn.initializers.xavier_uniform(),
      use_bias=False,
    )(x, x, mask=mask, deterministic=deterministic)


class UPDeTAgent(nn.Module):
  action_dim: int
  hidden_dim: int
  init_scale: int
  embed_scale_inputs: bool = True
  embed_activation: bool = True

  @nn.compact
  def __call__(self, hidden, x, train: bool = True):
    obs, dones = x
    embeddings = Embedder(
      self.hidden_dim,
      self.init_scale,
      self.embed_scale_inputs,
      self.embed_activation,
    )(obs, train)


class EpsilonGreedy:
  """Epsilon Greedy action selection"""

  def __init__(self, start_e: float, end_e: float, duration: int):
    self.start_e = start_e
    self.end_e = end_e
    self.duration = duration
    self.slope = (end_e - start_e) / duration

  @partial(jax.jit, static_argnums=0)
  def get_epsilon(self, t: int):
    e = self.slope * t + self.start_e
    return jnp.clip(e, self.end_e)

  @partial(jax.jit, static_argnums=0)
  def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):
    # fmt: off
    def explore(q, eps, key):
      key_a, key_e = jax.random.split(key, 2)  # a key for sampling random actions and one for picking
      greedy_actions = jnp.argmax(q, axis=-1)  # get the greedy actions
      random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1])  # sample random actions
      pick_random = jax.random.uniform(key_e, greedy_actions.shape) < eps  # pick which actions should be random
      chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
      return chosed_actions

    eps = self.get_epsilon(t)
    keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals))))  # get a key for each agent
    chosen_actions = jax.tree_map(lambda q, k: explore(q, eps, k), q_vals, keys)
    return chosen_actions
    # fmt: on


class Transition(NamedTuple):
  obs: dict
  actions: dict
  rewards: dict
  dones: dict
  infos: dict


def make_train(config, env):
  config["num_updates"] = (
    config["total_timesteps"] // config["num_steps"] // config["num_envs"]
  )

  def train(rng):
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    train_env = UPDeTCTRolloutManager(env, batch_size=config["num_envs"])
    init_obs, env_state = train_env.batch_reset(_rng)
    init_dones = {
      agent: jnp.zeros(config["num_envs"], dtype=jnp.bool_)
      for agent in env.agents + ["__all__"]
    }
    
    # INIT AGENT
    rng, _rng = jax.random.split(rng)
    agent = UPDeTAgent(
      action_dim=train_env.max_action_space,
      hidden_dim=config["agent_hidden_dim"],
      init_scale=config["agent_init_scale"]
    )
    init_x = (
      jnp.zeros((1, 1, len(env.agents), len(env.unit_features))), # (time_step, batch_size, n_entities, obs_size)
      jnp.zeros((1, 1), dtype=jnp.bool_)
    ) # fmt: skip
    init_hs = jnp.zeros((1, 1, config["agent_hidden_dim"])) # (batch_size, n_entities, hidden_dim)
    agent_params = agent.init(_rng, init_hs, init_x, train=False)


    # fmt: off
    def homogeneous_pass(params, hidden_state, obs, dones):
      # concatenate agents and parallel envs to process them in one batch
      agents, flatten_agents_obs = zip(*obs.items())
      original_shape = flatten_agents_obs[0].shape # assume obs shape is the same for all agents
      batch_input = (
        jnp.concatenate(flatten_agents_obs, axis=1), # (time_steps, n_agents * n_envs, n_entities, obs_size)
        jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
      )

    # fmt: on

    # TRAINING LOOP
    def _update_step(runner_state, _):
      return runner_state, _

    # TRAIN
    rng, _rng = jax.random.split(rng)
    runner_state = (

    )
    runner_state, metrics = jax.lax.scan(
      _update_step, runner_state, None, config["num_updates"]
    )

    return {"runner_state": None, "metrics": None}
  
  return train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
  config = OmegaConf.to_container(config)
  print("Config", config)

  if "smax" in config["env"]["env_name"].lower():
    from jaxmarl.environments.smax import map_name_to_scenario
    from jaxmarl.wrappers.baselines import SMAXLogWrapper

    scenario = map_name_to_scenario(config["env"]["map_name"])
    config["env"]["env_kwargs"]["scenario"] = scenario
    env = jaxmarl.make(config["env"]["env_name"], **config["env"]["env_kwargs"])
    env = SMAXLogWrapper(env)

  rng = jax.random.PRNGKey(config["seed"])
  rngs = jax.random.split(rng, config["num_seeds"])
  train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
  out = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
  main()

