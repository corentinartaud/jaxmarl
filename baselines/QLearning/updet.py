"""End-to-End Jax implementation of UPDeT.

This implementation closesly follows the original one: https://github.com/Theohhhu/UPDeT
"""
import sys, pathlib
from typing import NamedTuple, Any
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print
import chex

import flax.linen as nn
from flax.core import frozen_dict
from flax.training import train_state
import optax
import flashbax as fbx

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
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
      kernel_init=nn.initializers.orthogonal(self.init_scale),
      bias_init=nn.initializers.constant(0.0),
    )(x)
    x = nn.relu(x) if self.activation else x
    x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class EncoderBlock(nn.Module):
  hidden_dim: int
  init_scale: float
  num_heads: int
  feedforward_dim: int
  dropout_rate: float = 0.0
  use_fast_attention: bool = False

  @nn.compact
  def __call__(self, x, mask: bool = None, deterministic: bool = True):
    attention = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dropout_rate=self.dropout_rate,
      kernel_init=nn.initializers.xavier_uniform(),
      use_bias=False,
    )(x, x, mask=mask, deterministic=deterministic)
    x = nn.LayerNorm()(attention + x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    feedforward = nn.Sequential([
      nn.Dense(
        self.feedforward_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.constant(0.0),
      ),
      nn.relu,
      nn.Dense(
        self.hidden_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.constant(0.0),
      )
    ])(x)
    x = nn.LayerNorm()(feedforward + x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    return x, mask, deterministic


class ScannedTransformer(nn.Module):
  hidden_dim: int
  init_scale: int
  num_heads: int
  feedforward_dim: int
  num_layers: int
  dropout_rate: float = 0.0
  deterministic: bool = True

  @partial(
    nn.scan,
    variable_broadcast="params",
    in_axes=0,
    out_axes=0,
    split_rngs={"params": False},
  )
  @nn.compact
  def __call__(self, hidden, x):
    embeddings, mask, done = x
    hidden = jnp.where(
      done[:, jnp.newaxis, jnp.newaxis],
      jnp.zeros((*done.shape, 1, self.hidden_dim)),
      hidden
    )
    embeddings = jnp.concatenate((embeddings, hidden), axis=-2)
    embeddings, mask, _ = nn.Sequential([
      EncoderBlock(
        self.hidden_dim,
        self.init_scale,
        self.num_heads,
        self.feedforward_dim,
        self.dropout_rate, 
      )
      for _ in range(self.num_layers)
    ])(embeddings, mask, self.deterministic)
    embeddings = nn.Dense(
      self.hidden_dim,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.constant(0.0),
    )(embeddings)
    return embeddings[..., -1:, :], embeddings[..., :-1, :]


class UPDeTAgent(nn.Module):
  hidden_dim: int
  init_scale: int
  transf_num_heads: int
  transf_ff_dim: int
  transf_num_layers: int
  transf_dropout_rate: float
  num_enemies: int
  embed_scale_inputs: bool = True
  embed_activation: bool = True

  @nn.compact
  def __call__(self, hidden, x, train: bool = True):
    obs, dones = x
    obs_ = obs.reshape(obs.shape[0], -1, *obs.shape[-2:])
    embeddings = Embedder(
      self.hidden_dim,
      self.init_scale,
      self.embed_scale_inputs,
      self.embed_activation,
    )(obs_, train)
    hidden, embeddings = ScannedTransformer(
      self.hidden_dim,
      self.init_scale,
      self.transf_num_heads,
      self.transf_ff_dim,
      self.transf_num_layers,
      self.transf_dropout_rate,
      deterministic=True
    )(hidden, (embeddings, None, dones))
    embeddings = embeddings.reshape(*obs.shape[:-1], -1)

    fc = nn.Dense(
      5,
      kernel_init=nn.initializers.orthogonal(self.init_scale),
      bias_init=nn.initializers.constant(0.0),
    )
    q_mov = fc(embeddings[..., -1, :])
    q_attack = jnp.stack([
      jnp.mean(fc(embeddings[..., -i-2, :]), axis=-1)
      for i in range(self.num_enemies)
    ], axis=-1)
    q_vals = jnp.concatenate((q_mov, q_attack), axis=-1)
    
    return hidden, q_vals


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


class TrainState(train_state.TrainState):
  batch_stats: Any


def make_train(config, env):
  config["num_updates"] = (
    config["total_timesteps"] // config["num_steps"] // config["num_envs"]
  )

  def train(rng):
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    train_env = UPDeTCTRolloutManager(env, batch_size=config["num_envs"])
    test_env = UPDeTCTRolloutManager(env, batch_size=config["num_test_episodes"])
    init_obs, env_state = train_env.batch_reset(_rng)
    init_dones = {
      agent: jnp.zeros(config["num_envs"], dtype=jnp.bool_)
      for agent in env.agents + ["__all__"]
    }

    # INIT BUFFER
    # to initalize the buffer is necessary to sample a trajectory to know its strucutre
    def _env_sample_step(env_state, _):
      _, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
      key_a = jax.random.split(key_a, env.num_agents)
      actions = {agent: train_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
      obs, env_state, rewards, dones, infos = train_env.batch_step(key_s, env_state, actions)
      transition = Transition(obs, actions, rewards, dones, infos)
      return env_state, transition
    
    _, sample_traj = jax.lax.scan(
      _env_sample_step, env_state, None, config["num_steps"]
    )
    sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
    buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=config["buffer_size"] // config["num_envs"],
      min_length_time_axis=config["buffer_batch_size"],
      sample_batch_size=config["buffer_batch_size"],
      add_batch_size=config["num_envs"],
      sample_sequence_length=1,
      period=1,
    )
    buffer_state = buffer.init(sample_traj_unbatched)

    # INIT AGENT
    rng, _rng = jax.random.split(rng)
    agent = UPDeTAgent(
      hidden_dim=config["agent_hidden_dim"],
      init_scale=config["agent_init_scale"],
      transf_num_heads=config["agent_transf_num_heads"],
      transf_ff_dim=config["agent_transf_ff_dim"],
      transf_num_layers=config["agent_transf_num_layers"],
      transf_dropout_rate=0.,
      num_enemies=len(env.enemy_agents),
    )
    init_x = (
      jnp.zeros((1, 1, 1, len(env.all_agents), len(env.unit_features))), # (time_step, n_agents, batch_size, n_entities, obs_size)
      jnp.zeros((1, 1), dtype=jnp.bool_)
    )  # fmt: skip
    init_hs = jnp.zeros(
      (1, 1, config["agent_hidden_dim"])
    )  # (batch_size, n_entities, hidden_dim)
    agent_params = agent.init(_rng, init_hs, init_x, train=False)

    # init mixer
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((len(env.agents), 1, 1))
    state_size = sample_traj.obs["__all__"].shape[-1]  # get the state shape from the buffer
    init_state = jnp.zeros((1, 1, state_size))
    mixer = MixingNetwork(
      config["mixer_embedding_dim"], 
      config["mixer_hypernet_hidden_dim"], 
      config["mixer_init_scale"]
    )
    mixer_params = mixer.init(_rng, init_x, init_state)

    network_params = frozen_dict.freeze({"agent": agent_params["params"], "mixer": mixer_params})
    linear_schedule = lambda count: config["lr"] * (
      1.0 - (count / config["num_updates"])
    )
    lr = linear_schedule if config.get("lr_linear_decay", False) else config["lr"]
    tx = optax.chain(
      optax.clip_by_global_norm(config["max_grad_norm"]),
      optax.adamw(
        learning_rate=lr,
        eps=config["eps_adam"],
        weight_decay=config["weight_decay_adam"],
      ),
    )
    train_state = TrainState.create(
      apply_fn=None,
      params=network_params,
      batch_stats=agent_params["batch_stats"],
      tx=tx,
    )
    # target network params
    target_network_params = jax.tree_map(lambda x: jnp.copy(x), train_state.params)

    # INIT EXPLORATION STRATEGY
    explorer = EpsilonGreedy(
      start_e=config["epsilon_start"],
      end_e=config["epsilon_finish"],
      duration=config["epsilon_anneal_time"],
    )

    # fmt: off
    def homogeneous_pass(params, batch_stats, hidden_state, obs, dones, train: bool = False):
      # concatenate agents and parallel envs to process them in one batch
      agents, flatten_agents_obs = zip(*obs.items())
      batch_input = (
        jnp.stack(flatten_agents_obs, axis=1), # (time_steps, n_agents, n_envs, n_entities, obs_size)
        jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
      )
      apply_in = (
        {"hidden": hidden_state, "x": batch_input, "train": train}
        if not train
        else {"hidden": hidden_state, "x": batch_input, "train": train, "mutable": ["batch_stats"]}
      )
      out = agent.apply({"params": params, "batch_stats": batch_stats}, **apply_in)
      (hidden_state, q_vals), updates = out if train else (out, None)
      q_vals = {a: q_vals[:, i] for i, a in enumerate(agents)}
      return hidden_state, q_vals, updates
    # fmt: on

    # TRAINING LOOP
    def _update_step(runner_state, _):
      (
        train_state,
        target_network_params,
        env_state,
        buffer_state,
        init_obs,
        init_dones,
        time_state,
        test_metrics,
        rng,
      ) = runner_state

      # EPISODE STEP
      def _env_step(step_state, _):
        params, batch_stats, env_state, last_obs, last_dones, hstate, rng, t = step_state
        # prepare rngs for actions and step
        rng, key_a, key_s = jax.random.split(rng, 3)

        # SELECT ACTION
        # add a dummy time_step dimension to the agent input
        # ensure to not pass the global state (obs["__all__"]) to the network
        obs_ = {a: last_obs[a] for a in env.agents}
        obs_ = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
        # get the q_values from the agent netwoek
        hstate, q_vals, _ = homogeneous_pass(params, batch_stats, hstate, obs_, dones_)
        # remove the dummy time_step dimension and index qs by the valid actions of each agent
        valid_q_vals = jax.tree_map(lambda q: q.squeeze(0), q_vals)
        # explore with epsilon greedy_exploration
        actions = explorer.choose_actions(valid_q_vals, t, key_a)

        # STEP ENV
        obs, env_state, rewards, dones, infos = train_env.batch_step(
          key_s, env_state, actions
        )
        transition = Transition(last_obs, actions, rewards, dones, infos)

        step_state = (params, batch_stats, env_state, obs, dones, hstate, rng, t + 1)
        return step_state, transition

      rng, _rng = jax.random.split(rng)
      hstate = jnp.zeros(
        (len(env.agents) * config["num_envs"], 1, config["agent_hidden_dim"])
      )  # (batch_size, n_agents * n_envs, hidden_dim)
      step_state = (
        train_state.params["agent"],
        train_state.batch_stats,
        env_state,
        init_obs,
        init_dones,
        hstate,
        _rng,
        time_state["timesteps"],  # t is needed to compute epsilon
      )
      step_state, traj_batch = jax.lax.scan(
        _env_step, step_state, None, config["num_steps"]
      )

      # BUFFER UPDATE: save the collected trajectory in the buffer
      buffer_traj_batch = jax.tree_util.tree_map(
        # put the batch dim first and add a dummy sequence dim
        lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis],
        traj_batch
      ) # (num_envs, 1, time_steps, ...)
      buffer_state = buffer.add(buffer_state, buffer_traj_batch)

      # LEARN PHASE
      def q_of_action(q, u):
        """index the q_values with action indices"""
        q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
        return jnp.squeeze(q_u, axis=-1)
      
      def compute_target(target_max_qvals, rewards, dones):
        if not config.get("td_lambda_loss", True):
          return rewards[:-1] + config["gamma"] * (1 - dones[:-1]) * target_max_qvals
        
        # time difference loss
        def _td_lambda_target(ret, values):
            reward, done, target_qs = values
            ret = jnp.where(
              done,
              target_qs,
              ret * config["td_lambda"] * config["gamma"]
              + reward
              + (1 - config["td_lambda"]) * config["gamma"] * (1 - done) * target_qs
            )
            return ret, ret

        ret = target_max_qvals[-1] * (1 - dones[-1])
        ret, td_targets = jax.lax.scan(
            _td_lambda_target,
            ret,
            (rewards[-2::-1], dones[-2::-1], target_max_qvals[-1::-1])
        )
        return td_targets[::-1]

      def _network_update(carry, _):
        train_state, rng = carry

        rng, _rng = jax.random.split(rng)
        learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
        learn_traj = jax.tree_map(
          lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
          learn_traj
        ) # (max_time_steps, batch_size, ...)
        init_hs = jnp.zeros(
          (len(env.agents) * config["buffer_batch_size"], 1, config["agent_hidden_dim"])
        )  # (batch_size, n_agents * n_envs, hidden_dim)

        def _loss_fn(params, target_network_params, init_hstate, learn_traj):
          obs_ = {a: learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
          _, q_vals, updates = homogeneous_pass(
            params['agent'], train_state.batch_stats, init_hstate, obs_, learn_traj.dones, train=True
          )
          _, target_q_vals, _ = homogeneous_pass(
            target_network_params["agent"], train_state.batch_stats, init_hstate, obs_, learn_traj.dones, train=True
          )

          # get the q_vals of the taken actions (with exploration) for each agent
          chosen_action_qvals = jax.tree_map(
            lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
            q_vals,
            learn_traj.actions
          )

          # get the target q value of the greedy actions for each agent
          target_max_qvals = jax.tree_map(
            lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
            target_q_vals,
            jax.lax.stop_gradient(q_vals)
          )

          # compute q_tot with the mixer network
          chosen_action_qvals_mix = mixer.apply(
              params['mixer'], 
              jnp.stack(list(chosen_action_qvals.values())),
              learn_traj.obs['__all__'][:-1] # avoid last timestep
          )
          target_max_qvals_mix = mixer.apply(
              target_network_params["mixer"], 
              jnp.stack(list(target_max_qvals.values())),
              learn_traj.obs['__all__'][1:] # avoid first timestep
          )

          targets = compute_target(target_max_qvals_mix, learn_traj.rewards["__all__"], learn_traj.dones["__all__"])
          avg = 0.5 if config.get("td_lambda_loss", True) else 1.0
          loss = jnp.mean(avg * ((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2))
          return loss, updates["batch_stats"]

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, updates), grads = grad_fn(train_state.params, target_network_params, init_hs, learn_traj)
        train_state = train_state.apply_gradients(grads=grads)
        train_state = train_state.replace(batch_stats=updates)

        return (train_state, rng), {"loss": loss}

      # TRAINING - perform N updates over the network
      rng, _rng = jax.random.split(rng)
      (train_state, _), update_info = jax.lax.scan(
        _network_update, (train_state, _rng), None, config["num_mini_updates"]
      )

      # UPDATE THE VARIABLES AND RETURN
      # reset the environment
      rng, _rng = jax.random.split(rng)
      init_obs, env_state = train_env.batch_reset(_rng)
      init_dones = {agent:jnp.zeros((config["num_envs"]), dtype=bool) for agent in env.agents + ["__all__"]}

      # update the states
      time_state["timesteps"] = step_state[-1]
      time_state["updates"]   = time_state["updates"] + 1

      # update the target network if necessary
      target_network_params = jax.lax.cond(
        time_state["updates"] % config["target_update_interval"] == 0,
        lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
        lambda _: target_network_params,
        operand=None
      )

      # update the greedy rewards
      rng, _rng = jax.random.split(rng)
      test_metrics = jax.lax.cond(
        time_state['updates'] % (config["test_interval"] // config["num_steps"] // config["num_envs"]) == 0,
        lambda _: get_greedy_metrics(_rng, train_state.params["agent"], train_state.batch_stats, time_state),
        lambda _: test_metrics,
        operand=None
      ) # fmt: skip

      # update the returning metrics
      metrics = {
        "timesteps": time_state["timesteps"] * config["num_envs"],
        "updates": time_state["updates"],
        "loss": update_info["loss"].mean(),
        "rewards": jax.tree_util.tree_map(
          lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards
        ),
        "eps": explorer.get_epsilon(time_state["timesteps"]),
      }
      metrics["test_metrics"] = test_metrics  # add the test metrics dictionary

      def callback(metrics, infos):
        info_metrics = {
          k.replace("returned", "mean"): v[..., 0][infos["returned_episode"][..., 0]].mean()
          for k, v in infos.items()
          if k != "returned_episode"
        }
        test_metrics = {
          k.replace("returned", "mean"): v.mean() 
          for k, v in metrics["test_metrics"].items() 
          if k != "test_returned_episode"
        } 
        wandb.log(
          {
            "returns": metrics["rewards"]["__all__"].mean(),
            "timestep": metrics["timesteps"],
            "updates": metrics["updates"],
            "loss": metrics["loss"],
            "epsilon": metrics["eps"],
            **info_metrics,
            **test_metrics,
          }
        )

      jax.debug.callback(callback, metrics, traj_batch.infos)

      runner_state = (
        train_state,
        target_network_params,
        env_state,
        buffer_state,
        init_obs,
        init_dones,
        time_state,
        test_metrics,
        rng
      )
      return runner_state, _


    def get_greedy_metrics(rng, params, batch_stats, time_state):
      """Help function to test greedy policy during training"""
      def _greedy_env_step(step_state, _):
        params, env_state, last_obs, last_dones, hstate, rng = step_state
        rng, key_s = jax.random.split(rng)
        obs_   = {a:last_obs[a] for a in env.agents}
        obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
        hstate, q_vals, _ = homogeneous_pass(params, batch_stats, hstate, obs_, dones_)
        actions = jax.tree_map(lambda q: jnp.argmax(q.squeeze(0), axis=-1), q_vals)
        obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
        step_state = (params, env_state, obs, dones, hstate, rng)
        return step_state, (rewards, dones, infos)
      
      rng, _rng = jax.random.split(rng)
      init_obs, env_state = test_env.batch_reset(_rng)
      init_dones = {agent: jnp.zeros((config["num_test_episodes"]), dtype=bool) for agent in env.agents + ["__all__"]}
      rng, _rng = jax.random.split(rng)
      hstate = jnp.zeros(
        (len(env.agents) * config["num_test_episodes"], 1, config["agent_hidden_dim"])
      )  # (batch_size, n_agents * n_envs, hidden_dim) 
      step_state = (
          params,
          env_state,
          init_obs,
          init_dones,
          hstate, 
          _rng,
      )
      step_state, (rewards, dones, infos) = jax.lax.scan(
          _greedy_env_step, step_state, None, config["num_steps"]
      )
      # compute the metrics of the first episode that is done for each parallel env
      def first_episode_returns(rewards, dones):
          first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
          first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
          return jnp.where(first_episode_mask, rewards, 0.).sum()
      all_dones = dones['__all__']
      first_returns = jax.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
      first_infos   = jax.tree_map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
      metrics = {
        'test_returns': first_returns['__all__'],# episode returns
        **{'test_'+k:v for k,v in first_infos.items()}
      }
      if config.get('verbose', False):
        jax.debug.callback(
          lambda t, r: print(f"Timestep: {t}, return: {r}"), 
          time_state['timesteps'] * config['num_envs'], 
          first_returns['__all__'].mean()
        )
      return metrics

    time_state = {"timesteps": jnp.array(0), "updates": jnp.array(0)}
    rng, _rng = jax.random.split(rng)
    test_metrics = get_greedy_metrics(_rng, train_state.params["agent"], train_state.batch_stats, time_state) # initial greedy metrics
    
    # TRAIN
    rng, _rng = jax.random.split(rng)
    runner_state = (
      train_state,
      target_network_params,
      env_state,
      buffer_state,
      init_obs,
      init_dones,
      time_state,
      test_metrics,
      _rng,
    )
    runner_state, metrics = jax.lax.scan(
      _update_step, runner_state, None, config["num_updates"]
    )
    return {"runner_state": runner_state, "metrics": metrics}

  return train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
  config = OmegaConf.to_container(config)
  print("Config", config)

  env_name = config["env"]["env_name"]
  alg_name = f'updet_{"ps" if config["alg"].get("parameters_sharing", True) else "ns"}'

  if "smax" in env_name.lower():
    from jaxmarl.environments.smax import map_name_to_scenario
    from jaxmarl.wrappers.baselines import SMAXLogWrapper

    scenario = map_name_to_scenario(config["env"]["map_name"])
    config["env"]["env_kwargs"]["scenario"] = scenario
    env = jaxmarl.make(env_name, **config["env"]["env_kwargs"])
    env = SMAXLogWrapper(env)
  
  wandb.init(
    entity=config["entity"],
    project=config["project"],
    tags=[
      alg_name.upper(),
      env_name.upper(),
      "RNN",
      "TD_LOSS" if config["alg"].get("td_lambda_loss", True) else "DDQN_LOSS",
      f"jax_{jax.__version__}",
    ],
    name=f"{alg_name}_{env_name}",
    config=config,
    mode=config["wandb_mode"],
  )

  rng = jax.random.PRNGKey(config["seed"])
  rngs = jax.random.split(rng, config["num_seeds"])
  train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
  out = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
  main()
