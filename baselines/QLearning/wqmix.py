"""
End-to-End JAX Implementation of W-QMIX.

The implementation closely follows the original code: https://github.com/oxwhirl/wqmix.py
"""
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import numpy as np
import chex

import flax.linen as nn
from flax.core import frozen_dict
from flax.traverse_util import flatten_dict
from flax.training.train_state import TrainState
from safetensors.flax import save_file
import flashbax as fbx
import optax

import hydra
from omegaconf import OmegaConf
import wandb
from rich import print

from jaxmarl import make
from jaxmarl.wrappers.baselines import CTRolloutManager


class ScannedRNN(nn.Module):
  @partial(
    nn.scan,
    variable_broadcast="params",
    in_axes=0,
    out_axes=0,
    split_rngs={"params": False},
  )
  @nn.compact
  def __call__(self, carry, x):
    """Applies the module."""
    rnn_state = carry
    ins, resets = x
    rnn_state = jnp.where(
      resets[:, np.newaxis],
      self.initialize_carry(ins.shape[-1], *ins.shape[:-1]),
      rnn_state,
    )
    new_rnn_state, y = nn.GRUCell(ins.shape[-1])(rnn_state, ins)
    return new_rnn_state, y

  @staticmethod
  def initialize_carry(hidden_size, *batch_size):
    # Use a dummy key since the default state init fn is just zeros.
    return nn.GRUCell(hidden_size, parent=None).initialize_carry(
      jax.random.PRNGKey(0), (*batch_size, hidden_size)
    )


class AgentRNN(nn.Module):
  # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
  action_dim: int
  hidden_dim: int
  init_scale: float

  @nn.compact
  def __call__(self, hidden, x):
    obs, dones = x
    embedding = nn.Dense(
      self.hidden_dim, 
      kernel_init=nn.initializers.orthogonal(self.init_scale), 
      bias_init=nn.initializers.constant(0.0)
    )(obs)
    embedding = nn.relu(embedding)

    rnn_in = (embedding, dones)
    hidden, embedding = ScannedRNN()(hidden, rnn_in)

    q_vals = nn.Dense(
      self.action_dim, 
      kernel_init=nn.initializers.orthogonal(self.init_scale), 
      bias_init=nn.initializers.constant(0.0)
    )(embedding)

    return hidden, q_vals


class HyperNetwork(nn.Module):
  """HyperNetwork for generating weights of mixing network."""
  
  hidden_dim: int
  output_dim: int
  init_scale: float

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(
      self.hidden_dim, 
      kernel_init=nn.initializers.orthogonal(self.init_scale), 
      bias_init=nn.initializers.constant(0.0)
    )(x)
    x = nn.relu(x)
    x = nn.Dense(
      self.output_dim, 
      kernel_init=nn.initializers.orthogonal(self.init_scale), 
      bias_init=nn.initializers.constant(0.0)
    )(x)
    return x


class MixingNetwork(nn.Module):
  """
  Mixing network for projecting individual agent Q-values into Q_tot. 
  Follows the original QMIX implementation.
  """
  
  embedding_dim: int
  hypernet_hidden_dim: int
  init_scale: float

  @nn.compact
  def __call__(self, q_vals, states):
    n_agents, time_steps, batch_size = q_vals.shape
    q_vals = jnp.transpose(q_vals, (1, 2, 0))  # (time_steps, batch_size, n_agents)

    # hypernetwork
    w_1 = HyperNetwork(
      hidden_dim=self.hypernet_hidden_dim,
      output_dim=self.embedding_dim * n_agents,
      init_scale=self.init_scale,
    )(states)
    b_1 = nn.Dense(
      self.embedding_dim,
      kernel_init=nn.initializers.orthogonal(self.init_scale),
      bias_init=nn.initializers.constant(0.0),
    )(states)
    w_2 = HyperNetwork(
      hidden_dim=self.hypernet_hidden_dim,
      output_dim=self.embedding_dim,
      init_scale=self.init_scale,
    )(states)
    b_2 = HyperNetwork(
      hidden_dim=self.embedding_dim, 
      output_dim=1, 
      init_scale=self.init_scale
    )(states)

    # monotonicity and reshaping
    w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
    b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
    w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
    b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

    # mix
    hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
    q_tot = jnp.matmul(hidden, w_2) + b_2

    return q_tot.squeeze()  # (time_steps, batch_size)


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
    train_env = CTRolloutManager(env, batch_size=config["num_envs"])
    # NOTE: batched env for testing (has different batch size)
    test_env = CTRolloutManager(env, batch_size=config["num_test_episodes"])
    init_obs, env_state = train_env.batch_reset(_rng)
    init_dones = {
      agent: jnp.zeros((config["num_envs"]), dtype=bool)
      for agent in env.agents + ["__all__"]
    }

    # INIT BUFFER
    # to initalize the buffer is necessary to sample a trajectory to know its strucutre
    def _env_sample_step(env_state, _):
      _, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)  # use a dummy rng here
      key_a = jax.random.split(key_a, env.num_agents)
      actions = {agent: train_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
      obs, env_state, rewards, dones, infos = train_env.batch_step(key_s, env_state, actions)
      transition = Transition(obs, actions, rewards, dones, infos)
      return env_state, transition

    _, sample_traj = jax.lax.scan(
      _env_sample_step, env_state, None, config["num_steps"]
    )
    sample_traj_unbatched = jax.tree_map(
      lambda x: x[:, 0], sample_traj
    )  # remove the NUM_ENV dim
    buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=config["buffer_size"] // config["num_envs"],
      min_length_time_axis=config["buffer_batch_size"],
      sample_batch_size=config["buffer_batch_size"],
      add_batch_size=config["num_envs"],
      sample_sequence_length=1,
      period=1,
    )
    buffer_state = buffer.init(sample_traj_unbatched)

    # INIT NETWORK
    # init agent
    agent = AgentRNN(
      action_dim=train_env.max_action_space,
      hidden_dim=config["agent_hidden_dim"],
      init_scale=config["agent_init_scale"],
    )
    rng, _rng = jax.random.split(rng)
    if config["parameters_sharing"]:
      init_x = (
        jnp.zeros((1, 1, train_env.obs_size)),  # (time_step, batch_size, obs_size)
        jnp.zeros((1, 1)),  # (time_step, batch size)
      )
      init_hs = ScannedRNN.initialize_carry(
        config["agent_hidden_dim"], 1
      )  # (batch_size, hidden_dim)
      agent_params = agent.init(_rng, init_hs, init_x)
    else:
      init_x = (
        jnp.zeros(
          (len(env.agents), 1, 1, train_env.obs_size)
        ),  # (time_step, batch_size, obs_size)
        jnp.zeros((len(env.agents), 1, 1)),  # (time_step, batch size)
      )
      init_hs = ScannedRNN.initialize_carry(
        config["agent_hidden_dim"], len(env.agents), 1
      )  # (n_agents, batch_size, hidden_dim)
      rngs = jax.random.split(_rng, len(env.agents))  # a random init for each agent
      agent_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

    # init mixer
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((len(env.agents), 1, 1))
    state_size = sample_traj.obs["__all__"].shape[
      -1
    ]  # get the state shape from the buffer
    init_state = jnp.zeros((1, 1, state_size))
    mixer = MixingNetwork(
      config["mixer_embedding_dim"],
      config["mixer_hypernet_hidden_dim"],
      config["mixer_init_scale"],
    )
    mixer_params = mixer.init(_rng, init_x, init_state)

    # init optimizer
    network_params = frozen_dict.freeze({"agent": agent_params, "mixer": mixer_params})

    def linear_schedule(count):
      frac = 1.0 - (count / config["num_updates"])
      return config["lr"] * frac

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

    # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
    # fmt: off
    if config["parameters_sharing"]:

      def homogeneous_pass(params, hidden_state, obs, dones):
        # concatenate agents and parallel envs to process them in one batch
        agents, flatten_agents_obs = zip(*obs.items())
        original_shape = flatten_agents_obs[0].shape  # assumes obs shape is the same for all agents
        batched_input = (
          jnp.concatenate(flatten_agents_obs, axis=1),  # (time_step, n_agents*n_envs, obs_size)
          jnp.concatenate([dones[agent] for agent in agents], axis=1),  # ensure to not pass other keys (like __all__)
        )
        hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
        q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1)  # (time_steps, n_agents, n_envs, action_dim)
        q_vals = {a: q_vals[:, i] for i, a in enumerate(agents)}
        return hidden_state, q_vals
    else:

      def homogeneous_pass(params, hidden_state, obs, dones):
        # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
        agents, flatten_agents_obs = zip(*obs.items())
        batched_input = (
          jnp.stack(flatten_agents_obs, axis=0),  # (n_agents, time_step, n_envs, obs_size)
          jnp.stack([dones[agent] for agent in agents], axis=0),  # ensure to not pass other keys (like __all__)
        )
        # computes the q_vals with the params of each agent separately by vmapping
        hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
        q_vals = {a: q_vals[i] for i, a in enumerate(agents)}
        return hidden_state, q_vals
    # fmt: on

    # TRAINING LOOP
    def _update_step(runner_state, _):
      (
        train_state,
        target_network_params,
        env_state,
        buffer_state,
        time_state,
        init_obs,
        init_dones,
        test_metrics,
        rng,
      ) = runner_state

      # EPISODE STEP
      def _env_step(step_state, _):
        params, env_state, last_obs, last_dones, hstate, rng, t = step_state

        # prepare rngs for actions and step
        rng, key_a, key_s = jax.random.split(rng, 3)

        # SELECT ACTION
        # add a dummy time_step dimension to the agent input
        # ensure to not pass the global state (obs["__all__"]) to the network
        obs_ = {a: last_obs[a] for a in env.agents}
        obs_ = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
        # get the q_values from the agent netwoek
        hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
        # remove the dummy time_step dimension and index qs by the valid actions of each agent
        valid_q_vals = jax.tree_map(lambda q: q.squeeze(0), q_vals,)
        # explore with epsilon greedy_exploration
        actions = explorer.choose_actions(valid_q_vals, t, key_a)

        # STEP ENV
        obs, env_state, rewards, dones, infos = train_env.batch_step(
          key_s, env_state, actions
        )
        transition = Transition(last_obs, actions, rewards, dones, infos)

        step_state = (params, env_state, obs, dones, hstate, rng, t + 1)
        return step_state, transition

      # prepare the step state and collect the episode trajectory
      rng, _rng = jax.random.split(rng)
      if config["parameters_sharing"]:
        hstate = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents) * config["num_envs"]
        )  # (n_agents*n_envs, hs_size)
      else:
        hstate = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents), config["num_envs"]
        )  # (n_agents, n_envs, hs_size)

      step_state = (
        train_state.params["agent"],
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
        traj_batch,
      )  # (num_envs, 1, time_steps, ...)
      buffer_state = buffer.add(buffer_state, buffer_traj_batch)

      # LEARN PHASE
      def q_of_action(q, u):
        """index the q_values with action indices"""
        q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
        return jnp.squeeze(q_u, axis=-1)

      def compute_target(target_max_qvals, rewards, dones):
        if not config.get("td_lambda_loss", True):
          # standard DQN loss
          return rewards[:-1] + config["gamma"] * (1 - dones[:-1]) * target_max_qvals
        
        # time difference loss
        def _td_lambda_target(ret, values):
          reward, done, target_qs = values
          ret = jnp.where(
            done,
            target_qs,
            ret * config["td_lambda"] * config["gamma"]
            + reward
            + (1 - config["td_lambda"]) * config["gamma"] * (1 - done) * target_qs,
          )
          return ret, ret

        ret = target_max_qvals[-1] * (1 - dones[-1])
        ret, td_targets = jax.lax.scan(
          _td_lambda_target,
          ret,
          (rewards[-2::-1], dones[-2::-1], target_max_qvals[-1::-1]),
        )
        return td_targets[::-1]
      
      def _loss_fn(params, target_network_params, init_hstate, learn_traj):
        # ensure to not pass the global state (obs["__all__"]) to the network
        obs_ = {a: learn_traj.obs[a] for a in env.agents}
        _, q_vals = homogeneous_pass(
          params["agent"], init_hstate, obs_, learn_traj.dones
        )
        _, target_q_vals = homogeneous_pass(
          target_network_params["agent"], init_hstate, obs_, learn_traj.dones
        )

        # get the q_vals of the taken actions (with exploration) for each agent
        chosen_action_qvals = jax.tree_map(
          lambda q, u: q_of_action(q, u)[:-1],  # avoid last timestep
          q_vals,
          learn_traj.actions,
        )

        # get the target q value of the greedy actions for each agent
        target_max_qvals = jax.tree_map(
          lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1)),
          target_q_vals,
          jax.lax.stop_gradient(q_vals),
        )

        # compute q_tot with the mixer network
        chosen_action_qvals_mix = mixer.apply(
          params["mixer"],
          jnp.stack(list(chosen_action_qvals.values())),
          learn_traj.obs["__all__"][:-1],  # avoid last timestep
        )
        target_max_qvals_mix = mixer.apply(
          target_network_params["mixer"],
          jnp.stack(list(target_max_qvals.values()))[:, 1:], # avoid first timestep
          learn_traj.obs["__all__"][1:],  # avoid first timestep
        )

        # compute target
        targets = compute_target(
          target_max_qvals_mix, 
          learn_traj.rewards["__all__"],
          learn_traj.dones["__all__"],
        )
        td_error = (chosen_action_qvals_mix - jax.lax.stop_gradient(targets))

        # compute weighting
        if not config["optimistic_weighting"]:
          target_max_qvals_mix = mixer.apply(
            target_network_params["mixer"],
            jnp.stack(list(target_max_qvals.values()))[:, :-1],  # avoid last timestep
            learn_traj.obs["__all__"][:-1],  # avoid last timestep
          )
          is_max_action = jax.tree_map(
            lambda u, q: (u == jnp.argmax(q, axis=-1))[:-1],
            learn_traj.actions,
            jax.lax.stop_gradient(q_vals),
          )
          cond = (
            jnp.stack(list(is_max_action.values())).min(axis=0) 
            | (jax.lax.stop_gradient(targets) > target_max_qvals_mix)
          )
        else:
          cond = td_error < 0.

        ws = jnp.ones_like(td_error) * config["weighted_projection"]
        ws = jnp.mean(jnp.where(cond, jnp.ones_like(td_error) * 1., ws))  # target is greater than current max
        avg = 0.5 if config.get("td_lambda_loss", True) else 1.0
        loss = jnp.mean(jax.lax.stop_gradient(ws) * (avg * (td_error ** 2)))
        return loss

      # sample a batched trajectory from the buffer and set the time step dim in first axis
      # fmt: off
      rng, _rng = jax.random.split(rng)
      learn_traj = buffer.sample(buffer_state, _rng).experience  # (batch_size, 1, max_time_steps, ...)
      learn_traj = jax.tree_map(
        lambda x: jnp.swapaxes(x[:, 0], 0, 1),  # remove the dummy sequence dim (1) and swap batch and temporal dims
        learn_traj,
      )  # (max_time_steps, batch_size, ...)
      # fmt: on
      if config["parameters_sharing"]:
        init_hs = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents) * config["buffer_batch_size"]
        )  # (n_agents*batch_size, hs_size)
      else:
        init_hs = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents), config["buffer_batch_size"]
        )  # (n_agents, batch_size, hs_size)

      # compute loss and optimize grad
      grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
      loss, grads = grad_fn(
        train_state.params, target_network_params, init_hs, learn_traj
      )
      train_state = train_state.apply_gradients(grads=grads)

      # UPDATE THE VARIABLES AND RETURN
      # reset the environment
      rng, _rng = jax.random.split(rng)
      init_obs, env_state = train_env.batch_reset(_rng)
      init_dones = {
        agent: jnp.zeros((config["num_envs"]), dtype=bool)
        for agent in env.agents + ["__all__"]
      }

      # update the states
      time_state["timesteps"] = step_state[-1]
      time_state["updates"] = time_state["updates"] + 1

      # update the target network if necessary
      target_network_params = jax.lax.cond(
        time_state["updates"] % config["target_update_interval"] == 0,
        lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
        lambda _: target_network_params,
        operand=None,
      )

      # update the greedy rewards
      rng, _rng = jax.random.split(rng)
      test_metrics = jax.lax.cond(
        time_state["updates"] % (config["test_interval"] // config["num_steps"] // config["num_envs"]) == 0,
        lambda _: get_greedy_metrics(_rng, train_state.params["agent"], time_state),
        lambda _: test_metrics,
        operand=None,
      ) # fmt: skip

      # update the returning metrics
      metrics = {
        "timesteps": time_state["timesteps"] * config["num_envs"],
        "updates": time_state["updates"],
        "loss": loss,
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
        time_state,
        init_obs,
        init_dones,
        test_metrics,
        rng,
      )

      return runner_state, metrics

    def get_greedy_metrics(rng, params, time_state):
      """Help function to test greedy policy during training"""

      def _greedy_env_step(step_state, _):
        params, env_state, last_obs, last_dones, hstate, rng = step_state
        rng, key_s = jax.random.split(rng)
        obs_ = {a: last_obs[a] for a in env.agents}
        obs_ = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
        hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
        actions = jax.tree_map(lambda q: jnp.argmax(q.squeeze(0), axis=-1), q_vals)
        obs, env_state, rewards, dones, infos = test_env.batch_step(
          key_s, env_state, actions
        )
        step_state = (params, env_state, obs, dones, hstate, rng)
        return step_state, (rewards, dones, infos)

      rng, _rng = jax.random.split(rng)
      init_obs, env_state = test_env.batch_reset(_rng)
      init_dones = {
        agent: jnp.zeros((config["num_test_episodes"]), dtype=bool)
        for agent in env.agents + ["__all__"]
      }
      rng, _rng = jax.random.split(rng)
      if config["parameters_sharing"]:
        hstate = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents) * config["num_test_episodes"]
        )  # (n_agents*n_envs, hs_size)
      else:
        hstate = ScannedRNN.initialize_carry(
          config["agent_hidden_dim"], len(env.agents), config["num_test_episodes"]
        )  # (n_agents, n_envs, hs_size)
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
        first_done = jax.lax.select(
          jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones)
        )
        first_episode_mask = jnp.where(
          jnp.arange(dones.size) <= first_done, True, False
        )
        return jnp.where(first_episode_mask, rewards, 0.0).sum()

      all_dones = dones["__all__"]
      first_returns = jax.tree_map(
        lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards
      )
      first_infos = jax.tree_map(
        lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones),
        infos,
      )
      metrics = {
        "test_returns": first_returns["__all__"],  # episode returns
        **{"test_" + k: v for k, v in first_infos.items()},
      }
      if config.get("verbose", False):
        jax.debug.callback(
          lambda t, r: print(f"Timestep: {t}, return: {r}"),
          time_state["timesteps"] * config["num_envs"],
          first_returns["__all__"].mean(),
        )
      return metrics

    time_state = {"timesteps": jnp.array(0), "updates": jnp.array(0)}
    rng, _rng = jax.random.split(rng)
    test_metrics = get_greedy_metrics(
      _rng, train_state.params["agent"], time_state
    )  # initial greedy metrics

    # train
    rng, _rng = jax.random.split(rng)
    runner_state = (
      train_state,
      target_network_params,
      env_state,
      buffer_state,
      time_state,
      init_obs,
      init_dones,
      test_metrics,
      _rng,
    )
    runner_state, metrics = jax.lax.scan(
      _update_step, runner_state, None, config["num_updates"]
    )
    return {"runner_state": runner_state, "metrics": metrics}

  return train


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
  config = OmegaConf.to_container(config)
  print("Config:", config)

  env_name = config["env"]["env_name"]
  alg_name = f'qmix_{"ps" if config["alg"].get("parameters_sharing", True) else "ns"}'

  # smac init neeeds a scenario
  if "smax" in env_name.lower():
    from jaxmarl.environments.smax import map_name_to_scenario
    from jaxmarl.wrappers.baselines import SMAXLogWrapper

    config["env"]["env_kwargs"]["scenario"] = map_name_to_scenario(
      config["env"]["map_name"]
    )
    env_name = f"{config['env']['env_name']}_{config['env']['map_name']}"
    env = make(config["env"]["env_name"], **config["env"]["env_kwargs"])
    env = SMAXLogWrapper(env)

  wandb.init(
    entity=config["entity"],
    project=config["project"],
    tags=[
      alg_name.upper(),
      env_name.upper(),
      "RNN",
      "TD_LOSS" if config["alg"].get("td_lambda_loss", True) else "DQN_LOSS",
      f"jax_{jax.__version__}",
    ],
    name=f"{alg_name}_{env_name}",
    config=config,
    mode=config["wandb_mode"],
  )

  rng = jax.random.PRNGKey(config["seed"])
  rngs = jax.random.split(rng, config["num_seeds"])
  train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
  outs = jax.block_until_ready(train_vjit(rngs))

  # save params
  if config["save_path"] is not None:
    model_state = outs["runner_state"][0]
    params = jax.tree_map(lambda x: x[0], model_state.params)  # fmt: skip # save only params of the firt run
    save_dir = os.path.join(config["save_path"], env_name)
    os.makedirs(save_dir, exist_ok=True)
    fn = f"{save_dir}/{alg_name}.safetensors"
    save_file(flatten_dict(params, sep=","), fn)
    print(f"Parameters of first batch saved in {save_dir}/{alg_name}.safetensors")


if __name__ == "__main__":
  main()
