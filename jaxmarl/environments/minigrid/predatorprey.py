from typing import Tuple, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import chex

from jaxmarl.environments.minigrid.utils.minigrid_env import (
  MiniGridEnv, Tiles, State as MiniGridState
)
from jaxmarl.environments.minigrid.utils.constants import DIRECTIONS


@chex.dataclass
class State(MiniGridState):
  prey_pos: chex.Array


class PredatorPreyEnv(MiniGridEnv):
  def __init__(
    self,
    grid_size: int = 12,
    num_agents: int = 2,
    num_preys: int = 1,
    p_move_probs: tuple = jnp.array((0.175, 0.175, 0.175, 0.175, 0.3)),
    **kwargs
  ) -> None:
    super().__init__(
      grid_size=grid_size,
      num_agents=num_agents,
      **kwargs
    )
    self.num_preys = num_preys
    self.p_move_probs = p_move_probs

  @partial(jax.jit, static_argnums=(0,))
  def _generate(self, key: chex.Array, state: MiniGridState) -> State:
    grid = jnp.zeros((self.height, self.width), dtype=jnp.uint8)
    grid = MiniGridEnv.rectangle(grid, 0, 0, self.height, self.width, Tiles.WALL)
    state = state.replace(grid=grid)
    state = self.place_agents(key, state)
    
    def _place_prey(i, carry):
      key, state, pos = carry
      key, subkey = jax.random.split(key)
      coord = MiniGridEnv.sample_coordinates(subkey, state, 1)
      pos = pos.at[i].set(coord)
      state.grid = state.grid.at[coord[0], coord[1]].set(Tiles.BALL)
      return (key, state, pos)

    # update state to specific environment state to hold prey positions
    _, state, pos = jax.lax.fori_loop(
      0, self.num_preys, _place_prey, (key, state, jnp.zeros((self.num_preys, 2), dtype=jnp.int32)) 
    )
    state = State(**state, prey_pos=pos)
    return state
  
  @partial(jax.jit, static_argnums=(0,))
  def _heuristic_policy(self, key: chex.PRNGKey, state: State, pidx: int) -> chex.Array:
    view = self.agent_view_size // 2
    coords = jnp.mgrid[-view:view+1, -view:view+1]
    coords = coords.transpose(1, 2, 0).reshape(-1, 2)

    def _neighbourhood(key, pos, coords):
      m = jax.random.choice(key, jnp.arange(5), p=self.p_move_probs)
      next_pos = jax.lax.cond(m != 4, lambda: DIRECTIONS[m] + pos, lambda: pos)
      coords = jnp.clip( # rescale center and clip
        coords + next_pos,
        a_min=jnp.array((1, 1)), 
        a_max=jnp.array((self.height - 2, self.width - 2))
      )
      here = jax.vmap(
        lambda x, y: jnp.all(jnp.equal(x, y), axis=-1), in_axes=(None, 0)
      )(coords, state.agent_pos)
      count = jnp.count_nonzero(jnp.any(here, axis=-1), axis=-1)
      return count, next_pos
    
    # NOTE: Currently works, but it isn't great. Must find better way to do this.
    # The general idea is here though, probably use some sort of distance comparison.
    count, next_pos = jax.vmap(_neighbourhood, in_axes=(0, None, None))(
      jax.random.split(key, 5), state.prey_pos[pidx], coords
    )
    idx = jnp.argmin(count)
    next_pos = jax.lax.select(
      MiniGridEnv.check_walkable(state, next_pos[idx]),
      next_pos[idx],
      state.prey_pos[pidx]
    )
    return next_pos

  @partial(jax.jit, static_argnums=(0,))
  def _step(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
    obs, state, reward, done, info = super()._step(key, state, actions)
    
    for pidx in range(self.num_preys):
      key, subkey = jax.random.split(key)
      pos = self._heuristic_policy(subkey, state, pidx)
      state.grid = state.grid.at[*state.prey_pos[pidx]].set(Tiles.EMPTY)
      state.prey_pos = state.prey_pos.at[pidx].set(pos)
      state.grid = state.grid.at[*state.prey_pos[pidx]].set(Tiles.BALL)
    
    obs = self.get_obs(state)
    return obs, state, reward, done, info

