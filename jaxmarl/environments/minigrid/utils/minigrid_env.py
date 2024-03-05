from typing import Type, Tuple, Dict
from functools import partial
import abc
from dataclasses import make_dataclass, field

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import chex

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.minigrid.utils.constants import Tiles, Actions
from jaxmarl.environments.minigrid.utils.constants import (
  DIRECTIONS, WALKABLE
)
from jaxmarl.environments.spaces import Discrete


@chex.dataclass
class State:
  grid: chex.Array
  agent_pos: chex.Array
  agent_dir: chex.Array


class MiniGridEnv(MultiAgentEnv):
  def __init__(
    self,
    grid_size: int = None,
    width: int = None,
    height: int = None,
    num_agents: int = 2,
    agent_view_size: int = 5,
    max_steps: int = 100,
  ):
    super().__init__(num_agents=num_agents)
    if grid_size:
      assert width is None and height is None
      width = height = grid_size
    assert width is not None and height is not None    
    self.width, self.height = width, height

    self.agents = [f"agent_{i}" for i in range(num_agents)]
    self.atoi = {a: i for i, a in enumerate(self.agents)}
    self.itoa = {i: a for i, a in enumerate(self.agents)}
    
    self.action_spaces = {a: Discrete(len(Actions)) for a in self.agents}
    self.observation_spaces = {}

    self.agent_view_size = agent_view_size
    self.max_steps = max_steps
  
  @abc.abstractmethod
  def _generate(self):
    """Generate the environment grid."""
    raise NotImplementedError
  
  @partial(jax.jit, static_argnums=(0,))
  def reset(self, key: chex.Array) -> Type[State]:
    key, subkey = jax.random.split(key)

    # Create dummy state that will be replaced by _generate
    state = State(
      grid=jnp.zeros((self.height, self.width), dtype=jnp.uint8),
      agent_pos=jnp.array([[0, 0]] * self.num_agents),
      agent_dir=jax.random.randint(
        subkey, shape=(self.num_agents,), minval=0, maxval=4, dtype=jnp.uint8
      ), # NOTE: agent_dir will not be replaced for now
    )
    # Generate a new random grid at the start of each episode
    state = self._generate(subkey, state)
    # Return the first observation
    obs = self.get_obs(state)
    return obs, state

  @partial(jax.jit, static_argnums=(0,))
  def _step(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):

    def _clockwise(aidx: chex.Array, state: State):
      dir = (state.agent_dir[aidx] + 1) % 4
      dir = state.agent_dir.at[aidx].set(dir)
      state = state.replace(agent_dir=dir)
      return state

    def _anticlockwise(aidx: chex.Array, state: State):
      dir = (state.agent_dir[aidx] - 1) % 4
      dir = state.agent_dir.at[aidx].set(dir)
      state = state.replace(agent_dir=dir)
      return state

    def _forward(aidx: chex.Array, state: State):
      dir = jax.lax.dynamic_index_in_dim(DIRECTIONS, state.agent_dir[aidx], keepdims=False)
      fwd_pos = jnp.clip(
        state.agent_pos[aidx] + dir,
        a_min=jnp.array((0, 0)), 
        a_max=jnp.array((self.height - 1, self.width - 1))
      )
      fwd_pos = jax.lax.select(
        MiniGridEnv.check_walkable(state, fwd_pos),
        fwd_pos, 
        state.agent_pos[aidx],
      )
      fwd_pos = state.agent_pos.at[aidx].set(fwd_pos)
      state = state.replace(agent_pos=fwd_pos)
      return state

    def _interact(aidx: chex.Array, state: State):
      return state

    def _step_agent(carry: Tuple, aidx: chex.Array):
      actions, state = carry
      state = jax.lax.switch(actions[aidx], (
        lambda: _anticlockwise(aidx, state), # LEFT
        lambda: _clockwise(aidx, state), # RIGHT
        lambda: _forward(aidx, state), # FORWARD
        lambda: _interact(aidx, state), # INTERACT
        lambda: state, # NOOP
      ))
      return (actions, state), aidx

    key, _key = jax.random.split(key)
    permutation = jax.random.permutation(_key, self.num_agents)
    actions = jnp.array([actions[a] for a in self.agents])
    (_, state), _ = jax.lax.scan(
      _step_agent, (actions, state), permutation
    )

    obs = self.get_obs(state)
    reward = {a: jnp.zeros((), dtype=jnp.float32) for a in self.agents}
    done = {a: jnp.zeros((), dtype=jnp.bool_) for a in self.agents + ["__all__"]}
    return obs, state, reward, done, {}
  
  @partial(jax.jit, static_argnums=(0,))
  def get_obs(self, state: State) -> Dict:
    def _get_fov(aidx: chex.Array, state: State, height: int, width: int):
      grid = jnp.pad(
        state.grid, 
        pad_width=((height, height), (width, width)), 
        constant_values=Tiles.END
      )
      pos, dir = state.agent_pos[aidx], state.agent_dir[aidx]
      y, x = pos[0] + height, pos[1] + width # account for padding
      indices = jax.lax.switch(dir, (
        lambda: (y - height // 2, x), # RIGHT
        lambda: (y, x - width // 2), # DOWN
        lambda: (y - height // 2, x - width + 1), # LEFT
        lambda: (y - height + 1, x - width // 2), # UP
      )) 
      fov = jax.lax.dynamic_slice(grid, indices, (height, width))
      
      # rotate counter-clockwise to align agent direction with UP
      fov = jax.lax.switch(dir, (
        lambda: jnp.rot90(fov, 1), # RIGHT
        lambda: jnp.rot90(fov, 2), # DOWN
        lambda: jnp.rot90(fov, 3), # LEFT
        lambda: fov, # UP
      ))
      return fov
    
    height, width = self.agent_view_size, self.agent_view_size
    obs = jax.vmap(_get_fov, in_axes=(0, None, None, None))(
      jnp.arange(self.num_agents), state, height, width
    )
    return obs

  @partial(jax.jit, static_argnums=(0,))
  def place_agents(self, key: chex.Array, state: State):
    def _place_agent(i, carry):
      key, state = carry
      key, subkey = jax.random.split(key)
      coord = MiniGridEnv.sample_coordinates(subkey, state, 1)
      state.agent_pos = state.agent_pos.at[i].set(coord)
      return (key, state)
    
    _, state = jax.lax.fori_loop(
      0, self.num_agents, _place_agent, (key, state)
    )
    return state

  # ==== STATIC METHODS ====

  @staticmethod
  def horitonzal(
    grid: chex.Array, x: int, y: int, length: int, tile: chex.Array
  ) -> chex.Array:
    grid = grid.at[y, x: x + length].set(tile)
    return grid
  
  @staticmethod
  def vertical(
    grid: chex.Array, x: int, y: int, length: int, tile: chex.Array
  ) -> chex.Array:
    grid = grid.at[y: y + length, x].set(tile)
    return grid
  
  @staticmethod
  def rectangle(
    grid: chex.Array, x: int, y: int, h: int, w: int, tile: chex.Array
  ) -> chex.Array:
    grid = MiniGridEnv.vertical(grid, x, y, h, tile)
    grid = MiniGridEnv.vertical(grid, x + w - 1, y, h, tile)
    grid = MiniGridEnv.horitonzal(grid, x, y, w, tile)
    grid = MiniGridEnv.horitonzal(grid, x, y + h - 1, w, tile)
    return grid

  @staticmethod
  def check_available(state: State, pos: chex.Array) -> chex.Array:
    tile = state.grid[pos[0], pos[1]]
    empty = jnp.isin(tile, jnp.array((Tiles.EMPTY)), assume_unique=True)
    here = jnp.any(jnp.isin(pos, state.agent_pos, assume_unique=True))
    return empty & ~here
  
  @staticmethod
  def check_walkable(state: State, pos: chex.Array) -> chex.Array:
    tile = state.grid[pos[0], pos[1]]
    walkable = jnp.isin(tile, WALKABLE, assume_unique=True)
    here = jax.vmap(
      lambda x, y: jnp.all(jnp.equal(x, y)), in_axes=(None, 0)
    )(pos, state.agent_pos)
    return walkable & ~jnp.any(here)

  @staticmethod
  def relative_coordinates(grid: chex.Array) -> chex.Array:
    coords = jnp.mgrid[:grid.shape[0], :grid.shape[1]]
    coords = coords.transpose(1, 2, 0).reshape(-1, 2)
    return coords
  
  @staticmethod
  def available_tiles(state: State) -> chex.Array:
    coords = MiniGridEnv.relative_coordinates(state.grid)
    mask = jax.vmap(MiniGridEnv.check_available, in_axes=(None, 0))(state, coords)
    mask = mask.reshape(state.grid.shape[0], state.grid.shape[1])
    return mask

  @staticmethod
  def sample_coordinates(
    key: chex.PRNGKey, state: State, num: int, mask: chex.Array = None
  ) -> chex.Array:
    if mask is None:
      mask = jnp.ones((state.grid.shape[0], state.grid.shape[1]), dtype=jnp.bool_)
    
    coords = jax.random.choice(
      key=key,
      shape=(num,),
      a=jnp.arange(state.grid.shape[0] * state.grid.shape[1]),
      replace=False,
      p=(mask & MiniGridEnv.available_tiles(state)).flatten(),
    )
    coords = jnp.divmod(coords, state.grid.shape[1])
    coords = jnp.concatenate((coords[1], coords[0]), axis=-1)
    return coords

 