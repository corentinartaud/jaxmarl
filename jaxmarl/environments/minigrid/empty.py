
from functools import partial

import jax
import jax.numpy as jnp
import chex

from majax.envs.minigrid.utils import MiniGridEnv, Tiles, State


class EmptyEnv(MiniGridEnv):
  def __init__(
    self,
    grid_size: int = 7,
    **kwargs
  ) -> None:
    super().__init__(
      grid_size=grid_size,
      num_agents=1,
      **kwargs
    )

  @partial(jax.jit, static_argnums=(0,))
  def _generate(self, key: chex.Array, state: State):
    grid = jnp.zeros((self.height, self.width), dtype=jnp.uint8)
    grid = MiniGridEnv.rectangle(grid, 0, 0, self.height, self.width, Tiles.WALL)
    # place a goal square in the bottom-right corner
    grid = grid.at[self.height - 2, self.width - 2].set(Tiles.GOAL)
    state = state.replace(grid=grid)
    state = self.place_agents(key, state)
    return state
