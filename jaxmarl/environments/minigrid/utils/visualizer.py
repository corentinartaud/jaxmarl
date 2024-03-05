from typing import Tuple, Union, Sequence, Optional, Dict, Any
import math
import numpy as np

from jaxmarl.environments.minigrid.utils.constants import Tiles
from jaxmarl.environments.minigrid.utils.minigrid_env import State
from jaxmarl.environments.minigrid.utils import rendering


# Constants
TILE_SIZE = 32


class MiniGridVisualizer:
  # used for caching
  tile_cache: Dict[Tuple[Any, ...], Any] = {}

  def __init__(self, env):
    self._env = env
    self.view_size = env.agent_view_size
    self.window = None
  
  def animate(
    self, 
    state_seq: Sequence[State], 
    save_fname: Optional[str] = None,
    view: bool = True, 
  ):
    pass
  
  def render(self, state: State, highlight: bool = True) -> np.ndarray:
    highlight_mask = self._highlight_mask(state, self.view_size)

    # Compute the total grid size
    width_px = state.grid.shape[0] * TILE_SIZE
    height_px = state.grid.shape[1] * TILE_SIZE

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
    grid = np.array(state.grid)
    
    # Render the grid
    for y in range(0, grid.shape[0]):
      for x in range(0, grid.shape[1]):
        cell = grid[y, x]
        if cell == 0: cell = None
        agent_here = [np.array_equal(state.agent_pos[a], (y, x)) for a in range(self._env.num_agents)]
        agent_id = np.argmax(agent_here) if np.any(agent_here) else None
        tile_img = MiniGridVisualizer.render_tile(
          cell,
          agent_dir=int(state.agent_dir[agent_id]) if agent_id is not None else None,
          highlight=highlight_mask[y, x] if highlight else False,
          tile_size=TILE_SIZE,
        )
        ymin, ymax = y * TILE_SIZE, (y + 1) * TILE_SIZE
        xmin, xmax = x * TILE_SIZE, (x + 1) * TILE_SIZE
        img[ymin:ymax, xmin:xmax, :] = tile_img

    return img

  def _highlight_mask(self, state: State, size: int):
    mask = np.zeros((
      state.grid.shape[0] + 2 * size,
      state.grid.shape[1] + 2 * size
    ), dtype=np.bool_)
    for aidx in range(self._env.num_agents):
      y, x = state.agent_pos[aidx, 0] + size, state.agent_pos[aidx, 1] + size
      dir = state.agent_dir[aidx]

      if dir == 0: y = y - size // 2 # UP
      elif dir == 1: x = x - size // 2 # DOWN
      elif dir == 2: y, x = y - size // 2, x - size + 1 # LEFT
      elif dir == 3: y, x = y - size + 1, x - size // 2 # RIGHT

      mask[y:y + size, x:x + size] = True
    mask = mask[size:-size, size:-size]
    assert mask.shape == (state.grid.shape[0], state.grid.shape[1])
    return mask
  
  @classmethod
  def render_tile(
    cls,
    obj: int,
    agent_dir: Union[int, None] = None,
    highlight: bool = False,
    tile_size: int = TILE_SIZE,
    subdivs: int = 3
  ) -> np.ndarray:
    """Render a tile and cache the results."""
    # Hash map lookup key for the cache
    key: Tuple[Any, ...] = (agent_dir, highlight, tile_size)
    key = (obj, 0, 0) + key if obj else key

    if key in cls.tile_cache:
      return cls.tile_cache[key]
    
    img = np.zeros(
      shape=(tile_size * subdivs, tile_size * subdivs, 3), 
      dtype=np.uint8
    )

    rendering.fill_coords(
      img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
    )
    rendering.fill_coords(
      img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
    )

    if obj == Tiles.WALL:
      rendering.fill_coords(
        img, rendering.point_in_rect(0, 1, 0, 1), (127, 127, 127) # GREY-ish
      )
    elif obj == Tiles.GOAL:
      rendering.fill_coords(
        img, rendering.point_in_rect(0, 1, 0, 1), (0, 255, 0) # GREEN
      )
    elif obj == Tiles.BALL:
      rendering.fill_coords(
        img, rendering.point_in_circle(0.5, 0.5, 0.31), (255, 165, 0) # ORANGE
      )
    elif obj == Tiles.TREE:
      # Tree leaves
      rendering.fill_coords(
        img, rendering.point_in_rect(0.1, 0.35, 0.1, 0.4), (141, 203, 105)
      )
      rendering.fill_coords(
        img, rendering.point_in_rect(0.35, 0.65, 0.1, 0.4), (122, 175, 90)
      )
      rendering.fill_coords(
        img, rendering.point_in_rect(0.65, 0.9, 0.1, 0.4), (123, 194, 80)
      )
      rendering.fill_coords(
        img, rendering.point_in_rect(0.1, 0.35, 0.4, 0.7), (161, 217, 127)
      )
      rendering.fill_coords(
        img, rendering.point_in_rect(0.65, 0.9, 0.4, 0.7), (102, 162, 67)
      )
      # Tree trunk
      rendering.fill_coords(
        img, rendering.point_in_rect(0.35, 0.65, 0.4, 0.7), (167, 106, 43)
      )
      rendering.fill_coords(
        img, rendering.point_in_rect(0.35, 0.65, 0.7, 0.9), (189, 128, 65)
      )
    
    if agent_dir is not None:
      tri_fn = rendering.point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
      )
      # Rotate the agent based on its direction
      tri_fn = rendering.rotate_fn(
        tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir
      )
      rendering.fill_coords(img, tri_fn, (229, 52, 52))
    
    if highlight: rendering.highlight_img(img)
    # Downsample the image to perform supersampling/anti-aliasing
    img = rendering.downsample(img, subdivs)
    # Cache the rendered tile
    cls.tile_cache[key] = img
    return img

