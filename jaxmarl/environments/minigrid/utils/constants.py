from enum import IntEnum
import jax.numpy as jnp


class Tiles(IntEnum):
  EMPTY = 0
  WALL = 1
  GOAL = 2
  BALL = 3
  TREE = 4
  END = 127


class Actions(IntEnum):
  LEFT = 0
  RIGHT = 1
  FORWARD = 2
  INTERACT = 3
  NOOP = 4


DIRECTIONS = jnp.array((
  (0, 1), # RIGHT (positive X)
  (1, 0), # DOWN (positive Y)
  (0, -1), # LEFT (negative X)
  (-1, 0), # UP (negative Y)
))


WALKABLE = jnp.array((
  Tiles.EMPTY,
  Tiles.GOAL,
))