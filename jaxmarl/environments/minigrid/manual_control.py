from typing import Dict

import argparse
import pygame
import numpy as np

import jax
import jax.numpy as jnp
import chex

from jaxmarl.environments.minigrid.utils.minigrid_env import MiniGridEnv, State
from jaxmarl.environments.minigrid.utils.constants import Actions
from jaxmarl.environments.minigrid.utils.visualizer import MiniGridVisualizer


class ManualControl:
  def __init__(
    self, 
    env: MiniGridEnv, 
    screen_size: int = 640, 
    render_fps: int = 30,
    seed: int = 0,
  ) -> None:
    self._env = env
    self._viz = MiniGridVisualizer(self._env)

    self._reset = jax.jit(self._env.reset)
    self._step = jax.jit(self._env.step)
    self._key = jax.random.PRNGKey(seed)

    self.screen_size = screen_size
    self.render_fps = render_fps
    self.window = None
    self.clock = None
    self.closed = False
    self.aidx = "agent_0"
    self.step_count = None
  
  def reset(self) -> None:
    self._key, subkey = jax.random.split(self._key)
    _, env_state = self._reset(subkey)
    self.render(env_state)
    self.step_count = 0
    return env_state

  def step(self, state: State, actions: Dict[str, chex.Array]) -> None:
    self._key, subkey = jax.random.split(self._key)
    _, env_state, reward, done, _ = self._step(subkey, state, actions)
    print(f"step={self.step_count}, reward={sum(list(reward.values()))}, done={done['__all__']}")
    
    if done["__all__"]:
      print("terminated")
      self._key, subkey = jax.random.split(self._key)
      env_state = self.reset(subkey)
    else:
      self.render(env_state)
    
    return env_state

  def render(self, state: State) -> None:
    img = self._viz.render(state)
    img = np.transpose(img, axes=(1, 0, 2)) # (h, w, c) -> (w, h, c)
    
    if self.window is None:
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.screen_size, self.screen_size)
      )
      pygame.display.set_caption("minigrid")
    if self.clock is None:
      self.clock = pygame.time.Clock()
    surf = pygame.surfarray.make_surface(img)

    offset = surf.get_size()[0] * 0.025
    bg = pygame.Surface(
      (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
    )
    bg.convert()
    bg.fill((255, 255, 255))
    bg.blit(surf, (offset / 2, offset / 2))

    bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

    self.window.blit(bg, (0, 0))
    pygame.event.pump()
    self.clock.tick(self.render_fps)
    pygame.display.flip()
  
  def run(self) -> None:
    env_state = self.reset()
    """Start the window display with blocking event loop."""
    while not self.closed:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.close()
          break
        if event.type == pygame.KEYDOWN:
          event.key = pygame.key.name(int(event.key))
          env_state = self.key_handler(event, env_state)
    
  def key_handler(self, event, env_state) -> None:
    key: str = event.key
    print("pressed", key)

    if key == "escape":
      self.closed = True
      self.close()
      return
    if key == "backspace":
      env_state = self.reset()
      return env_state

    key_to_action = {
      "left": Actions.LEFT,
      "right": Actions.RIGHT,
      "up": Actions.FORWARD,
      "return": Actions.NOOP,
    }
    if key in key_to_action.keys():
      self.step_count += 1
      action = key_to_action[key]
      actions = {a: Actions.NOOP for a in self._env.agents}
      actions[self.aidx] = jnp.array(action)
      env_state = self.step(env_state, actions)
    
    if key in [str(i + 1) for i in self._env.itoa.keys()]:
      self.aidx = self._env.itoa[int(key) - 1]

    return env_state

  def close(self) -> None:
    if self.window is not None:
      pygame.quit()


if __name__ == "__main__":
  # NOTE: Until we don't have a proper register/make just testing by env-name import
  from jaxmarl.environments.minigrid.predatorprey import PredatorPreyEnv
  from jaxmarl.environments.minigrid.empty import EmptyEnv

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--screen-size",
    type=int,
    help="set the resolution for pygame rendering (width and height)",
    default=640
  )
  parser.add_argument(
    "--seed", 
    type=int, 
    help="random seed to generate the environment with", 
    default=0
  )
  args = parser.parse_args()

  env = PredatorPreyEnv()
  control = ManualControl(env, **vars(args))
  control.run()