import collections

import numpy as np
from dm_control import mjcf, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_robotics.transformations import transformations as tr

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e


def _build_arena(name: str, disable_gravity: bool = False) -> Arena:
    arena = Arena(name)
    arena.mjcf_model.option.timestep = 0.001
    if disable_gravity:
        arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)
    else:
        arena.mjcf_model.option.gravity = (0.0, 0.0, -9.81)
    arena.mjcf_model.size.nconmax = 1_000
    arena.mjcf_model.size.njmax = 2_000
    arena.mjcf_model.visual.__getattr__("global").offheight = 480
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 5e-4
    return arena


def _add_hand(arena: Arena) -> shadow_hand_e.ShadowHandSeriesE:
    axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    quat = tr.axisangle_to_quat(axis_angle)
    attachment_site = arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=[0, 0, 0.1],
        quat=quat,
        rgba="0 0 0 0",
        size="0.01",
    )
    hand = shadow_hand_e.ShadowHandSeriesE()
    arena.attach(hand, attachment_site)
    return hand


class SimpleTask(base.Task):
    def __init__(self, random=None) -> None:
        self._arena = _build_arena("simple_task", disable_gravity=False)
        self._hand = _add_hand(self._arena)
        self._ball = self._arena.mjcf_model.worldbody.add(
            "body", name="ball", pos="0 -0.3 0.2"
        )
        self._ball.add("freejoint")
        self._ball.add(
            "geom", type="sphere", size="0.028", group="0", mass="0.043", condim="4"
        )
        super().__init__(random)

    def initialize_episode(self, physics):
        return super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        return obs

    def get_reward(self, physics) -> float:
        return 0.0

    def action_spec(self, physics):
        return super().action_spec(physics)


if __name__ == "__main__":

    def get_env():
        task = SimpleTask()
        physics = mjcf.Physics.from_mjcf_model(task._arena.mjcf_model)
        env = control.Environment(physics=physics, task=task)
        return env

    # Launch the viewer application.
    viewer.launch(get_env)
