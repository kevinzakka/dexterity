import numpy as np
from dm_control import composer  # , viewer
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


class SimpleTask(composer.Task):
    def __init__(self) -> None:
        self._arena = _build_arena("simple_task", disable_gravity=False)
        self._hand = _add_hand(self._arena)
        self._ball = self._arena.mjcf_model.worldbody.add(
            "body", name="ball", pos="0 -0.3 0.2"
        )
        self._ball.add("freejoint")
        self._ball.add(
            "geom", type="sphere", size="0.028", group="0", mass="0.043", condim="4"
        )

        self._hand.observables.enable_all()
        self._hand.observables.joint_torques.enabled = False

    @property
    def root_entity(self):
        return self._arena

    def get_reward(self, physics):
        return 0.0


if __name__ == "__main__":

    def get_env():
        task = SimpleTask()
        env = composer.Environment(task=task)
        return env

    # Launch the viewer application.
    # viewer.launch(get_env)

    env = get_env()
    timestep = env.reset()

    for key, value in timestep.observation.items():
        print(f"{key}: {value.shape}")
