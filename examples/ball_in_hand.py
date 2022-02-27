import dataclasses
from typing import List

import dcargs
import imageio
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e


@dataclasses.dataclass
class Args:
    compensate_gravity: bool = False


def render_scene(
    physics: mjcf.Physics, cam_id: str = "cam0", transparent: bool = False
) -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def plot(image: np.ndarray, window: str = "") -> None:
    plt.figure(window)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def animate(
    physics: mjcf.Physics,
    duration: float = 2.0,
    framerate: float = 30,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    while physics.data.time < duration:
        physics.step()
        if len(frames) < physics.data.time * framerate:
            pixels = render_scene(physics, transparent=False)
            frames.append(pixels)
    return frames


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


def main(args: Args) -> None:
    # Build the scene.
    arena = _build_arena("hand_ik", disable_gravity=False)
    hand = _add_hand(arena)

    # Add ball.
    ball = arena.mjcf_model.worldbody.add("body", name="ball", pos="0 -0.3 0.2")
    ball.add("freejoint")
    ball.add("geom", type="sphere", size="0.028", group="0", mass="0.043", condim="4")

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    if args.compensate_gravity:
        hand.compensate_gravity(physics)

    frames = animate(physics, duration=2.0)
    imageio.mimsave("temp/ball_in_hand.mp4", frames, fps=30)


if __name__ == "__main__":
    main(dcargs.parse(Args))
