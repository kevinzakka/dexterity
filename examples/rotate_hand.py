import imageio
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_robotics.transformations import transformations as tr

from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e

enums = mjbindings.enums
mjlib = mjbindings.mjlib


def render(physics: mjcf.Physics, cam_id: str = "fixed_viewer") -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.frame = enums.mjtFrame.mjFRAME_GEOM
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


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


def main() -> None:
    # Build the arena.
    arena = _build_arena("hand_arena")

    # Load the hand and add it to the arena.
    attachment_site = arena.mjcf_model.worldbody.add(
        "site",
        type="sphere",
        pos=[0, 0, 0.08],
        rgba="0 0 0 0",
        size="0.01",
    )
    hand = shadow_hand_e.ShadowHandSeriesE()
    arena.attach(hand, attachment_site)

    # Compile.
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    frames = []
    axis_angle = np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    for mag in np.linspace(0, np.pi):
        hand.set_pose(physics, quaternion=tr.axisangle_to_quat(mag * axis_angle))
        physics.step()
        frames.append(render(physics))
    imageio.mimsave("temp/rotate_hand.mp4", frames, fps=30)


if __name__ == "__main__":
    main()
