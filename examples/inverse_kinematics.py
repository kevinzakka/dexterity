"""Placing fingertip locations at target sites using inverse kinematics."""

import dataclasses
import time
from typing import List, Optional

import dcargs
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.ik import ik_solver
from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


def render_scene(
    physics: mjcf.Physics, cam_id: str = "closeup", transparent: bool = False
) -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640,
        height=480,
        camera_id=cam_id,
        scene_option=scene_option,
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
            pixels = render_scene(physics, transparent=True)
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


def _add_hand(arena: Arena):
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
    hand = shadow_hand_e.ShadowHandSeriesE(actuation=consts.Actuation.POSITION)
    arena.attach(hand, attachment_site)
    return hand


@dataclasses.dataclass
class Args:
    seed: Optional[int] = None
    num_solves: int = 1
    linear_tol: float = 1e-3
    disable_plot: bool = False


def main(args: Args) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)

    successes: int = 0
    for _ in range(args.num_solves):
        # Build the scene.
        arena = _build_arena("shadow_hand_inverse_kinematics")
        hand = _add_hand(arena)

        # The fingers we'd like to control.
        fingers = (
            consts.Components.FF,
            consts.Components.MF,
            consts.Components.RF,
            consts.Components.LF,
            consts.Components.TH,
        )

        solver = ik_solver.IKSolver(
            model=arena.mjcf_model,
            fingers=fingers,
            prefix=hand.mjcf_model.model,
        )

        # Set the configuration and query fingertip sites.
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        for finger in fingers:
            cntrl_jnts = solver._controllable_joints[finger]
            joint_binding = physics.bind(cntrl_jnts)
            joint_binding.qpos[:] = np.random.uniform(
                joint_binding.range[:, 0],
                joint_binding.range[:, 1],
            )
        target_positions = {}
        for finger, fingertip_site in hand._fingertip_site_elem_mapping.items():
            if finger in fingers:
                target_positions[finger] = physics.bind(fingertip_site).xpos.copy()
        assert set(tuple(target_positions.keys())) == set(fingers)
        im_desired = render_scene(physics, transparent=False)

        # Add the target sites to the MJCF model for visualization purposes.
        for name, position in target_positions.items():
            arena.mjcf_model.worldbody.add(
                "site",
                name=f"{name}_target",
                type="sphere",
                pos=position,
                rgba="0 0 1 .7",
                size="0.001",
            )

        # Recreate physics instance since we changed the MJCF.
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        im_start = render_scene(physics, transparent=False)

        ik_start = time.time()
        qpos = solver.solve(
            target_positions=target_positions,
            linear_tol=args.linear_tol,
            max_steps=1_000,
            early_stop=True,
            num_attempts=15,
            stop_on_first_successful_attempt=True,
        )
        ik_end = time.time()

        if qpos is not None:
            solve_time_ms = (ik_end - ik_start) * 1000
            print(f"Full IK solved in {solve_time_ms} ms.")

            # Directly set joint angles and visualize.
            hand.set_joint_angles(physics, qpos)
            physics.step()
            im_actual = render_scene(physics, transparent=False)
            im_actual_tr = render_scene(physics, transparent=True)

            if not args.disable_plot:
                _, axes = plt.subplots(1, 4, figsize=(12, 4))
                axes[0].imshow(im_start)
                axes[0].set_title("Starting")
                axes[1].imshow(im_desired)
                axes[1].set_title("Desired")
                axes[2].imshow(im_actual)
                axes[2].set_title("Actual")
                axes[3].imshow(im_actual_tr)
                axes[3].set_title("Actual (transparent)")
                for ax in axes:
                    ax.axis("off")
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                plt.show()
                plt.close()

                plt.figure()
                plt.imshow(im_actual_tr)
                plt.show()
                plt.close()

            successes += 1
        else:
            plot(im_desired, "failed")

    print(f"solve success rate: {successes}/{args.num_solves}.")


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
