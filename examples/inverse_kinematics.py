"""Placing fingertip locations at target sites using inverse kinematics."""

import time

import numpy as np
from absl import app
from absl import flags
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco import wrapper
from dm_robotics.transformations import transformations as tr
from matplotlib import pyplot as plt

from shadow_hand.inverse_kinematics import ik_solver
from shadow_hand.manipulation.arenas import Standard
from shadow_hand.manipulation.shared import cameras
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_integer("num_solves", 1, "Number of IK solves.")
flags.DEFINE_float("linear_tol", 1e-4, "Linear tolerance.")
flags.DEFINE_boolean("disable_plot", False, "Angular tolerance.")

FLAGS = flags.FLAGS


def render_scene(
    physics: mjcf.Physics,
    transparent: bool = False,
    cam_id: str = cameras.FRONT_CLOSE.name,
) -> np.ndarray:
    scene_option = wrapper.core.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640,
        height=480,
        camera_id=cam_id,
        scene_option=scene_option,
    )


def main(_) -> None:
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)

    successes: int = 0
    for _ in range(FLAGS.num_solves):
        # Build the scene.
        arena = Standard()
        axis_angle = np.radians(180) * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        quat = tr.axisangle_to_quat(axis_angle)
        hand = shadow_hand_e.ShadowHandSeriesE()
        arena.attach_offset(hand, position=(0, 0.2, 0.1), quaternion=quat)

        # Add camera.
        arena.mjcf_model.worldbody.add(
            "camera",
            name=cameras.FRONT_CLOSE.name,
            pos=cameras.FRONT_CLOSE.pos,
            xyaxes=cameras.FRONT_CLOSE.xyaxes,
        )

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
                size="0.005",
            )

        # Recreate physics instance since we changed the MJCF.
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        im_start = render_scene(physics, transparent=False)

        ik_start = time.time()
        qpos = solver.solve(
            target_positions=target_positions,
            linear_tol=FLAGS.linear_tol,
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

            if not FLAGS.disable_plot:
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

            successes += 1
        else:
            plt.figure("failed")
            plt.imshow(im_desired)
            plt.axis("off")
            plt.show()

    print(f"solve success rate: {successes}/{FLAGS.num_solves}.")


if __name__ == "__main__":
    app.run(main)
