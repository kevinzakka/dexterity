"""Place fingertip locations at target sites using inverse kinematics.

Example usage:
$ python examples/inverse_kinematics.py --num_solves 2 --linear_tol 1e-3
"""

import time

import numpy as np
from absl import app
from absl import flags
from dm_control import mjcf
from matplotlib import pyplot as plt

from dexterity.inverse_kinematics import ik_solver
from dexterity.manipulation.shared import cameras
from dexterity.manipulation.shared import workspaces
from dexterity.models import hands
from dexterity.models.arenas import Arena

flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_solves", 1, "Number of IK solves.")
flags.DEFINE_float("linear_tol", 1e-3, "Linear tolerance.")
flags.DEFINE_boolean("disable_plot", False, "Angular tolerance.")
flags.DEFINE_boolean("optimize_nullspace", False, "Nullspace optimization.")

FLAGS = flags.FLAGS

_SITE_SIZE = 1e-2
_SITE_ALPHA = 0.1
_SITE_COLORS = (
    (1.0, 0.0, 0.0),  # Red.
    (0.0, 1.0, 0.0),  # Green.
    (0.0, 0.0, 1.0),  # Blue.
    (0.0, 1.0, 1.0),  # Cyan.
    (1.0, 0.0, 1.0),  # Magenta.
)
_TARGET_SIZE = 5e-3
_TARGET_ALPHA = 1.0


def main(_) -> None:
    random_state = np.random.RandomState(seed=FLAGS.seed)

    # Build the scene.
    arena = Arena()
    hand = hands.ShadowHandSeriesE()
    arena.attach_offset(
        hand,
        position=hand.palm_upright_pose.xpos,
        quaternion=hand.palm_upright_pose.xquat,
    )

    # Add camera.
    arena.mjcf_model.worldbody.add(
        "camera",
        name=cameras.FRONT_CLOSE.name,
        pos=cameras.FRONT_CLOSE.pos,
        xyaxes=cameras.FRONT_CLOSE.xyaxes,
    )

    solver = ik_solver.IKSolver(hand=hand)

    # Create target sites for each fingertip.
    target_sites = []
    for i, site in enumerate(hand.fingertip_sites):
        target_sites.append(
            workspaces.add_target_site(
                body=arena.mjcf_model.worldbody,
                radius=_TARGET_SIZE,
                visible=True,
                rgba=_SITE_COLORS[i] + (_TARGET_ALPHA,),
                name=f"target_{site.name}",
            )
        )

    # Customize hand fingertip sites.
    for i, site in enumerate(hand.fingertip_sites):
        site.group = None  # Make the sites visible.
        site.size = (_SITE_SIZE,) * 3  # Increase their size.
        site.rgba = _SITE_COLORS[i] + (_SITE_ALPHA,)  # Change their color.

    render_kwargs = dict(width=640, height=480, camera_id=0)

    successes: int = 0
    for _ in range(FLAGS.num_solves):
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

        # Randomly sample a joint configuration.
        qpos_initial = physics.bind(hand.joints).qpos.copy()
        qpos_desired = hand.sample_collision_free_joint_angles(physics, random_state)
        physics.bind(hand.joints).qpos[:] = qpos_desired

        # Forward kinematics to compute Cartesian fingertip positions.
        target_positions = physics.bind(hand.fingertip_sites).xpos.copy()

        # Set target sites to their respective locations.
        physics.bind(target_sites).pos = target_positions

        # Restore joints to their initial configuration.
        physics.bind(hand.joints).qpos[:] = qpos_initial

        # Forward dynamics to update the target site positions.
        physics.forward()
        im_start = physics.render(**render_kwargs)

        ik_start = time.time()
        qpos = solver.solve(
            target_positions=target_positions,
            linear_tol=FLAGS.linear_tol,
            early_stop=True,
            stop_on_first_successful_attempt=(not FLAGS.optimize_nullspace),
        )
        ik_end = time.time()

        if qpos is not None:
            print(f"Full IK solved in {(ik_end - ik_start) * 1000} ms.")

            # Directly set joint angles and visualize.
            hand.set_joint_angles(physics, qpos)
            physics.forward()
            im_actual = physics.render(**render_kwargs)

            if not FLAGS.disable_plot:
                _, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].imshow(im_start)
                axes[0].set_title("Starting")
                axes[1].imshow(im_actual)
                axes[1].set_title("Solution")
                for ax in axes:
                    ax.axis("off")
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                plt.show()

            successes += 1

    print(f"solve success rate: {successes}/{FLAGS.num_solves}.")


if __name__ == "__main__":
    app.run(main)
