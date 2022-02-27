import matplotlib.pyplot as plt
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums

from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import mujoco_utils


def main() -> None:
    hand = shadow_hand_e.ShadowHandSeriesE()
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)

    # Joint positions.
    joint_pos = physics.bind(hand.joints).qpos
    assert joint_pos.shape == (consts.NUM_JOINTS,)

    # Joint velocities.
    joint_vel = physics.bind(hand.joints).qvel
    assert joint_vel.shape == (consts.NUM_JOINTS,)

    # Joint torques.
    # Mujoco torques are 3-axis, but we only care about torques acting on the axis of
    # rotation.
    torques = physics.bind(hand.joint_torque_sensors).sensordata
    joint_axes = physics.bind(hand.joints).axis
    joint_torques = np.einsum("ij,ij->i", torques.reshape(-1, 3), joint_axes)
    assert joint_torques.shape == (consts.NUM_JOINTS,)

    # Fingertip poses, in the world frame.
    poses = []
    for fingertip_site in hand.fingertip_sites:
        pose = mujoco_utils.get_site_pose(physics, fingertip_site)
        poses.append(pose)
        print(f"{fingertip_site.name} position", pose[:3, 3])

    # Fingertip velocities, in the world frame.
    velocities = []
    for fingertip_site in hand.fingertip_sites:
        velocity = mujoco_utils.get_site_velocity(physics, fingertip_site)
        velocities.append(velocity)
        print(f"{fingertip_site.name} velocity", velocity)

    for fingertip_site in hand.fingertip_sites:
        # Get pose of fingertip relative to itself. This should be the 4x4 identity.
        pose = mujoco_utils.get_site_relative_pose(
            physics,
            fingertip_site,
            fingertip_site,
        )
        assert np.allclose(pose, np.eye(4))

    # Render and visualize sites by making model transparent.
    # Fingertip sites are red.
    # Joint torque sensors are green.
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = True
    pixels = physics.render(
        width=640,
        height=480,
        camera_id="cam1",
        scene_option=scene_option,
    )
    plt.imshow(pixels)
    plt.show()


if __name__ == "__main__":
    main()
