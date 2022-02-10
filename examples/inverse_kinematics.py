from typing import Dict, List, Tuple

import imageio
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_robotics.transformations import transformations as tr

from shadow_hand.ik import differential_ik
from shadow_hand.models.arenas.empty import Arena
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts

TARGET_POSITIONS: Dict[consts.Components, Tuple[float, float, float]] = {
    consts.Components.RF: (0.01, -0.41, 0.16),
    consts.Components.MF: (-0.01, -0.41, 0.16),
    consts.Components.FF: (-0.03, -0.41, 0.16),
}


def render(
    physics: mjcf.Physics, cam_id: str = "fixed_viewer1", transparent: bool = False
) -> np.ndarray:
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    return physics.render(
        width=640, height=480, camera_id=cam_id, scene_option=scene_option
    )


def animate(
    physics: mjcf.Physics,
    duration: float = 2.0,
    framerate: float = 30,
) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    while physics.data.time < duration:
        physics.step()
        if len(frames) < physics.data.time * framerate:
            pixels = render(physics)
            frames.append(pixels)
    return frames


def main() -> None:
    # Build the arena.
    arena = Arena("hand_arena")
    arena.mjcf_model.option.gravity = (0.0, 0.0, 0.0)  # Disable gravity.
    arena.mjcf_model.size.nconmax = 1_000
    arena.mjcf_model.size.njmax = 2_000

    # Create sites for target fingertip positions.
    target_site_elems = []
    for name, position in TARGET_POSITIONS.items():
        site_elem = arena.mjcf_model.worldbody.add(
            "site",
            name=f"{name}_target",
            type="sphere",
            pos=position,
            rgba="0 0 1 1",
            size="0.005",
        )
        target_site_elems.append(site_elem)

    axis_angle = np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
    quat = tr.axisangle_to_quat(axis_angle)

    # Load the hand and add it to the arena.
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

    # Get elem associated with fingertip.
    finger = consts.Components.FF
    fingertip_name = finger.name.lower() + "tip"
    fingertip_site_name = f"{hand.mjcf_model.model}/{fingertip_name}_site"
    fingertip_site_elem = arena.mjcf_model.find("site", fingertip_site_name)
    assert fingertip_site_elem is not None

    # Get controllable joints for the hand given finger pose.
    wrist_joints = consts.JOINT_GROUP[consts.Components.WR]
    finger_joints = consts.JOINT_GROUP[finger]
    joints = finger_joints + wrist_joints
    controllable_joints = []
    for joint, joint_elem in hand._joint_elem_mapping.items():
        if joint in joints:
            controllable_joints.append(joint_elem)
    assert len(controllable_joints) == len(joints)

    wrist_actuators = consts.ACTUATOR_GROUP[consts.Components.WR]
    finger_actuators = consts.ACTUATOR_GROUP[finger]
    actuators = finger_actuators + wrist_actuators
    controllable_actuators = []
    for actuator, actuator_elem in hand._actuator_elem_mapping.items():
        if actuator in actuators:
            controllable_actuators.append(actuator_elem)
    assert len(controllable_actuators) == len(actuators)

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    physics_joints = physics.bind(controllable_joints)

    controller = differential_ik.DifferentialIK(
        model=arena.mjcf_model,
        controllable_joints=controllable_joints,
        site_name=fingertip_site_name,
    )

    # Solve.
    target_position = np.array(TARGET_POSITIONS[finger])
    qpos = controller.solve(
        target_position=target_position,
        linear_tol=1e-6,
        max_steps=100,
        early_stop=True,
        num_attempts=30,
        stop_on_first_successful_attempt=False,
        inital_joint_configuration=None,
        nullspace_reference=None,
        regularization_weight=1e-3,
    )

    # # Directly set joint angles.
    # physics_joints = physics.bind(controllable_joints)
    # physics_joints.qpos[:] = qpos
    # physics.step()

    # Command the actuators.
    joint_angles = np.zeros(len(hand.joints))
    joint_angles[physics_joints.dofadr] = qpos
    ctrl = hand.joint_positions_to_control(joint_angles)
    hand.set_position_control(physics, ctrl)
    frames = animate(physics, duration=5.0)
    imageio.mimsave("temp/differential_kinematics.mp4", frames, fps=30)


if __name__ == "__main__":
    main()
