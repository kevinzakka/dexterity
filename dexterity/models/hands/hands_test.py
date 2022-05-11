import itertools
from typing import Callable, Tuple

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from dexterity.inverse_kinematics import ik_solver
from dexterity.models.hands import adroit_hand
from dexterity.models.hands import adroit_hand_constants
from dexterity.models.hands import dexterous_hand
from dexterity.models.hands import mpl_hand
from dexterity.models.hands import mpl_hand_constants
from dexterity.models.hands import shadow_hand_e
from dexterity.models.hands import shadow_hand_e_constants
from dexterity.utils import mujoco_collisions

HandCls = Callable[[], dexterous_hand.DexterousHand]


@parameterized.named_parameters(
    {"testcase_name": "shadow_hand", "constants": shadow_hand_e_constants},
    {"testcase_name": "mpl_hand", "constants": mpl_hand_constants},
)
class ConstantsTest(absltest.TestCase):
    def test_projection_matrices(self, constants) -> None:
        # Matrix multiplication of these two matrices should be the identity.
        actual = constants.POSITION_TO_CONTROL @ constants.CONTROL_TO_POSITION
        expected = np.eye(constants.NUM_ACTUATORS)
        np.testing.assert_array_equal(actual, expected)


@parameterized.named_parameters(
    {
        "testcase_name": "shadow_hand",
        "hand_cls": shadow_hand_e.ShadowHandSeriesE,
        "constants": shadow_hand_e_constants,
    },
    {
        "testcase_name": "adroit_hand",
        "hand_cls": adroit_hand.AdroitHand,
        "constants": adroit_hand_constants,
    },
    {
        "testcase_name": "mpl_hand",
        "hand_cls": mpl_hand.MPLHand,
        "constants": mpl_hand_constants,
    },
)
class HandTest(parameterized.TestCase):
    def test_can_compile_and_step_model(self, hand_cls: HandCls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        for _ in range(100):
            physics.step()

    def test_initialize_episode(self, hand_cls: HandCls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        hand.initialize_episode(physics, np.random.RandomState(0))

    def test_joints(self, hand_cls: HandCls, constants) -> None:
        hand = hand_cls()
        self.assertLen(hand.joints, constants.NUM_JOINTS)
        for joint in hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self, hand_cls: HandCls, constants) -> None:
        hand = hand_cls()
        self.assertLen(hand.actuators, constants.NUM_ACTUATORS)
        for actuator in hand.actuators:
            self.assertIn(actuator.tag, ["general", "position"])

    def test_raises_when_control_wrong_len(self, hand_cls: HandCls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.control_to_joint_positions(control)

    def test_raises_when_qpos_wrong_len(self, hand_cls: HandCls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.joint_positions_to_control(qpos)

    def test_set_joint_angles(self, hand_cls: HandCls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        rand_qpos = np.random.uniform(*physics.bind(hand.joints).range.T)
        hand.set_joint_angles(physics, rand_qpos)
        physics_joints_qpos = physics.bind(hand.joints).qpos
        np.testing.assert_array_equal(physics_joints_qpos, rand_qpos)

    def test_sample_collision_free_joint_angles(
        self, hand_cls: HandCls, constants
    ) -> None:
        del constants  # Unused.
        hand = hand_cls()
        random_state = np.random.RandomState(12345)
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        for _ in range(50):
            rand_qpos = hand.sample_collision_free_joint_angles(physics, random_state)
            hand.set_joint_angles(physics, rand_qpos)
            self.assertFalse(mujoco_collisions.has_self_collision(physics, hand.name))


class DexterousHandObservablesTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(joint_index=0, joint_pos=0),
        dict(joint_index=1, joint_pos=0.5),
    )
    def test_joint_positions_observable(
        self, joint_index: int, joint_pos: float
    ) -> None:
        hand = adroit_hand.AdroitHand()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        physics.bind(hand.joints).qpos[joint_index] = joint_pos
        actual_obs = hand.observables.joint_positions(physics)[joint_index]
        np.testing.assert_array_almost_equal(actual_obs, joint_pos)

    @parameterized.parameters(
        dict(joint_index=0, joint_pos=0, expected_obs=(0.0, 1.0)),
        dict(
            joint_index=0, joint_pos=0.175, expected_obs=(np.sin(0.175), np.cos(0.175))
        ),
    )
    def test_joint_positions_sin_cos_observable(
        self, joint_index: int, joint_pos: float, expected_obs: Tuple[float, float]
    ) -> None:
        hand = adroit_hand.AdroitHand()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        physics.bind(hand.joints).qpos[joint_index] = joint_pos
        actual_obs = hand.observables.joint_positions_sin_cos(physics).reshape(-1, 2)[
            joint_index
        ]
        np.testing.assert_array_almost_equal(actual_obs, expected_obs)

    @parameterized.parameters(
        dict(joint_index=0, joint_vel=0),
        dict(joint_index=1, joint_vel=0.5),
    )
    def test_joint_velocities_observable(
        self,
        joint_index: int,
        joint_vel: float,
    ) -> None:
        hand = adroit_hand.AdroitHand()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        physics.bind(hand.joints).qvel[joint_index] = joint_vel
        actual_obs = hand.observables.joint_velocities(physics)[joint_index]
        np.testing.assert_array_almost_equal(actual_obs, joint_vel)

    @parameterized.parameters(
        dict(joint_index=idx, applied_torque=t)
        for idx, t in itertools.product([0, 2, 4], [0, -6, 5])
    )
    def test_joint_torques_observable(
        self, joint_index: int, applied_torque: float
    ) -> None:

        hand = adroit_hand.AdroitHand()
        joint = hand.joints[joint_index]
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)

        with physics.model.disable("contact", "gravity", "actuation"):
            # Project the torque onto the joint axis and apply it to the joint's parent
            # body.
            physics.bind(joint.parent).xfrc_applied[3:] = (
                applied_torque * physics.bind(joint).xaxis
            )

            # Run the simulation forward until the joint stops moving.
            physics.step()
            qvel_thresh = 1e-2
            while max(abs(physics.bind(joint).qvel)) > qvel_thresh:
                physics.step()

            # Read the torque sensor reading.
            observed_torque = hand.observables.joint_torques(physics)[joint_index]

            # Flip the sign since the sensor reports values in the child->parent
            # direction.
            observed_torque = -1.0 * observed_torque

        # Note the change in sign, since the sensor measures torques in the
        # child->parent direction.
        self.assertAlmostEqual(observed_torque, applied_torque, delta=1e-2)

    @parameterized.parameters(
        dict(
            fingertip_positions=np.asarray(
                [
                    [-0.003572, -0.020904, 0.371999],
                    [-0.028277, -0.036063, 0.391271],
                    [-0.052305, -0.006066, 0.393481],
                    [-0.089808, -0.042816, 0.423813],
                    [0.026246, -0.017261, 0.416314],
                ]
            )
        )
    )
    def test_fingertip_positions_observable(
        self, fingertip_positions: np.ndarray
    ) -> None:
        hand = adroit_hand.AdroitHand()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)

        solver = ik_solver.IKSolver(hand)

        qpos = solver.solve(
            target_positions=fingertip_positions,
            linear_tol=1e-3,
            early_stop=True,
            stop_on_first_successful_attempt=True,
        )
        self.assertIsNotNone(qpos)

        assert qpos is not None  # Appease mypy.
        hand.set_joint_angles(physics, qpos)

        observed_pos = hand.observables.fingertip_positions(physics).reshape(-1, 3)
        np.testing.assert_allclose(observed_pos, fingertip_positions, atol=1e-3)


if __name__ == "__main__":
    absltest.main()
