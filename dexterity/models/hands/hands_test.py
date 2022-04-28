import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from dexterity.models.hands import adroit_hand
from dexterity.models.hands import adroit_hand_constants
from dexterity.models.hands import mpl_hand
from dexterity.models.hands import mpl_hand_constants
from dexterity.models.hands import shadow_hand_e
from dexterity.models.hands import shadow_hand_e_constants


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
    def test_can_compile_and_step_model(self, hand_cls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        for _ in range(100):
            physics.step()

    def test_initialize_episode(self, hand_cls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        hand.initialize_episode(physics, np.random.RandomState(0))

    def test_joints(self, hand_cls, constants) -> None:
        hand = hand_cls()
        self.assertLen(hand.joints, constants.NUM_JOINTS)
        for joint in hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self, hand_cls, constants) -> None:
        hand = hand_cls()
        self.assertLen(hand.actuators, constants.NUM_ACTUATORS)
        for actuator in hand.actuators:
            self.assertIn(actuator.tag, ["general", "position"])

    def test_raises_when_control_wrong_len(self, hand_cls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.control_to_joint_positions(control)

    def test_raises_when_qpos_wrong_len(self, hand_cls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.joint_positions_to_control(qpos)

    def test_set_joint_angles(self, hand_cls, constants) -> None:
        del constants  # Unused.
        hand = hand_cls()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        rand_qpos = np.random.uniform(*physics.bind(hand.joints).range.T)
        hand.set_joint_angles(physics, rand_qpos)
        physics_joints_qpos = physics.bind(hand.joints).qpos
        np.testing.assert_array_equal(physics_joints_qpos, rand_qpos)


if __name__ == "__main__":
    absltest.main()
