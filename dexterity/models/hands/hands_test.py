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


class ShadowHandEConstantsTest(absltest.TestCase):
    def test_projection_matrices(self) -> None:
        # Matrix multiplication of these two matrices should be the identity.
        actual = (
            shadow_hand_e_constants.POSITION_TO_CONTROL
            @ shadow_hand_e_constants.CONTROL_TO_POSITION
        )
        expected = np.eye(shadow_hand_e_constants.NUM_ACTUATORS)
        np.testing.assert_array_equal(actual, expected)


class ShadowHandSeriesETest(parameterized.TestCase):
    def setUp(self) -> None:
        self.hand = shadow_hand_e.ShadowHandSeriesE()
        self.physics = mjcf.Physics.from_mjcf_model(self.hand.mjcf_model)

    def test_can_compile_and_step_model(self) -> None:
        for _ in range(100):
            self.physics.step()

    def test_initialize_episode(self) -> None:
        self.hand.initialize_episode(self.physics, np.random.RandomState(0))

    def test_set_name(self) -> None:
        name = "hand_of_glory"
        hand = shadow_hand_e.ShadowHandSeriesE(name=name)
        self.assertEqual(hand.mjcf_model.model, name)

    def test_joints(self) -> None:
        self.assertLen(self.hand.joints, shadow_hand_e_constants.NUM_JOINTS)
        for joint in self.hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self) -> None:
        self.assertLen(self.hand.actuators, shadow_hand_e_constants.NUM_ACTUATORS)
        for actuator in self.hand.actuators:
            self.assertEqual(actuator.tag, "position")

    def test_control_to_joint_pos(self) -> None:
        # Randomly generate a control for the hand.
        wr_ctrl = np.random.randn(2)
        ff_ctrl = np.random.randn(3)
        mf_ctrl = np.random.randn(3)
        rf_ctrl = np.random.randn(3)
        lf_ctrl = np.random.randn(4)
        th_ctrl = np.random.randn(5)
        control = np.concatenate(
            [
                wr_ctrl,
                ff_ctrl,
                mf_ctrl,
                rf_ctrl,
                lf_ctrl,
                th_ctrl,
            ]
        )

        # The qpos commands should be the same as the controls except for the coupled
        # joints. Those should have the control evenly split between them.
        def _split_last(ctrl: np.ndarray) -> np.ndarray:
            qpos = np.zeros((len(ctrl) + 1,))
            qpos[:-2] = ctrl[:-1]
            qpos[-2] = ctrl[-1] / 2
            qpos[-1] = ctrl[-1] / 2
            return qpos

        expected = np.concatenate(
            [
                wr_ctrl.copy(),
                _split_last(ff_ctrl),
                _split_last(mf_ctrl),
                _split_last(rf_ctrl),
                _split_last(lf_ctrl),
                th_ctrl.copy(),
            ]
        )

        actual = self.hand.control_to_joint_positions(control)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (shadow_hand_e_constants.NUM_JOINTS,))

    def test_raises_when_control_wrong_len(self) -> None:
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.control_to_joint_positions(control)

    def test_joint_pos_to_control(self) -> None:
        # Randomly generate joint positions for the hand.
        wr_qpos = np.random.randn(2)
        ff_qpos = np.random.randn(4)
        mf_qpos = np.random.randn(4)
        rf_qpos = np.random.randn(4)
        lf_qpos = np.random.randn(5)
        th_qpos = np.random.randn(5)
        qpos = np.concatenate(
            [
                wr_qpos,
                ff_qpos,
                mf_qpos,
                rf_qpos,
                lf_qpos,
                th_qpos,
            ]
        )

        # The control commands should be the same as the qpos except for the coupled
        # joints. Those should have the qpos summed over them.
        def _sum_last(qpos: np.ndarray) -> np.ndarray:
            ctrl = np.zeros((len(qpos) - 1,))
            ctrl[:-1] = qpos[:-2]
            ctrl[-1] = qpos[-1] + qpos[-2]
            return ctrl

        expected = np.concatenate(
            [
                wr_qpos.copy(),
                _sum_last(ff_qpos),
                _sum_last(mf_qpos),
                _sum_last(rf_qpos),
                _sum_last(lf_qpos),
                th_qpos.copy(),
            ]
        )

        actual = self.hand.joint_positions_to_control(qpos)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (shadow_hand_e_constants.NUM_ACTUATORS,))

    def test_raises_when_qpos_wrong_len(self) -> None:
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.joint_positions_to_control(qpos)

    def test_set_joint_angles(self) -> None:
        rand_qpos = np.random.uniform(*self.physics.bind(self.hand.joints).range.T)
        self.hand.set_joint_angles(self.physics, rand_qpos)
        physics_joints_qpos = self.physics.bind(self.hand.joints).qpos
        np.testing.assert_array_equal(physics_joints_qpos, rand_qpos)


class AdroitHandTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.hand = adroit_hand.AdroitHand()
        self.physics = mjcf.Physics.from_mjcf_model(self.hand.mjcf_model)

    def test_can_compile_and_step_model(self) -> None:
        for _ in range(100):
            self.physics.step()

    def test_initialize_episode(self) -> None:
        self.hand.initialize_episode(self.physics, np.random.RandomState(0))

    def test_set_name(self) -> None:
        name = "hand_of_glory"
        hand = adroit_hand.AdroitHand(name=name)
        self.assertEqual(hand.mjcf_model.model, name)

    def test_joints(self) -> None:
        self.assertLen(self.hand.joints, adroit_hand_constants.NUM_JOINTS)
        for joint in self.hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self) -> None:
        self.assertLen(self.hand.actuators, adroit_hand_constants.NUM_ACTUATORS)
        for actuator in self.hand.actuators:
            self.assertEqual(actuator.tag, "general")

    def test_raises_when_control_wrong_len(self) -> None:
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.control_to_joint_positions(control)

    def test_raises_when_qpos_wrong_len(self) -> None:
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.joint_positions_to_control(qpos)

    def test_set_joint_angles(self) -> None:
        rand_qpos = np.random.uniform(*self.physics.bind(self.hand.joints).range.T)
        self.hand.set_joint_angles(self.physics, rand_qpos)
        physics_joints_qpos = self.physics.bind(self.hand.joints).qpos
        np.testing.assert_array_equal(physics_joints_qpos, rand_qpos)


class MPLHandTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.hand = mpl_hand.MPLHand()
        self.physics = mjcf.Physics.from_mjcf_model(self.hand.mjcf_model)

    def test_can_compile_and_step_model(self) -> None:
        for _ in range(100):
            self.physics.step()

    def test_initialize_episode(self) -> None:
        self.hand.initialize_episode(self.physics, np.random.RandomState(0))

    def test_set_name(self) -> None:
        name = "hand_of_glory"
        hand = mpl_hand.MPLHand(name=name)
        self.assertEqual(hand.mjcf_model.model, f"right_{name}")

    def test_joints(self) -> None:
        self.assertLen(self.hand.joints, mpl_hand_constants.NUM_JOINTS)
        for joint in self.hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self) -> None:
        self.assertLen(self.hand.actuators, mpl_hand_constants.NUM_ACTUATORS)
        for actuator in self.hand.actuators:
            self.assertEqual(actuator.tag, "general")

    def test_raises_when_control_wrong_len(self) -> None:
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.control_to_joint_positions(control)

    def test_raises_when_qpos_wrong_len(self) -> None:
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            self.hand.joint_positions_to_control(qpos)

    def test_set_joint_angles(self) -> None:
        rand_qpos = np.random.uniform(*self.physics.bind(self.hand.joints).range.T)
        self.hand.set_joint_angles(self.physics, rand_qpos)
        physics_joints_qpos = self.physics.bind(self.hand.joints).qpos
        np.testing.assert_array_equal(physics_joints_qpos, rand_qpos)


if __name__ == "__main__":
    absltest.main()
