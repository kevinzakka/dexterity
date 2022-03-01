import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf

from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.models.hands import shadow_hand_e_constants as consts


class ShadowHandEConstantsTest(absltest.TestCase):
    def test_projection_matrices(self) -> None:
        # Matrix multiplication of these two matrices should be the identity.
        actual = consts.POSITION_TO_CONTROL @ consts.CONTROL_TO_POSITION
        expected = np.eye(consts.NUM_ACTUATORS)
        np.testing.assert_array_equal(actual, expected)


class ShadowHandSeriesETest(parameterized.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        for _ in range(100):
            physics.step()

    def test_set_name(self) -> None:
        name = "hand_of_glory"
        hand = shadow_hand_e.ShadowHandSeriesE(name=name)
        self.assertEqual(hand.mjcf_model.model, name)

    def test_joints(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        self.assertLen(hand.joints, consts.NUM_JOINTS)
        for joint in hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        self.assertLen(hand.actuators, consts.NUM_ACTUATORS)
        for actuator in hand.actuators:
            self.assertEqual(actuator.tag, "general")

    def test_mjcf_model(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        self.assertIsInstance(hand.mjcf_model, mjcf.RootElement)

    def test_control_to_joint_pos(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()

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

        actual = hand.control_to_joint_positions(control)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (consts.NUM_JOINTS,))

    def test_raises_when_control_wrong_len(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.control_to_joint_positions(control)

    def test_joint_pos_to_control(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()

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

        actual = hand.joint_positions_to_control(qpos)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (consts.NUM_ACTUATORS,))

    def test_raises_when_qpos_wrong_len(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.joint_positions_to_control(qpos)


if __name__ == "__main__":
    absltest.main()
