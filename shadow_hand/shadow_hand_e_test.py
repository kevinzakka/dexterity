import numpy as np
from absl.testing import absltest, parameterized
from dm_control import mjcf

from shadow_hand import shadow_hand_e
from shadow_hand import shadow_hand_e_constants as consts


@parameterized.named_parameters(
    {
        "testcase_name": "position_control",
        "actuation": consts.Actuation.POSITION,
    },
)
class ShadowHandSeriesETest(absltest.TestCase):
    def test_physics_step(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
        physics.step()

    def test_joints(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        self.assertLen(hand.joints, consts.NUM_JOINTS)
        for joint in hand.joints:
            self.assertEqual(joint.tag, "joint")

    def test_actuators(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        self.assertLen(hand.actuators, consts.NUM_ACTUATORS)
        for actuator in hand.actuators:
            if actuation == consts.Actuation.POSITION:
                self.assertEqual(actuator.tag, "position")

    def test_mjcf_model(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        self.assertIsInstance(hand.mjcf_model, mjcf.RootElement)

    def test_zero_joint_pos(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        np.testing.assert_array_equal(
            hand.zero_joint_positions(),
            np.zeros(consts.NUM_JOINTS),
        )

    def test_zero_control(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        np.testing.assert_array_equal(
            hand.zero_control(),
            np.zeros(consts.NUM_ACTUATORS),
        )

    def test_control_to_joint_pos(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)

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

    def test_raises_when_control_wrong_len(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        control = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.control_to_joint_positions(control)

    def test_joint_pos_to_control(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)

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

    def test_raises_when_qpos_wrong_len(self, actuation: consts.Actuation) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE(actuation=actuation)
        qpos = np.array([0.0])
        with self.assertRaises(ValueError):
            hand.joint_positions_to_control(qpos)


if __name__ == "__main__":
    absltest.main()
