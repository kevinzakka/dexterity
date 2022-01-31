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


if __name__ == "__main__":
    absltest.main()
