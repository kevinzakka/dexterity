"""Tests for JugglingBall."""

from absl.testing import absltest
from dm_control import mjcf

from dexterity.manipulation.props import JugglingBall


class JugglingBallTest(absltest.TestCase):
    def test_can_compile_and_step_model(self) -> None:
        prop = JugglingBall()
        physics = mjcf.Physics.from_mjcf_model(prop.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()
