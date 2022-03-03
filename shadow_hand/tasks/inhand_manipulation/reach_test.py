"""Tests for inhand_manipulation.reach."""

import numpy as np
from absl.testing import absltest

from shadow_hand.tasks import inhand_manipulation


class ReachTest(absltest.TestCase):
    def test_observables(self) -> None:
        env = inhand_manipulation.load(environment_name="reach", seed=12345)

        timestep = env.reset()

        for key in [
            "shadow_hand_e/joint_positions",
            "shadow_hand_e/joint_velocities",
            "shadow_hand_e/fingerip_positions",
            "shadow_hand_e/fingertip_linear_velocities",
            "target_positions",
            "fingertip_positions",
        ]:
            self.assertIn(key, timestep.observation)

    def test_reward(self) -> None:
        env = inhand_manipulation.load(environment_name="reach", seed=12345)
        env.reset()

        zero_action = np.zeros_like(env.physics.data.ctrl)
        timestep = env.step(zero_action)

        target_positions = timestep.observation["target_positions"]
        fingertip_positions = timestep.observation["fingertip_positions"]
        expected_reward = -1.0 * np.linalg.norm(target_positions - fingertip_positions)
        np.testing.assert_equal(expected_reward, timestep.reward)


if __name__ == "__main__":
    absltest.main()
