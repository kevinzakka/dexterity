"""Tests for reach.py."""

import numpy as np
from absl.testing import absltest

from dexterity.environment import Environment
from dexterity.manipulation.shared import observations
from dexterity.manipulation.tasks import reach_task


class ReachTaskTest(absltest.TestCase):
    def test_sparse_reward(self) -> None:
        task = reach_task(
            observations.ObservationSet.STATE_ONLY,
            use_dense_reward=False,
            visualize_reward=False,
        )

        random_state = np.random.RandomState(12345)
        env = Environment(task, random_state=random_state)
        action_spec = env.action_spec()

        env.reset()
        timestep = env.step(np.zeros(action_spec.shape, action_spec.dtype))
        self.assertEqual(timestep.reward, -1.0)

        qpos_sol = env.task.goal_generator.qpos.copy()  # type: ignore
        ctrl_sol = env.task.hand.joint_positions_to_control(qpos_sol)
        ctrl_sol = ctrl_sol.astype(action_spec.dtype)
        while True:
            timestep = env.step(ctrl_sol)
            if env.task.successes > 0:
                break
        expected_reward = 0.0
        np.testing.assert_equal(timestep.reward, expected_reward)


if __name__ == "__main__":
    absltest.main()
