"""Tests for reorient.py."""

import numpy as np
from absl.testing import absltest

from dexterity import environment
from dexterity.manipulation.shared import observations
from dexterity.manipulation.tasks import reorient
from dexterity.manipulation.tasks import reorient_task


class ReOrientTaskTest(absltest.TestCase):
    def test_dense_reward(self) -> None:
        task = reorient_task(observations.ObservationSet.STATE_ONLY)

        random_state = np.random.RandomState(12345)
        env = environment.GoalEnvironment(task, random_state=random_state)
        action_spec = env.action_spec()
        timestep = env.reset()
        self.assertIsNone(timestep.reward)

        # Manually set the prop's orientation to be equal to the goal orientation.
        env.task._prop.set_pose(physics=env.physics, quaternion=env.task._goal)

        # Artificially set the ctrl to test its reward component.
        rand_ctrl = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )
        env.step(rand_ctrl)

        # Compute shaped reward.
        shaped_reward = reorient._get_shaped_reorientation_reward(
            physics=env.physics,
            goal_distance=env.task._goal_distance,
        )

        # Check individual reward components.
        np.testing.assert_equal(
            shaped_reward["orientation"].value, 1 / reorient._ORIENTATION_THRESHOLD
        )
        np.testing.assert_equal(shaped_reward["success_bonus"].value, 1.0)
        np.testing.assert_equal(
            shaped_reward["action_smoothing"].value, np.linalg.norm(rand_ctrl) ** 2
        )

        # # Check final weighted sum.
        expected_reward = sum([s.value * s.weight for s in shaped_reward.values()])
        actual_reward = env.task.get_reward(env.physics)
        np.testing.assert_equal(actual_reward, expected_reward)


if __name__ == "__main__":
    absltest.main()
