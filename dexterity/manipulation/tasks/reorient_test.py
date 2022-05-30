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
        rand_ctrl = rand_ctrl.astype(action_spec.dtype)
        env.task.hand_effector.set_control(env.physics, rand_ctrl)

        # Compute shaped reward.
        goal_distance = env.task.goal_generator.goal_distance(
            env.task._goal,
            env.task.goal_generator.current_state(env.physics),
        )
        shaped_reward = reorient._get_shaped_reorientation_reward(
            goal_distance=goal_distance,
            action=env.task.hand_effector.previous_action,
            has_fallen=env.task._is_prop_fallen(env.physics),
        )

        # Check individual reward components.
        np.testing.assert_equal(
            shaped_reward["orientation"].value, 1 / reorient._ORIENTATION_THRESHOLD
        )
        np.testing.assert_equal(shaped_reward["success_bonus"].value, 1.0)
        np.testing.assert_equal(
            shaped_reward["action_smoothing"].value, np.linalg.norm(rand_ctrl) ** 2
        )


if __name__ == "__main__":
    absltest.main()
