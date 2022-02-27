"""Tests for `dm_control.manipulation_suite`."""

import collections
from typing import Mapping

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from shadow_hand.tasks import inhand_manipulation

_SEED = 0
_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


class InHandManipulationTest(parameterized.TestCase):
    """Tests run on all the registered tasks."""

    @parameterized.parameters(*inhand_manipulation.ALL)
    def test_task_runs(self, task_name: str) -> None:
        """Tests that the environment runs and is coherent with its specs."""
        env = inhand_manipulation.load(task_name, seed=_SEED)
        random_state = np.random.RandomState(_SEED)

        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
        self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

        # Run a partial episode, check observations, rewards, discount.
        for _ in range(_NUM_EPISODES):
            time_step = env.reset()
            for _ in range(_NUM_STEPS_PER_EPISODE):
                self._validate_observation(time_step.observation, observation_spec)
                if time_step.first():
                    self.assertIsNone(time_step.reward)
                    self.assertIsNone(time_step.discount)
                else:
                    self._validate_reward_range(time_step.reward)
                    self._validate_discount(time_step.discount)
                action = random_state.uniform(action_spec.minimum, action_spec.maximum)
                env.step(action)

    # Helper methods.

    def _validate_observation(
        self,
        observation: Mapping[str, np.ndarray],
        observation_spec: collections.OrderedDict,
    ) -> None:
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def _validate_reward_range(self, reward: float) -> None:
        self.assertIsInstance(reward, float)
        self.assertBetween(reward, 0.0, 1.0)

    def _validate_discount(self, discount: float) -> None:
        self.assertIsInstance(discount, float)
        self.assertBetween(discount, 0.0, 1.0)


if __name__ == "__main__":
    absltest.main()
