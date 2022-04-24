import dm_env
from absl import logging
from dm_control import composer

from dexterity import exception
from dexterity import task


class GoalEnvironment(composer.Environment):
    """A composer.Environment for `GoalTask`s."""

    task: task.GoalTask

    def reset(self) -> dm_env.TimeStep:
        failed_attempts = 0
        while True:
            try:
                return super().reset()
            except exception.GoalInitializationError as e:
                failed_attempts += 1
                logging.error(
                    "Error %d during episode reset: %s", failed_attempts, repr(e)
                )

    def step(self, action) -> dm_env.TimeStep:
        failed_attempts = 0
        while True:
            try:
                return super().step(action)
            except exception.GoalInitializationError as e:
                failed_attempts += 1
                logging.error(
                    "Error %d during episode step: %s", failed_attempts, repr(e)
                )
