"""A set of in-hand manipulation tasks."""

from typing import Optional, Tuple

from dm_control import composer as _composer

from shadow_hand.tasks.inhand_manipulation import reach as _reach
from shadow_hand.tasks.inhand_manipulation import reorient as _reorient
from shadow_hand.tasks.inhand_manipulation.shared import registry as _registry
from shadow_hand.utils import mujoco_collisions

_registry.done_importing_tasks()

# A list of strings representing all the registered tasks.
ALL: Tuple[str, ...] = tuple(_registry.get_all_names())


def load(environment_name: str, seed: Optional[int] = None) -> _composer.Environment:
    # Build the task.
    task = _registry.get_constructor(environment_name)()

    # Ensure MuJoCo will not check for collisions between geoms that can never collide.
    mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(
        task.root_entity.mjcf_model
    )

    # NOTE(kevin): By default, the time limit is infinite.
    return _composer.Environment(task=task, random_state=seed)
