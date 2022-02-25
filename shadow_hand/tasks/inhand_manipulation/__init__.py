"""A set of in-hand dexterous manipulation tasks."""

from typing import Optional

from dm_control import composer as _composer

from shadow_hand.tasks.inhand_manipulation import registry as _registry
from shadow_hand.tasks.inhand_manipulation import reorient as _reorient


_registry.done_importing_tasks()

ALL = tuple(_registry.get_all_names())
TAGS = tuple(_registry.get_tags())


def load(environment_name: str, seed: Optional[int] = None) -> _composer.Environment:
    task = _registry.get_constructor(environment_name)()
    return _composer.Environment(task=task, random_state=seed)
