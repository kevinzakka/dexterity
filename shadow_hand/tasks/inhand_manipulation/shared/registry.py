"""A global registry of constructors for in-hand manipulation environments."""

from dm_control.utils import containers

_ALL_CONSTRUCTORS = containers.TaggedTasks(allow_overriding_keys=False)

add = _ALL_CONSTRUCTORS.add
get_constructor = _ALL_CONSTRUCTORS.__getitem__
get_all_names = _ALL_CONSTRUCTORS.keys
get_tags = _ALL_CONSTRUCTORS.tags
get_names_by_tag = _ALL_CONSTRUCTORS.tagged


def done_importing_tasks() -> None:
    _ALL_CONSTRUCTORS.allow_overriding_keys = True
