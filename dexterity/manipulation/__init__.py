import collections
import inspect
from typing import Optional

from dm_control import composer as _composer

from dexterity import environment as _environment
from dexterity import task as _task
from dexterity.manipulation.tasks import juggle
from dexterity.manipulation.tasks import reach
from dexterity.manipulation.tasks import reorient
from dexterity.utils import mujoco_collisions

# Find all domains imported.
_DOMAINS = {
    name: module
    for name, module in locals().items()
    if inspect.ismodule(module) and hasattr(module, "SUITE")
}


def _get_tasks(tag):
    """Returns a sequence of (domain name, task name) pairs for the given tag."""
    result = []
    for domain_name in sorted(_DOMAINS.keys()):
        domain = _DOMAINS[domain_name]
        if tag is None:
            tasks_in_domain = domain.SUITE
        else:
            tasks_in_domain = domain.SUITE.tagged(tag)
        for task_name in tasks_in_domain.keys():
            result.append((domain_name, task_name))
    return tuple(result)


def _get_tasks_by_domain(tasks):
    """Returns a dict mapping from task name to a tuple of domain names."""
    result = collections.defaultdict(list)

    for domain_name, task_name in tasks:
        result[domain_name].append(task_name)

    return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# A sequence containing all `domain_name.task_name` pairs.
ALL_NAMES = [".".join(domain_task) for domain_task in ALL_TASKS]

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(
    domain_name: str,
    task_name: str,
    seed: Optional[int] = None,
    strip_singleton_obs_buffer_dim: bool = True,
    time_limit: Optional[float] = None,
) -> _composer.Environment:
    if domain_name not in _DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}")
    domain = _DOMAINS[domain_name]

    if task_name not in domain.SUITE:
        raise ValueError(f"Unknown task: {task_name}")
    task = domain.SUITE[task_name]()

    # Ensure MuJoCo will not check for collisions between geoms that can never collide.
    mujoco_collisions.exclude_bodies_based_on_contype_conaffinity(
        task.root_entity.mjcf_model
    )

    if isinstance(task, _task.GoalTask):
        env_cls = _environment.GoalEnvironment
    else:
        env_cls = _composer.Environment

    return env_cls(
        task=task,
        time_limit=time_limit or task.time_limit,
        random_state=seed,
        strip_singleton_obs_buffer_dim=strip_singleton_obs_buffer_dim,
    )
