"""Tools for defining and visualizing workspaces for in-hand manipulation tasks."""

import dataclasses
from typing import Sequence, Tuple

import numpy as np
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations

from shadow_hand import hints
from shadow_hand.tasks.inhand_manipulation.shared import constants

_MIN_SITE_DIMENSION = 1e-6  # Ensures that all site dimensions are positive.


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]


uniform_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(-np.pi, np.pi, single_sample=True),
)


def add_bbox_site(
    body: hints.MjcfElement,
    lower: Sequence[float],
    upper: Sequence[float],
    visible: bool = False,
    **kwargs,
) -> hints.MjcfElement:
    """Adds a site for visualizing a bounding box to an MJCF model."""
    assert len(lower) == len(upper) == 3
    lower_arr = np.array(lower)
    upper_arr = np.array(upper)
    assert np.all(lower_arr <= upper_arr)
    pos = (upper_arr + lower_arr) / 2.0
    size = np.maximum((upper_arr - lower_arr) / 2.0, _MIN_SITE_DIMENSION)
    group = None if visible else constants.TASK_SITE_GROUP
    return body.add(
        "site",
        type="box",
        pos=pos,
        size=size,
        group=group,
        **kwargs,
    )


def add_target_site(
    body: hints.MjcfElement,
    radius: float,
    visible: bool = False,
    **kwargs,
) -> hints.MjcfElement:
    """Adds a site for visualizing a target location to an MJCF model."""
    assert radius > 0.0
    group = None if visible else constants.TASK_SITE_GROUP
    return body.add(
        "site",
        type="sphere",
        size=[radius],
        group=group,
        **kwargs,
    )
