"""Tools for adding custom cameras to the arena."""

import collections
import dataclasses
from typing import Tuple

from dm_control import composer
from dm_control.composer.observation import observable

from shadow_hand.tasks.inhand_manipulation.shared import observations


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    name: str
    pos: Tuple[float, float, float]
    xyaxes: Tuple[float, float, float, float, float, float]


# Custom cameras that can be added to the arena.

FRONT_CLOSE = CameraConfig(
    name="front_close",
    pos=(0.0, -0.5, 0.5),
    xyaxes=(1.0, 0.0, 0.0, 0.0, 0.7, 0.75),
)

LEFT_CLOSE = CameraConfig(
    name="left_close",
    pos=(-0.6, 0.0, 0.5),
    xyaxes=(0.0, -1.0, 0.0, 0.7, 0.0, 0.75),
)

RIGHT_CLOSE = CameraConfig(
    name="right_close",
    pos=(0.6, 0.0, 0.5),
    xyaxes=(0.0, 1.0, 0.0, -0.7, 0.0, 0.75),
)

FRONT_FAR = CameraConfig(
    name="front_far",
    pos=(0.0, -1.0, 0.7),
    xyaxes=(1.0, 0.0, 0.0, 0.0, 0.7, 0.75),
)

TOP_DOWN = CameraConfig(
    name="top_down",
    pos=(0.0, 0.0, 2.5),
    xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
)


def add_camera_observables(
    entity: composer.Entity,
    obs_settings: observations.ObservationSettings,
    *camera_configs: CameraConfig,
) -> collections.OrderedDict:
    obs_dict = collections.OrderedDict()
    for config in camera_configs:
        camera = entity.mjcf_model.worldbody.add("camera", **dataclasses.asdict(config))
        obs = observable.MJCFCamera(camera)
        obs.configure(**dataclasses.asdict(obs_settings.camera))
        obs_dict[config.name] = obs
    return obs_dict
