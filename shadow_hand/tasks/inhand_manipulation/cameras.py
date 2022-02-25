"""Tools for adding custom cameras to the arena."""

import collections
import dataclasses
from typing import Sequence, Tuple

from dm_control import composer
from dm_control.composer.observation import observable


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    name: str
    position: Tuple[float, float, float]
    xyaxes: Tuple[float, float, float, float, float, float]


# Custom cameras that can be added to the arena.
FRONT_CLOSE = CameraConfig(
    name="front_close",
    position=(0.0, -0.6, 0.75),
    xyaxes=(1.0, 0.0, 0.0, 0.0, 0.7, 0.75),
)

FRONT_FAR = CameraConfig(
    name="front_far", position=(0.0, -0.8, 1.0), xyaxes=(1.0, 0.0, 0.0, 0.0, 0.7, 0.75)
)

TOP_DOWN = CameraConfig(
    name="top_down", position=(0.0, 0.0, 2.5), xyaxes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
)

LEFT_CLOSE = CameraConfig(
    name="left_close",
    position=(-0.6, 0.0, 0.75),
    xyaxes=(0.0, -1.0, 0.0, 0.7, 0.0, 0.75),
)

RIGHT_CLOSE = CameraConfig(
    name="right_close",
    position=(0.6, 0.0, 0.75),
    xyaxes=(0.0, 1.0, 0.0, -0.7, 0.0, 0.75),
)


def add_camera_observables(
    entity: composer.Entity, obs_settings, camera_configs: Sequence[CameraConfig]
) -> collections.OrderedDict:
    obs_dict = collections.OrderedDict()
    for config in camera_configs:
        camera = entity.mjcf_model.worldbody.add("camera", **dataclasses.asdict(config))
        obs = observable.MJCFCamera(camera)
        obs.configure(**obs_settings.camera._asdict())
        obs_dict[config.name] = obs
    return obs_dict
