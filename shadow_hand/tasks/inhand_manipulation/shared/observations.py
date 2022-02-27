"""Shared configuration options for observations."""

import dataclasses
from typing import Callable, Optional, Tuple, Union


@dataclasses.dataclass(frozen=True)
class ObservableSpec:
    """Configuration options for generic observables."""

    enabled: bool
    update_interval: Union[int, Callable[..., int]]
    buffer_size: int
    delay: Union[int, Callable[..., int]]
    aggregator: Optional[Union[str, Callable[..., int]]]
    corruptor: Optional[Callable[..., int]]


@dataclasses.dataclass(frozen=True)
class CameraObservableSpec(ObservableSpec):
    """Configuration options for camera observables."""

    height: int
    width: int


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """Container for `ObservableSpec`s grouped by category."""

    proprio: ObservableSpec
    prop_pose: ObservableSpec
    camera: CameraObservableSpec


@dataclasses.dataclass(frozen=True)
class ObservableNames:
    """Container that groups the names of observables by category."""

    proprio: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    prop_pose: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    camera: Tuple[str, ...] = dataclasses.field(default_factory=tuple)


# Global defaults for feature observables (i.e., anything that isn't a camera).
_DISABLED_FEATURE = ObservableSpec(
    enabled=False,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None,
)
_ENABLED_FEATURE = dataclasses.replace(_DISABLED_FEATURE, enabled=True)

# Global defaults for camera observables.
_DISABLED_CAMERA = CameraObservableSpec(
    height=84,
    width=84,
    enabled=False,
    update_interval=1,
    buffer_size=1,
    delay=0,
    aggregator=None,
    corruptor=None,
)
_ENABLED_CAMERA = dataclasses.replace(_DISABLED_CAMERA, enabled=True)

# Predefined observation settings.
PERFECT_FEATURES = ObservationSettings(
    proprio=_ENABLED_FEATURE,
    prop_pose=_ENABLED_FEATURE,
    camera=_DISABLED_CAMERA,
)
VISION = ObservationSettings(
    proprio=_ENABLED_FEATURE,
    prop_pose=_DISABLED_FEATURE,
    camera=_ENABLED_CAMERA,
)


def make_options(obs_settings: ObservationSettings, obs_names: ObservableNames):
    """Constructs a dict of configuration options for a set of named observables."""
    observable_options = {}
    for category, spec in dataclasses.asdict(obs_settings).items():
        for observable_name in getattr(obs_names, category):
            observable_options[observable_name] = spec
    return observable_options
