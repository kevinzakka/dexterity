"""Shared configuration options for observations."""

import dataclasses
from typing import Callable, Optional, Union


@dataclasses.dataclass(frozen=True)
class ObservableSpec:
    """Configuration options for generic observables."""

    enabled: bool
    """Whether the observable is computed and returned to the agent."""

    update_interval: Union[int, Callable[..., int]]
    """The interval, in simulation steps, at which the values of the observable will be
    updated. The last value will be repeated between updates. This parameter may be used
    to simulate sensors with different sample rates. Sensors with stochastic rates may
    be modelled by passing a callable that returns a random integer."""

    buffer_size: int
    """Controls the size of the internal FIFO buffer used to store observations that
    were sampled on previous simulation time-steps. In the default case where no
    aggregator is provided (see below), the entire contents of the buffer is returned as
    an observation at each control timestep. This can be used to avoid discarding
    observations from sensors whose values may change significantly within the control
    timestep. If the buffer size is sufficiently large, it will contain observations
    from previous control timesteps, endowing the environment with a simple form of
    memory."""

    delay: Union[int, Callable[..., int]]
    """Specifies a delay (in terms of simulation timesteps) between when the value of
    the observable is sampled, and when it will appear in the observations returned by
    the environment. This parameter can be used to model sensor latency. Stochastic
    latencies may be modelled by passing a callable that returns a randomly sampled
    integer."""

    aggregator: Optional[Union[str, Callable[..., int]]]
    """Performs a reduction over all of the elements in the observation buffer. For
    example this can be used to take a moving average over previous observation values.
    """

    corruptor: Optional[Callable[..., int]]
    """Performs a point-wise transformation of each observation value before it is
    inserted into the buffer. Corruptors are most commonly used to simulate observation
    noise"""


@dataclasses.dataclass(frozen=True)
class CameraObservableSpec(ObservableSpec):
    """Configuration options for camera observables."""

    height: int
    width: int


@dataclasses.dataclass(frozen=True)
class ObservationSettings:
    """Container of `ObservableSpec`s grouped by category."""

    proprio: ObservableSpec
    prop_pose: ObservableSpec
    camera: CameraObservableSpec


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
