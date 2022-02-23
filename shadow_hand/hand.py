import abc
from typing import List

import numpy as np
from dm_control import composer, mjcf

from shadow_hand.hints import MjcfElement


class Hand(abc.ABC, composer.Entity):
    """Abstract base class for a robotic hand."""

    @abc.abstractmethod
    def _build(self) -> None:
        """Entity initialization method to be overridden by subclasses."""

    @property
    @abc.abstractmethod
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to the hand."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the hand."""

    @property
    @abc.abstractmethod
    def actuators(self) -> List[MjcfElement]:
        """List of actuator elements belonging to the hand."""

    @property
    def joints(self) -> List[MjcfElement]:
        """List of joint elements belonging to the hand."""

    @abc.abstractmethod
    def set_joint_angles(
        self, physics: mjcf.Physics, joint_angles: np.ndarray
    ) -> None:
        """Sets the joints of the hand to a given configuration."""
