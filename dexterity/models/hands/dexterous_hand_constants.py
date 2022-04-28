import enum
from dataclasses import dataclass
from typing import Tuple

from dexterity import hints


class HandSide(enum.Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()


@dataclass(frozen=True)
class JointGrouping:
    """A collection of joints belonging to a hand part (wrist or finger)."""

    name: str
    joints: Tuple[hints.MjcfElement, ...]

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple([joint.name for joint in self.joints])
