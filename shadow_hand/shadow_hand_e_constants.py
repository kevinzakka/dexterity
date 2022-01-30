"""Shadow hand constants.

## DoFs
The thumb has 5 degrees of freedom and 5 joints. Each finger has 3 degrees of freedom and 4 joints.

## Weight
The Hand and forearm have a total weight of 4.3 kg.
The Hand, while in a power-grasp, can hold up to 5 Kg.
"""

import enum
from typing import Dict, Tuple

from pathlib import Path

# Path to the root of the src files, i.e. `shadow_hand/`.
_SRC_ROOT = Path(__file__).parent.parent / "shadow_hand"


class Actuation(enum.Enum):
    POSITION_CONTROL = enum.auto()


class Parts(enum.Enum):
    WR = "wrist"
    FF = "first_finger"
    MF = "middle_finger"
    RF = "ring_finger"
    LF = "little_finger"
    TH = "thumb"


class Joints(enum.Enum):
    """Joints of the Shadow Hand.

    Joints are numbered from fingertip to palm, e.g., FFJ0 is the first knuckle, FFJ1 is the second knuckle, etc.

    The first two joints of each of the main fingers are coupled, which means there is only one actuator controlling them via a single tendon.
    """

    # Wrist.
    # 2 degrees of freedom for the wrist (horizontal and vertical).
    WRJ1 = enum.auto()
    WRJ0 = enum.auto()
    # First finger.
    # 3 degrees of freedom for the first finger.
    FFJ3 = enum.auto()
    FFJ2 = enum.auto()
    FFJ1 = enum.auto()  # Tendon "FFT1", coupled joint.
    FFJ0 = enum.auto()  # Tendon "FFT1", coupled joint.
    # Middle finger.
    # 3 degrees of freedom for the middle finger.
    MFJ3 = enum.auto()
    MFJ2 = enum.auto()
    MFJ1 = enum.auto()  # Tendon "MFT1", coupled joint.
    MFJ0 = enum.auto()  # Tendon "MFT1", coupled joint.
    # Ring finger.
    RFJ3 = enum.auto()
    RFJ2 = enum.auto()
    RFJ1 = enum.auto()  # Tendon "RFT1", coupled joint.
    RFJ0 = enum.auto()  # Tendon "RFT1", coupled joint.
    # Little finger.
    # 4 degrees of freedom for the little finger.
    LFJ4 = enum.auto()
    LFJ3 = enum.auto()
    LFJ2 = enum.auto()
    LFJ1 = enum.auto()  # Tendon "LFT1", coupled joint.
    LFJ0 = enum.auto()  # Tendon "LFT1", coupled joint.
    # Thumb.
    # 5 degrees of freedom for the thumb.
    THJ4 = enum.auto()
    THJ3 = enum.auto()
    THJ2 = enum.auto()
    THJ1 = enum.auto()
    THJ0 = enum.auto()


class Actuators(enum.Enum):
    """Actuators of the Shadow Hand."""

    # Wrist.
    A_WRJ1 = enum.auto()  # Horizontal movement.
    A_WRJ0 = enum.auto()  # Vertical movement.
    # First finger.
    A_FFJ3 = enum.auto()  # Horizontal movement.
    A_FFJ2 = enum.auto()  # Vertical movement.
    A_FFJ1 = enum.auto()  # Vertical movement, bending coupled joints.
    # Middle finger.
    A_MFJ3 = enum.auto()  # Horizontal movement.
    A_MFJ2 = enum.auto()  # Vertical movement.
    A_MFJ1 = enum.auto()  # Vertical movement, bending coupled joints.
    # Ring finger.
    A_RFJ3 = enum.auto()  # Horizontal movement.
    A_RFJ2 = enum.auto()  # Vertical movement.
    A_RFJ1 = enum.auto()  # Vertical movement, bending coupled joints.
    # Little finger.
    A_LFJ4 = enum.auto()  # Vertical movement, towards center of palm.
    A_LFJ3 = enum.auto()  # Horizontal movement.
    A_LFJ2 = enum.auto()  # Vertical movement.
    A_LFJ1 = enum.auto()  # Vertical movement, bending coupled joints.
    # Thumb.
    A_THJ4 = enum.auto()  # Rotational movement of the thumb.
    A_THJ3 = enum.auto()  # Bending movement.
    A_THJ2 = enum.auto()  # Bending movement.
    A_THJ1 = enum.auto()  # Bending movement.
    A_THJ0 = enum.auto()  # Bending movement.


# A mapping from physical part to actuators that control it.
ACTUATOR_GROUPS: Dict[Parts, Tuple[Actuators]] = {
    Parts.WR: (Actuators.A_WRJ1, Actuators.A_WRJ0),
    Parts.FF: (Actuators.A_FFJ3, Actuators.A_FFJ2, Actuators.A_FFJ1),
    Parts.MF: (Actuators.A_MFJ3, Actuators.A_MFJ2, Actuators.A_MFJ1),
    Parts.RF: (Actuators.A_RFJ3, Actuators.A_RFJ2, Actuators.A_RFJ1),
    Parts.LF: (Actuators.A_LFJ4, Actuators.A_LFJ3, Actuators.A_LFJ2, Actuators.A_LFJ1),
    Parts.TH: (
        Actuators.A_THJ4,
        Actuators.A_THJ3,
        Actuators.A_THJ2,
        Actuators.A_THJ1,
        Actuators.A_THJ0,
    ),
}


ACTUATOR_JOINT_MAP = {}
ACTUATOR_CTRLRANGE = []
JOINT_LIMITS = []

