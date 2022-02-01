"""Shadow hand constants."""

import enum
from math import radians as rad
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Path to the root of the src files, i.e. `shadow_hand/`.
_SRC_ROOT: Path = Path(__file__).parent.parent / "shadow_hand"

# Path to the shadow hand E series XML file.
SHADOW_HAND_E_XML: Path = _SRC_ROOT / "shadow_hand_series_e.xml"


class Components(enum.Enum):
    """The actuated components of the hand: wrist and fingers."""

    WR = "wrist"
    FF = "first_finger"
    MF = "middle_finger"
    RF = "ring_finger"
    LF = "little_finger"
    TH = "thumb"


# ====================== #
# Joint constants
# ====================== #


class Joints(enum.Enum):
    """Joints of the Shadow Hand.

    There are a total of 24 joints:
        * 2 joints for the wrist
        * 4 joints for the first, middle and ring fingers
        * 5 joints for the little finger and thumb

    The joint numbering is increasing from fingertip to palm, i.e. FFJ0 is the first
    knuckle of the first finger, FFJ3 is the last knuckle of the first finger, etc.

    The first two joints (*FJ0, *FJ1) of the main fingers (exlcuding the thumb) are
    coupled, which means 1 actuator controls both of them via a tendon. This means there
    is 1 less DoF for each of these fingers, which means the total number of degrees of
    freedom for the hand is: 24 - 4 = 20.
    """

    # Wrist: 2 joints, 2 degrees of freedom.
    WRJ1 = enum.auto()
    WRJ0 = enum.auto()
    # First finger: 4 joints, 3 degrees of freedom.
    FFJ3 = enum.auto()
    FFJ2 = enum.auto()
    FFJ1 = enum.auto()  # c
    FFJ0 = enum.auto()  # c
    # Middle finger: 4 joints, 3 degrees of freedom.
    MFJ3 = enum.auto()
    MFJ2 = enum.auto()
    MFJ1 = enum.auto()  # c
    MFJ0 = enum.auto()  # c
    # Ring finger: 4 joints, 3 degrees of freedom.
    RFJ3 = enum.auto()
    RFJ2 = enum.auto()
    RFJ1 = enum.auto()  # c
    RFJ0 = enum.auto()  # c
    # Little finger: 5 joints, 4 degrees of freedom.
    LFJ4 = enum.auto()
    LFJ3 = enum.auto()
    LFJ2 = enum.auto()
    LFJ1 = enum.auto()  # c
    LFJ0 = enum.auto()  # c
    # Thumb: 5 joints, 5 degrees of freedom.
    THJ4 = enum.auto()
    THJ3 = enum.auto()
    THJ2 = enum.auto()
    THJ1 = enum.auto()
    THJ0 = enum.auto()


# The total number of joints.
NUM_JOINTS: int = len(Joints)

# A list of joint names, as strings.
JOINT_NAMES: List[str] = [j.name for j in Joints]

# ====================== #
# Actuation constants
# ====================== #


class Actuation(enum.Enum):
    """Available actuation methods for the hand."""

    POSITION = enum.auto()
    """In position mode, the joint actuators receive a position and a position
    controller is used to maintain the joint configuration.
    """

    TORQUE = enum.auto()


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


# The total number of actuators.
NUM_ACTUATORS: int = len(Actuators)

# A list of actuator names, as strings.
ACTUATOR_NAMES: List[str] = [a.name for a in Actuators]

# A mapping from `Components` to the list of `Actuators` that belong to it.
ACTUATOR_GROUP: Dict[Components, Tuple[Actuators, ...]] = {
    # Wrist has 2 actuators.
    Components.WR: (Actuators.A_WRJ1, Actuators.A_WRJ0),
    # First, middle and ring fingers have 3 actuators.
    Components.FF: (Actuators.A_FFJ3, Actuators.A_FFJ2, Actuators.A_FFJ1),
    Components.MF: (Actuators.A_MFJ3, Actuators.A_MFJ2, Actuators.A_MFJ1),
    Components.RF: (Actuators.A_RFJ3, Actuators.A_RFJ2, Actuators.A_RFJ1),
    # Little finger has 4 actuators.
    Components.LF: (
        Actuators.A_LFJ4,
        Actuators.A_LFJ3,
        Actuators.A_LFJ2,
        Actuators.A_LFJ1,
    ),
    # Thumb has 5 actuators.
    Components.TH: (
        Actuators.A_THJ4,
        Actuators.A_THJ3,
        Actuators.A_THJ2,
        Actuators.A_THJ1,
        Actuators.A_THJ0,
    ),
}


# Note: It seems there's a discrepancy between the values reported in the spec sheet^[1]
# and the values reported in the company's github repo^[2]. I'm going to follow the ones
# on the spec sheet, since it seems those are the ones OpenAI^[3] used for their project
# as well.
# References:
#   [1]: Shadow Robot spec sheet: https://www.shadowrobot.com/wp-content/uploads/shadow_dexterous_hand_technical_specification_E_20190221.pdf
#   [2]: Shadow Robot Company code: github.com/shadow-robot/sr_common
#   [3]: OpenAI code: github.com/openai/robogym
#
# A mapping from `Actuators` to the corresponding control range, in radians.
ACTUATOR_CTRLRANGE: Dict[Actuators, Tuple[float, float]] = {
    # Wrist.
    Actuators.A_WRJ1: (rad(-28), rad(8)),  # (-0.4886921905584123, 0.13962634015954636)
    Actuators.A_WRJ0: (rad(-40), rad(28)),  # (-0.6981317007977318, 0.4886921905584123)
    # First finger.
    Actuators.A_FFJ3: (rad(-20), rad(20)),
    Actuators.A_FFJ2: (rad(0), rad(90)),
    Actuators.A_FFJ1: (rad(0), rad(180)),
    # Middle finger.
    Actuators.A_MFJ3: (rad(-20), rad(20)),
    Actuators.A_MFJ2: (rad(0), rad(90)),
    Actuators.A_MFJ1: (rad(0), rad(180)),
    # Ring finger.
    Actuators.A_RFJ3: (rad(-20), rad(20)),
    Actuators.A_RFJ2: (rad(0), rad(90)),
    Actuators.A_RFJ1: (rad(0), rad(180)),
    # Little finger.
    Actuators.A_LFJ4: (rad(0), rad(45)),
    Actuators.A_LFJ3: (rad(-20), rad(20)),
    Actuators.A_LFJ2: (rad(0), rad(90)),
    Actuators.A_LFJ1: (rad(0), rad(180)),
    # Thumb.
    Actuators.A_THJ4: (rad(-60), rad(60)),
    Actuators.A_THJ3: (rad(0), rad(70)),
    Actuators.A_THJ2: (rad(-12), rad(12)),
    Actuators.A_THJ1: (rad(-30), rad(30)),
    Actuators.A_THJ0: (rad(0), rad(90)),  # OpenAI uses (-90, 0) here. Why?
}

# One-to-many mapping from `Actuators` to the joint(s) it controls.
# The first two joints of each of the main fingers are coupled, which means there is
# only one actuator controlling them via a single tendon.
ACTUATOR_JOINT_MAPPING: Dict[Actuators, Tuple[Joints, ...]] = {
    # Wrist.
    Actuators.A_WRJ1: (Joints.WRJ1,),
    Actuators.A_WRJ0: (Joints.WRJ0,),
    # First finger.
    Actuators.A_FFJ3: (Joints.FFJ3,),
    Actuators.A_FFJ2: (Joints.FFJ2,),
    Actuators.A_FFJ1: (Joints.FFJ1, Joints.FFJ0),
    # Middle finger.
    Actuators.A_MFJ3: (Joints.MFJ3,),
    Actuators.A_MFJ2: (Joints.MFJ2,),
    Actuators.A_MFJ1: (Joints.MFJ1, Joints.MFJ0),
    # Ring finger.
    Actuators.A_RFJ3: (Joints.RFJ3,),
    Actuators.A_RFJ2: (Joints.RFJ2,),
    Actuators.A_RFJ1: (Joints.RFJ1, Joints.RFJ0),
    # Little finger.
    Actuators.A_LFJ4: (Joints.LFJ4,),
    Actuators.A_LFJ3: (Joints.LFJ3,),
    Actuators.A_LFJ2: (Joints.LFJ2,),
    Actuators.A_LFJ1: (Joints.LFJ1, Joints.LFJ0),
    # Thumb.
    Actuators.A_THJ4: (Joints.THJ4,),
    Actuators.A_THJ3: (Joints.THJ3,),
    Actuators.A_THJ2: (Joints.THJ2,),
    Actuators.A_THJ1: (Joints.THJ1,),
    Actuators.A_THJ0: (Joints.THJ0,),
}

# Reverse mapping of `ACTUATOR_JOINT_MAPPING`.
JOINT_ACTUATOR_MAPPING: Dict[Joints, Actuators] = {
    v: k for k, vs in ACTUATOR_JOINT_MAPPING.items() for v in vs
}

# Joint position limits, in radians.
# These are derived from `ACTUATOR_CTRLRANGE`, since we don't have a 1:1 mapping between
# actuators and joints. Coupled joints share the full ctrlrange, so their range is split
# in half.
JOINT_LIMITS: Dict[Joints, Tuple[float, float]] = {}
for actuator, ctrlrange in ACTUATOR_CTRLRANGE.items():
    joints = ACTUATOR_JOINT_MAPPING[actuator]
    for joint in joints:
        JOINT_LIMITS[joint] = (
            ctrlrange[0] / len(joints),
            ctrlrange[1] / len(joints),
        )

# Effort limits.
# For a `hinge` (revolute) joint, this is equivalent to the torque limit, in N-m.
# For a `slide` (prismatic) joint, this is equivalent to the force limit, in N.
EFFORT_LIMITS: Dict[Actuators, Tuple[float, float]] = {
    # Wrist.
    Actuators.A_WRJ1: (-10.0, 10.0),
    Actuators.A_WRJ0: (-30.0, 30.0),
    # First finger.
    Actuators.A_FFJ3: (-2.0, 2.0),
    Actuators.A_FFJ2: (-2.0, 2.0),
    Actuators.A_FFJ1: (-2.0, 2.0),
    # Middle finger.
    Actuators.A_MFJ3: (-2.0, 2.0),
    Actuators.A_MFJ2: (-2.0, 2.0),
    Actuators.A_MFJ1: (-2.0, 2.0),
    # Ring finger.
    Actuators.A_RFJ3: (-2.0, 2.0),
    Actuators.A_RFJ2: (-2.0, 2.0),
    Actuators.A_RFJ1: (-2.0, 2.0),
    # Little finger.
    Actuators.A_LFJ4: (-2.0, 2.0),
    Actuators.A_LFJ3: (-2.0, 2.0),
    Actuators.A_LFJ2: (-2.0, 2.0),
    Actuators.A_LFJ1: (-2.0, 2.0),
    # Thumb.
    Actuators.A_THJ4: (-5.0, 5.0),
    Actuators.A_THJ3: (-3.0, 3.0),
    Actuators.A_THJ2: (-2.0, 2.0),
    Actuators.A_THJ1: (-2.0, 2.0),
    Actuators.A_THJ0: (-1.0, 1.0),
}

# Joint velocity limits, in rad/s.
VELOCITY_LIMITS: Dict[Actuators, Tuple[float, float]] = {
    # Wrist.
    Actuators.A_WRJ1: (-2.0, 2.0),
    Actuators.A_WRJ0: (-2.0, 2.0),
    # First finger.
    Actuators.A_FFJ3: (-2.0, 2.0),
    Actuators.A_FFJ2: (-2.0, 2.0),
    Actuators.A_FFJ1: (-2.0, 2.0),
    # Middle finger.
    Actuators.A_MFJ3: (-2.0, 2.0),
    Actuators.A_MFJ2: (-2.0, 2.0),
    Actuators.A_MFJ1: (-2.0, 2.0),
    # Ring finger.
    Actuators.A_RFJ3: (-2.0, 2.0),
    Actuators.A_RFJ2: (-2.0, 2.0),
    Actuators.A_RFJ1: (-2.0, 2.0),
    # Little finger.
    Actuators.A_LFJ4: (-2.0, 2.0),
    Actuators.A_LFJ3: (-2.0, 2.0),
    Actuators.A_LFJ2: (-2.0, 2.0),
    Actuators.A_LFJ1: (-2.0, 2.0),
    # Thumb.
    Actuators.A_THJ4: (-4.0, 4.0),
    Actuators.A_THJ3: (-4.0, 4.0),
    Actuators.A_THJ2: (-4.0, 4.0),
    Actuators.A_THJ1: (-2.0, 2.0),
    Actuators.A_THJ0: (-4.0, 4.0),
}

# Actuation limits of the hand.
ACTUATION_LIMITS: Dict[Actuation, Dict[Actuators, Tuple[float, float]]] = {
    Actuation.TORQUE: EFFORT_LIMITS,
    Actuation.POSITION: ACTUATOR_CTRLRANGE,
}


def _compute_projection_matrices() -> Tuple[np.ndarray, np.ndarray]:
    position_to_control = np.zeros((NUM_ACTUATORS, NUM_JOINTS))
    control_to_position = np.zeros((NUM_JOINTS, NUM_ACTUATORS))
    actuator_ids = dict(zip(Actuators, range(NUM_ACTUATORS)))
    joint_ids = dict(zip(Joints, range(NUM_JOINTS)))
    for actuator, joints in ACTUATOR_JOINT_MAPPING.items():
        value = 1.0 / len(joints)
        a_id = actuator_ids[actuator]
        j_ids = np.array([joint_ids[joint] for joint in joints])
        position_to_control[a_id, j_ids] = 1.0
        control_to_position[j_ids, a_id] = value
    return position_to_control, control_to_position


# Projection matrices for mapping control space to joint space and vice versa.
# POSITION_TO_CONTROL maps a control vector to a joint vector.
# CONTROL_TO_POSITION maps a joint vector to a control vector.
POSITION_TO_CONTROL, CONTROL_TO_POSITION = _compute_projection_matrices()

# ====================== #
# Tendon constants
# ====================== #


class Tendons(enum.Enum):
    """Tendons of the Shadow Hand.

    These are used to model the underactuation of the *FJ0 and *FJ1 joints of the main
    fingers. A tendon is defined for each *FJ0-*FJ1 pair, and an actuator is used to
    drive it.
    """

    FFT1 = enum.auto()  # First finger.
    MFT1 = enum.auto()  # Middle finger.
    RFT1 = enum.auto()  # Ring finger.
    LFT1 = enum.auto()  # Little finger.


# Mapping from `Tendons` to `Joints` pair.
TENDON_JOINT_MAPPING: Dict[Tendons, Tuple[Joints, Joints]] = {
    Tendons.FFT1: (Joints.FFJ0, Joints.FFJ1),  # First finger.
    Tendons.MFT1: (Joints.MFJ0, Joints.MFJ1),  # Middle finger.
    Tendons.RFT1: (Joints.RFJ0, Joints.RFJ1),  # Ring finger.
    Tendons.LFT1: (Joints.LFJ0, Joints.LFJ1),  # Little finger.
}

# Mapping from `Tendons` to the `Actuators` that drives it.
TENDON_ACTUATOR_MAPPING: Dict[Tendons, Actuators] = {
    Tendons.FFT1: Actuators.A_FFJ1,  # First finger.
    Tendons.MFT1: Actuators.A_MFJ1,  # Middle finger.
    Tendons.RFT1: Actuators.A_RFJ1,  # Ring finger.
    Tendons.LFT1: Actuators.A_LFJ1,  # Little finger.
}
# Reverse mapping of `TENDON_ACTUATOR_MAPPING`.
ACTUATOR_TENDON_MAPPING: Dict[Actuators, Tendons] = {
    v: k for k, v in TENDON_ACTUATOR_MAPPING.items()
}


# ====================== #
# Other constants
# ====================== #


# Names of the <geom> tags in the XML file whose color can be changed. This is useful
# for dynamically changing the colors of the hand components.
COLORED_GEOMS: Tuple[str, ...] = (
    "forearm",
    "wrist",
    "palm",
    "ffproximal",
    "ffmiddle",
    "ffdistal",
    "mfproximal",
    "mfmiddle",
    "mfdistal",
    "rfproximal",
    "rfmiddle",
    "rfdistal",
    "lfmetacarpal",
    "lfproximal",
    "lfmiddle",
    "lfdistal",
    "thproximal",
    "thmiddle",
    "thdistal",
)


if __name__ == "__main__":
    for k, v in ACTUATOR_CTRLRANGE.items():
        print(f"{k} -> {v}")
