"""Shadow hand constants."""

from math import radians as rad
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dexterity import _SRC_ROOT

# Path to the shadow hand E series XML file.
SHADOW_HAND_E_XML: Path = (
    _SRC_ROOT
    / "models"
    / "vendor"
    / "shadow_robot"
    / "shadow_hand_description"
    / "mjcf"
    / "shadow_hand_series_e.xml"
)


# ====================== #
# Joint constants
# ====================== #


"""Joints of the Shadow Hand.

There are a total of 24 joints:
    * 2 joints for the wrist
    * 4 joints for the first, middle and ring fingers (4 * 3 = 12)
    * 5 joints for the little finger and thumb (5 * 2 = 10)

The joint numbering is increasing from fingertip to palm, i.e. FFJ0 is the first
knuckle of the first finger, FFJ3 is the last knuckle of the first finger, etc.

The first two joints (*FJ0, *FJ1) of the main fingers (exlcuding the thumb) are
coupled, which means 1 actuator controls both of them via a tendon. This means there
is 1 less DoF for each of these fingers, which means the total number of degrees of
freedom for the hand is: 24 - 4 = 20.
"""
JOINTS: Tuple[str, ...] = (
    "WRJ1",
    "WRJ0",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "FFJ0",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "MFJ0",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "RFJ0",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "LFJ0",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "THJ0",
)


# The total number of joints.
NUM_JOINTS: int = len(JOINTS)

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("WRJ1", "WRJ0"),
    "thumb": ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"),
    "first": ("FFJ3", "FFJ2", "FFJ1", "FFJ0"),
    "middle": ("MFJ3", "MFJ2", "MFJ1", "MFJ0"),
    "ring": ("RFJ3", "RFJ2", "RFJ1", "RFJ0"),
    "little": ("LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"),
}

# ====================== #
# Actuation constants
# ====================== #

"""Actuators of the Shadow Hand."""
ACTUATORS: Tuple[str, ...] = (
    "A_WRJ1",
    "A_WRJ0",
    "A_FFJ3",
    "A_FFJ2",
    "A_FFJ1",
    "A_MFJ3",
    "A_MFJ2",
    "A_MFJ1",
    "A_RFJ3",
    "A_RFJ2",
    "A_RFJ1",
    "A_LFJ4",
    "A_LFJ3",
    "A_LFJ2",
    "A_LFJ1",
    "A_THJ4",
    "A_THJ3",
    "A_THJ2",
    "A_THJ1",
    "A_THJ0",
)


# The total number of actuators.
NUM_ACTUATORS: int = len(ACTUATORS)

ACTUATOR_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("A_WRJ1", "A_WRJ0"),
    "thumb": ("A_THJ4", "A_THJ3", "A_THJ2", "A_THJ1", "A_THJ0"),
    "first": ("A_FFJ3", "A_FFJ2", "A_FFJ1"),
    "middle": ("A_MFJ3", "A_MFJ2", "A_MFJ1"),
    "ring": ("A_RFJ3", "A_RFJ2", "A_RFJ1"),
    "little": ("A_LFJ4", "A_LFJ3", "A_LFJ2", "A_LFJ1"),
}

# One-to-many mapping from `Actuators` to the joint(s) it controls.
# The first two joints of each of the main fingers are coupled, which means there is
# only one actuator controlling them via a single tendon.
ACTUATOR_JOINT_MAPPING: Dict[str, Tuple[str, ...]] = {
    # Wrist.
    "A_WRJ1": ("WRJ1",),
    "A_WRJ0": ("WRJ0",),
    # First finger.
    "A_FFJ3": ("FFJ3",),
    "A_FFJ2": ("FFJ2",),
    "A_FFJ1": ("FFJ1", "FFJ0"),
    # Middle finger.
    "A_MFJ3": ("MFJ3",),
    "A_MFJ2": ("MFJ2",),
    "A_MFJ1": ("MFJ1", "MFJ0"),
    # Ring finger.
    "A_RFJ3": ("RFJ3",),
    "A_RFJ2": ("RFJ2",),
    "A_RFJ1": ("RFJ1", "RFJ0"),
    # Little finger.
    "A_LFJ4": ("LFJ4",),
    "A_LFJ3": ("LFJ3",),
    "A_LFJ2": ("LFJ2",),
    "A_LFJ1": ("LFJ1", "LFJ0"),
    # Thumb.
    "A_THJ4": ("THJ4",),
    "A_THJ3": ("THJ3",),
    "A_THJ2": ("THJ2",),
    "A_THJ1": ("THJ1",),
    "A_THJ0": ("THJ0",),
}

# Reverse mapping of `ACTUATOR_JOINT_MAPPING`.
JOINT_ACTUATOR_MAPPING: Dict[str, str] = {
    v: k for k, vs in ACTUATOR_JOINT_MAPPING.items() for v in vs
}


def _compute_projection_matrices() -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    position_to_control = np.zeros((NUM_ACTUATORS, NUM_JOINTS))
    control_to_position = np.zeros((NUM_JOINTS, NUM_ACTUATORS))
    coupled_joint_ids = []
    actuator_ids = dict(zip(ACTUATORS, range(NUM_ACTUATORS)))
    joint_ids = dict(zip(JOINTS, range(NUM_JOINTS)))
    for actuator, joints in ACTUATOR_JOINT_MAPPING.items():
        value = 1.0 / len(joints)
        a_id = actuator_ids[actuator]
        j_ids = np.array([joint_ids[joint] for joint in joints])
        if len(joints) > 1:
            coupled_joint_ids.append([joint_ids[joint] for joint in joints])
        position_to_control[a_id, j_ids] = 1.0
        control_to_position[j_ids, a_id] = value
    return position_to_control, control_to_position, coupled_joint_ids


# Projection matrices for mapping control space to joint space and vice versa. These
# matrices should premultiply the vector to be projected.
# POSITION_TO_CONTROL maps a control vector to a joint vector.
# CONTROL_TO_POSITION maps a joint vector to a control vector.
(
    POSITION_TO_CONTROL,
    CONTROL_TO_POSITION,
    COUPLED_JOINT_IDS,
) = _compute_projection_matrices()

# ====================== #
# Limits
# ====================== #

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
ACTUATOR_CTRLRANGE: Dict[str, Tuple[float, float]] = {
    # Wrist.
    "A_WRJ1": (rad(-28), rad(8)),
    "A_WRJ0": (rad(-40), rad(28)),
    # First finger.
    "A_FFJ3": (rad(-20), rad(20)),
    "A_FFJ2": (rad(0), rad(90)),
    "A_FFJ1": (rad(0), rad(180)),
    # Middle finger.
    "A_MFJ3": (rad(-20), rad(20)),
    "A_MFJ2": (rad(0), rad(90)),
    "A_MFJ1": (rad(0), rad(180)),
    # Ring finger.
    "A_RFJ3": (rad(-20), rad(20)),
    "A_RFJ2": (rad(0), rad(90)),
    "A_RFJ1": (rad(0), rad(180)),
    # Little finger.
    "A_LFJ4": (rad(0), rad(45)),
    "A_LFJ3": (rad(-20), rad(20)),
    "A_LFJ2": (rad(0), rad(90)),
    "A_LFJ1": (rad(0), rad(180)),
    # Thumb.
    "A_THJ4": (rad(-60), rad(60)),
    "A_THJ3": (rad(0), rad(70)),
    "A_THJ2": (rad(-12), rad(12)),
    "A_THJ1": (rad(-30), rad(30)),
    # TODO(kevin): OpenAI uses (-90, 0) here, figure out why.
    "A_THJ0": (rad(0), rad(90)),
}

# Joint position limits, in radians.
# Coupled joints share the full ctrlrange, so their range is split in half.
# Note: These values match the values reported in the spec sheet^[1], page 7.
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {}
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
# Taken from company's github repo^[2] by parsing the XACRO files.
EFFORT_LIMITS: Dict[str, Tuple[float, float]] = {
    # Wrist.
    "A_WRJ1": (-10.0, 10.0),
    "A_WRJ0": (-30.0, 30.0),
    # First finger.
    "A_FFJ3": (-2.0, 2.0),
    "A_FFJ2": (-2.0, 2.0),
    "A_FFJ1": (-2.0, 2.0),
    # Middle finger.
    "A_MFJ3": (-2.0, 2.0),
    "A_MFJ2": (-2.0, 2.0),
    "A_MFJ1": (-2.0, 2.0),
    # Ring finger.
    "A_RFJ3": (-2.0, 2.0),
    "A_RFJ2": (-2.0, 2.0),
    "A_RFJ1": (-2.0, 2.0),
    # Little finger.
    "A_LFJ4": (-2.0, 2.0),
    "A_LFJ3": (-2.0, 2.0),
    "A_LFJ2": (-2.0, 2.0),
    "A_LFJ1": (-2.0, 2.0),
    # Thumb.
    "A_THJ4": (-2.0, 2.0),
    "A_THJ3": (-2.0, 2.0),
    "A_THJ2": (-2.0, 2.0),
    "A_THJ1": (-2.0, 2.0),
    "A_THJ0": (-2.0, 2.0),
}

# Joint velocity limits, in rad/s.
# Taken from company's github repo^[2] by parsing the XACRO files.
# NOTE(kevin): It seems all the actuators have the same velocity limits.
VELOCITY_LIMITS: Dict[str, Tuple[float, float]] = {
    # Wrist.
    "A_WRJ1": (-2.0, 2.0),
    "A_WRJ0": (-2.0, 2.0),
    # First finger.
    "A_FFJ3": (-2.0, 2.0),
    "A_FFJ2": (-2.0, 2.0),
    "A_FFJ1": (-2.0, 2.0),
    # Middle finger.
    "A_MFJ3": (-2.0, 2.0),
    "A_MFJ2": (-2.0, 2.0),
    "A_MFJ1": (-2.0, 2.0),
    # Ring finger.
    "A_RFJ3": (-2.0, 2.0),
    "A_RFJ2": (-2.0, 2.0),
    "A_RFJ1": (-2.0, 2.0),
    # Little finger.
    "A_LFJ4": (-2.0, 2.0),
    "A_LFJ3": (-2.0, 2.0),
    "A_LFJ2": (-2.0, 2.0),
    "A_LFJ1": (-2.0, 2.0),
    # Thumb.
    "A_THJ4": (-2.0, 2.0),
    "A_THJ3": (-2.0, 2.0),
    "A_THJ2": (-2.0, 2.0),
    "A_THJ1": (-2.0, 2.0),
    "A_THJ0": (-2.0, 2.0),
}

# ====================== #
# Tendon constants
# ====================== #


"""Tendons of the Shadow Hand.

These are used to model the underactuation of the *FJ0 and *FJ1 joints of the main
fingers. A tendon is defined for each *FJ0-*FJ1 pair, and an actuator is used to
drive it.
"""
TENDONS: Tuple[str, ...] = (
    "FFT1",
    "MFT1",
    "RFT1",
    "LFT1",
)


# The total number of tendons.
NUM_TENDONS: int = len(TENDONS)

# Mapping from `Tendons` to `Joints` pair.
TENDON_JOINT_MAPPING: Dict[str, Tuple[str, str]] = {
    "FFT1": ("FFJ0", "FFJ1"),  # First finger.
    "MFT1": ("MFJ0", "MFJ1"),  # Middle finger.
    "RFT1": ("RFJ0", "RFJ1"),  # Ring finger.
    "LFT1": ("LFJ0", "LFJ1"),  # Little finger.
}

# Mapping from `Tendons` to the `Actuators` that drives it.
TENDON_ACTUATOR_MAPPING: Dict[str, str] = {
    "FFT1": "A_FFJ1",  # First finger.
    "MFT1": "A_MFJ1",  # Middle finger.
    "RFT1": "A_RFJ1",  # Ring finger.
    "LFT1": "A_LFJ1",  # Little finger.
}
# Reverse mapping of `TENDON_ACTUATOR_MAPPING`.
ACTUATOR_TENDON_MAPPING: Dict[str, str] = {
    v: k for k, v in TENDON_ACTUATOR_MAPPING.items()
}

# ====================== #
# Fingertip constants
# ====================== #

FINGERTIP_NAMES: Tuple[str, ...] = (
    "fftip",
    "mftip",
    "rftip",
    "lftip",
    "thtip",
)

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

# Mapping from finger `Components` to its associated geoms.
FINGER_GEOM_MAPPING: Dict[str, Tuple[str, ...]] = {
    "thumb": ("thproximal", "thmiddle", "thdistal"),
    "first": ("ffproximal", "ffmiddle", "ffdistal"),
    "middle": ("mfproximal", "mfmiddle", "mfdistal"),
    "ring": ("rfproximal", "rfmiddle", "rfdistal"),
    "little": ("lfproximal", "lfmiddle", "lfdistal"),
}
