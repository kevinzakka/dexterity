from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dexterity import _SRC_ROOT

# Path to the MPL XML file.
MPL_HAND_RIGHT_XML: Path = (
    _SRC_ROOT
    / "models"
    / "vendor"
    / "mpl"
    / "mpl_hand_description"
    / "mjcf"
    / "mpl_right.xml"
)
MPL_HAND_LEFT_XML: Path = (
    _SRC_ROOT
    / "models"
    / "vendor"
    / "mpl"
    / "mpl_hand_description"
    / "mjcf"
    / "mpl_left.xml"
)

JOINTS: Tuple[str, ...] = (
    "wrist_PRO",
    "wrist_UDEV",
    "wrist_FLEX",
    "thumb_ABD",
    "thumb_MCP",
    "thumb_PIP",
    "thumb_DIP",
    "index_ABD",
    "index_MCP",
    "index_PIP",
    "index_DIP",
    "middle_MCP",
    "middle_PIP",
    "middle_DIP",
    "ring_ABD",
    "ring_MCP",
    "ring_PIP",
    "ring_DIP",
    "pinky_ABD",
    "pinky_MCP",
    "pinky_PIP",
    "pinky_DIP",
)

NUM_JOINTS: int = len(JOINTS)

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("wrist_PRO", "wrist_UDEV", "wrist_FLEX"),
    "first": ("index_ABD", "index_MCP", "index_PIP", "index_DIP"),
    "middle": ("middle_MCP", "middle_PIP", "middle_DIP"),
    "ring": ("ring_ABD", "ring_MCP", "ring_PIP", "ring_DIP"),
    "little": ("pinky_ABD", "pinky_MCP", "pinky_PIP", "pinky_DIP"),
    "thumb": ("thumb_ABD", "thumb_MCP", "thumb_PIP", "thumb_DIP"),
}

ACTUATORS: Tuple[str, ...] = (
    "A_wrist_PRO",
    "A_wrist_UDEV",
    "A_wrist_FLEX",
    "A_thumb_ABD",
    "A_thumb_MCP",
    "A_thumb_PIP",
    "A_thumb_DIP",
    "A_index_ABD",
    "A_index_MCP",
    "A_middle_MCP",
    "A_ring_MCP",
    "A_pinky_ABD",
    "A_pinky_MCP",
)

NUM_ACTUATORS: int = len(ACTUATORS)

ACTUATOR_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("A_wrist_PRO", "A_wrist_UDEV", "A_wrist_FLEX"),
    "thumb": ("A_thumb_ABD", "A_thumb_MCP", "A_thumb_PIP", "A_thumb_DIP"),
    "first": ("A_index_ABD", "A_index_MCP"),
    "middle": ("A_middle_MCP",),
    "ring": ("A_ring_MCP",),
    "little": ("A_pinky_ABD", "A_pinky_MCP"),
}

# One-to-many mapping from `Actuators` to the joint(s) it controls.
# The first two joints of each of the main fingers are coupled, which means there is
# only one actuator controlling them via a single tendon.
ACTUATOR_JOINT_MAPPING: Dict[str, Tuple[str, ...]] = {
    # Wrist.
    "A_wrist_PRO": ("wrist_PRO",),
    "A_wrist_UDEV": ("wrist_UDEV",),
    "A_wrist_FLEX": ("wrist_FLEX",),
    # Thumb.
    "A_thumb_ABD": ("thumb_ABD",),
    "A_thumb_MCP": ("thumb_MCP",),
    "A_thumb_PIP": ("thumb_PIP",),
    "A_thumb_DIP": ("thumb_DIP",),
    # First finger.
    "A_index_ABD": ("index_ABD",),
    "A_index_MCP": ("index_MCP", "index_PIP", "index_DIP"),
    # Middle finger.
    "A_middle_MCP": ("middle_MCP", "middle_PIP", "middle_DIP"),
    # Ring finger.
    "A_ring_MCP": ("ring_ABD", "ring_MCP", "ring_PIP", "ring_DIP"),
    # Little finger.
    "A_pinky_ABD": ("pinky_ABD",),
    "A_pinky_MCP": ("pinky_MCP", "pinky_PIP", "pinky_DIP"),
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

FINGERTIP_SITE_NAMES: Tuple[str, ...] = (
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
    "thumb_distal",
)

# Mapping from finger `Components` to its associated geoms.
FINGER_GEOM_MAPPING: Dict[str, Tuple[str, ...]] = {
    "thumb": ("thumb0", "thumb1", "thumb2", "thumb3"),
    "first": ("index0", "index1", "index2", "index3"),
    "middle": ("middle0", "middle1", "middle2", "middle3"),
    "ring": ("ring0", "ring1", "ring2", "ring3"),
    "little": ("pinky0", "pinky1", "pinky2", "pinky3"),
}
