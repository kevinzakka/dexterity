from pathlib import Path
from typing import Tuple

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

FINGERTIP_SITE_NAMES: Tuple[str, ...] = (
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
    "thumb_distal",
)
