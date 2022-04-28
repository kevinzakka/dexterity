"""Adroit hand constants."""

from pathlib import Path
from typing import Dict, Tuple

from dexterity import _SRC_ROOT

# Path to the Adroit hand MJCF XML file.
ADROIT_HAND_E_XML: Path = (
    _SRC_ROOT
    / "models"
    / "vendor"
    / "adroit"
    / "adroit_hand_description"
    / "mjcf"
    / "adroit_hand.xml"
)

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

NUM_JOINTS: int = len(JOINTS)

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("WRJ1", "WRJ0"),
    "first": ("FFJ3", "FFJ2", "FFJ1", "FFJ0"),
    "middle": ("MFJ3", "MFJ2", "MFJ1", "MFJ0"),
    "ring": ("RFJ3", "RFJ2", "RFJ1", "RFJ0"),
    "little": ("LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"),
    "thumb": ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"),
}

ACTUATORS: Tuple[str, ...] = (
    "A_WRJ1",
    "A_WRJ0",
    "A_FFJ3",
    "A_FFJ2",
    "A_FFJ1",
    "A_FFJ0",
    "A_MFJ3",
    "A_MFJ2",
    "A_MFJ1",
    "A_MFJ0",
    "A_RFJ3",
    "A_RFJ2",
    "A_RFJ1",
    "A_RFJ0",
    "A_LFJ4",
    "A_LFJ3",
    "A_LFJ2",
    "A_LFJ1",
    "A_LFJ0",
    "A_THJ4",
    "A_THJ3",
    "A_THJ2",
    "A_THJ1",
    "A_THJ0",
)

NUM_ACTUATORS: int = len(ACTUATORS)

FINGERTIP_SITE_NAMES: Tuple[str, ...] = (
    "S_fftip",
    "S_mftip",
    "S_rftip",
    "S_lftip",
    "S_thtip",
)

# Mapping from finger `Components` to its associated geoms.
FINGER_GEOM_MAPPING: Dict[str, Tuple[str, ...]] = {
    "thumb": ("V_thproximal", "V_thmiddle", "V_thdistal"),
    "first": ("V_ffproximal", "V_ffmiddle", "V_ffdistal"),
    "middle": ("V_mfproximal", "V_mfmiddle", "V_mfdistal"),
    "ring": ("V_rfproximal", "V_rfmiddle", "V_rfdistal"),
    "little": ("V_lfproximal", "V_lfmiddle", "V_lfdistal"),
}
