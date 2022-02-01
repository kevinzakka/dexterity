"""Print out the joint ranges in radians or degrees."""

import dataclasses
import math
from typing import Dict, Tuple

import dcargs


@dataclasses.dataclass
class Args:
    radians: bool = False
    """If True, prints the joint ranges in radians, otherwise degrees."""

    num_digits: int = 6
    """The number of significant digits to print."""


# Taken from spec sheet.
JOINT_RANGES_DEGREES: Dict[str, Tuple[float, float]] = {
    "WRJ1": (-28, 8),
    "WRJ0": (-40, 28),
    "FFJ3": (-20, 20),
    "FFJ2": (0, 90),
    "FFJ1": (0, 90),
    "FFJ0": (0, 90),
    "MFJ3": (-20, 20),
    "MFJ2": (0, 90),
    "MFJ1": (0, 90),
    "MFJ0": (0, 90),
    "RFJ3": (-20, 20),
    "RFJ2": (0, 90),
    "RFJ1": (0, 90),
    "RFJ0": (0, 90),
    "LFJ4": (0, 45),
    "LFJ3": (-20, 20),
    "LFJ2": (0, 90),
    "LFJ1": (0, 90),
    "LFJ0": (0, 90),
    "THJ4": (-60, 60),
    "THJ3": (0, 70),
    "THJ2": (-12, 12),
    "THJ1": (-30, 30),
    "THJ0": (0, 90),
}


def main(args: Args) -> None:
    for joint, _joint_range in JOINT_RANGES_DEGREES.items():
        if args.radians:
            joint_range = tuple([math.radians(x) for x in _joint_range])
        else:
            joint_range = _joint_range
        print(
            f"{joint}:\t{joint_range[0]:.{args.num_digits}f} {joint_range[1]:.{args.num_digits}f}"
        )


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
