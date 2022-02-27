"""Global constants used in the inhand_manipulation tasks."""

from typing import Tuple

_Rgba = Tuple[float, float, float, float]

# MuJoCo physics timestep.
PHYSICS_TIMESTEP: float = 0.001

# Interval between agent actions, in seconds.
CONTROL_TIMESTEP: float = 0.04

# Predefined RGBA values
RED: _Rgba = (1.0, 0.0, 0.0, 0.3)
GREEN: _Rgba = (0.0, 1.0, 0.0, 0.3)
BLUE: _Rgba = (0.0, 0.0, 1.0, 0.3)
CYAN: _Rgba = (0.0, 1.0, 1.0, 0.3)
MAGENTA: _Rgba = (1.0, 0.0, 1.0, 0.3)
YELLOW: _Rgba = (1.0, 1.0, 0.0, 0.3)

# Invisible group for task-related sites.
TASK_SITE_GROUP: int = 3
