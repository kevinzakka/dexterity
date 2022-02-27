"""Global constants used in the inhand_manipulation tasks."""

PHYSICS_TIMESTEP: float = 0.001

# Interval between agent actions, in seconds.
CONTROL_TIMESTEP: float = 0.04

# Predefined RGBA values
RED = (1.0, 0.0, 0.0, 0.3)
GREEN = (0.0, 1.0, 0.0, 0.3)
BLUE = (0.0, 0.0, 1.0, 0.3)
CYAN = (0.0, 1.0, 1.0, 0.3)
MAGENTA = (1.0, 0.0, 1.0, 0.3)
YELLOW = (1.0, 1.0, 0.0, 0.3)

# Invisible group for task-related sites.
TASK_SITE_GROUP = 3
