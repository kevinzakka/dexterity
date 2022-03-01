"""Utility functions for dealing with MuJoCo actuators."""

from typing import Tuple

from shadow_hand import hints


def add_position_actuator(
    elem: hints.MjcfElement,
    elem_type: str,
    qposrange: Tuple[float, float],
    ctrlrange: Tuple[float, float] = (-1.0, 1.0),
    kp: float = 1.0,
    **kwargs,
) -> hints.MjcfElement:
    """Adds a scaled position actuator that is bound to the specified element.

    This is equivalent to MuJoCo's built-in `<position>` actuator where an affine
    transformation is pre-applied to the control signal, such that the minimum control
    value corresponds to the minimum desired position, and the maximum control value
    corresponds to the maximum desired position.

    Args:
        elem: The joint or tendon element that is to be actuated.
        elem_type: The type of the element. Must be either `"joint"` or `"tendon"`.
        qposrange: A sequence of two numbers specifying the allowed range of target
            position.
        ctrlrange: A sequence of two numbers specifying the allowed range of this
            actuator's control signal.
        kp: The gain parameter of this position actuator.
        **kwargs:

    Returns:
        The actuator element.
    """
    kwargs[elem_type] = elem
    slope = (qposrange[1] - qposrange[0]) / (ctrlrange[1] - ctrlrange[0])
    g0 = kp * slope
    b0 = kp * (qposrange[0] - slope * ctrlrange[0])
    b1 = -kp
    b2 = 0
    return elem.root.actuator.add(
        "general",
        biastype="affine",
        gainprm=[g0],
        biasprm=[b0, b1, b2],
        ctrllimited=True,
        ctrlrange=ctrlrange,
        **kwargs,
    )
