from typing import List

import numpy as np
from dm_control import mjcf
from typing_extensions import TypeAlias

from shadow_hand import shadow_hand_e_constants as consts

MjcfElement: TypeAlias = mjcf.element._ElementImpl

_POSITION_DEFAULT_DCLASS = {
    "actuator": {
        "general": {
            "ctrllimited": "true",
            "forcelimited": "false",
        },
    },
}



class ShadowHandSeriesE:
    """Shadow Dexterous Hand E Series."""

    def __init__(
        self,
        name: str = "shadow_hand_e",
        actuation: consts.Actuation = consts.Actuation.POSITION,
        randomize_color: bool = False,
    ) -> None:
        """Initializes the hand.

        Args:
            name: The name of the hand. Used as a prefix in the MJCF name attributes.
            actuation: Instance of `shadow_hand_e_constants.Actuation` specifying which
                actuation method to use.
            randomize_color: Whether to randomize the color of the hand.
        """
        self._mjcf_root = mjcf.from_path(str(consts.SHADOW_HAND_E_XML))

        self._mjcf_root.model = name
        self._actuation = actuation
        self._randomize_color = randomize_color

        self._add_mjcf_elements()
        self._add_actuators()

        if self._randomize_color:
            self._color_hand()

    # Accessors.

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def joints(self) -> List[MjcfElement]:
        """List of joint elements belonging to the hand."""
        return self._joints

    @property
    def actuators(self) -> List[MjcfElement]:
        """List of actuator elements belonging to the hand."""
        return self._actuators

    # Public methods.

    # Private methods.

    def _add_mjcf_elements(self) -> None:
        self._joints = [self._mjcf_root.find("joint", j) for j in consts.JOINT_NAMES]

    def _add_actuators(self) -> None:
        """Adds actuators to the hand."""
        if self._actuation not in consts.Actuation:
            raise ValueError(f"Actuation {self._actuation} is not a valid actuation.")

        if self._actuation == consts.Actuation.TORQUE:
            self._add_torque_actuators()
        elif self._actuation == consts.Actuation.POSITION:
            self._add_position_actuators()

    def _add_torque_actuators(self) -> None:
        ...

    def _add_position_actuators(self) -> None:
        """Adds position actuators and default class attributes to the mjcf model."""
        # Add default class attributes.
        for name, defaults in _POSITION_DEFAULT_DCLASS.items():
            default_dclass = self._mjcf_root.default.add("default", dclass=name)
            for tag, attributes in defaults.items():
                element = getattr(default_dclass, tag)
                for attr_name, attr_val in attributes.items():
                    setattr(element, attr_name, attr_val)

        # Construct list of ctrlrange tuples from act limits and actuation mode.
        ctrl_ranges = [r for r in consts.ACTUATION_LIMITS[self._actuation].values()]

        # Construct list of forcerange tuples from effort limits.

        def add_actuator(i: int) -> MjcfElement:
            actuator = self._sawyer_root.actuator.add(
                "position",
                name=f'j{i}',
                ctrllimited=True,
                forcelimited=True,
                ctrlrange=ctrl_ranges[i],
            )
            # actuator.joint = self._joints[i]
            return actuator

        self._actuators = [add_actuator(i) for i in range(consts.NUM_ACTUATORS)]

    def _color_hand(self) -> None:
        for geom_name in consts.COLORED_GEOMS:
            geom = self._mjcf_root.find("geom", geom_name)
            rgb = np.random.uniform(size=3).flatten()
            rgba = np.append(rgb, 1)
            geom.rgba = rgba


if __name__ == "__main__":
    hand = ShadowHandSeriesE(actuation=consts.Actuation.POSITION, randomize_color=True)

    # Check we can step the physics.
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    physics.step()

    # print("actuators: ", hand.actuators)

    # # Render.
    # import matplotlib.pyplot as plt

    # pixels = physics.render(width=640, height=480)
    # plt.imshow(pixels)
    # plt.show()
