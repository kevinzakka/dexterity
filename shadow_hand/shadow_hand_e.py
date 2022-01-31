import dataclasses
from typing import List

import numpy as np
from dm_control import mjcf
from typing_extensions import TypeAlias

from shadow_hand import shadow_hand_e_constants as consts

MjcfElement: TypeAlias = mjcf.element._ElementImpl


@dataclasses.dataclass
class _ActuatorParams:
    # Position feedback gain.
    kp: float = 1.0


# NOTE(kevin): The values of the constants below are not really tuned. At the moment,
# I just loaded the model in the MuJoCo viewer and set the values manually to get
# visually reasonable results.
_ACTUATOR_PARAMS = {
    consts.Actuation.POSITION: {
        # Wrist.
        consts.Actuators.A_WRJ1: _ActuatorParams(kp=20.0),
        consts.Actuators.A_WRJ0: _ActuatorParams(kp=20.0),
        # First finger.
        consts.Actuators.A_FFJ3: _ActuatorParams(kp=2.0),
        consts.Actuators.A_FFJ2: _ActuatorParams(kp=1.0),
        consts.Actuators.A_FFJ1: _ActuatorParams(kp=1.0),
        # Middle finger.
        consts.Actuators.A_MFJ3: _ActuatorParams(kp=2.0),
        consts.Actuators.A_MFJ2: _ActuatorParams(kp=1.0),
        consts.Actuators.A_MFJ1: _ActuatorParams(kp=1.0),
        # Ring finger.
        consts.Actuators.A_RFJ3: _ActuatorParams(kp=2.0),
        consts.Actuators.A_RFJ2: _ActuatorParams(kp=1.0),
        consts.Actuators.A_RFJ1: _ActuatorParams(kp=1.0),
        # Little finger.
        consts.Actuators.A_LFJ4: _ActuatorParams(kp=1.0),
        consts.Actuators.A_LFJ3: _ActuatorParams(kp=2.0),
        consts.Actuators.A_LFJ2: _ActuatorParams(kp=1.0),
        consts.Actuators.A_LFJ1: _ActuatorParams(kp=1.0),
        # Thumb.
        consts.Actuators.A_THJ4: _ActuatorParams(kp=1.0),
        consts.Actuators.A_THJ3: _ActuatorParams(kp=1.0),
        consts.Actuators.A_THJ2: _ActuatorParams(kp=1.0),
        consts.Actuators.A_THJ1: _ActuatorParams(kp=1.0),
        consts.Actuators.A_THJ0: _ActuatorParams(kp=1.0),
    }
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
        self._add_tendons()
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

    @classmethod
    def zero_joint_positions(cls) -> np.ndarray:
        return np.zeros(consts.NUM_JOINTS, dtype=float)

    @classmethod
    def zero_control(cls) -> np.ndarray:
        return np.zeros(consts.NUM_ACTUATORS, dtype=float)

    @classmethod
    def control_to_joint_positions(cls, control: np.ndarray) -> np.ndarray:
        """Map a 20-D position control command to a 24-D joint position array.

        This method will evenly split the control command over the coupled joints.
        """
        # TODO(kevin): Sanity check the control array.
        return consts.CONTROL_TO_POSITION @ control

    @classmethod
    def joint_positions_to_control(cls, qpos: np.ndarray) -> np.ndarray:
        """Map a 24-D joint position array to a 20-D control command array.

        This method will sum up the joint positions for the coupled joints.
        """
        # TODO(kevin): Sanity check the joint position array.
        return consts.POSITION_TO_CONTROL @ qpos

    def set_position_control(self, physics: mjcf.Physics, control: np.ndarray) -> None:
        # Is the control array valid?
        physics_joints = physics.bind(self._joints)
        physics_joints.qpos[:] = control

    # Private methods.

    def _add_mjcf_elements(self) -> None:
        self._joints = [self._mjcf_root.find("joint", j) for j in consts.JOINT_NAMES]

        # TODO(kevin): Add sensors to XML file and parse them here.

    def _add_tendons(self) -> None:
        """Add tendons to the hand."""
        self._tendons = []
        for tendon, joints in consts.TENDON_JOINT_MAPPING.items():
            tendon_elem = self._mjcf_root.tendon.add("fixed", name=tendon.name)
            for joint in joints:
                # We set `coef=1` to make the tendon's length equal to the sum of the
                # joint positions.
                tendon_elem.add("joint", joint=joint.name, coef=1.0)
            self._tendons.append(tendon)

    def _add_actuators(self) -> None:
        """Adds actuators to the hand."""
        if self._actuation not in consts.Actuation:
            raise ValueError(f"Actuation {self._actuation} is not a valid actuation.")

        if self._actuation == consts.Actuation.TORQUE:
            self._add_torque_actuators()
        elif self._actuation == consts.Actuation.POSITION:
            self._add_position_actuators()

    def _add_torque_actuators(self) -> None:
        raise NotImplementedError

    def _add_position_actuators(self) -> None:
        """Adds position actuators to the mjcf model."""

        def add_actuator(act: consts.Actuators, params: _ActuatorParams) -> MjcfElement:
            if act in consts.ACTUATOR_TENDON_MAPPING:
                joint_or_tendon_kwarg = {
                    "tendon": consts.ACTUATOR_TENDON_MAPPING[act].name,
                }
            else:
                joint_or_tendon_kwarg = {
                    "joint": consts.ACTUATOR_JOINT_MAPPING[act][0].name,
                }
            actuator = self._mjcf_root.actuator.add(
                "position",
                name=act.name,
                ctrllimited=True,
                ctrlrange=consts.ACTUATION_LIMITS[self._actuation][act],
                forcelimited=False,
                # forcerange=None,
                kp=params.kp,
                **joint_or_tendon_kwarg,
            )
            return actuator

        self._actuators = []
        for actuator, actuator_params in _ACTUATOR_PARAMS[self._actuation].items():
            self._actuators.append(add_actuator(actuator, actuator_params))

    def _color_hand(self) -> None:
        """Randomly assign an RGB color to the hand components."""
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

    # print(hand.mjcf_model.to_xml_string())

    # with open("./mjcf_hand.xml", "w") as f:
    # f.write(hand.mjcf_model.to_xml_string())

    # qpos = hand.zero_joint_positions()
    # qpos[0] += 1
    # qpos = np.random.randn(consts.NUM_JOINTS)
    # hand.set_position_control(physics, qpos)
    # physics.step()

    # Render.
    import matplotlib.pyplot as plt

    pixels = physics.render(width=640, height=480)
    plt.imshow(pixels)
    plt.show()

    # hand.set_position_control(hand.zero_control())
