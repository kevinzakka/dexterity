import dataclasses
from typing import Dict, List

import numpy as np
from dm_control import mjcf

from shadow_hand import hand
from shadow_hand.hints import MjcfElement
from shadow_hand.models.hands import shadow_hand_e_constants as consts


# NOTE(kevin): See if we can use the cached_property to speed up property access.
# from dm_control.composer import cached_property


# NOTE(kevin): There's a damping parameter at the <joint> level, which means we in fact
# have a PD-based position controller under the hood. So `kp` here is the proportional
# gain and the damping parameter is related to the derivative gain.
@dataclasses.dataclass(frozen=True)
class _ActuatorParams:
    kp: float = 1.0
    """Position feedback gain for the MuJoCo actuator."""


# TODO(kevin): Need to tune these.
# NOTE(kevin): Some terminology:
# little finger: metacarpal - knuckle - proximal - middle - distal
# thumb: base - proximal - middle - distal
# others: knuckle - proximal - middle - distal
_WR_GAIN = 20.0
_TH_GAIN = 3.0
_KNUCKLE_GAIN = 1.2
_PROXIMAL_GAIN = 1.2
_MIDDLE_DISTAL_GAIN = 0.8
_METACARPAL_GAIN = 1.2
_ACTUATOR_PARAMS = {
    consts.Actuation.POSITION: {
        # Wrist.
        consts.Actuators.A_WRJ1: _ActuatorParams(kp=_WR_GAIN),
        consts.Actuators.A_WRJ0: _ActuatorParams(kp=_WR_GAIN),
        # First finger.
        consts.Actuators.A_FFJ3: _ActuatorParams(kp=_KNUCKLE_GAIN),
        consts.Actuators.A_FFJ2: _ActuatorParams(kp=_PROXIMAL_GAIN),
        consts.Actuators.A_FFJ1: _ActuatorParams(kp=_MIDDLE_DISTAL_GAIN),
        # Middle finger.
        consts.Actuators.A_MFJ3: _ActuatorParams(kp=_KNUCKLE_GAIN),
        consts.Actuators.A_MFJ2: _ActuatorParams(kp=_PROXIMAL_GAIN),
        consts.Actuators.A_MFJ1: _ActuatorParams(kp=_MIDDLE_DISTAL_GAIN),
        # Ring finger.
        consts.Actuators.A_RFJ3: _ActuatorParams(kp=_KNUCKLE_GAIN),
        consts.Actuators.A_RFJ2: _ActuatorParams(kp=_PROXIMAL_GAIN),
        consts.Actuators.A_RFJ1: _ActuatorParams(kp=_MIDDLE_DISTAL_GAIN),
        # Little finger.
        consts.Actuators.A_LFJ4: _ActuatorParams(kp=_METACARPAL_GAIN),
        consts.Actuators.A_LFJ3: _ActuatorParams(kp=_KNUCKLE_GAIN),
        consts.Actuators.A_LFJ2: _ActuatorParams(kp=_PROXIMAL_GAIN),
        consts.Actuators.A_LFJ1: _ActuatorParams(kp=_MIDDLE_DISTAL_GAIN),
        # Thumb.
        consts.Actuators.A_THJ4: _ActuatorParams(kp=_TH_GAIN),
        consts.Actuators.A_THJ3: _ActuatorParams(kp=_TH_GAIN),
        consts.Actuators.A_THJ2: _ActuatorParams(kp=_TH_GAIN),
        consts.Actuators.A_THJ1: _ActuatorParams(kp=_TH_GAIN),
        consts.Actuators.A_THJ0: _ActuatorParams(kp=_TH_GAIN),
    },
}


class ShadowHandSeriesE(hand.Hand):
    """Shadow Dexterous Hand E Series."""

    def _build(
        self,
        name: str = "shadow_hand_e",
        actuation: consts.Actuation = consts.Actuation.POSITION,
    ) -> None:
        """Initializes the hand.

        Args:
            name: The name of the hand. Used as a prefix in the MJCF name attributes.
            actuation: Instance of `shadow_hand_e_constants.Actuation` specifying which
                actuation method to use.
        """
        self._mjcf_root = mjcf.from_path(str(consts.SHADOW_HAND_E_XML))

        self._mjcf_root.model = name
        self._actuation = actuation

        self._parse_mjcf_elements()
        self._add_fingertip_sites()
        self._add_tendons()
        self._add_actuators()
        self._add_sensors()

    # ================= #
    # Accessors.
    # ================= #

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

    @property
    def tendons(self) -> List[MjcfElement]:
        """List of tendon elements belonging to the hand."""
        return self._tendons

    @property
    def joint_torque_sensors(self) -> List[MjcfElement]:
        """List of joint torque sensor elements belonging to the hand."""
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> List[MjcfElement]:
        """List of fingertip site elements belonging to the hand."""
        return self._fingertip_sites

    # ================= #
    # Public methods.
    # ================= #

    @classmethod
    def zero_joint_positions(cls) -> np.ndarray:
        return np.zeros(consts.NUM_JOINTS, dtype=float)

    @classmethod
    def zero_control(cls) -> np.ndarray:
        return np.zeros(consts.NUM_ACTUATORS, dtype=float)

    @classmethod
    def control_to_joint_positions(cls, control: np.ndarray) -> np.ndarray:
        """Maps a 20-D position control command to a 24-D joint position command.

        The control commands for the coupled joints are evenly split amongst them.
        """
        if control.shape != (consts.NUM_ACTUATORS,):
            raise ValueError(
                f"Expected control of shape ({consts.NUM_ACTUATORS}), got"
                f" {control.shape}"
            )
        return consts.CONTROL_TO_POSITION @ control

    @classmethod
    def joint_positions_to_control(cls, qpos: np.ndarray) -> np.ndarray:
        """Maps a 24-D joint position command to a 20-D control command.

        The position commands for the coupled joints are summed up to form the control
        for their corresponding actuator.
        """
        if qpos.shape != (consts.NUM_JOINTS,):
            raise ValueError(
                f"Expected qpos of shape ({consts.NUM_JOINTS}), got {qpos.shape}"
            )
        return consts.POSITION_TO_CONTROL @ qpos

    def actuator_ctrl_range(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns lower and upper bounds on the actuator controls.

        Args:
            physics: An `mjcf.Physics` instance.

        Returns:
            A (20, 2) ndarray containing (lower, upper) control bounds.
        """
        # These are queried from the mjcf instead of the hard-coded constants. This is
        # to account for any possible runtime randomizations.
        return np.array(physics.bind(self._actuators).ctrlrange, copy=True)

    def joint_limits(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns lower and upper bounds on the joint positions.

        Args:
            physics: An `mjcf.Physics` instance.

        Returns:
            A (24, 2) ndarray containing (lower, upper) position bounds.
        """
        # These are queried from the mjcf instead of the hard-coded constants. This is
        # to account for any possible runtime randomizations.
        return np.array(physics.bind(self._joints).range, copy=True)

    def clip_position_control(
        self, physics: mjcf.Physics, control: np.ndarray
    ) -> np.ndarray:
        """Clips the position control vector to the supported range.

        Args:
            physics: An `mjcf.Physics` instance.
            control: The position control vector, of shape (20,).
        """
        bounds = self.actuator_ctrl_range(physics)
        return np.clip(
            a=control,
            a_min=bounds[:, 0],
            a_max=bounds[:, 1],
        )

    def set_position_control(self, physics: mjcf.Physics, control: np.ndarray) -> None:
        """Sets the positions of the joints to the given control command.

        Each coordinate in the control vector is the desired joint angle, or sum of
        joint angles for coupled joints.

        Args:
            physics: An `mjcf.Physics` instance.
            control: The position control vector, of shape (20,).
        """
        if not self.is_position_control_valid(physics, control):
            raise ValueError("Position control command is invalid.")

        physics_actuators = physics.bind(self._actuators)
        physics_actuators.ctrl[:] = control

    def set_joint_angles(self, physics: mjcf.Physics, joint_angles: np.ndarray) -> None:
        """Sets the joints of the hand to a given configuration.

        Also sets the controller to prevent the hand from moving back to the previous
        configuration.
        """
        physics_joints = physics.bind(self._joints)
        physics_actuators = physics.bind(self._actuators)

        physics_joints.qpos[:] = joint_angles
        physics_actuators.ctrl[:] = self.joint_positions_to_control(joint_angles)

    def is_position_control_valid(
        self, physics: mjcf.Physics, control: np.ndarray
    ) -> bool:
        """Returns True if the given position control command is valid."""
        ctrl_bounds = self.actuator_ctrl_range(physics)
        shape_cond = control.shape == (consts.NUM_ACTUATORS,)
        lower_cond = bool(np.all(control >= ctrl_bounds[:, 0] - consts.EPSILON))
        upper_cond = bool(np.all(control <= ctrl_bounds[:, 1] + consts.EPSILON))
        return shape_cond and lower_cond and upper_cond

    def compensate_gravity(self, physics: mjcf.Physics) -> None:
        """Applies Cartesian forces to the bodies of the hand in order to counteract
        gravity.

        This will affect the output of pressure, force, or torque sensors within the
        kinematic chain leading from the worldbody to the bodies that are being
        compensated.

        Args:
            physics: An `mjcf.Physics` instance.
        """
        body_elements = self.mjcf_model.find_all("body")
        gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
        physics_bodies = physics.bind(body_elements)
        physics_bodies.xfrc_applied[:] = -gravity * physics_bodies.mass[..., None]

    # TODO(kevin): Add method to remove gravity compensation.

    # ================= #
    # Private methods.
    # ================= #

    def _parse_mjcf_elements(self) -> None:
        """Parses MJCF elements that will be exposed as attributes."""
        # Parse joints.
        self._joints: List[mjcf.Element] = []
        self._joint_elem_mapping: Dict[consts.Joints, mjcf.Element] = {}
        for joint in consts.Joints:
            joint_elem = self._mjcf_root.find("joint", joint.name)
            if joint_elem is None:
                raise ValueError(f"Could not find joint {joint.name} in MJCF model.")
            self._joints.append(joint_elem)
            self._joint_elem_mapping[joint] = joint_elem

    def _add_palm_site(self) -> None:
        """Adds a site to the palm of the hand."""
        palm_elem = self._mjcf_root.find("body", "palm")
        if palm_elem is None:
            raise ValueError("Could not find palm in MJCF model.")
        self._palm_site: mjcf.Element = palm_elem.add(
            "site",
            name="palm_site",
            pos="0 0 0.0475",
            size="0.001 0.001 0.001",
            type="sphere",
            rgba="1 0 0 1",
        )

    def _add_fingertip_sites(self) -> None:
        """Adds sites to the tips of the fingers of the hand."""
        self._fingertip_sites: List[mjcf.Element] = []
        self._fingertip_site_elem_mapping: Dict[consts.Components, mjcf.Element] = {}
        for finger, tip_name in consts.FINGER_FINGERTIP_MAPPING.items():
            tip_elem = self._mjcf_root.find("body", tip_name)
            if tip_elem is None:
                raise ValueError(f"Could not find fingertip {tip_name} in MJCF model.")
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos="0 0 0",
                # NOTE(kevin): The kwargs below are for visualization purposes.
                size="0.001 0.001 0.001",
                type="sphere",
                rgba="1 0 0 1",
            )
            self._fingertip_sites.append(tip_site)
            self._fingertip_site_elem_mapping[finger] = tip_site

    def _add_tendons(self) -> None:
        """Add tendons to the hand."""
        self._tendons: List[mjcf.Element] = []
        self._tendon_elem_mapping: Dict[consts.Tendons, mjcf.Element] = {}
        for tendon, joints in consts.TENDON_JOINT_MAPPING.items():
            tendon_elem = self._mjcf_root.tendon.add("fixed", name=tendon.name)
            for joint in joints:
                # We set `coef=1` to make the tendon's length equal to the sum of the
                # joint positions.
                tendon_elem.add("joint", joint=joint.name, coef=1.0)
            self._tendons.append(tendon_elem)
            self._tendon_elem_mapping[tendon] = tendon_elem

    def _add_actuators(self) -> None:
        """Adds actuators to the hand."""
        if self._actuation not in consts.Actuation:
            raise ValueError(
                f"Actuation {self._actuation} is not a valid actuation mode."
            )

        if self._actuation == consts.Actuation.POSITION:
            self._add_position_actuators()

    def _add_position_actuators(self) -> None:
        """Adds position actuators to the mjcf model."""

        def add_actuator(act: consts.Actuators, params: _ActuatorParams) -> MjcfElement:
            # Create a position actuator mjcf elem.
            actuator = self._mjcf_root.actuator.add(
                "position",
                name=act.name,
                ctrllimited=True,
                ctrlrange=consts.ACTUATION_LIMITS[self._actuation][act],
                forcelimited=True,
                forcerange=consts.EFFORT_LIMITS[act],
                kp=params.kp,
            )

            # NOTE(kevin): When specifying the joint or tendon to which an actuator is
            # attached, the mjcf element itself is used, rather than the name string.
            if act in consts.ACTUATOR_TENDON_MAPPING:
                actuator.tendon = self._tendon_elem_mapping[
                    consts.ACTUATOR_TENDON_MAPPING[act]
                ]
            else:
                actuator.joint = self._joint_elem_mapping[
                    consts.ACTUATOR_JOINT_MAPPING[act][0]
                ]

            return actuator

        self._actuators: List[mjcf.Element] = []
        self._actuator_elem_mapping: Dict[consts.Actuators, mjcf.Element] = {}
        for actuator, actuator_params in _ACTUATOR_PARAMS[self._actuation].items():
            actuator_elem = add_actuator(actuator, actuator_params)
            self._actuator_elem_mapping[actuator] = actuator_elem
            self._actuators.append(actuator_elem)

    def _add_sensors(self) -> None:
        """Add sensors to the mjcf model."""
        self._add_torque_sensors()

    def _add_torque_sensors(self) -> None:
        """Adds torque sensors to the joints of the hand."""
        self._joint_torque_sensors = []
        self._joint_torque_sensor_elem_mapping = {}
        for joint in consts.Joints:
            joint_elem = self._joint_elem_mapping[joint]
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                # NOTE(kevin): The kwargs below are for visualization purposes.
                size="0.001 0.001 0.001",
                type="box",
                rgba="0 1 0 1",
            )
            # Create a 3-axis torque sensor.
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            self._joint_torque_sensors.append(torque_sensor_elem)
            self._joint_torque_sensor_elem_mapping[joint] = torque_sensor_elem
