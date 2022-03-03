from typing import Dict, List

import numpy as np
from dm_control import composer
from dm_control import mjcf

from shadow_hand.hints import MjcfElement
from shadow_hand.models.hands import fingered_hand
from shadow_hand.models.hands import shadow_hand_e_actuation as sh_actuation
from shadow_hand.models.hands import shadow_hand_e_constants as consts
from shadow_hand.utils import mujoco_actuation

_EPSILON: float = 1e-6


class ShadowHandSeriesE(fingered_hand.FingeredHand):
    """Shadow Dexterous Hand E Series."""

    def _build(
        self,
        name: str = "shadow_hand_e",
        actuation: sh_actuation.Actuation = sh_actuation.Actuation.POSITION,
    ) -> None:
        """Initializes the hand.

        Args:
            name: The name of the hand. Used as a prefix in the MJCF name attributes.
            actuation: Instance of `shadow_hand_e_actuation.Actuation` specifying which
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
        lower_cond = bool(np.all(control >= ctrl_bounds[:, 0] - _EPSILON))
        upper_cond = bool(np.all(control <= ctrl_bounds[:, 1] + _EPSILON))
        return shape_cond and lower_cond and upper_cond

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
                group=composer.SENSOR_SITES_GROUP,
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
        if self._actuation not in sh_actuation.Actuation:
            raise ValueError(
                f"Actuation {self._actuation} is not a valid actuation mode."
            )

        if self._actuation == sh_actuation.Actuation.POSITION:
            self._add_position_actuators()
        elif self._actuation == sh_actuation.Actuation.EFFORT:
            raise NotImplementedError("Effort actuation is not yet implemented.")

    def _add_position_actuators(self) -> None:
        """Adds position actuators to the mjcf model."""

        self._mjcf_root.default.general.forcelimited = "true"
        self._mjcf_root.actuator.motor.clear()

        self._actuators: List[mjcf.Element] = []
        self._actuator_elem_mapping: Dict[consts.Actuators, mjcf.Element] = {}
        for actuator, actuator_params in sh_actuation.ACTUATOR_PARAMS[
            self._actuation
        ].items():
            if actuator in consts.ACTUATOR_TENDON_MAPPING:
                elem = self._tendon_elem_mapping[
                    consts.ACTUATOR_TENDON_MAPPING[actuator]
                ]
                elem_type = "tendon"
            else:
                elem = self._joint_elem_mapping[
                    consts.ACTUATOR_JOINT_MAPPING[actuator][0]
                ]
                elem_type = "joint"
                elem.damping = actuator_params.damping

            qposrange = sh_actuation.ACTUATION_LIMITS[self._actuation][actuator]
            actuator_elem = mujoco_actuation.add_position_actuator(
                elem=elem,
                elem_type=elem_type,
                qposrange=qposrange,
                ctrlrange=qposrange,
                kp=actuator_params.kp,
                forcerange=consts.EFFORT_LIMITS[actuator],
                name=actuator.name,
            )

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
                group=composer.SENSOR_SITES_GROUP,
            )
            # Create a 3-axis torque sensor.
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            self._joint_torque_sensors.append(torque_sensor_elem)
            self._joint_torque_sensor_elem_mapping[joint] = torque_sensor_elem
