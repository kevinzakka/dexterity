from typing import List

import numpy as np
from dm_control import composer
from dm_control import mjcf

from dexterity.hints import MjcfElement
from dexterity.models.hands import dexterous_hand
from dexterity.models.hands import dexterous_hand_constants
from dexterity.models.hands import shadow_hand_e_constants as consts
from dexterity.utils import mujoco_utils


class ShadowHandSeriesE(dexterous_hand.DexterousHand):
    """Shadow Dexterous Hand E Series."""

    def _build(self, name: str = "shadow_hand_e") -> None:
        """Initializes the hand.

        Args:
            name: The name of the hand. Used as a prefix in the MJCF name attributes.
        """
        self._mjcf_root = mjcf.from_path(str(consts.SHADOW_HAND_E_XML))
        self._mjcf_root.model = name

        self._parse_mjcf_elements()
        self._add_fingertip_sites()
        self._add_sensors()

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.

        # Apply gravity compensation.
        mujoco_utils.compensate_gravity(physics, self.mjcf_model.find_all("body"))

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
        return self._joints

    @property
    def actuators(self) -> List[MjcfElement]:
        return self._actuators

    @property
    def tendons(self) -> List[MjcfElement]:
        return self._tendons

    @property
    def joint_torque_sensors(self) -> List[MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> List[MjcfElement]:
        return self._fingertip_sites

    @property
    def joint_groups(self) -> List[dexterous_hand_constants.JointGrouping]:
        return self._joint_groups

    # ================= #
    # Public methods.
    # ================= #

    def control_to_joint_positions(self, control: np.ndarray) -> np.ndarray:
        """Maps a 20-D position control command to a 24-D joint position command.

        The control commands for the coupled joints are evenly split amongst them.
        """
        if control.shape != (self.num_actuators,):
            raise ValueError(
                f"Expected control of shape ({self.num_actuators}), got"
                f" {control.shape}"
            )
        return consts.CONTROL_TO_POSITION @ control

    def joint_positions_to_control(self, qpos: np.ndarray) -> np.ndarray:
        """Maps a 24-D joint position command to a 20-D control command.

        The position commands for the coupled joints are summed up to form the control
        for their corresponding actuator.
        """
        if qpos.shape != (self.num_joints,):
            raise ValueError(
                f"Expected qpos of shape ({self.num_joints}), got {qpos.shape}"
            )
        return consts.POSITION_TO_CONTROL @ qpos

    def set_joint_angles(self, physics: mjcf.Physics, joint_angles: np.ndarray) -> None:
        """Sets the joints of the hand to a given configuration."""
        physics.bind(self._joints).qpos = joint_angles

    def sample_joint_angles(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> np.ndarray:
        qpos = random_state.uniform(*physics.bind(self._joints).range.T)

        # Ensure coupled joints have the same joint values.
        for coupled_ids in consts.COUPLED_JOINT_IDS:
            val = qpos[coupled_ids[-1]]
            qpos[coupled_ids] = val

        return qpos

    # ================= #
    # Private methods.
    # ================= #

    def _parse_mjcf_elements(self) -> None:
        """Parses MJCF elements that will be exposed as attributes."""
        # Parse joints.
        self._joints = self._mjcf_root.find_all("joint")
        if not self._joints:
            raise ValueError("No joints found in the MJCF model.")

        # Parse actuators.
        self._actuators = self._mjcf_root.find_all("actuator")
        if not self._actuators:
            raise ValueError("No actuators found in the MJCF model.")

        # Parse tendons.
        self._tendons = self._mjcf_root.find_all("tendon")
        if not self._tendons:
            raise ValueError("No tendons found in the MJCF model.")

        # Create joint groups.
        self._joint_groups = []
        for name, group in consts.JOINT_GROUP.items():
            joint_group = dexterous_hand_constants.JointGrouping(
                name=name,
                joints=tuple([joint for joint in self._joints if joint.name in group]),
            )
            self._joint_groups.append(joint_group)

    def _add_fingertip_sites(self) -> None:
        """Adds sites to the tips of the fingers of the hand."""
        self._fingertip_sites: List[mjcf.Element] = []

        for tip_name in consts.FINGERTIP_NAMES:
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

    def _add_sensors(self) -> None:
        """Add sensors to the hand's MJCF model."""

        self._add_torque_sensors()

    def _add_torque_sensors(self) -> None:
        """Adds torque sensors to the joints of the hand."""
        self._joint_torque_sensors = []

        for joint_elem in self._joints:
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
