from typing import Tuple

import numpy as np
from dm_control import composer
from dm_control import mjcf

from dexterity.hints import MjcfElement
from dexterity.models.hands import dexterous_hand
from dexterity.models.hands import mpl_hand_constants as consts
from dexterity.utils import mujoco_utils

HandSide = dexterous_hand.HandSide


class MPLHand(dexterous_hand.DexterousHand):
    """Modular Prosthetic Limb Hand."""

    def _build(
        self,
        side: HandSide = HandSide.RIGHT,
        name: str = "mpl_hand",
    ) -> None:
        super()._build()

        if side == HandSide.RIGHT:
            self._mjcf_root = mjcf.from_path(str(consts.MPL_HAND_RIGHT_XML))
            self._mjcf_root.model = f"right_{name}"
        else:
            self._mjcf_root = mjcf.from_path(str(consts.MPL_HAND_LEFT_XML))
            self._mjcf_root.model = f"left_{name}"

        self._parse_mjcf_elements()
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

    @composer.cached_property
    def root_body(self) -> MjcfElement:
        return self._mjcf_root.find("body", "forearm")

    @composer.cached_property
    def bodies(self) -> Tuple[MjcfElement, ...]:
        return tuple(self.mjcf_model.find_all("body"))

    @property
    def joints(self) -> Tuple[MjcfElement, ...]:
        """List of joint elements belonging to the hand."""
        return self._joints

    @property
    def actuators(self) -> Tuple[MjcfElement, ...]:
        """List of actuator elements belonging to the hand."""
        return self._actuators

    @property
    def tendons(self) -> Tuple[MjcfElement, ...]:
        """List of tendon elements belonging to the hand."""
        return self._tendons

    @property
    def joint_torque_sensors(self) -> Tuple[MjcfElement, ...]:
        """List of joint torque sensor elements belonging to the hand."""
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Tuple[MjcfElement, ...]:
        """List of fingertip site elements belonging to the hand."""
        return self._fingertip_sites

    @property
    def joint_groups(self) -> Tuple[dexterous_hand.JointGrouping, ...]:
        return self._joint_groups

    # ================= #
    # Public methods.
    # ================= #

    def control_to_joint_positions(self, control: np.ndarray) -> np.ndarray:
        if control.shape != (self.num_actuators,):
            raise ValueError(
                f"Expected control of shape ({self.num_actuators}), got"
                f" {control.shape}"
            )
        return consts.CONTROL_TO_POSITION @ control

    def joint_positions_to_control(self, qpos: np.ndarray) -> np.ndarray:
        if qpos.shape != (self.num_joints,):
            raise ValueError(
                f"Expected qpos of shape ({self.num_joints}), got {qpos.shape}"
            )
        return consts.POSITION_TO_CONTROL @ qpos

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        physics.bind(self._joints).qpos = qpos

    def _postprocess_sampled_joint_angles(self, qpos: np.ndarray) -> np.ndarray:
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
        joints = self._mjcf_root.find_all("joint")
        if not joints:
            raise ValueError("No joints found in the MJCF model.")
        self._joints = tuple(joints)

        # Parse actuators.
        actuators = self._mjcf_root.find_all("actuator")
        if not actuators:
            raise ValueError("No actuators found in the MJCF model.")
        self._actuators = tuple(actuators)

        # Parse tendons.
        tendons = self._mjcf_root.find_all("tendon")
        if not tendons:
            raise ValueError("No tendons found in the MJCF model.")
        self._tendons = tuple(tendons)

        # Parse fingertip sites.
        fingertip_sites = []
        for fingertip_site_name in consts.FINGERTIP_SITE_NAMES:
            fingertip_site_elem = self._mjcf_root.find("site", fingertip_site_name)
            if fingertip_site_elem is None:
                raise ValueError(
                    f"No fingertip site found with name {fingertip_site_name}."
                )
            fingertip_sites.append(fingertip_site_elem)
        self._fingertip_sites = tuple(fingertip_sites)

        # Create joint groups.
        joint_groups = []
        for name, group in consts.JOINT_GROUP.items():
            joint_group = dexterous_hand.JointGrouping(
                name=name,
                joints=tuple([joint for joint in self._joints if joint.name in group]),
            )
            joint_groups.append(joint_group)
        self._joint_groups = tuple(joint_groups)

    def _add_sensors(self) -> None:
        """Add sensors to the hand's MJCF model."""

        self._add_torque_sensors()

    def _add_torque_sensors(self) -> None:
        joint_torque_sensors = []
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
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)
