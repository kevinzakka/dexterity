from typing import List

import numpy as np
from dm_control import composer
from dm_control import mjcf

from dexterity.hints import MjcfElement
from dexterity.models.hands import adroit_hand_constants as consts
from dexterity.models.hands import dexterous_hand
from dexterity.utils import mujoco_utils

_PALM_UPRIGHT_POS = (0.0, 0.2, 0.1)
_PALM_UPRIGHT_QUAT = (0.0, 0.0, 0.707106781186, -0.707106781186)


class AdroitHand(dexterous_hand.DexterousHand):
    def _build(
        self,
        name: str = "adroit_hand",
    ) -> None:
        super()._build()

        self._mjcf_root = mjcf.from_path(str(consts.ADROIT_HAND_E_XML))
        self._mjcf_root.model = name

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
    def palm_upright_pose(self) -> dexterous_hand.HandPose:
        return dexterous_hand.HandPose.create(
            xpos=_PALM_UPRIGHT_POS, xquat=_PALM_UPRIGHT_QUAT
        )

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find("body", "forearm")

    @composer.cached_property
    def bodies(self) -> List[MjcfElement]:
        return self.mjcf_model.find_all("body")

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

    @property
    def joint_groups(self) -> List[dexterous_hand.JointGrouping]:
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
        # The Adroit hand is fully-actuated, so qpos = ctrl.
        return control

    def joint_positions_to_control(self, qpos: np.ndarray) -> np.ndarray:
        if qpos.shape != (self.num_joints,):
            raise ValueError(
                f"Expected qpos of shape ({self.num_joints}), got {qpos.shape}"
            )
        # The Adroit hand is fully-actuated, so ctrl = qpos.
        return qpos

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        physics.bind(self._joints).qpos = qpos

    # ================= #
    # Private methods.
    # ================= #

    def _parse_mjcf_elements(self) -> None:
        """Parses MJCF elements that will be exposed as attributes."""
        # Parse joints.
        self._joints: List[mjcf.Element] = self._mjcf_root.find_all("joint")
        if not self._joints:
            raise ValueError("No joints found in the MJCF model.")

        # Parse actuators.
        self._actuators: List[mjcf.Element] = self._mjcf_root.find_all("actuator")
        if not self._actuators:
            raise ValueError("No actuators found in the MJCF model.")

        # Parse tendons.
        self._tendons: List[mjcf.Element] = self._mjcf_root.find_all("tendon")
        if not self._tendons:
            raise ValueError("No tendons found in the MJCF model.")

        # Parse fingertip sites.
        self._fingertip_sites: List[mjcf.Element] = []
        for fingertip_site_name in consts.FINGERTIP_SITE_NAMES:
            fingertip_site_elem = self._mjcf_root.find("site", fingertip_site_name)
            if fingertip_site_elem is None:
                raise ValueError(
                    f"No fingertip site found with name {fingertip_site_name}."
                )
            self._fingertip_sites.append(fingertip_site_elem)

        # Create joint groups.
        self._joint_groups = []
        for name, group in consts.JOINT_GROUP.items():
            joint_group = dexterous_hand.JointGrouping(
                name=name,
                joints=tuple([joint for joint in self._joints if joint.name in group]),
            )
            self._joint_groups.append(joint_group)

    def _add_sensors(self) -> None:
        """Add sensors to the hand's MJCF model."""

        self._add_torque_sensors()

    def _add_torque_sensors(self) -> None:
        self._joint_torque_sensors: List[mjcf.Element] = []
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
