"""Tasks involving in-hand object re-orientation."""

import collections
import dataclasses
from typing import Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.utils import rewards
from dm_robotics.transformations import transformations as tr

from shadow_hand import arena
from shadow_hand import hand
from shadow_hand.models.hands import shadow_hand_e
from shadow_hand.tasks.inhand_manipulation import props
from shadow_hand.tasks.inhand_manipulation.shared import arenas
from shadow_hand.tasks.inhand_manipulation.shared import cameras
from shadow_hand.tasks.inhand_manipulation.shared import constants
from shadow_hand.tasks.inhand_manipulation.shared import observations
from shadow_hand.tasks.inhand_manipulation.shared import registry
from shadow_hand.tasks.inhand_manipulation.shared import tags
from shadow_hand.tasks.inhand_manipulation.shared import workspaces
from shadow_hand.utils import geometry_utils


@dataclasses.dataclass(frozen=True)
class Workspace:
    prop_bbox: workspaces.BoundingBox


# The position of the hand relative in the world frame, in meters.
_HAND_POS = (0, 0.2, 0.1)
# The orientation of the hand relative to the world frame.
_HAND_QUAT = tr.axisangle_to_quat(
    np.pi * np.array([0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
)

# Alpha value of the visual goal hint.
_HINT_ALPHA = 0.4
# Position of the hint in the world frame, in meters.
_HINT_POS = (0.12, 0.0, 0.15)

# Size of the prop, in meters.
_PROP_SIZE = 0.02

# Fudge factor for taking the inverse of the orientation error, in radians.
_ORIENTATION_EPS = 0.1
# Threshold for successful orientation, in radians.
_ORIENTATION_THRESHOLD = 0.1
_ORIENTATION_COEF = 1.0
_SUCCESS_BONUS_COEF = 800.0
_ACTION_SMOOTHING_COEF = -0.1  # NOTE(kevin): negative sign.

_WORKSPACE = Workspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.02, -0.22, 0.16),
        upper=(+0.02, -0.15, 0.19),
    ),
)

# Observable settings.
_FREEPROP_OBSERVABLES = observations.ObservableNames(
    prop_pose=("position", "orientation", "linear_velocity", "angular_velocity"),
)
_TARGETPROP_OBSERVABLES = observations.ObservableNames(
    prop_pose=("orientation",),
)
_HAND_OBSERVABLES = observations.ObservableNames(
    proprio=(
        "joint_positions",
        "joint_velocities",
        "fingerip_positions",
        "fingertip_orientations",
        "fingertip_linear_velocities",
        "fingertip_angular_velocities",
    ),
)


class ReOrient(composer.Task):
    """Manipulate an object to a goal orientation.

    The goal orientation is sampled from SO(3) or
    """

    def __init__(
        self,
        arena: arena.Arena,
        hand: hand.Hand,
        obs_settings: observations.ObservationSettings,
        workspace: Workspace,
        restrict_orientation: bool = False,
        control_timestep: float = constants.CONTROL_TIMESTEP,
        physics_timestep: float = constants.PHYSICS_TIMESTEP,
    ) -> None:

        self._arena = arena
        self._hand = hand
        self._restrict_orientation = restrict_orientation

        # Attach the hand to the arena.
        hand_attachment_site = arena.mjcf_model.worldbody.add(
            "site",
            type="sphere",
            pos=_HAND_POS,
            quat=_HAND_QUAT,
            size="0.01",
        )
        self._arena.attach(hand, hand_attachment_site)

        # Add custom cameras obserbables.
        self._task_observables = cameras.add_camera_observables(
            arena,
            obs_settings,
            cameras.FRONT_CLOSE,
            cameras.TOP_DOWN,
            cameras.LEFT_CLOSE,
            cameras.RIGHT_CLOSE,
        )

        # Add prop.
        prop_obs_options = observations.make_options(
            obs_settings, _FREEPROP_OBSERVABLES
        )
        self._prop = props.OpenAICube(
            size=[_PROP_SIZE] * 3, observable_options=prop_obs_options, name="prop"
        )
        arena.add_free_entity(self._prop)

        # Translucent, contactless prop with no observables. This is used to provide a
        # visual hint of the goal state.
        target_prop_obs_options = observations.make_options(
            obs_settings, _TARGETPROP_OBSERVABLES
        )
        self._hint_prop = props.OpenAICube(
            size=[_PROP_SIZE] * 3,
            observable_options=target_prop_obs_options,
            name="target_prop",
        )
        _hintify(self._hint_prop, _HINT_ALPHA)
        arena.attach_offset(self._hint_prop, offset=_HINT_POS)

        # Place the prop slightly above the hand.
        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(
                workspace.prop_bbox.lower,
                workspace.prop_bbox.upper,
            ),
            quaternion=rotations.UniformQuaternion(),
            settle_physics=False,
        )

        # Add angular difference between prop and target prop as an observable.
        angular_diff_observable = observable.Generic(self._get_quaternion_difference)
        angular_diff_observable.configure(**dataclasses.asdict(obs_settings.prop_pose))
        self._task_observables["angular_difference"] = angular_diff_observable

        # Visual debugging.
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower,
            upper=workspace.prop_bbox.upper,
            rgba=constants.GREEN,
            name="prop_spawn_area",
            visible=False,
        )

        self.set_timesteps(
            control_timestep=control_timestep,
            physics_timestep=physics_timestep,
        )

        if restrict_orientation:
            self._prop_orientation_sampler = workspaces.uniform_z_rotation
        else:
            self._prop_orientation_sampler = rotations.UniformQuaternion()

    def _get_quaternion_difference(self, physics: mjcf.Physics) -> np.ndarray:
        """Returns the quaternion difference between the prop and the target prop."""
        prop_quat = physics.bind(self._prop.orientation).sensordata
        target_prop_quat = physics.bind(self._hint_prop.orientation).sensordata
        return tr.quat_diff_active(source_quat=prop_quat, target_quat=target_prop_quat)

    @property
    def task_observables(self) -> collections.OrderedDict:
        return self._task_observables

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def hand(self) -> composer.Entity:
        return self._hand

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        # Randomly sample a starting configuration for the prop.
        self._prop_placer(physics=physics, random_state=random_state)

        # Randomly sample a goal orientation and use it to configure the orientation of
        # the translucent hint prop.
        self._goal_quat = self._prop_orientation_sampler(random_state=random_state)
        self._hint_prop.set_pose(physics=physics, quaternion=self._goal_quat)

    def get_reward(self, physics: mjcf.Physics) -> float:
        return _get_shaped_reorientation_reward(
            physics,
            prop_quat=physics.bind(self._prop.orientation).sensordata,
            goal_quat=self._goal_quat,
        )


def _get_shaped_reorientation_reward(
    physics: mjcf.Physics,
    prop_quat: np.ndarray,
    goal_quat: np.ndarray,
) -> float:
    """Returns a shaped reward for re-orientation, as described in [1].

    The reward is a weighted sum of the following components:
        - orientation reward: The inverse of the absolute value of the angular error
            between the prop's current orientation and the goal orientation.
        - success reward: 1.0 if the the angular error is within a tolerance and 0.0
            otherwise.
        - action smoothness reward: The negative of the squared L2 norm of the control
            action.

    Args:
        physics: An `mjcf.Physics` instance.
        prop_quat: The current orientation of the prop, as a quaternion.
        goal_quat: The goal orientation of the prop, as a quaternion.

    References:
        [1]: A System for General In-Hand Object Re-Orientation,
        https://arxiv.org/abs/2111.03043
    """
    # Orientation component.
    angular_error = np.linalg.norm(
        geometry_utils.get_orientation_error(to_quat=prop_quat, from_quat=goal_quat)
    )
    angular_error_abs = np.abs(angular_error)
    orientation_reward = 1.0 / (angular_error_abs + _ORIENTATION_EPS)

    # Success bonus component.
    success_bonus_reward = rewards.tolerance(
        x=angular_error_abs,
        bounds=(0, _ORIENTATION_THRESHOLD),
        margin=0.0,
    )

    # Action smoothing component.
    action_smoothing_reward = np.linalg.norm(physics.data.ctrl) ** 2

    return (
        orientation_reward * _ORIENTATION_COEF
        + success_bonus_reward * _SUCCESS_BONUS_COEF
        + action_smoothing_reward * _ACTION_SMOOTHING_COEF
    )


def _replace_alpha(rgba: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Replaces the alpha value of a color tuple."""
    new_rgba = rgba.copy()
    new_rgba[3] = alpha
    return new_rgba


def _hintify(entity: composer.Entity, alpha: Optional[float] = None) -> None:
    """Modifies an entity for use as a visual hint.

    Specifically, contacts are disabled for all geoms within the entity, and its bodies
    are converted to mocap bodies which are viewed as fixed from the perspective of the
    dynamics. Additionally, the geom alpha values can be overriden to render the geoms
    as translucent.
    """
    for subentity in entity.iter_entities():
        if (
            alpha is not None
            and subentity.mjcf_model.default.geom is not None
            and subentity.mjcf_model.default.geom.rgba is not None
        ):
            subentity.mjcf_model.default.geom.rgba = _replace_alpha(
                subentity.mjcf_model.default.geom.rgba, alpha=alpha
            )
        for body in subentity.mjcf_model.find_all("body"):
            body.mocap = "true"
        for geom in subentity.mjcf_model.find_all("geom"):
            if alpha is not None and geom.rgba is not None:
                geom.rgba = _replace_alpha(geom.rgba, alpha=alpha)
            # This deals with textures.
            if alpha is not None and geom.material is not None:
                material = subentity.mjcf_model.find("material", geom.material)
                material.rgba = _replace_alpha(material.rgba, alpha=alpha)
            geom.contype = 0
            geom.conaffinity = 0


def _reorient(
    obs_settings: observations.ObservationSettings, restrict_orientation: bool
) -> composer.Task:
    """Configure and instantiate a `ReOrient` task."""
    arena = arenas.Standard()
    hand = shadow_hand_e.ShadowHandSeriesE(
        observable_options=observations.make_options(obs_settings, _HAND_OBSERVABLES),
    )
    return ReOrient(
        arena=arena,
        hand=hand,
        obs_settings=obs_settings,
        workspace=_WORKSPACE,
        restrict_orientation=restrict_orientation,
        control_timestep=constants.CONTROL_TIMESTEP,
        physics_timestep=constants.PHYSICS_TIMESTEP,
    )


@registry.add(tags.FEATURES)
def reorient_so3():
    return _reorient(
        obs_settings=observations.PERFECT_FEATURES, restrict_orientation=False
    )


@registry.add(tags.FEATURES, tags.EASY)
def reorient_z():
    return _reorient(
        obs_settings=observations.PERFECT_FEATURES, restrict_orientation=True
    )
