"""Tasks involving in-hand object re-orientation."""

import collections
import dataclasses
from typing import Dict, Optional

import numpy as np
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.entities.props import primitive
from dm_control.utils import containers
from dm_control.utils import rewards as reward_utils
from dm_robotics.transformations import transformations as tr

from dexterity import effector
from dexterity import effectors
from dexterity import goal
from dexterity import task
from dexterity.manipulation import arenas
from dexterity.manipulation import props
from dexterity.manipulation.goals import prop_orientation
from dexterity.manipulation.shared import cameras
from dexterity.manipulation.shared import constants
from dexterity.manipulation.shared import observations
from dexterity.manipulation.shared import rewards
from dexterity.manipulation.shared import tags
from dexterity.manipulation.shared import workspaces
from dexterity.models.hands import dexterous_hand
from dexterity.models.hands import shadow_hand_e
from dexterity.utils import mujoco_collisions


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
# Reward shaping coefficients.
_ORIENTATION_WEIGHT = 1.0
_SUCCESS_BONUS_WEIGHT = 800.0
_ACTION_SMOOTHING_WEIGHT = -0.1  # NOTE(kevin): negative sign.

# Timestep of the physics simulation.
_PHYSICS_TIMESTEP: float = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP: float = 0.025

# The maximum number of consecutive solves until the task is terminated.
_SUCCESSED_NEEDED: int = 1

# The maximum allowed time for reaching the current target, in seconds.
_MAX_STEPS_SINGLE_SOLVE: int = 300
_MAX_TIME_SINGLE_SOLVE: float = _MAX_STEPS_SINGLE_SOLVE * _CONTROL_TIMESTEP

_STEPS_BEFORE_MOVING_TARGET: int = 5

_BBOX_SIZE = 0.05
_WORKSPACE = Workspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-_BBOX_SIZE / 2, -0.13 - _BBOX_SIZE / 2, 0.16),
        upper=(+_BBOX_SIZE / 2, -0.13 + _BBOX_SIZE / 2, 0.16),
    ),
)

# Observable settings.
_FREEPROP_OBSERVABLES = observations.ObservableNames(
    prop_pose=("position", "orientation", "linear_velocity", "angular_velocity"),
)
_TARGETPROP_OBSERVABLES = observations.ObservableNames(
    prop_pose=("orientation",),
)

SUITE = containers.TaggedTasks()


class ReOrient(task.GoalTask):
    """Manipulate an object to a goal orientation."""

    def __init__(
        self,
        arena: composer.Arena,
        hand: dexterous_hand.DexterousHand,
        hand_effector: effector.Effector,
        goal_generator: goal.GoalGenerator,
        prop: primitive.Primitive,
        hint_prop: primitive.Primitive,
        workspace: Workspace = _WORKSPACE,
        fall_termination: bool = True,
        success_threshold: float = _ORIENTATION_THRESHOLD,
        successes_needed: int = _SUCCESSED_NEEDED,
        steps_before_changing_goal: int = _STEPS_BEFORE_MOVING_TARGET,
        max_time_per_goal: Optional[float] = _MAX_TIME_SINGLE_SOLVE,
        control_timestep: float = _CONTROL_TIMESTEP,
        physics_timestep: float = _PHYSICS_TIMESTEP,
    ) -> None:
        """Construct a new `ReOrient` task."""

        super().__init__(
            arena=arena,
            hands=[hand],
            hand_effectors=[hand_effector],
            goal_generator=goal_generator,
            success_threshold=success_threshold,
            successes_needed=successes_needed,
            steps_before_changing_goal=steps_before_changing_goal,
            max_time_per_goal=max_time_per_goal,
        )

        self._fall_termination = fall_termination

        # Attach the hand to the arena.
        arena.attach_offset(hand, position=_HAND_POS, quaternion=_HAND_QUAT)

        # Add prop to the arena.
        arena.add_free_entity(prop)
        self._prop = prop

        # Translucent, contactless prop with no observables. This is used to provide a
        # visual hint of the goal state.
        _hintify(hint_prop, _HINT_ALPHA)
        arena.attach_offset(hint_prop, position=_HINT_POS)
        self._hint_prop = hint_prop

        self._prop_placer = initializers.PropPlacer(
            props=[self._prop],
            position=distributions.Uniform(
                workspace.prop_bbox.lower,
                workspace.prop_bbox.upper,
            ),
            quaternion=rotations.UniformQuaternion(),
            settle_physics=False,
        )

        # Add a closeup camera, used for rendering.
        arena.mjcf_model.worldbody.add(
            "camera", **dataclasses.asdict(cameras.FRONT_CLOSE)
        )

        # Visual debugging.
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower,
            upper=workspace.prop_bbox.upper,
            rgba=constants.GREEN,
            name="prop_spawn_area",
            visible=False,
        )

        self.set_timesteps(control_timestep, physics_timestep)

    @property
    def hand(self) -> dexterous_hand.DexterousHand:
        return self.hands[0]

    @property
    def hand_effector(self) -> effector.Effector:
        return self.hand_effectors[0]

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().initialize_episode(physics, random_state)

        self._hint_prop.set_pose(physics=physics, quaternion=self._goal)
        self._prop_placer(physics=physics, random_state=random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        super().before_step(physics, action, random_state)

        if self._goal_changed:
            self._hint_prop.set_pose(physics=physics, quaternion=self._goal)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        super().after_step(physics, random_state)

        self._failure_termination = False
        if self._fall_termination:
            if self._is_prop_fallen(physics):
                self._failure_termination = True

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        should_terminate = super().should_terminate_episode(physics)
        return should_terminate or self._failure_termination

    def get_reward(self, physics: mjcf.Physics) -> float:
        shaped_reward = _get_shaped_reorientation_reward(
            physics,
            self._goal_distance,
        )
        return rewards.weighted_average(shaped_reward)

    def get_discount(self, physics: mjcf.Physics) -> float:
        if self._failure_termination:
            return 1.0
        return super().get_discount(physics)

    # Helper methods.

    def _is_prop_fallen(self, physics: mjcf.Physics) -> bool:
        """Returns True if the prop has fallen from the hand."""
        return mujoco_collisions.has_collision(
            physics=physics,
            collision_geom_prefix_1=[f"{self._prop.name}/"],
            collision_geom_prefix_2=[self._arena.ground.full_identifier],
        )


def _get_shaped_reorientation_reward(
    physics: mjcf.Physics, goal_distance: np.ndarray
) -> Dict[str, rewards.Reward]:
    """Returns a tuple of shaping reward components, as defined in [1].

    The reward is a weighted sum of the following components:
        - orientation reward: The inverse of the absolute value of the angular error
            between the prop's current orientation and the goal orientation.
        - success reward: 1.0 if the the angular error is within a tolerance and 0.0
            otherwise.
        - action smoothness reward: The negative of the squared L2 norm of the control
            action.

    References:
        [1]: A System for General In-Hand Object Re-Orientation,
        https://arxiv.org/abs/2111.03043
    """
    shaped_reward = collections.OrderedDict()

    # Orientation component.
    distance = float(goal_distance[0])
    orientation_reward = 1.0 / (distance + _ORIENTATION_EPS)
    shaped_reward["orientation"] = rewards.Reward(
        value=orientation_reward, weight=_ORIENTATION_WEIGHT
    )

    # Success bonus component.
    success_bonus_reward = reward_utils.tolerance(
        x=distance,
        bounds=(0, _ORIENTATION_THRESHOLD),
        margin=0.0,
    )
    assert isinstance(success_bonus_reward, float)
    shaped_reward["success_bonus"] = rewards.Reward(
        value=success_bonus_reward,
        weight=_SUCCESS_BONUS_WEIGHT,
    )

    # Action smoothing component.
    action_smoothing_reward = np.linalg.norm(physics.data.ctrl) ** 2
    assert isinstance(action_smoothing_reward, float)
    shaped_reward["action_smoothing"] = rewards.Reward(
        value=action_smoothing_reward,
        weight=_ACTION_SMOOTHING_WEIGHT,
    )

    return shaped_reward


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


def reorient_task(
    observation_set: observations.ObservationSet,
) -> composer.Task:
    """Configure and instantiate a `ReOrient` task."""
    arena = arenas.Standard()

    hand = shadow_hand_e.ShadowHandSeriesE(
        observable_options=observations.make_options(
            observation_set.value,
            observations.HAND_OBSERVABLES,
        ),
    )

    hand_effector = effectors.HandEffector(hand=hand, hand_name=hand.name)

    prop_obs_options = observations.make_options(
        observation_set.value, _FREEPROP_OBSERVABLES
    )
    prop = props.OpenAICube(
        size=_PROP_SIZE, observable_options=prop_obs_options, name="prop"
    )

    target_prop_obs_options = observations.make_options(
        observation_set.value, _TARGETPROP_OBSERVABLES
    )
    hint_prop = props.OpenAICube(
        size=_PROP_SIZE,
        observable_options=target_prop_obs_options,
        name="target_prop",
    )

    goal_generator = prop_orientation.PropOrientation(prop=prop)

    return ReOrient(
        arena=arena,
        hand=hand,
        hand_effector=hand_effector,
        prop=prop,
        hint_prop=hint_prop,
        goal_generator=goal_generator,
    )


@SUITE.add(tags.STATE)
def state_dense() -> composer.Task:
    return reorient_task(
        observation_set=observations.ObservationSet.STATE_ONLY,
    )
