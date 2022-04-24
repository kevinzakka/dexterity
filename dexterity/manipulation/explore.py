"""A standalone app for visualizing manipulation tasks."""

from typing import Sequence

from absl import app
from absl import flags
from dm_control import viewer

from dexterity import manipulation
from dexterity.manipulation.wrappers import ActionNoise

flags.DEFINE_enum(
    "environment_name",
    None,
    manipulation.ALL_NAMES,
    "Optional 'domain_name.task_name' pair specifying the environment to load.",
)
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_bool("timeout", True, "Whether episodes should have a time limit.")
flags.DEFINE_float("action_noise", 0.0, "Action noise scale.")

FLAGS = flags.FLAGS


def prompt_environment_name(values: Sequence[str]) -> str:
    environment_name = None
    while not environment_name:
        environment_name = input("Please enter the environment name: ")
        if not environment_name or environment_name not in values:
            print(f"'{environment_name}' is not a valid environment name.")
            environment_name = None
    return environment_name


def main(_) -> None:
    if FLAGS.environment_name is None:
        print("\n ".join(["Available environments:"] + manipulation.ALL_NAMES))
        environment_name = prompt_environment_name(manipulation.ALL_NAMES)
    else:
        environment_name = FLAGS.environment_name

    index = manipulation.ALL_NAMES.index(environment_name)
    domain_name, task_name = manipulation.ALL_TASKS[index]

    env = manipulation.load(
        domain_name=domain_name,
        task_name=task_name,
        seed=FLAGS.seed,
        time_limit=float("inf") if not FLAGS.timeout else None,
    )

    if FLAGS.action_noise > 0.0:
        # By default, the viewer will apply the midpoint action since the action spec is
        # bounded. So this wrapper will add noise centered around this midpoint.
        env = ActionNoise(env, scale=FLAGS.action_noise)

    # Print entity and task observables.
    timestep = env.reset()
    for k, v in timestep.observation.items():
        print(f"{k}: {v.shape}")

    import dm_env
    import numpy as np

    action_spec = env.action_spec()

    def policy(timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused
        qpos = env.task.goal_generator.qpos  # type: ignore
        ctrl = env.task.hand.joint_positions_to_control(qpos)
        ctrl = ctrl.astype(action_spec.dtype)
        return ctrl

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    app.run(main)
