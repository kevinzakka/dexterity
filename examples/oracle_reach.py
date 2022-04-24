"""Rollout and visualize an oracle policy for the reaching domain."""

import time

import dm_env
import imageio
import numpy as np
from absl import app
from absl import flags

from dexterity import manipulation

_DOMAIN = "reach"
_REACH_TASKS = manipulation.TASKS_BY_DOMAIN[_DOMAIN]

flags.DEFINE_enum("task_name", "state_dense", _REACH_TASKS, "Which reach task to load.")
flags.DEFINE_integer("seed", None, "RNG seed.")
flags.DEFINE_integer("num_episodes", 1, "Number of episodes to run.")
flags.DEFINE_boolean("render", False, "Whether to render the episode and save to disk.")

FLAGS = flags.FLAGS


def main(_) -> None:
    env = manipulation.load(
        domain_name=_DOMAIN,
        task_name=FLAGS.task_name,
        strip_singleton_obs_buffer_dim=True,
        seed=FLAGS.seed,
    )
    action_spec = env.action_spec()

    def oracle(timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused
        qpos = env.task.goal_generator.qpos  # type: ignore
        ctrl = env.task.hand.joint_positions_to_control(qpos)
        ctrl = ctrl.astype(action_spec.dtype)
        return ctrl

    render_kwargs = dict(height=480, width=640, camera_id=0)
    frames = []

    for _ in range(FLAGS.num_episodes):
        timestep = env.reset()
        if FLAGS.render:
            frames.append(env.physics.render(**render_kwargs))
        actions = []
        num_steps = 0
        returns = 0.0
        rewards = []
        episode_start = time.time()
        while True:
            action = oracle(timestep)
            actions.append(action)
            timestep = env.step(action)
            if FLAGS.render:
                frames.append(env.physics.render(**render_kwargs))
            returns += timestep.reward
            rewards.append(timestep.reward)
            num_steps += 1
            if timestep.last():
                break
        episode_time_ms = time.time() - episode_start

        # Print info.
        print(f"Episode time: {episode_time_ms:.2f} seconds.")
        print(f"{num_steps} steps taken.")
        print(f"Total reward: {returns}")
        print(f"Success rate: {env.task.successes}/{env.task.successes_needed}")

    if FLAGS.render:
        imageio.mimsave("temp/oracle_reach.mp4", frames, fps=60, quality=8)


if __name__ == "__main__":
    app.run(main)
