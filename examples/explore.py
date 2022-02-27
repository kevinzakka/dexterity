"""A standalone app for visualizing in-hand manipulation tasks."""

import dataclasses
from typing import Optional, Sequence

import dcargs
import dm_env
import numpy as np
from dm_control import viewer

from shadow_hand.tasks import inhand_manipulation

_PROMPT = "Please enter the environment name: "


@dataclasses.dataclass
class Args:
    environment_name: Optional[str] = None


def prompt_environment_name(prompt: str, values: Sequence[str]) -> str:
    environment_name = None
    while not environment_name:
        environment_name = input(prompt)
        if not environment_name or values.index(environment_name) < 0:
            print(f"'{environment_name}' is not a valid environment name.")
            environment_name = None
    return environment_name


def main(args: Args) -> None:
    all_names = list(inhand_manipulation.ALL)

    if args.environment_name is None:
        print("\n ".join(["Available environments:"] + all_names))
        environment_name = prompt_environment_name(_PROMPT, all_names)
    else:
        environment_name = args.environment_name

    env = inhand_manipulation.load(environment_name=environment_name)
    spec = env.action_spec()

    def random_policy(timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused.
        action = np.random.uniform(spec.minimum, spec.maximum, size=spec.shape)
        action[:2] = 0.0  # Disable wrist movement.
        return action

    viewer.launch(env, policy=random_policy)

    timestep = env.reset()
    for key, value in timestep.observation.items():
        print(f"{key}: {value.shape}")
        if key == "angular_difference":
            print(value)


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
