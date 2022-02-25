"""A standalone app for visualizing in-hand manipulation tasks."""

import dataclasses

import dcargs
import numpy as np
from dm_control import viewer

from shadow_hand.tasks import inhand_manipulation


@dataclasses.dataclass
class Args:
    environment_name: str = "reorient_so3"


def main(args: Args) -> None:
    env = inhand_manipulation.load(environment_name=args.environment_name)
    spec = env.action_spec()

    def random_policy(timestep) -> np.ndarray:
        del timestep  # Unused.
        action = np.random.uniform(spec.minimum, spec.maximum, size=spec.shape)
        action[:2] = 0.0  # Disable wrist movement.
        return action

    viewer.launch(env, policy=random_policy)


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
