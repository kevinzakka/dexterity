"""A standalone app for visualizing in-hand manipulation tasks."""

import dataclasses
import functools
from typing import Optional, Sequence

import dcargs
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
        if not environment_name or environment_name not in values:
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

    loader = functools.partial(
        inhand_manipulation.load, environment_name=environment_name
    )
    viewer.launch(loader)


if __name__ == "__main__":
    main(dcargs.parse(Args, description=__doc__))
