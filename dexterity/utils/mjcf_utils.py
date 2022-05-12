from typing import Tuple

from dm_control import mjcf

from dexterity.hints import MjcfElement


def safe_find_all(root: mjcf.RootElement, feature_name: str) -> Tuple[MjcfElement, ...]:
    features = root.find_all(feature_name)
    if not features:
        raise ValueError(f"No {feature_name} found in the MJCF model.")
    return tuple(features)
