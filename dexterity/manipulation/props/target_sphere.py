from typing import Tuple

from dm_control import composer
from dm_control import mjcf


class TargetSphere(composer.Entity):
    """A non-colliding spherical site that can be used as a fingertip target."""

    def _build(
        self,
        radius: float,
        rgba: Tuple[float, float, float, float],
        name: str = "target",
    ) -> None:
        self._mjcf_root = mjcf.RootElement(model=name)

        self._site = self._mjcf_root.worldbody.add(
            "site",
            type="sphere",
            size=[radius],
            rgba=rgba,
            group=None,  # Make the site visible.
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def site(self):
        return self._site
