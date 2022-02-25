from pathlib import Path
from typing import Optional

from dm_control import composer, mjcf

from shadow_hand import _SRC_ROOT


ARENA_XML_PATH: Path = _SRC_ROOT / "models" / "arenas" / "assets" / "arena.xml"


class Arena(composer.Arena):
    """An empty arena with a ground plane and a camera."""

    def _build(self, name: Optional[str] = None) -> None:
        super()._build(name=name)

        # Remove the default visual settings in `dm_control.Arena`.
        self._mjcf_root.remove("visual")

        self._mjcf_root.include_copy(
            mjcf.from_path(ARENA_XML_PATH), override_attributes=True
        )

        self._ground = self._mjcf_root.find("geom", "ground")

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to this arena."""
        return self._mjcf_root

    @property
    def ground(self) -> mjcf.Element:
        """Returns the ground plane mjcf element."""
        return self._ground
