from pathlib import Path
from typing import Any, Optional

from dm_control import composer, mjcf

from shadow_hand import _SRC_ROOT


ARENA_XML_PATH: Path = _SRC_ROOT / "models" / "arenas" / "assets" / "arena.xml"


class Arena(composer.Entity):
    """An empty arena with a ground plane and a camera."""

    def _build(self, name: Optional[str] = None) -> None:
        self._mjcf_root = mjcf.from_path(str(ARENA_XML_PATH))
        if name:
            self._mjcf_root.model = name

        self._ground = self._mjcf_root.find("geom", "ground")

    def add_free_entity(self, entity: composer.Entity) -> Any:
        """Includes an entity in the arena as a free-moving body."""
        frame = self.attach(entity)
        frame.add("freejoint")
        return frame

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """Returns the `mjcf.RootElement` object corresponding to this arena."""
        return self._mjcf_root

    @property
    def ground(self) -> mjcf.Element:
        """Returns the ground plane mjcf element."""
        return self._ground
