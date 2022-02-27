from typing import Optional, Sequence

from dm_control import composer

from shadow_hand import hints


class Arena(composer.Arena):
    """Standard composer arena with added functionality."""

    def attach_offset(
        self,
        entity: composer.Entity,
        offset: Sequence[float],
        attach_site: Optional[hints.MjcfAttachmentFrame] = None,
    ):
        frame = self.attach(entity, attach_site=attach_site)
        frame.pos = offset
        return frame
