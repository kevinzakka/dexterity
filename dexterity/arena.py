from typing import Optional, Sequence, Tuple

from dm_control import composer

from dexterity import hints


class Arena(composer.Arena):
    """Standard composer arena with added functionality."""

    def attach_offset(
        self,
        entity: composer.Entity,
        position: Optional[Sequence[float]] = None,
        quaternion: Optional[Sequence[float]] = None,
        attach_site: Optional[hints.MjcfAttachmentFrame] = None,
    ):
        frame = self.attach(entity, attach_site=attach_site)
        if position is not None:
            if len(position) != 3:
                raise ValueError("Position must be a sequence of length 3.")
            frame.pos = position
        if quaternion is not None:
            if len(quaternion) != 4:
                raise ValueError("Quaternion must be a sequence of length 4.")
            frame.quat = quaternion
        return frame

    def add_mocap(
        self,
        entity: composer.Entity,
        position: Optional[Sequence[float]] = None,
        quaternion: Optional[Sequence[float]] = None,
        visible: bool = False,
        color: Tuple[float, float, float, float] = (0.9, 0.5, 0.5, 1.0),
        name: str = "mocap",
    ) -> hints.MjcfElement:
        # Add mocap body.
        mocap_body = self.mjcf_model.worldbody.add(
            "body",
            name=name,
            mocap="true",
            pos=position,
            quat=quaternion,
        )

        # Give it a square shape.
        if visible:
            mocap_body.add(
                "geom",
                type="box",
                group="1",
                size="0.02 0.02 0.02",
                contype="0",
                conaffinity="0",
                rgba=color,
            )

        # Make root body of entity's pose same as mocap pose.
        root_body = entity.mjcf_model.find_all("body")[0]
        root_body.pos = position
        root_body.quat = quaternion

        # Add the entity to the arena as a free body.
        self.add_free_entity(entity)

        # Add a weld constraint between the mocap body and the root body.
        self.mjcf_model.equality.add(
            "weld",
            body1=name,
            body2=root_body.full_identifier,
            solref="0.01 1",
            solimp=".9 .9 0.01",
        )

        return mocap_body
