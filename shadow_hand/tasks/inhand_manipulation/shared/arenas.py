"""Arenas for the in-hand manipulation suite of tasks."""

from typing import Optional

from shadow_hand import arena


class Standard(arena.Arena):
    """Suite-specific subclass of the standard Composer arena."""

    def _build(self, name: Optional[str] = None) -> None:
        super()._build(name=name)

        # Remove the default visual headlight setting in `dm_control.Arena`.
        self._mjcf_root.visual.remove("headlight")

        # Add visual assets.
        self.mjcf_model.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=(0.4, 0.6, 0.8),
            rgb2=(0.0, 0.0, 0.0),
            width=100,
            height=100,
        )
        groundplane_texture = self.mjcf_model.asset.add(
            "texture",
            name="groundplane",
            type="2d",
            builtin="checker",
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.1, 0.2, 0.3),
            width=300,
            height=300,
            mark="edge",
            markrgb=(0.8, 0.8, 0.8),
        )
        groundplane_material = self.mjcf_model.asset.add(
            "material",
            name="groundplane",
            texture=groundplane_texture,
            texrepeat=(5, 5),
            texuniform="true",
            reflectance=0.2,
        )

        # Add ground plane.
        self.mjcf_model.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            material=groundplane_material,
            size=(1, 1, 0.1),
            friction=(0.4,),
            solimp=(0.95, 0.99, 0.001),
            solref=(0.002, 1),
        )

        # Add lighting
        self.mjcf_model.worldbody.add(
            "light",
            pos=(0, 0, 1.5),
            dir=(0, 0, -1),
            diffuse=(0.7, 0.7, 0.7),
            specular=(0.3, 0.3, 0.3),
            directional="true",
            castshadow="true",
        )

        # Always initialize the free camera so that it points at the origin.
        self.mjcf_model.statistic.center = (0.0, 0.0, 0.0)
