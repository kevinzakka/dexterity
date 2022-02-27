"""The OpenAI cube."""

from typing import Sequence

from dm_control.entities.props import primitive

from shadow_hand import _SRC_ROOT

_TEXTURE_PATH = (
    _SRC_ROOT
    / "tasks"
    / "inhand_manipulation"
    / "props"
    / "openai_cube"
    / "openai_cube.png"
)


class OpenAICube(primitive.Primitive):
    """A cube with OpenAI letters on each face."""

    def _build(
        self,
        size: Sequence[float],
        name: str = "openai_cube",
    ) -> None:
        """Builds the cube.

        Args:
            size: [x_half_length, y_half_length, z_half_length]
            name: Optional name for the cube prop.
        """
        super()._build(geom_type="box", size=size, name=name)

        self.mjcf_model.asset.add(
            "texture",
            name="texture_openai",
            file=str(_TEXTURE_PATH),
            gridsize="3 4",
            gridlayout=".U..LFRB.D..",
        )
        self.mjcf_model.asset.add(
            "material",
            name="material_openai",
            texture="texture_openai",
            specular="1",
            shininess=".3",
            reflectance="0.0",
            rgba="1 1 1 1",
        )

        setattr(self.mjcf_model.find("geom", "geom"), "material", "material_openai")
