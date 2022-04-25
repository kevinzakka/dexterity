from dm_control.entities.props import primitive

from dexterity import _SRC_ROOT

_TEXTURE_PATH = _SRC_ROOT / "manipulation" / "props" / "juggling_ball" / "rgby.png"


class JugglingBall(primitive.Primitive):
    def _build(
        self,
        radius: float = 0.01,
        name: str = "ball",
    ) -> None:
        """Builds the ball."""

        super()._build(
            geom_type="sphere",
            size=[radius] * 3,
            condim=6,
            friction="1 .001 .001",
            name=name,
        )

        self.mjcf_model.asset.add(
            "texture",
            name="texture_ball",
            file=str(_TEXTURE_PATH),
        )
        self.mjcf_model.asset.add(
            "material",
            name="texture_ball",
            texture="texture_ball",
            specular="1",
            shininess=".3",
            reflectance="0.0",
            rgba="1 1 1 1",
        )

        setattr(self.mjcf_model.find("geom", "geom"), "material", "texture_ball")

    @property
    def name(self) -> str:
        return self.mjcf_model.model
