from dm_control import mjcf

from shadow_hand import shadow_hand_e_constants as consts


class ShadowHandSeriesE:
    """Shadow Dexterous Hand E Series."""

    def __init__(self, name: str = "shadow_hand_e") -> None:
        self._mjcf_root = mjcf.from_path("./shadow_hand_series_e.xml")
        self._mjcf_root.model = name

        self._add_actuators()

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def joints(self):
        pass

    @property
    def actuators(self):
        pass

    def _add_actuators(self) -> None:
        pass



if __name__ == "__main__":
    hand = ShadowHandSeriesE()

    # Check we can step the physics.
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    physics.step()

    # Render.
    pixels = physics.render(height=480, width=640)
    import matplotlib.pyplot as plt
    plt.imshow(pixels)
    plt.show()
