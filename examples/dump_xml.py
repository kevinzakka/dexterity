from dm_control.mjcf import export_with_assets

from shadow_hand.models.hands import shadow_hand_e


def main() -> None:
    hand = shadow_hand_e.ShadowHandSeriesE()
    export_with_assets(hand.mjcf_model, "./temp/mjcf/", "shadow_hand_e_generated.xml")


if __name__ == "__main__":
    main()
