"""Pipes an MJCF file through the MuJoCo XML parser and spits out the processed XML.

Example usage:
$ python scripts/clean_xml.py -i dexterity/models/vendor/mpl/mpl_hand_description/mjcf/mpl_right.xml -o mpl_right_clean.xml
"""

from pathlib import Path

import mujoco
from absl import app
from absl import flags

flags.DEFINE_string("input", None, "Path to the MJCF file to clean.", short_name="i")
flags.DEFINE_string(
    "model_name",
    None,
    "Name of the output XML file. It will be \n"
    "saved in the same directory as the input file to correctly load assets.",
    short_name="o",
)


FLAGS = flags.FLAGS


def main(_) -> None:
    output_filename = Path(FLAGS.input).parent / FLAGS.model_name

    if output_filename.exists():
        raise FileExistsError(f"{output_filename} already exists.")

    model = mujoco.MjModel.from_xml_path(FLAGS.input)
    mujoco.mj_saveLastXML(str(output_filename), model)


if __name__ == "__main__":
    flags.mark_flags_as_required(["input", "model_name"])
    app.run(main)
