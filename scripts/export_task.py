"""Exports a `dexterity.Task` as a standalone MJCF (XML) file with assets.

Example usage:
$ python scripts/export_task.py --env_name "reach.state_dense" --save_dir "./temp/"
"""

from absl import app
from absl import flags
from dm_control.mjcf import export_with_assets_as_zip

from dexterity import manipulation

flags.DEFINE_enum(
    "env_name",
    "reach.state_dense",
    manipulation.ALL_NAMES,
    "Which environment to load.",
)
flags.DEFINE_string("out_dir", "./temp/mjcf/", "Where to save the exported task.")
flags.DEFINE_string(
    "model_name",
    None,
    "What to name the saved ZIP file and XML file within the ZIP.\n"
    "If not specified, defaults {environment_name}_export.zip",
)


FLAGS = flags.FLAGS


def main(_) -> None:
    # Load the environment.
    domain_name, task_name = FLAGS.env_name.split(".")
    env = manipulation.load(domain_name=domain_name, task_name=task_name)

    if FLAGS.model_name is None:
        model_name = f"{FLAGS.env_name}_export"
    else:
        model_name = FLAGS.model_name

    # Note: This function takes care of creating the directory if it doesn't exist.
    export_with_assets_as_zip(
        mjcf_model=env.task.root_entity.mjcf_model,
        out_dir=FLAGS.out_dir,
        model_name=model_name,
    )


if __name__ == "__main__":
    app.run(main)
