from setuptools import find_packages
from setuptools import setup

core_requirements = [
    "absl-py",
    "numpy",
    "typing_extensions",
    "mujoco",
    "dm_control",
    "dm_robotics-geometry",
    "dm_robotics-transformations",
]

examples_requirements = [
    "matplotlib",
    "imageio",
    "imageio-ffmpeg",
]

testing_requirements = [
    "pytest-xdist",
]

dev_requirements = (
    [
        "black",
        "isort",
        "flake8",
        "mypy",
        "ipdb",
        "jupyter",
    ]
    + testing_requirements
    + examples_requirements
)


setup(
    name="shadow_hand",
    version="0.0.0",
    author="Kevin Zakka",
    license_files=("LICENSE",),
    packages=find_packages(),
    package_data={
        "shadow_hand": [
            "py.typed",
            "models/*.xml",
            "models/*.stl",
        ],
    },
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "testing": testing_requirements,
        "examples": examples_requirements,
        "dev": dev_requirements,
    },
)
