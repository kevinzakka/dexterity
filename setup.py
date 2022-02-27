from setuptools import find_packages
from setuptools import setup

setup(
    name="shadow_hand",
    version="0.0.0",
    author="Kevin Zakka",
    license_files=("LICENSE",),
    packages=find_packages(),
    package_data={
        "shadow_hand": [
            "models/*.xml",
            "models/*.stl",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "typing_extensions",
        "dm_control @ git+git://github.com/deepmind/dm_control.git",
        "dm_robotics-geometry",
        "dm_robotics-transformations",
        "dm_robotics-agentflow",
    ],
    extras_require={
        "testing": [
            "pytest",
            "absl-py",
        ],
    },
)
