"""Install the two robot task packages and shared runner/play helpers."""

import os
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Isaac Sim 5.0 ships Python 3.11; older installs may use tomli.
    import tomli as tomllib

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as _f:
    EXTENSION_TOML_DATA = tomllib.load(_f)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "scipy",
    "trimesh",
    # "torch==2.4.0",
    # "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=3.20.2, < 5.0.0",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

PACKAGES = find_packages(where=EXTENSION_PATH, include=["bipedal_locomotion_*"])
# Include prepared assets AND their raw sources for an independently usable
# checkout/wheel. Exclude Python caches and converter cache fingerprints.
ASSETS = {}
for name in ("bipedal_locomotion_motor43", "bipedal_locomotion_motor35"):
    base = Path(EXTENSION_PATH) / name
    ASSETS[name] = [str(path.relative_to(base)) for path in (base / "assets").rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts
                   and path.suffix not in (".py", ".pyc") and not path.name.startswith(".")]

# Installation operation
setup(
    name="bipedal_locomotion",
    packages=PACKAGES,
    package_data=ASSETS,
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
