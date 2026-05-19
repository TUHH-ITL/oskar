import os
from glob import glob

from setuptools import setup

package_name = "oskar_pruning"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="OSKAR Developer",
    author_email="user@example.com",
    maintainer="OSKAR Developer",
    maintainer_email="user@example.com",
    keywords=["ROS2", "OSKAR", "apple", "pruning"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
    ],
    description="OSKAR Phase 2: Pruning execution pipeline",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "local_refinement_node = oskar_pruning.local_refinement_node:main",
            "pruning_decision_node = oskar_pruning.pruning_decision_node:main",
        ],
    },
)
