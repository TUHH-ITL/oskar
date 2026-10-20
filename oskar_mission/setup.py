import os
from glob import glob

from setuptools import setup

package_name = 'oskar_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Markus Knitt',
    maintainer_email='markusknitt@web.de',
    description='Drive, scan, thin, move on: the mission and the top-level launch files',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mission_node = oskar_mission.mission_node:main',
        ],
    },
)
