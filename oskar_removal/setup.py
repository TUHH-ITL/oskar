import os
from glob import glob

from setuptools import setup

package_name = 'oskar_removal'

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
    description='Approach geometry, removal order, and the removal loop on MoveIt',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'removal_planner_node = oskar_removal.removal_planner_node:main',
            'removal_executor_node = oskar_removal.removal_executor_node:main',
        ],
    },
)
