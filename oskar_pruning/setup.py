import os
from glob import glob

from setuptools import setup

package_name = 'oskar_pruning'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))
                                                             + glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Markus Knitt',
    maintainer_email='markusknitt@web.de',
    description='Real-time flower localization and tree scanning for the pruning phase',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flower_perception_node = oskar_pruning.flower_perception_node:main',
            'flower_map_node = oskar_pruning.flower_map_node:main',
            'scan_controller_node = oskar_pruning.scan_controller_node:main',
            'mission_node = oskar_pruning.mission_node:main',
            'gt_eval_node = oskar_pruning.gt_eval_node:main',
            'dataset_capture_node = oskar_pruning.dataset_capture_node:main',
        ],
    },
)
