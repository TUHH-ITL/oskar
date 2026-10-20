import os
from glob import glob

from setuptools import setup

package_name = 'oskar_online_perception'

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
    description='Online perception: detect, localize and map objects while scanning a tree',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_3d_node = oskar_online_perception.detection_3d_node:main',
            'landmark_map_node = oskar_online_perception.landmark_map_node:main',
            'scan_controller_node = oskar_online_perception.scan_controller_node:main',
            'gt_eval_node = oskar_online_perception.gt_eval_node:main',
            'dataset_capture_node = oskar_online_perception.dataset_capture_node:main',
        ],
    },
)
