from setuptools import setup
from glob import glob
import os

package_name = 'oskar_mapping'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='OSKAR Developer',
    author_email='user@example.com',
    maintainer='OSKAR Developer',
    maintainer_email='user@example.com',
    keywords=['ROS2', 'OSKAR', 'apple', 'flower', 'mapping'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
    ],
    description='OSKAR Phase 1: Apple flower mapping pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_sync_node = oskar_mapping.stereo_sync_node:main',
            'disparity_node = oskar_mapping.disparity_node:main',
            'segmentation_node = oskar_mapping.segmentation_node:main',
            'depth_fusion_node = oskar_mapping.depth_fusion_node:main',
            'backprojection_node = oskar_mapping.backprojection_node:main',
            'multiview_fusion_node = oskar_mapping.multiview_fusion_node:main',
            'bio_sanity_node = oskar_mapping.bio_sanity_node:main',
            'tree_assignment_node = oskar_mapping.tree_assignment_node:main',
            'thinning_decision_node = oskar_mapping.thinning_decision_node:main',
        ],
    },
)
