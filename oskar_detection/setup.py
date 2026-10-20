import os
from glob import glob

from setuptools import setup

package_name = 'oskar_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Markus Knitt',
    maintainer_email='markusknitt@web.de',
    description='Model files and the one loader for instance segmentation models',
    license='Apache-2.0',
    tests_require=['pytest'],
)
