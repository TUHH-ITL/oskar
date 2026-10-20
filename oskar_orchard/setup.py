import os
from glob import glob

from setuptools import setup

package_name = 'oskar_orchard'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml')) + glob(os.path.join('config', '*.json'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Markus Knitt',
    maintainer_email='markusknitt@web.de',
    description='Orchard facts: surveyed tree maps and simulated ground truth',
    license='Apache-2.0',
)
