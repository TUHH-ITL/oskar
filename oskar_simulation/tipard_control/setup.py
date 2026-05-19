import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tipard_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='workstation',
    maintainer_email='markus.knitt@tuhh.de',
    description='Control package for the Tipard 4WSWD robot',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tipard_joystick_teleop = tipard_control.tipard_joystick_teleop:main',
            'arm_joystick_servo = tipard_control.arm_joystick_servo:main',
        ],
    },
)
