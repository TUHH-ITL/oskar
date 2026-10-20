from setuptools import find_packages, setup

package_name = 'oskar_scene'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/scene.yaml']),
        ('share/' + package_name + '/launch', ['launch/scene.launch.py']),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='vz',
    maintainer_email='vz@example.com',
    description='Fixture-first local collision scene construction for OSKAR.',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'local_scene_node = oskar_scene.local_scene_node:main',
            'depth_cloud_node = oskar_scene.depth_cloud_node:main',
        ],
    },
)
