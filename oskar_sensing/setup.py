from setuptools import find_packages, setup

package_name = 'oskar_sensing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/zed_x_live.launch.py',
            'launch/zed_x_svo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vz',
    maintainer_email='vz@example.com',
    description='Source-agnostic PointCloud2 snapshot boundary for OSKAR.',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'snapshot_node = oskar_sensing.snapshot_node:main',
        ],
    },
)
