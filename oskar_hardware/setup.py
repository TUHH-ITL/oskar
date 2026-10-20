from setuptools import find_packages, setup

package_name = 'oskar_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/oskar_hardware.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vz',
    maintainer_email='vz@example.com',
    description='Hardware skeleton for the OSKAR blossom thinning robot.',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'tool_node = oskar_hardware.tool_node:main',
            'execute_approach_node = oskar_hardware.execute_approach_node:main',
        ],
    },
)
