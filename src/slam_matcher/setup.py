from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'slam_matcher'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    py_modules=[],
    data_files=[
        # Install package-level resources
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # (os.path.join('share', package_name), glob('slam_matcher/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools', 
        'opencv-python', 
        'rclpy', 
        'cv_bridge', 
        'sensor_msgs', 
        'nav_msgs', 
        'geometry_msgs', 
        'pcl_msgs',
    ],
    zip_safe=False,
    maintainer='YOUR_NAME',
    maintainer_email='your@email.com',
    description='BEV map builder and matcher for global localization using RTAB-Map',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_mapper_node = slam_matcher.bev_mapper_node:main',
            # Add this if/when you write the matcher
            # 'bev_matcher = slam_matcher.bev_matcher_node:main',
        ],
    },
)
