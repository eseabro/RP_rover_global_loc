from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'custom_slam'

def package_files(directory):
    paths_dict = {}
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            install_path = os.path.join('share', package_name, path)
            
            if install_path in paths_dict:
                paths_dict[install_path].append(file_path)
            else:
                paths_dict[install_path] = [file_path]
                
    data_files = []
    for key in paths_dict:
        data_files.append((key, paths_dict[key]))
        
    return data_files
# Generate the list of files
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ]+ package_files('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_mapper_node = custom_slam.bev_mapper_node:main',
            'rgbd_mapper_node = custom_slam.rgbd_mapper_node:main',
            'controller_node = custom_slam.controller_node:main',
            'controller_node_ackermann = custom_slam.controller_node_ackermann:main',
            'image_saver = custom_slam.image_saver:main',
            'map_exporter = custom_slam.map_exporter:main',
            'matcher_node = custom_slam.matcher_node:main',
            'odom_to_path = custom_slam.odom_to_path:main'
            
        ],
    },
)
