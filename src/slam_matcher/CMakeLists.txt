cmake_minimum_required(VERSION 3.0.2)
project(slam_matcher)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

# Install Python scripts (be sure to adjust the paths correctly)
catkin_install_python(PROGRAMS
  src/slam_matcher/bev_mapper_node.py
  src/slam_matcher/bev_builder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
