# Use the official ROS 2 Humble image as a base
FROM osrf/ros:jazzy-desktop-full

ARG USERNAME=bernat
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN id -u $USER_UID || true \
    && userdel $(id -un $USER_UID) || true
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && groupadd -f video && usermod -aG video $USERNAME

# Base tools
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    wget \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    ros-jazzy-xacro \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-route \
    ros-jazzy-nav2-bringup \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-ros2-control-test-assets \
    ros-jazzy-tf-transformations \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description \
    # Gazebo Harmonic integration (ros-gz bridge)
    ros-jazzy-rtabmap-ros \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-gz-ros2-control-demos \
    ros-jazzy-slam-toolbox \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ultralytics "numpy<2.0" --break-system-packages
    
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,graphics,utility



ENV XDG_RUNTIME_DIR=/tmp/runtime-${USERNAME}
RUN mkdir -p $XDG_RUNTIME_DIR


ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

# Source ROS 2 Jazzy in every new terminal
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/${USERNAME}/.bashrc \
 && echo "if [ -f /home/ws/install/setup.bash ]; then source /home/ws/install/setup.bash; fi" >> /home/${USERNAME}/.bashrc \
 && chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc

USER $USERNAME
CMD ["/bin/bash"]
