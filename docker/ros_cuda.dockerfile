ARG UBUNTU_VERSION=20.04
ARG CUDA_IMG_VERSION=11.3.1
FROM nvidia/cuda:${CUDA_IMG_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS ros_setup
# Arguments not required to get docker image must come after FROM
# source: https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG UBUNTU_NAME=focal
ARG ROS_DISTRO_NAME=noetic
# Get Args from before the FROM command
ARG UBUNTU_VERSION
ARG CUDA_IMG_VERSION

# Set timezone needed for apt installs later on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install packages
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb http://packages.ros.org/ros2/ubuntu ${UBUNTU_NAME} main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
# ENV LANG=C.UTF-8
# ENV LC_ALL=C.UTF-8

# Needed for ROS graphical applications like gazebo
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# remove /usr/bin/man redirect, man page exclusion, and install manpages so new packages have man pages
RUN if [ -f /usr/bin/man ]; then \
      rm /usr/bin/man; \
      dpkg-divert --remove --rename /usr/bin/man; \
    fi && \
    if [ -f /etc/dpkg/dpkg.cfg.d/excludes ]; then \
      sed -i '/path-exclude=\/usr\/share\/man/d' /etc/dpkg/dpkg.cfg.d/excludes; \
    fi && \
    apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    man-db \
    manpages \
    && rm -rf /var/lib/apt/lists/*

# Setup graphical libraries
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    libxau6 \
    libxdmcp6 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/misc/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV ROS_DISTRO=${ROS_DISTRO_NAME}

# Nice to have programs, separated from the
# other installations because the other ones
# take forever and these might change
# Last line about deleting docker-clean allows for tab autocomplete. Source:
# https://dev.to/tetractius/tetraquicky09-get-back-autocomplete-in-a-docker-container-gob
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    autoconf \
    automake \
    bash-completion \
    build-essential \
    cmake \
    curl \
    g++ \
    gdb \
    git \
    git-lfs \
    htop \
    iputils-ping \
    less \
    libssl-dev \
    libtool \
    locales \
    lsb-release \
    net-tools \
    pigz \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-pip \
    python3-rosdep \
    python3-tk \
    python3-vcstool \
    software-properties-common \
    sudo \
    tmux \
    tzdata \
    udev \
    unzip \
    valgrind \
    vim \
    wget \
    xdg-user-dirs \
    && rm -rf /var/lib/apt/lists/* \
    && rm /etc/apt/apt.conf.d/docker-clean

# Install ROS-specific packages and install navigation2 if we are in ROS 2
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    ros-${ROS_DISTRO_NAME}-desktop \
    ros-dev-tools \
    # ros-${ROS_DISTRO_NAME}-perception \
    # ros-${ROS_DISTRO_NAME}-viz \
    protobuf-compiler \
    libprotobuf-dev && \
    if dpkg --compare-versions "${UBUNTU_VERSION}" ge "22.04" ; then \
      DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
        ros-${ROS_DISTRO_NAME}-navigation2; \
    fi \
    # && rosdep init \
    && rm -rf /var/lib/apt/lists/*


# Set up the locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Pip packages to get basic ROS and rosbag working in python3
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends --no-install-suggests \
    python3-pycryptodome \
    python3-gnupg \
    python3-rospkg \
    && rm -rf /var/lib/apt/lists/*
# RUN pip3 install pycryptodomex \
#   python-gnupg \
#   rospkg

# Create a script to run command with ros params
RUN echo \#'\
!/bin/bash\n\
set -e\n\
# setup ros environment\n\
source "/opt/ros/$ROS_DISTRO/setup.bash"\n\
exec "$@"' >> ros_entrypoint.sh && chmod +x ros_entrypoint.sh

RUN echo "source /opt/ros/${ROS_DISTRO_NAME}/setup.bash" >> /etc/bash.bashrc

########## Build code from scratch ##########
# Protobuf does not have a deb package on 16.04
WORKDIR /repos
RUN if [ "${UBUNTU_VERSION}" = "16.04" ] ; then git clone https://github.com/protocolbuffers/protobuf.git --branch v3.17.0 --depth 1 && \
    cd protobuf && \
    git submodule update --init --recursive && \
    ./autogen.sh && \
    ./configure && \
    make -j$(expr $(nproc) - 2) && \
    make check && \
    make install && \
    make clean && \
    ldconfig; fi

# We need a newer CMake on 16.04
WORKDIR /repos
RUN if [ "${UBUNTU_VERSION}" = "16.04" ] ; then git clone https://github.com/Kitware/CMake.git --branch v3.19.4 --depth 1 && \
    cd CMake && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(expr $(nproc) - 2) install && \
    make clean; fi

# Install an eigen version compatible with CUDA 10+
WORKDIR /repos
RUN git clone https://gitlab.com/libeigen/eigen.git --branch 3.3.7 --depth 1 && \
    sed -i "s/#include <host_defines.h>/#include <cuda_runtime.h>/" /repos/eigen/Eigen/Core && \
    cd eigen && \
    mkdir build && cd build && \
    cmake .. && \
    make install && \
    make clean

#Ensure that the cuda version is set properly
SHELL ["/bin/bash", "-c"]
RUN update-alternatives --set cuda /usr/local/cuda-${CUDA_IMG_VERSION::-2}
# Environment variable necessary to get QT to play nice in GUIs
ENV QT_X11_NO_MITSHM=1

############### Rosdep update ###############
# We would like to run rosdep update inside of the image creation process
# as it would take up a lot of time to run everytime we open a container.
# However, it writes to a folder in the home directory ~/.ros/rosdep.
# The mapping to the home directory has to be done outside of the image creation
# so how do we handle this? We can create a docker volume for ~/.ros. This

# Rosdep update writes to ~/.ros/rosdep so we create a docker volume for that folder
# to prevent overwriting the host machine's ~/.ros/rosdep
# WORKDIR /home/${USER_NAME}/.ros
# RUN chown -R ${USER_NAME}:${GROUP_NAME} /home/${USER_NAME}
# # Must be declared a volume after its permissions are set or else they don't
# # actually get set
# VOLUME /home/${USER_NAME}/.ros

# Create Users
ARG USER_ID=902819
ARG GROUP_ID=2626
ARG USER_NAME=docker_user
ARG GROUP_NAME=gtperson

USER root
# Fake audio setup
RUN apt-get update && apt-get install -y alsa-utils pulseaudio-utils pulseaudio

### setup user with same UID/GID and group name as the user outside the container ###

# Make group with specified name and id. If a group exists (new to ubuntu24.04 images), rename
RUN if getent group ${GROUP_NAME} ; then \
      groupdel ${GROUP_NAME}; \
    fi && \
    PREV_GROUP="$(getent group ${GROUP_ID} | cut -d: -f1)"; \
    if [[ -z "${PREV_GROUP}" ]]; then \
      groupadd -g ${GROUP_ID} ${GROUP_NAME}; \
    else \
      groupmod -n ${GROUP_NAME} ${PREV_GROUP}; \
    fi; \
    unset PREV_GROUP

# Make user with specified name and id. If a user exists (new to ubuntu24.04 images), rename
RUN PREV_USER="$(id -u -n ${USER_ID})"; \
    if [ $? -ne 0 ]; then \
      useradd --create-home -r -p "" -l -u ${USER_ID} -g ${GROUP_ID} --shell /bin/bash ${USER_NAME}; \
    else \
      usermod -l ${USER_NAME} ${PREV_USER}; mv /home/${PREV_USER} /home/${USER_NAME}; \
    fi; \
    unset PREV_USER && adduser ${USER_NAME} sudo

# Set no  password
RUN echo "${USER_NAME}:docker" | chpasswd && \
    echo "${USER_NAME} ALL=NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}


RUN chown -R ${USER_NAME}:${GROUP_NAME} /repos
USER ${USER_NAME}

# Now actually run rosdep update as user
# RUN rosdep update
WORKDIR /home/${USER_NAME}
ENV HOME=/home/${USER_NAME}

ENV TERM=xterm-256color

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
