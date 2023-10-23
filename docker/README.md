# Contents
- [Contents](#contents)
- [Prerequisites Installation](#prerequisites-installation)
  - [1. Install Docker (Source):](#1-install-docker-source)
  - [2. Install Nividia Docker (Source):](#2-install-nividia-docker-source)
  - [3. Docker Compose Installation (Source):](#3-docker-compose-installation-source)
- [Get inside the Docker container TLDR Version](#get-inside-the-docker-container-tldr-version)

# Prerequisites Installation
## 1. Install Docker [(Source)](https://docs.docker.com/engine/install/ubuntu/):
Create a directory that you will be using as the home directory for the docker installation.
```bash
export DOCKER_USER_HOMEDIR=~/racer-jpl-docker
mkdir -p $DOCKER_USER_HOMEDIR
```
```bash
# Clone this repo (core_environment) to ${DOCKER_USER_HOMEDIR}/Admin/core_environment (with LFS installed)
git clone URL_OF_THIS_REPO ${DOCKER_USER_HOMEDIR}/Admin/core_environment

cd ${DOCKER_USER_HOMEDIR}/Admin/core_environment/docker_cuda

# remove previous Docker installations
sudo apt-get remove docker docker-engine docker.io containerd runc

# Setup Docker apt repository
#curl -fsSL https://download.docker.com/linux/ubuntu/gpg > misc/docker.key
sudo apt-key add misc/docker.key

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
# Setup user to have access to Docker without sudo
sudo groupadd docker
sudo usermod -aG docker $USER
```
After adding your user to the docker group, it is important step to log out and back in to update your user groups.

## 2. Install Nividia Docker [(Source)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):

```bash
# Setup Nvidia Docker apt repository
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey > misc/nvidia.key
sudo apt-key add misc/nvidia.key
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list > misc/nvidia-docker.list
cat misc/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

We should now be able to download a CUDA docker image and see if it is working by running `nvidia-smi` inside of it:
```bash
nvidia-docker run --rm nvidia/cuda:11.2.0-devel-ubuntu20.04 nvidia-smi
docker run --rm --runtime=nvidia nvidia/cuda:11.2.0-devel-ubuntu20.04 nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.2.0-devel-ubuntu20.04 nvidia-smi
```

## 3. Docker Compose Installation [(Source)](https://docs.docker.com/compose/install/):

Docker-compose is a nice wrapper for creating Docker containers that is used to setup displays and users inside of the docker container.
```bash
#curl -SL "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o misc/docker-compose
#chmod +x misc/docker-compose
cp misc/docker-compose ~/.local/bin/docker-compose
```

# Get inside the Docker container
* install `jq` using `sudo apt install jq` as this is needed to get docker tags
## Create docker container config using create_cuda_docker:

By default, create_cuda_docker will use a DATA_PATH of `/data`, WORKSPACE_PATH of `~/workspaces`, and DOCKER_USER_HOME_DIRECTORY=`(DIRECTORY_OF_create_cuda_docker)/../../..`.  If you don't have access to `/data` it is recommended that you change those parameters.

```
usage: create_cuda_docker.py [-h] [-d] [-c CUDA] [-b] [--data_path DATA_PATH] [--workspace_path WORKSPACE_PATH] [--docker_user_home_directory DOCKER_USER_HOME_DIRECTORY] [-u]

Script to help configure the docker container by rewriting the .env file used by docker-compose for building. It will automatically set up the current user and their home directory as
available in the docker container.

optional arguments:
  -h, --help            show this help message and exit
  -d, --default         Use default version of Cuda and Paths
  -c CUDA, --cuda CUDA  Version of CUDA to use (Default: 11.3.1)
  -b, --build           Build docker container in addition to configuring it (Builds cans take around 30 minutes)
  --data_path DATA_PATH
                        Which host OS folder to link to docker container as /data (Default: /data/)
  --workspace_path WORKSPACE_PATH
                        Which host OS folder to link to docker container as ~/workspaces (Default: ~/workspaces/)
  --docker_user_home_directory DOCKER_USER_HOME_DIRECTORY
                        Which host OS folder to link to docker container as the home directory. (Default: /home/jaedlund/racer-jpl-docker)
  -u, --user            Use current user as the username in the docker container
```
### Example Option 1: Use your current user and group inside the container and set the workspaces path

Create the docker image.  Note that currently the image is dependent on your current user.
```bash
# Create directories for data and workspaces
mkdir -p ${DOCKER_USER_HOMEDIR}/data
mkdir -p ${DOCKER_USER_HOMEDIR}/workspaces

# create the docker .env file
./create_cuda_docker.py -c 11.3.1 --user --data_path ${DOCKER_USER_HOMEDIR}/data --workspace_path ${DOCKER_USER_HOMEDIR}/workspaces --docker_user_home_directory  ${DOCKER_USER_HOMEDIR} 

# Build the docker image
docker-compose up --no-start --build

# To start a bash session inside the container run:
./docker_bash.sh racer_cuda
# docker_bash.sh may be used repeatedly in other terminals to get multiple shells in the docker container.
```

#### Setup the homedirectory and racer dependencies inside the docker image
```bash
jaedlund@racer_cuda:~$ cp /etc/skel/.[bp]* .
jaedlund@racer_cuda:~$ pip3 install --user -e $HOME/Admin/core_environment/racerenv
jaedlund@racer_cuda:~$ export PATH="$PATH:$HOME/.local/bin"
jaedlund@racer_cuda:~$ racerenv install-dotfiles --yes
jaedlund@racer_cuda:~$ racerenv install-dependencies
```

Create the workspace:
```bash
# Inside the docker container create a workspace:

username@racer_cuda:~$ racerenv create-workspace -d ~/workspaces/racer-jpl-1.9_ws -c racer_sim_1.9-feb-miix.yaml 

user@racer_cuda:~$ cd ~/workspaces/racer-jpl-1.9_ws

# building our stack can take about 100GB of RAM, so you should likely reduce the number of parallel jobs and limit memory usage.

user@racer_cuda:~/workspaces/racer-jpl-1.9_ws$ catkin build -j 5 -l 1 --mem-limit 60

# There is currently a dependency issue that requires the stack to be built twice, so if you get a failed package, just run it again,

user@racer_cuda:~/workspaces/racer-jpl-1.9_ws$ catkin build -j 5 -l 1 --mem-limit 60

 
# You will need to add these lines to the end of the .bashrc
# These are needed because racerenv run assumes each new bash shell sources ROS.
echo -e '\nexport ROS_WORKSPACE=$HOME/workspaces/racer-jpl-1.9_ws/src' >> ~/.bashrc
echo '[[ -f ${ROS_WORKSPACE}/../devel/setup.bash ]] && . ${ROS_WORKSPACE}/../devel/setup.bash' >> ~/.bashrc

# Source the workspace so that racerenv run can work.
source ~/.bashrc

# Source the workspace so that racerenv run can work.
user@racer_cuda:~/workspaces/racer-jpl-1.9_ws$ source ~/workspaces/racer-jpl-1.9_ws/devel/setup.bash
```
#### Running simple sim:
```bash
# Run SimpleSim 
racerenv run-simple --site unit/slope_01_up
```
A tmux session will start and launch our stack. RVIZ and node manager should appear and after a bit the vehicle should start to move. To exit the tmux session, press CTRL-b CTRL-x to trigger the cleanup script.  If that doesn't work for any reason, you can also run `racerenv cleanup`. There are many other worlds to try you can look at the help or try command line completion after the --site.
```bash
jaedlund@racer_cuda:~$ racerenv run-simple -h
usage: racerenv run-simple [-h] [-t TEST]
                           [-c {racer_sim_simple_comms.yaml,racer_sim_simple.yaml}]
                           [--data_dir_parent DATA_DIR_PARENT] -s
                           {dem/parkfield_01_midrange,dem/parkfield_02_valleys,dem/parkfield_03_w_ridge,dem/parkfield_04_e_ridge,dem/parkfield_05_e_slopes,integration/01_bush_hills,sandbox/eastlot,sandbox/eastlot_dem,sandbox/parkfield_left,sandbox/trail,terrain/med_01_shallow_sparse,terrain/med_02_shallow_dense,terrain/med_03_steep_sparse,terrain/med_04_steep_dense,tests/00_test_no_offset_00,tests/00_test_offset_00,tests/00_test_out_of_bounds_01,tests/00_test_out_of_bounds_02,tests/00_test_out_of_bounds_03,tests/00_test_out_of_bounds_04,tests/demo_world,unit/mission_01_waypoint_distance_large,unit/mission_02_waypoint_distance_planner_trap_large,unit/obstacle_01_dense,unit/obstacle_02_narrow_corridor,unit/obstacle_03_rock_crawl,unit/obstacle_04_low_vegetation_crawl,unit/obstacle_05_rock_crawl_slope,unit/obstacle_06_cliff,unit/obstacle_07_narrow_corridor_wiggle,unit/obstacle_08_dense_large,unit/obstacle_09_mid_range_trap_large,unit/obstacle_10_narrow_opening,unit/obstacle_11_slope_forest_large,unit/obstacle_12_desert_bushes,unit/slope_01_up,unit/slope_02_up_down,unit/slope_03_orthogonal,unit/slope_04_cross_wash,unit/slope_05_ridge,unit/slope_06_valley,unit/slope_07_avoid_steep_slope,unit/slope_08_large,unit/slope_09_narrow_opening,unit/slope_10_narrow_opening_cliff,unit/slope_11_gully,unit/slope_12_gully_trees,unit/slope_13_v_ditch,unit/speed_01_open_dense,unit/speed_02_open_slope,unit/speed_03_open,unit/speed_04_open_large,unit/speed_05_narrow_opening,unit/speed_06_open_large_more_waypoints,unit/stuck_01_backup,unit/stuck_02_backtrack,unit/trail_01_various_widths,unit/trail_02_slope,unit/trail_03_intermittent,unit/trail_04_cross_road,unit/trail_05_ridge,unit/trail_06_large_forest,unit/trail_07_valley,unit/trail_08_desert,unit/trail_09_large_desert,unit/trail_10_follow_near,unit/trail_11_follow_far,unit/trail_12_follow_zigzag,unit/trail_13_fork,unit/vegetation_01_avoid_bush,unit/vegetation_02_avoid_tree_drive_bush,unit/vegetation_03_avoid_rock_drive_bush}
                           [--degraded_gps | --no-degraded_gps]
                           [--respawn | --no-respawn] [--force-cpu]

optional arguments:
  -h, --help            show this help message and exit
  -t TEST, --test TEST  Test name
  -c {racer_sim_simple_comms.yaml,racer_sim_simple.yaml}, --config {racer_sim_simple_comms.yaml,racer_sim_simple.yaml}
                        tmuxp configuration file
  --data_dir_parent DATA_DIR_PARENT
                        prefix for storing all the logs.
  -s {dem/parkfield_01_midrange,dem/parkfield_02_valleys,dem/parkfield_03_w_ridge,dem/parkfield_04_e_ridge,dem/parkfield_05_e_slopes,integration/01_bush_hills,sandbox/eastlot,sandbox/eastlot_dem,sandbox/parkfield_left,sandbox/trail,terrain/med_01_shallow_sparse,terrain/med_02_shallow_dense,terrain/med_03_steep_sparse,terrain/med_04_steep_dense,tests/00_test_no_offset_00,tests/00_test_offset_00,tests/00_test_out_of_bounds_01,tests/00_test_out_of_bounds_02,tests/00_test_out_of_bounds_03,tests/00_test_out_of_bounds_04,tests/demo_world,unit/mission_01_waypoint_distance_large,unit/mission_02_waypoint_distance_planner_trap_large,unit/obstacle_01_dense,unit/obstacle_02_narrow_corridor,unit/obstacle_03_rock_crawl,unit/obstacle_04_low_vegetation_crawl,unit/obstacle_05_rock_crawl_slope,unit/obstacle_06_cliff,unit/obstacle_07_narrow_corridor_wiggle,unit/obstacle_08_dense_large,unit/obstacle_09_mid_range_trap_large,unit/obstacle_10_narrow_opening,unit/obstacle_11_slope_forest_large,unit/obstacle_12_desert_bushes,unit/slope_01_up,unit/slope_02_up_down,unit/slope_03_orthogonal,unit/slope_04_cross_wash,unit/slope_05_ridge,unit/slope_06_valley,unit/slope_07_avoid_steep_slope,unit/slope_08_large,unit/slope_09_narrow_opening,unit/slope_10_narrow_opening_cliff,unit/slope_11_gully,unit/slope_12_gully_trees,unit/slope_13_v_ditch,unit/speed_01_open_dense,unit/speed_02_open_slope,unit/speed_03_open,unit/speed_04_open_large,unit/speed_05_narrow_opening,unit/speed_06_open_large_more_waypoints,unit/stuck_01_backup,unit/stuck_02_backtrack,unit/trail_01_various_widths,unit/trail_02_slope,unit/trail_03_intermittent,unit/trail_04_cross_road,unit/trail_05_ridge,unit/trail_06_large_forest,unit/trail_07_valley,unit/trail_08_desert,unit/trail_09_large_desert,unit/trail_10_follow_near,unit/trail_11_follow_far,unit/trail_12_follow_zigzag,unit/trail_13_fork,unit/vegetation_01_avoid_bush,unit/vegetation_02_avoid_tree_drive_bush,unit/vegetation_03_avoid_rock_drive_bush}, --site {dem/parkfield_01_midrange,dem/parkfield_02_valleys,dem/parkfield_03_w_ridge,dem/parkfield_04_e_ridge,dem/parkfield_05_e_slopes,integration/01_bush_hills,sandbox/eastlot,sandbox/eastlot_dem,sandbox/parkfield_left,sandbox/trail,terrain/med_01_shallow_sparse,terrain/med_02_shallow_dense,terrain/med_03_steep_sparse,terrain/med_04_steep_dense,tests/00_test_no_offset_00,tests/00_test_offset_00,tests/00_test_out_of_bounds_01,tests/00_test_out_of_bounds_02,tests/00_test_out_of_bounds_03,tests/00_test_out_of_bounds_04,tests/demo_world,unit/mission_01_waypoint_distance_large,unit/mission_02_waypoint_distance_planner_trap_large,unit/obstacle_01_dense,unit/obstacle_02_narrow_corridor,unit/obstacle_03_rock_crawl,unit/obstacle_04_low_vegetation_crawl,unit/obstacle_05_rock_crawl_slope,unit/obstacle_06_cliff,unit/obstacle_07_narrow_corridor_wiggle,unit/obstacle_08_dense_large,unit/obstacle_09_mid_range_trap_large,unit/obstacle_10_narrow_opening,unit/obstacle_11_slope_forest_large,unit/obstacle_12_desert_bushes,unit/slope_01_up,unit/slope_02_up_down,unit/slope_03_orthogonal,unit/slope_04_cross_wash,unit/slope_05_ridge,unit/slope_06_valley,unit/slope_07_avoid_steep_slope,unit/slope_08_large,unit/slope_09_narrow_opening,unit/slope_10_narrow_opening_cliff,unit/slope_11_gully,unit/slope_12_gully_trees,unit/slope_13_v_ditch,unit/speed_01_open_dense,unit/speed_02_open_slope,unit/speed_03_open,unit/speed_04_open_large,unit/speed_05_narrow_opening,unit/speed_06_open_large_more_waypoints,unit/stuck_01_backup,unit/stuck_02_backtrack,unit/trail_01_various_widths,unit/trail_02_slope,unit/trail_03_intermittent,unit/trail_04_cross_road,unit/trail_05_ridge,unit/trail_06_large_forest,unit/trail_07_valley,unit/trail_08_desert,unit/trail_09_large_desert,unit/trail_10_follow_near,unit/trail_11_follow_far,unit/trail_12_follow_zigzag,unit/trail_13_fork,unit/vegetation_01_avoid_bush,unit/vegetation_02_avoid_tree_drive_bush,unit/vegetation_03_avoid_rock_drive_bush}
                        Site name (formerly world)
  --degraded_gps
  --no-degraded_gps
  --respawn
  --no-respawn
```

Currently there are 2 tmuxp configs that work with run-simple: `racer_sim_simple_comms.yaml` and `racer_sim_simple.yaml`. `racer_sim_simple.yaml` is the default. 

#### Analyzing rosbag data:

Download the rosbags/logs to a directory in `~/racer-jpl-docker/data` on the host or `/data` inside the docker. The structure should be like this:
```
crl_rzr_hw_2023-01-12-19-56-00_jpl6_helendale_trail_trav_t4/
├── [4.0K]  config
│   ├── [   4]  admin_diff.txt
│   ├── [ 505]  admin_exact.yaml
│   ├── [393K]  apt_exact.txt
│   ├── [9.9K]  env.txt
│   ├── [ 15K]  pip_exact.txt
│   ├── [ 13K]  ws_diff.txt
│   └── [6.0K]  ws_exact.yaml
├── [204K]  crl_rzr_rosparam.yaml
├── [ 12K]  log
│   ├── [ 42M]  *.log
└── [ 12K]  rosbag
    ├── [512M]  crl_rzr_color_*.bag
    ├── [213M]  crl_rzr_comms_2023-01-12-20-01-01_0.bag
    ├── [ 13M]  crl_rzr_controller_2023-01-12-20-01-01_0.bag
    ├── [4.2M]  crl_rzr_gps_2023-01-12-20-01-01_0.bag
    ├── [ 13M]  crl_rzr_health_2023-01-12-20-01-02_0.bag
    ├── [ 44M]  crl_rzr_imu_2023-01-12-20-01-01_0.bag
    ├── [513M]  crl_rzr_lidar_*.bag
    ├── [514M]  crl_rzr_mapping_*.bag
    ├── [3.1M]  crl_rzr_mission_2023-01-12-20-01-01_0.bag
    ├── [432M]  crl_rzr_ndvi_2023-01-12-20-01-01_0.bag
    ├── [124M]  crl_rzr_odometry_2023-01-12-20-01-01_0.bag
    ├── [523M]  crl_rzr_oryx_*.bag
    ├── [505M]  crl_rzr_planner_micro_2023-01-12-20-01-01_0.bag
    ├── [ 84K]  crl_rzr_planner_mid_2023-01-12-20-01-01_0.bag
    ├── [ 22M]  crl_rzr_planner_short_2023-01-12-20-01-01_0.bag
    ├── [513M]  crl_rzr_stereo_*.bag
    ├── [ 31M]  crl_rzr_tf_2023-01-12-20-01-01_0.bag
    ├── [513M]  crl_rzr_trav_*.bag
    └── [121M]  crl_rzr_vehicle_2023-01-12-20-01-01_0.bag
```

The analysis requires a few additional packages inside the docker
```
python3 -m pip install --user moviepy
# The following are needed, but should now already be installed.
# python3 -m pip install --user geopy
# python3 -m pip install --user pykml
# sudo apt-get install ffmpeg
```
`python3 -m pip install --user PyQtWebEngine` seems to hang in docker.  I modified the scripts to skip some functions if it's not available.

```
user@racer_cuda:/data$ rosrun logging_analysis batch_analysis.sh crl_rzr_hw_2023-01-12-19-56-00_jpl6_helendale_trail_trav_t4
```
This will create the following:

```
crl_rzr_hw_2023-01-12-19-56-00_jpl6_helendale_trail_trav_t4
├── [       4096]  analysis
│   ├── [       4096]  controller_diagnostics
│   │   ├── [       1453]  crl_rzr_controller_2023-01-12-20-01-01_0_diagnostics.csv
│   │   └── [     206933]  crl_rzr_controller_2023-01-12-20-01-01_0_diagnostics.png
│   ├── [       4096]  control_smoothness
│   │   ├── [    7215008]  control_smoothness.npy
│   │   └── [     225321]  control_smoothness.png
│   ├── [       4096]  interventions
│   │   ├── [     271953]  data.png
│   │   ├── [      90603]  intervention_0.png
│   │   ├── [        916]  interventions.csv
│   │   └── [       1474]  interventions.pkl
│   ├── [       4096]  speed
│   │   ├── [    1797632]  speed.npy
│   │   ├── [     179532]  speed_roll_pitch_elevation.png
│   │   ├── [      99061]  speed_traj.png
│   │   ├── [      51334]  speed_traj_transparent.png
│   │   ├── [     150583]  speed_traj_z_color.png
│   │   ├── [     106684]  speed_traj_z_color_transparent.png
│   │   └── [     200974]  speed_vel.png
│   └── [       4096]  topic_size
│       ├── [    3631020]  sunburst.html
│       ├── [      11722]  topics.csv
│       └── [    4115227]  topics.png
├── [       4096]  config
│   ├── [          ?]  *
├── [     209163]  crl_rzr_rosparam.yaml
├── [      12288]  log
│   ├── [          ?]  *.log
├── [       4096]  processed_data
│   └── [  779573541]  odometry_vehicle_merged.bag
├── [       4096]  processed_videos
│   ├── [    5827996]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_backx16.mp4
│   ├── [   21946662]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_backx4.mp4
│   ├── [    1586498]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_backx64.mp4
│   ├── [   11233783]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_frontx16.mp4
│   ├── [   41601186]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_frontx4.mp4
│   ├── [    3067869]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_frontx64.mp4
│   ├── [   12920276]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_leftx16.mp4
│   ├── [   48687785]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_leftx4.mp4
│   ├── [    3594794]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_leftx64.mp4
│   ├── [   12319874]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_rightx16.mp4
│   ├── [   45735907]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_rightx4.mp4
│   └── [    3344773]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_rightx64.mp4
├── [       4096]  raw_videos
│   ├── [   82418495]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_back.mp4
│   ├── [  145173451]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_front.mp4
│   ├── [  163623837]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_left.mp4
│   └── [  150832533]  crl_rzr_color_2023-01-12-20-01-01_0_multisense_right.mp4
├── [      12288]  rosbag
│   ├── [          ?]  *.bag
│   └── [   65177327]  interventions.pkl
└── [       4096]  visualization
    └── [      54948]  gps_map.html
```

### Example Option 2: Use racer user and the default installation path

* To create link the `./create_cuda_docker.py --default`
* run `docker-compose up --no-start --build` (might take a while to build the docker image the first time - maybe 20 minutes?)
* run `./docker_bash.sh racer_cuda` to be dropped inside the container and to open multiple terminals into the same container
