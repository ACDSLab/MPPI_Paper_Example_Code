#! /bin/bash
docker_container="ros2_cuda_11.3.1"
# Use argument passed in as container name
if [ -n "$1" ]; then
  docker_container="$1"
fi

# check if docker container is already running
container_running=$(docker ps --filter status=running | grep -cw "$docker_container")
if [ "$container_running" -lt "1" ]; then
    # Container needs to be started
    docker start $docker_container
    echo "Starting container..."
fi

# open a new terminal into the container
docker exec -it $docker_container /bin/bash
