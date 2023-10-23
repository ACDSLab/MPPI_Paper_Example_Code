#! /bin/bash
ubuntu_version=$1 #20.04
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

image=nvidia/cuda

# Get All tags using new Docker Hub API
source $SCRIPT_DIR/docker_tags.sh
json_array=$(docker-tags $image | jq --raw-output '.["tags"]')
readarray -t tag_array < <(echo "$json_array" | jq -c '.[]' | sed -e 's/"//g')
for tag_i in "${tag_array[@]}"; do
  if grep -q "devel-ubuntu$ubuntu_version" <<< "$tag_i"; then
    echo "$tag_i"
  fi
done
