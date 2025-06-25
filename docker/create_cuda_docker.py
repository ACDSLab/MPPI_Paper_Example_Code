#! /usr/bin/env python3

import argparse
import grp
import os
import subprocess

from pathlib import Path

ubuntu_dict = {"16.04": "xenial", "18.04": "bionic", "20.04": "focal", "22.04": "jammy", "24.04": "noble"}

ros_dict = {"16.04": "kinetic", "18.04": "melodic", "20.04": "noetic", "22.04": "iron", "24.04": "kilted"}

cuda_dict = {"20.04": [], "22.04": [], "24.04": []}

default_ubuntu_vers = "22.04"
default_cuda_vers = "11.8.0"
default_data_path = "/data/"
default_workspace_path = "~/workspaces/"

default_docker_user_homedir = (Path(__file__).parent / "../../../").resolve()


def blue_wrapper(input):
    return "\033[1;36m{}\033[0m".format(input)


def red_wrapper(input):
    return "\033[1;31m{}\033[0m".format(input)


def create_folder(folder_path):
    if folder_path and not os.path.isdir(folder_path):
        print('"{}" does not exist yet. Creating it now'.format(folder_path))
        Path(folder_path).mkdir(parents=True, exist_ok=True)


def main(args):
    # Get available cuda tags
    cuda_tags_cmd = "./get_cuda_tags.sh"
    process = subprocess.run(
        cuda_tags_cmd.split(), stdout=subprocess.PIPE, encoding="utf-8"
    )
    os_name = "ubuntu"
    for tag in process.stdout.split("\n"):
        os_version = tag[tag.find(os_name) + len(os_name) :]
        cuda_version = tag[: tag.find("-devel")]
        if os_version in cuda_dict:
            cuda_dict[os_version].append(cuda_version)
    # Get User info
    user_id = os.geteuid()
    # Check if sudo was used
    if user_id == 0:
        print(red_wrapper("DON'T RUN THIS AS SUDO!!!"))
        exit(0)
    group_id = os.getegid()

    if args.user:
        user_name = os.environ["USER"]
        group_name = grp.getgrgid(group_id).gr_name
    else:
        user_name = "docker_user"
        group_name = "docker_user"

    if args.default:
        ubuntu_input = default_ubuntu_vers
        cuda_input = default_cuda_vers
        data_input = default_data_path
        workspace_input = default_workspace_path
    else:
        ubuntu_input = args.ubuntu
        cuda_input = args.cuda
        data_input = args.data_path
        workspace_input = args.workspace_path
        # Get Ubuntu Version from user
        if ubuntu_input == "" or not ubuntu_input in ubuntu_dict:
            question_str = "Version of Ubuntu would you like to use (Default: {}): ".format(default_ubuntu_vers)
            question_str = blue_wrapper(question_str)
            # print("What version of Ubuntu would you like to use?")
            ubuntu_input = input(question_str)
            while ubuntu_input and not ubuntu_input in ubuntu_dict:
                print("\"{}\" is not available from {}".format(ubuntu_input,
                                                           list(ubuntu_dict.keys())))
                ubuntu_input = input(question_str)

            if not ubuntu_input:
                ubuntu_input = default_ubuntu_vers
        # Get CUDA version from user
        if cuda_input == "" or not cuda_input in cuda_dict[ubuntu_input]:
            question_str = "Version of CUDA would you like to use (Default: {}): ".format(
                default_cuda_vers
            )
            question_str = blue_wrapper(question_str)
            # print("What version of cuda would you like to use?")
            cuda_input = input(question_str)
            while cuda_input and not cuda_input in cuda_dict[ubuntu_input]:
                print(
                    '"{}" is not available from {}'.format(
                        cuda_input, cuda_dict[ubuntu_input]
                    )
                )
                cuda_input = input(question_str)

            if not cuda_input:
                cuda_input = default_cuda_vers

        # Get data folder from user
        if data_input == default_data_path:
            question_str = "What folder would you like to be `/data` in the Docker Container (Default: {})? ".format(
                default_data_path
            )
            question_str = blue_wrapper(question_str)
            data_input = input(question_str)
            if not data_input:
                data_input = default_data_path
        # Get workspace folder from user
        if workspace_input == default_workspace_path:
            question_str = "What folder would you like to be `~/workspaces` in the Docker Container (Default: {})? ".format(
                default_workspace_path
            )
            question_str = blue_wrapper(question_str)
            workspace_input = input(question_str)
            if not workspace_input:
                workspace_input = default_workspace_path
    workspace_input = os.path.expanduser(workspace_input)
    data_input = os.path.expanduser(data_input)
    create_folder(data_input)
    create_folder(workspace_input)

    env_file = (
        "################ CAN BE OVERWRITTEN BY create_cuda_docker.py ################\n"
        + "USER_NAME={}\n".format(user_name)
        + "GROUP_NAME={}\n".format(group_name)
        + "USER_ID={}\n".format(user_id)
        + "GROUP_ID={}\n".format(group_id)
        + "UBUNTU_VERSION={}\n".format(ubuntu_input)
        + "UBUNTU_NAME={}\n".format(ubuntu_dict[ubuntu_input])
        + "CUDA_IMG_VERSION={}\n".format(cuda_input)
        + "ROS_DISTRO_NAME={}\n".format(ros_dict[ubuntu_input])
        + "HOME_DIRECTORY={}\n".format(os.path.expanduser("~"))
        + "WORKSPACE_DIRECTORY={}\n".format(workspace_input)
        + "DATA_DIRECTORY={}\n".format(data_input)
        + "DOCKER_USER_HOME_DIRECTORY={}\n".format(args.docker_user_home_directory)
    )

    with open(".env", "w") as file:
        file.write(env_file)
    print(blue_wrapper(".env has been rewritten to:"))
    print(env_file)

    docker_cmd = "docker compose up --no-start --build"
    if args.build:
        subprocess.run(docker_cmd.split())
        print(blue_wrapper("Finished building docker container"))
        docker_clean_cmd = "docker image prune"
    else:
        print('Don\'t forget to run "{}" after this!'.format(blue_wrapper(docker_cmd)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
                    Script to help configure the docker container
                    by rewriting the .env file used by docker compose
                    for building. It will automatically set up the
                    current user and their home directory as available
                    in the docker container.
                                                """
    )
    parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="Use default version of Cuda and Paths",
    )
    # parser.add_argument("-u", "--ubuntu", type=str, default="",
    #                     help="Version of Ubuntu to use (Default: {})".format(default_ubuntu_vers))
    parser.add_argument(
        "-c",
        "--cuda",
        type=str,
        default="",
        help="Version of CUDA to use (Default: {})".format(default_cuda_vers),
    )
    parser.add_argument(
        "--ubuntu",
        type=str,
        default="",
        help="Version of Ubuntu to use (Default: {})".format(default_ubuntu_vers),
    )
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="""Build docker container in addition to configuring it
                                (Builds cans take around 30 minutes)\n""",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="""Which host OS folder to link to docker container as /data
                                (Default: {})""".format(
            default_data_path
        ),
    )
    parser.add_argument(
        "--workspace_path",
        type=str,
        default=default_workspace_path,
        help="""Which host OS folder to link to docker container as ~/workspaces
                                (Default: {})""".format(
            default_workspace_path
        ),
    )
    parser.add_argument(
        "--docker_user_home_directory",
        type=str,
        default=default_docker_user_homedir,
        help=f"""Which host OS folder to link to docker container as the home directory.
                                (Default: {default_docker_user_homedir})""",
    )
    parser.add_argument(
        "-u",
        "--user",
        action="store_true",
        help="""Use current user as the username in the docker container""",
    )

    args = parser.parse_args()
    main(args)
