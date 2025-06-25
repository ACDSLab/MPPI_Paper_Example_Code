#! /usr/bin/env bash

# Based on this tutorial https://iridakos.com/programming/2018/03/01/bash-programmable-completion-tutorial
_docker_bash_completions() {
  COMPREPLY=($(compgen -W "$(docker ps -a --format \"{{.Names}}\")" "${COMP_WORDS[1]}" ))
}
complete -F _docker_bash_completions  docker_bash.sh
