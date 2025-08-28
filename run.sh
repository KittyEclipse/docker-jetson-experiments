#!/bin/bash
cd $(dirname $(realpath $0))
export DISPLAY=:0
xhost +
docker compose run --rm -it app
