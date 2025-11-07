#!/bin/bash
cd $(dirname $(realpath $0))
export DISPLAY
xhost +
docker compose up -d
docker compose exec -it app bash
