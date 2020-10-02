#!/bin/bash
DIR="$(dirname "$(readlink -f "$0")")"

docker run --gpus all -v $DIR:/sdk -v $DIR/../data:/data --shm-size=16gb -it --rm catkit-container
