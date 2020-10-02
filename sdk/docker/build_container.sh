#!/bin/bash
DIR="$(dirname "$(readlink -f "$0")")"

echo "$DIR"
docker build -t catkit-container $DIR/.