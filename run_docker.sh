#!/bin/bash

docker run -it \
  --name="gps" \
  --gpus="all" \
  --ipc="host" \
  --volume=".:/app" \
  hsun:gps-denied-torch2.1 /bin/bash
