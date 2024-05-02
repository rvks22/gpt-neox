#!/bin/bash
docker run \
    --runtime=nvidia \
    --rm \
    -it \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --mount type=bind,src=$PWD,dst=/gpt-neox \
    -v $(pwd):/workspace/ \
    -v $(pwd)/../pythia_deduped_pile_idxmaps:/pythia_deduped_pile_idxmaps \
    -v $(pwd)/../pythia_pile_idxmaps:/pythia_pile_idxmaps \
    pythia:latest bash #run-pythia.sh