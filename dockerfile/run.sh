#!/bin/bash

docker run --name seeocr -d \
    --restart always \
    --publish 8181:8888 \
    --shm-size 10g  \
    --volume /root/seeocr:/ckpts \
    --volume /data/k8s-nfs:/data \
    --volume ${PWD}/jupyter_config/jupyter:/root/.jupyter \
    --volume ${PWD}/jupyter_config/local/share/jupyter:/root/.local/share/jupyter \
    --volume ${PWD}/entrypoint.sh:/entrypoint.sh --entrypoint /entrypoint.sh \
    hzcsk8s.io/seeocr_jupyter
