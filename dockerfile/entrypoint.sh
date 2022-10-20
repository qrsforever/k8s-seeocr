#!/bin/bash

umask 0000

while (( 1 ))
do
    jupyter notebook --no-browser --allow-root --notebook-dir=/data --ip=0.0.0.0 --port=8888
    sleep 3
done
