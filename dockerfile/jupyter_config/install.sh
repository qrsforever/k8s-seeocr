#!/bin/bash

CUR_DIR=`pwd $(cd $(dirname ${BASH_SOURCE[0]}))`

NB_CONFIG_DIR=$(jupyter --config-dir)
NB_DATA_DIR=$(jupyter --data-dir)

ln -s $CUR_DIR/jupyter $NB_CONFIG_DIR
ln -s $CUR_DIR/local/share/jupyter $NB_DATA_DIR

# cp -aprf $CUR_DIR/jupyter/* $NB_CONFIG_DIR/
# cp -aprf $CUR_DIR/local/share/jupyter/* $NB_DATA_DIR/
