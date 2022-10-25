#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
APP_DIR=$(cd $CUR_DIR/..; pwd)
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && SOURCED=1 || SOURCED=0

cd $APP_DIR

if [[ x$1 == x ]]
then
    echo "1. paddle cpu (det)"
    echo "2. paddle gpu (rec)"
    echo "3. paddle cpu & gpu"
    read -p "Select:" select
    case $select in
        1)
            echo "paddle cpu"
            docker-compose build paddle_cpu && docker-compose push paddle_cpu
            ;;
        2)
            echo "paddle gpu"
            docker-compose build paddle_gpu && docker-compose push paddle_gpu
            ;;
        3)
            echo "paddle cpu & paddle gpu"
            docker-compose build paddle_cpu && docker-compose push paddle_cpu
            docker-compose build paddle_gpu && docker-compose push paddle_gpu
            ;;
        *)
            echo "select error"
            [[ $SOURCED == 1 ]] && return || exit 0
    esac
fi

cd - > /dev/null
