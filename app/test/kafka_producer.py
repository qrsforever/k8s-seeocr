#!/usr/bin/python3
# -*- coding: utf-8 -*-

# kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic messages
# kafka-console-producer.sh --broker-list kafka:19092 --topic messages
# kafka-console-consumer.sh --bootstrap-server kafka:19092 --topic messages --from-beginning

import os
import json
import time
from kafka import KafkaProducer

import socket
import sys
hostname = socket.gethostname()

if hostname == 'storage':
    host = '172.21.0.4'
else:
    host = '82.157.36.183'

producer = KafkaProducer(
        bootstrap_servers=[f'{host}:19092'],
        # key_serializer=str.encode,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'))


if __name__ == "__main__":
    test_json = 'test.json'
    if len(sys.argv) == 2:
        test_json = sys.argv[1]
        
    with open(test_json, 'r') as fr:
        value = json.load(fr)

    N = int(os.environ.get('N', 1))

    print('Count:', N)

    try:
        t0 = int(time.time())
        for i in range(N):
            token = int(time.time()) - t0
            value[0]['cfg']['pigeon']['token'] = token
            value[0]['cfg']['pigeon']['order'] = i
            value[0]['cfg']['pigeon']['total'] = N
            value[0]['cfg']['pigeon']['msgkey'] = 'seeocr_output'
            future = producer.send(
                    'seeocr_input',
                    value=value[0])
            time.sleep(2)
            result = future.get(timeout=10)
            print(f'1: {i} {token} {result}')

        producer.flush()

    except Exception:
        producer.close()
