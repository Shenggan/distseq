#!/bin/bash

export MASTER_IP=127.0.0.1
export MASTER_PORT=12346
export WORLD_SISE=$1

python -m torch.distributed.launch \
       --rdzv_endpoint 127.0.0.1:12332 \
       --nproc_per_node=${WORLD_SISE} \
       --nnodes=1 \
       --use_env \
       test_dist.py
