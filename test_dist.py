import os
import time
from datetime import timedelta

import torch.distributed as c10d
from torch.autograd import Variable

import torch
torch.manual_seed(2048)
import random
random.seed(2048)
import numpy as np
np.random.seed(2048)

from distseq.training import LSTransformerEncoderLayer

local_rank = int(os.environ.get("LOCAL_RANK", -1))

torch.cuda.set_device(local_rank)
c10d.init_process_group(backend='nccl', timeout=timedelta(minutes=60))

rank = c10d.get_rank()
world_size = c10d.get_world_size()

pg_ = c10d.distributed_c10d._get_default_group()

config = LSTransformerEncoderLayer.get_config(
    max_batch_tokens=12800,
    max_seq_len=50,
    hidden_size=1024,
    intermediate_size=4096,
    nhead=16,
    attn_prob_dropout_ratio=0.0,
    activation_dropout_ratio=0.0,
    hidden_dropout_ratio=0.0,
    pre_layer_norm=True,
    fp16=True,
    local_rank=local_rank,
)

hidden_states = Variable(torch.randn(256, 50, 1024).cuda(), requires_grad=True)
encoder_padding_mask = torch.ones(256, 50).cuda()
label = torch.empty(256, dtype=torch.long).random_(5).cuda()

# a = torch.ones_like(enc_layer.para)
# enc_layer.para.data.copy_(a)

enc_layer = LSTransformerEncoderLayer(config, pg_).cuda()
mlp = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 128)).cuda()
loss_fn = torch.nn.CrossEntropyLoss()

# warm-up steps
for _ in range(10):
    x = enc_layer(hidden_states, encoder_padding_mask)
    x = x.mean(dim=1)
    x = mlp(x)
    loss = loss_fn(x, label)
    loss.backward()

iter = 1000
enc_time = mlp_time = backward_time = 0
for _ in range(iter):
    torch.cuda.synchronize()
    start = time.time()
    x = enc_layer(hidden_states, encoder_padding_mask)

    torch.cuda.synchronize()
    enc_time += time.time() - start

    x = x.mean(dim=1)

    torch.cuda.synchronize()
    start = time.time()
    x = mlp(x)

    torch.cuda.synchronize()
    mlp_time += time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    loss = loss_fn(x, label)
    loss.backward()

    torch.cuda.synchronize()
    backward_time += time.time() - start

print('Encoder: {:.3f} ms | MLP: {:.3f} ms | Backward: {:.3f} ms'.format(
    enc_time, mlp_time, backward_time))
