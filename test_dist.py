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
    max_batch_tokens=16,
    max_seq_len=8,
    hidden_size=32,
    intermediate_size=64,
    nhead=4,
    attn_prob_dropout_ratio=0.0,
    activation_dropout_ratio=0.0,
    hidden_dropout_ratio=0.0,
    pre_layer_norm=True,
    fp16=False,
    local_rank=local_rank,
)

hidden_states = Variable(torch.randn(2, 8, 32).cuda(), requires_grad=True)
encoder_padding_mask = torch.ones(2, 8).cuda()
label = torch.empty(2, dtype=torch.long).random_(5).cuda()

# a = torch.ones_like(enc_layer.para)
# enc_layer.para.data.copy_(a)

enc_layer = LSTransformerEncoderLayer(config, pg_).cuda()
mlp = torch.nn.Sequential(torch.nn.LayerNorm(32), torch.nn.Linear(32, 10)).cuda()
loss_fn = torch.nn.CrossEntropyLoss()

iter = 10000000
enc_time = mlp_time = backward_time = 0
for _ in range(iter):
    start = time.time()
    x = enc_layer(hidden_states, encoder_padding_mask)
    enc_time += time.time() - start

    x = x.mean(dim=1)

    start = time.time()
    x = mlp(x)
    mlp_time += time.time() - start

    start = time.time()
    loss = loss_fn(x, label)
    loss.backward()
    backward_time += time.time() - start

    torch.cuda.sync()

print('Encoder: {:.3f} us | MLP: {:.3f} us | Backward: {:.3f} us'.format(
    enc_time * 1e6/1e5, mlp_time * 1e6/1e5, backward_time * 1e6/1e5))
