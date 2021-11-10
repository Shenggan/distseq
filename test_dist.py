import os
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

enc_layer = LSTransformerEncoderLayer(config, pg_).cuda()

hidden_states = Variable(torch.randn(2, 8, 32).cuda(), requires_grad=True)
encoder_padding_mask = torch.ones(2, 8).cuda()
label = torch.empty(2, dtype=torch.long).random_(5).cuda()

# a = torch.ones_like(enc_layer.para)
# enc_layer.para.data.copy_(a)

x = enc_layer(hidden_states, encoder_padding_mask)
x = x.mean(dim=1)


mlp = torch.nn.Sequential(torch.nn.LayerNorm(32), torch.nn.Linear(32, 10)).cuda()
x = mlp(x)

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(x, label)
# print(loss)

loss.backward()

print(hidden_states.grad)