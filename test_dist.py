import os
from datetime import timedelta

import torch.distributed as c10d

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
    hidden_size=16,
    intermediate_size=64,
    nhead=4,
    attn_prob_dropout_ratio=0.1,
    activation_dropout_ratio=0.1,
    hidden_dropout_ratio=0.1,
    pre_layer_norm=True,
    fp16=True,
    local_rank=local_rank,
)

enc_layer = LSTransformerEncoderLayer(config, pg_).cuda()

hidden_states = torch.randn(2, 8, 16).cuda()
hidden_states = torch.autograd.Variable(hidden_states, requires_grad=True)
encoder_padding_mask = torch.ones(2, 8).cuda()
encoder_padding_mask = torch.autograd.Variable(encoder_padding_mask)

output = enc_layer(hidden_states, encoder_padding_mask)
print(output)

# enc_layer.config.training = True
# enc_layer.config.is_grad_enabled = True
# print(torch.autograd.gradcheck(LSTransformerEncoderFunc.apply, [hidden_states, encoder_padding_mask, enc_layer.para, enc_layer.config]))