import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


from distseq.training.ops.pytorch.builder import TransformerBuilder
from distseq.training.ops.pytorch.util import (
    copy_para,
    state_dict,
    MODEL_ARCH,
    check_config,
    calc_offset,
)

transformer_cuda_module = None
_all_layer_grads = dict()


class LSTransformerEncoderFunc(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        parameters,
        config
    ):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.transformer_encoder_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_encoder_layer_fw_fp32
        )
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(
            config.layer_id, input, input_mask, config.training, config.pre_layer_norm
        )

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(output, input, input_mask)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.config.training

        cuda_module = transformer_cuda_module
        backward_func = (
            cuda_module.transformer_encoder_layer_bw_fp16
            if ctx.config.fp16
            else cuda_module.transformer_encoder_layer_bw_fp32
        )

        output, input, input_mask = ctx.saved_tensors
        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)
            output = output.to(torch.half)
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)
        (grad_input,) = backward_func(
            ctx.config.layer_id, grad_output, output, input, input_mask
        )

        grad = _all_layer_grads[ctx.config.layer_id]

        return (grad_input, None, grad, None)


class LSTransformerEncoderLayer(nn.Module):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(self, config, pg=None, initial_weights=None, initial_biases=None):
        super(LSTransformerEncoderLayer, self).__init__()

        self.config = config
        self.pg = pg
        self.pg_size = pg.size()
        self.config.layer_id = LSTransformerEncoderLayer.layer_id
        LSTransformerEncoderLayer.layer_id = LSTransformerEncoderLayer.layer_id + 1

        print("Lightseq Transformer config is ", self.config.__dict__)

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        # Load cuda modules if needed
        global transformer_cuda_module
        if transformer_cuda_module is None:
            transformer_cuda_module = TransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_transformer_encoder_layer_fp16
            if self.config.fp16
            else cuda_module.create_transformer_encoder_layer_fp32
        )

        create_layer_func(
            self.config.layer_id,
            self.config.max_batch_tokens,
            self.config.max_seq_len,
            self.config.hidden_size,
            self.config.nhead,
            self.config.intermediate_size,
            self.config.attn_prob_dropout_ratio,
            self.config.activation_dropout_ratio,
            self.config.hidden_dropout_ratio,
            self.config.pre_layer_norm,
            self.config.activation_fn,
            self.pg,
        )

        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        self.para_offset = LSTransformerEncoderLayer.gen_offset(hs, ims, self.pg_size)
        self.para = nn.Parameter(torch.Tensor(self.para_offset[-1]))

        if initial_weights is None and initial_biases is None:
            self.init_transformer_weights()
            return

        # For testing only.
        qkv_w = [ele.detach().clone() for ele in initial_weights[:3]]
        qkv_w = torch.cat(qkv_w, dim=0)
        weights = [qkv_w] + [copy_para(ele) for ele in initial_weights[3:]]

        qkv_b = [ele.detach().clone() for ele in initial_biases[:3]]
        qkv_b = torch.cat(qkv_b, dim=0)
        biases = [qkv_b] + [copy_para(ele) for ele in initial_biases[3:]]

        idx = 0
        for w, b in zip(weights, biases):
            cur_para = self._get_weights(idx)
            assert cur_para.numel() == w.numel()
            cur_para.copy_(w.view(-1))
            idx += 1

            cur_para = self._get_weights(idx)
            assert cur_para.numel() == b.numel()
            cur_para.copy_(b.view(-1))
            idx += 1

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            activation_fn: str = "relu"  # relu or gelu

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        config = Config(**kwargs)
        check_config(config)
        return config

    @staticmethod
    def gen_offset(hidden_size, intermediate_size, pg_size):
        hs, ims = hidden_size, intermediate_size
        sizes = [
            int(hs * hs * 3 / pg_size),  # attn_qkvw
            int(hs * 3 / pg_size),  # attn_qkvb
            int(hs * hs / pg_size),  # attn_ow
            hs,  # attn_ob
            hs,  # attn_nw
            hs,  # attn_nb
            int(hs * ims / pg_size),  # inter_w
            int(ims / pg_size),  # inter_b
            int(hs * ims / pg_size),  # output_w
            hs,  # output_b
            hs,  # ffn_nw
            hs,  # ffn_nb
        ]
        offsets = calc_offset(sizes)
        return offsets

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def calc_bound(self, w):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def init_transformer_weights(self):
        hs = self.config.hidden_size
        ims = self.config.intermediate_size

        nn.init.zeros_(self._get_weights(3))

        nn.init.ones_(self._get_weights(4))
        nn.init.zeros_(self._get_weights(5))

        if self.pg_size > 1:
            attn_qkvw_global = torch.empty(hs * 3, hs)
            attn_qkvb_global = torch.empty(hs * 3)
            nn.init.xavier_uniform_(attn_qkvw_global, 1.0 / math.sqrt(2.0))
            bound = self.calc_bound(attn_qkvw_global)
            nn.init.uniform_(attn_qkvb_global, -bound, bound)

            attn_qkvw_global = attn_qkvw_global.cuda()
            attn_qkvb_global = attn_qkvb_global.cuda()
            torch.distributed.broadcast(attn_qkvw_global, src=0)
            torch.distributed.broadcast(attn_qkvb_global, src=0)
            attn_qkvw_global = attn_qkvw_global.cpu()
            attn_qkvb_global = attn_qkvb_global.cpu()

            attn_qkvw = self._get_weights(0).view(3, int(hs / self.pg_size), hs)
            attn_qkvb = self._get_weights(1).view(3, int(hs / self.pg_size))

            attn_qkvw.copy_(
                attn_qkvw_global.view(3, hs, hs)[:, int(hs * self.config.local_rank /
                                     self.pg_size):int(hs * (self.config.local_rank + 1) /
                                                       self.pg_size), :])
            attn_qkvb.copy_(
                attn_qkvb_global.view(3, hs)[:, int(hs * self.config.local_rank /
                                     self.pg_size):int(hs * (self.config.local_rank + 1) /
                                                       self.pg_size)])

            attn_ow_global = torch.empty(hs, hs)
            nn.init.xavier_uniform_(attn_ow_global, 1.0)
            attn_ow_global = attn_ow_global.cuda()
            torch.distributed.broadcast(attn_ow_global, src=0)
            attn_ow_global = attn_ow_global.cpu()
            attn_ow = self._get_weights(2).view(-1, int(hs / self.pg_size))
            attn_ow.copy_(attn_ow_global[:, int(hs * self.config.local_rank /
                                     self.pg_size):int(hs * (self.config.local_rank + 1) /
                                                       self.pg_size)])

            inter_w_global = torch.empty(ims, hs)
            inter_b_global = torch.empty(ims)
            nn.init.kaiming_uniform_(inter_w_global, math.sqrt(5.0))
            bound = self.calc_bound(inter_w_global)
            nn.init.uniform_(inter_b_global, -bound, bound)

            inter_w_global = inter_w_global.cuda()
            inter_b_global = inter_b_global.cuda()

            torch.distributed.broadcast(inter_w_global, src=0)
            torch.distributed.broadcast(inter_b_global, src=0)

            inter_w_global = inter_w_global.cpu()
            inter_b_global = inter_b_global.cpu()

            inter_w = self._get_weights(6).view(int(ims / self.pg_size), hs)
            inter_b = self._get_weights(7)

            inter_w.copy_(inter_w_global[int(ims * (self.config.local_rank) /
                                             self.pg_size):int(ims * (self.config.local_rank + 1) /
                                                               self.pg_size), :])
            inter_b.copy_(inter_b_global[int(ims * (self.config.local_rank) /
                                             self.pg_size):int(ims * (self.config.local_rank + 1) /
                                                               self.pg_size)])

            output_w_global = torch.empty(hs, ims)
            output_b_global = torch.empty(hs)
            nn.init.kaiming_uniform_(output_w_global, math.sqrt(5.0))
            bound = self.calc_bound(output_w_global)
            nn.init.uniform_(output_b_global, -bound, bound)

            output_w_global = output_w_global.cuda()
            output_b_global = output_b_global.cuda()

            torch.distributed.broadcast(output_w_global, src=0)
            torch.distributed.broadcast(output_b_global, src=0)

            output_w_global = output_w_global.cpu()
            output_b_global = output_b_global.cpu()

            output_w = self._get_weights(8).view(hs, int(ims / self.pg_size))
            output_b = self._get_weights(9)

            output_w.copy_(output_w_global[:, int(ims * (self.config.local_rank) / self.pg_size): int(ims * (self.config.local_rank + 1) / self.pg_size)])
            output_b.copy_(output_b_global)
        else:
            attn_qkvw = self._get_weights(0).view(-1, hs)
            nn.init.xavier_uniform_(attn_qkvw, 1.0 / math.sqrt(2.0))
            bound = self.calc_bound(attn_qkvw)
            nn.init.uniform_(self._get_weights(1), -bound, bound)

            nn.init.xavier_uniform_(self._get_weights(2).view(-1, hs), 1.0)

            inter_w = self._get_weights(6).view(int(ims / self.pg_size), hs)
            nn.init.kaiming_uniform_(inter_w, math.sqrt(5.0))
            bound = self.calc_bound(inter_w)
            nn.init.uniform_(self._get_weights(7), -bound, bound)

            output_w = self._get_weights(8).view(hs, int(ims / self.pg_size))
            nn.init.kaiming_uniform_(output_w, math.sqrt(5.0))
            bound = self.calc_bound(output_w)
            nn.init.uniform_(self._get_weights(9), -bound, bound)

        nn.init.ones_(self._get_weights(10))
        nn.init.zeros_(self._get_weights(11))


    def __assign_layer_weight_grad(self):
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )
        if self.config.layer_id in _all_layer_grads:
            return
        cuda_module = transformer_cuda_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.empty_like(param)
        func(param, grad, "TransformerEncoderLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(self, hidden_states, encoder_padding_mask, **kwargs):
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        hidden_states = hidden_states.contiguous()
        encoder_padding_mask = (
            (encoder_padding_mask * -1e8).type_as(hidden_states).contiguous()
        )
        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        self.__assign_layer_weight_grad()
        bs, sl, dim = hidden_states.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        if len(encoder_padding_mask.size()) == 1:
            assert bs == 1 and sl == encoder_padding_mask.size(0)
        else:
            assert bs == encoder_padding_mask.size(
                0
            ) and sl == encoder_padding_mask.size(1)
        output = LSTransformerEncoderFunc.apply(
            hidden_states,
            encoder_padding_mask,
            self.para,
            self.config
        )

        return output.to(self.para)
