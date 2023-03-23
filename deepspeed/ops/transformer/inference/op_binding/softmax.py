'''Copyright The Microsoft DeepSpeed Team'''

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class SoftmaxOp(BaseOp):
    def __init__(self, config: DeepSpeedInferenceConfig):
        super(SoftmaxOp, self).__init__(config)
        self.num_attention_heads_per_partition = config.heads // config.mp_size
        try:
            if self.config.fp16:
                self.softmax_func = self.inference_module.softmax_fp16
            elif self.config.bf16:
                self.softmax_func = self.inference_module.softmax_bf16
            else:
                self.softmax_func = self.inference_module.softmax_fp32
        except AttributeError:
            self.softmax_func = None

    def forward(self,
                attn_scores: torch.Tensor,
                attn_mask: torch.Tensor,
                alibi: torch.Tensor,
                triangular: bool,
                recompute: bool,
                local_attention: bool,
                window_size: int,
                async_op: bool,
                layer_scale: float,
                head_offset: int):
        if self.softmax_func != None:
            output = self.softmax_func(attn_scores,
                                       attn_mask,
                                       alibi,
                                       triangular,
                                       recompute,
                                       local_attention,
                                       window_size,
                                       async_op,
                                       layer_scale,
                                       head_offset,
                                       self.config.mp_size)
        else:
            # fallback
            if alibi is not None:
                bs = attn_scores.shape[0]
                total_heads = alibi.shape[0]
                heads = int(total_heads / bs)
                seq_length = alibi.shape[-1]
                alibi = alibi.view(bs, heads, 1, seq_length)

                alibi_split = alibi[:, head_offset:head_offset +
                          self.num_attention_heads_per_partition, :, :]
                attn_scores = attn_scores * layer_scale + alibi_split
            input_dtype = attn_scores.dtype
            if (triangular):
                tri = ~torch.tril(
                    torch.ones(attn_scores.size(),
                               device=attn_scores.device)).to(bool)
                attn_scores = torch.masked_fill(attn_scores,
                                                tri,
                                                torch.finfo(input_dtype).min)
            if attn_mask is not None:
                # expand atten_mask from two dim into 4 dim, insert two dims in the middle
                attn_mask = attn_mask[:, None, None, :]
                attn_scores += attn_mask
            output = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(input_dtype)

        return output
