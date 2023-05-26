# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F

class InferenceBuilderObject():
    def __init__(self):
        pass

    # qkv functions
    # the function name that exposed to OpBuilder caller
    def qkv_gemm_bf16(self, input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        return self.qkv_gemm_func(input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose)

    # the real function that do the work
    def qkv_gemm_func(self, input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
        if not transpose:
            inp_norm = F.layer_norm(input, (input.shape[2], ), gamma, beta, eps)
            tmp = torch.matmul(inp_norm, weight)
            if add_bias:
                tmp += bias
                return tmp, inp_norm
            else:
                raise NotImplementedError

    def softmax_bf16(self, attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale, head_offset, mp_size):
        return self.softmax_func(attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale, head_offset, mp_size)

    def softmax_func(self, attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size, async_op, layer_scale, head_offset, mp_size):

        # get heads, algo from kernel code
        len_ = len(attn_scores.size())
        heads = 1
        if len_ > 1:
            heads = attn_scores.size()[1]

        alibi = alibi[head_offset:head_offset + heads]
        input_dtype = attn_scores.dtype
        if (triangular):
            tri = ~torch.tril(torch.ones(attn_scores.size(), device=attn_scores.device)).to(bool)
            attn_scores = torch.masked_fill(attn_scores * layer_scale, tri, torch.finfo(input_dtype).min)
        if alibi is not None:
            attn_scores += alibi
        if attn_mask is not None:
            # expand atten_mask from two dim into 4 dim, insert two dims in the middle
            attn_mask = attn_mask[:, None, None, :]
            attn_scores += attn_mask
        output = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(input_dtype)
        return output

    def vector_matmul_bf16(self, input, weight, async_op, q_scale, q_int8, transpose):
        return self.vector_matmul_func(input, weight, async_op, q_scale, q_int8, transpose)

    def vector_matmul_func(self, input, weight, async_op, q_scale, q_int8, transpose):
        if not transpose:
            return torch.matmul(input, weight)
        else:
            raise NotImplementedError

    def mlp_gemm_bf16(self, input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                      transpose):
        return self.mlp_gemm_func(input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                      transpose)

    def mlp_gemm_func(self, input, residual, input_bias, weight_interm, weight_out, bias, gamma, beta, eps,
                      pre_layer_norm, mlp_after_attn, interm_scale, out_scale, dtype, mlp_act_func_type,
                      transpose):
         if mlp_after_attn and not transpose:
             residual_add = F.layer_norm(input + residual + input_bias, (input.shape[2], ), gamma, beta, eps)
             tmp = torch.matmul(residual_add, weight_interm)
             tmp = F.gelu(tmp + bias)
             output = torch.matmul(tmp, weight_out)
             return output, residual_add
         else:
             raise NotImplementedError

    def residual_add_bias_bf16(self, hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, pre_layer_norm):
        return self.residual_add_bias_func(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, pre_layer_norm)

    def residual_add_bias_func(self, hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_bias, pre_layer_norm):
        if mlp_after_attn:
            if pre_layer_norm:
                tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                       final_bias.float()) / mp_size + hidden_state.float()
            else:
                tmp = residual.float() + hidden_state.float() + final_bias.float()

            input_dtype = hidden_state.dtype
            residual.copy_(tmp.to(input_dtype))
        else:
            raise NotImplementedError

    def _vector_add(self):
        raise NotImplemented


class InferenceBuilder():
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def load(self):
        return InferenceBuilderObject()
'''
    def is_compatible(self, verbose=True):
        try:
            import torch
        except ImportError:
            self.warning("Please install torch if trying to pre-compile inference kernels")
            return False

        cuda_okay = True
        if not self.is_rocm_pytorch() and torch.cuda.is_available():
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split('.')[0])
            cuda_capability = torch.cuda.get_device_properties(0).major
            if cuda_capability < 6:
                self.warning("NVIDIA Inference is only supported on Pascal and newer architectures")
                cuda_okay = False
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in ccs:
            if int(cc[0]) >= 6:
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
            'csrc/transformer/inference/csrc/relu.cu',
            'csrc/transformer/inference/csrc/layer_norm.cu',
            'csrc/transformer/inference/csrc/rms_norm.cu',
            'csrc/transformer/inference/csrc/softmax.cu',
            'csrc/transformer/inference/csrc/dequantize.cu',
            'csrc/transformer/inference/csrc/apply_rotary_pos_emb.cu',
            'csrc/transformer/inference/csrc/transform.cu',
            'csrc/transformer/inference/csrc/pointwise_ops.cu',
        ]

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def include_paths(self):
        return ['csrc/transformer/inference/includes', 'csrc/includes']
'''
