from typing import Optional
import time
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from dataclasses import dataclass
from transformers import Qwen2Config
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from qwen2 import Qwen2ForCausalLM, Qwen2RotaryEmbedding, Qwen2RMSNorm, Qwen2MLP, apply_rotary_pos_emb
from models.mixin import PretrainedModelParallelPreSplitMixin
from triton_tree_attn import attention as tree_attention


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    llm_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

class GlideAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.max_len = 512
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.softmax_scale = 1 / (self.head_dim ** 0.5)
        self.range_indices = torch.arange(1024)
        
        self.set_torch_mask()
    
    def set_torch_mask(self, max_len=4096, block_size=4):
        q_idx = torch.arange(max_len).view(-1, 1)
        kv_idx = torch.arange(max_len).view(1, -1)
        self.torch_mask = q_idx // block_size > kv_idx // block_size
        self.torch_mask = self.torch_mask.cuda()
        self.torch_mask[:4, :4] = True

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        exec_type="training",
        k_cache=None,
        v_cache=None,
        llm_kv_len=None,
        tree_mask=None,
    ):
        
        if exec_type in ["prefill", "sa_prefill"]:
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "sa_training":
            y = self.sa_training(hidden_states, position_embeddings)
        elif exec_type == "sa_decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens, K_Cache=None, V_Cache=None)
        elif exec_type in ["decoding", "ca_decoding", "ca_prefill"]:
            y = self.decoding(hidden_states, position_embeddings, cache_lens, k_cache, v_cache, llm_kv_len)
        elif exec_type in ["sa_tree_decoding"]:
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, None, None, llm_kv_len, tree_mask)
        elif exec_type in ["ca_tree_decoding"]:
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, k_cache, v_cache, llm_kv_len, tree_mask)
        elif exec_type == "ca_training":
            y = self.flash_glide_cross_attn_training(hidden_states, position_embeddings, k_cache, v_cache)
        else:
            raise ValueError(f"Unknown inference_type: {exec_type}")
        return y

    def flash_glide_cross_attn_training(
            self,
            hidden_states,
            position_embeddings,
            k_cache,  # LLM key cache, size (bsz, seqlen, num_heads, head_dim)
            v_cache,  # LLM value cache, size (bsz, seqlen, num_heads, head_dim)
            ):
        """
        Args:
            hidden_states: current hiddend
            position_embeddings
            k_cache: LLM key cache, (batch_size, sequence_length, num_heads, head_dimension)
            v_cache: LLM value cache, (batch_size, sequence_length, num_heads, head_dimension)
        """
        bsz, seqlen, numheads, headdim = k_cache.size()
        k_cache = k_cache.clone().requires_grad_(True)
        v_cache = v_cache.clone().requires_grad_(True)

        pad_size = random.randint(1, 4)
        # pad_size = 4
        pad_out = k_cache.new_zeros((bsz, pad_size, self.num_heads, headdim))
        
        k_cache = k_cache[:, :-pad_size]
        v_cache = v_cache[:, :-pad_size]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states[:, pad_size:]
        attn_output = flash_attn_func(query_states, k_cache, v_cache, causal=True)
        attn_output = torch.cat([pad_out, attn_output], dim=1)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def glide_cross_attn_training(
            self,
            hidden_states,
            position_embeddings,
            k_cache,
            v_cache,
            ):
        k_cache = k_cache.clone().requires_grad_(True)
        v_cache = v_cache.clone().requires_grad_(True)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k_cache.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v_cache.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, unsqueeze_dim=1)

        key_states = key_states.view(bsz, self.num_key_value_heads, 1, q_len, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(query_states.size())
        value_states = value_states.view(bsz, self.num_key_value_heads, 1, q_len, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(query_states.size())
        mask = self.torch_mask[None, None, :q_len, :q_len]
        scores = torch.matmul(query_states, key_states.transpose(3, 2)) / (key_states.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def sa_training(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attn_output = flash_attn_func(query_states, key_states, value_states, window_size=(512, -1), causal=True)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

    def prefill(
            self,
            hidden_states,
            position_embeddings,
            ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        self.K_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.K_Cache[:, :q_len] = key_states
        self.V_Cache[:, :q_len] = value_states
        attn_output = flash_attn_func(query_states, key_states, value_states, window_size=(512, -1), causal=True)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        self.range_indices = self.range_indices.to(self.K_Cache.device)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
    def decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            K_Cache,
            V_Cache,
            llm_kv_len=None
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if K_Cache is None:
            K_Cache = self.K_Cache
            V_Cache = self.V_Cache
            attn_output = flash_attn_with_kvcache(query_states, K_Cache, V_Cache, key_states, value_states, 
                                                 window_size=(512,-1), causal=True, cache_seqlens=cache_lens.int())
        else:
            cache_lens = llm_kv_len
            attn_output = flash_attn_with_kvcache(query_states, K_Cache, V_Cache, causal=True, cache_seqlens=cache_lens.int())

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            K_Cache, # from LLM
            V_Cache, # from LLM
            llm_kv_len=None,
            tree_mask=None,
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if K_Cache is not None:
            attn_output = flash_attn_with_kvcache(query_states, K_Cache, V_Cache, causal=False, cache_seqlens=llm_kv_len.int())
        
        else:
            prefix_o, prefix_lse = flash_attn_with_kvcache(query_states, self.K_Cache, self.V_Cache, window_size=(512,-1), cache_seqlens=cache_lens, return_softmax_lse=True)
            current_out, weight = self.triton_tree_part_fwd(query_states, key_states, value_states, tree_mask, cache_lens, prefix_lse, bsz, q_len)
            attn_output = prefix_o.to(torch.float32) * weight + current_out * (1 - weight)

        attn_output = attn_output.view(bsz, q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output
        
    def triton_tree_part_fwd(self, query_states, key_states, value_states, tree_mask, cache_lens, prefix_lse, bsz, q_len):
        # update kv cache
        _, current_kv_len, all_kv_len = tree_mask.size()
        range_indices = cache_lens.unsqueeze(-1) + self.range_indices[all_kv_len - current_kv_len : all_kv_len].unsqueeze(0)
        bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
        self.K_Cache[bsz_indices, range_indices] = key_states
        self.V_Cache[bsz_indices, range_indices] = value_states

        all_cache_indices = cache_lens.unsqueeze(-1) + self.range_indices[0 :all_kv_len].unsqueeze(0)
        key_states = self.K_Cache[bsz_indices, all_cache_indices]
        value_states = self.V_Cache[bsz_indices, all_cache_indices]
        current_out, current_lse = tree_attention(
            query_states.permute(0, 2, 1, 3), 
            key_states.permute(0, 2, 1, 3), 
            value_states.permute(0, 2, 1, 3), 
            tree_mask
        )
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse)
        current_out = current_out.transpose(1, 2)
        weight = weight.transpose(1, 2).unsqueeze(-1)
        return current_out, weight

    # A non-triton version of tree_part_fwd
    @torch.compile
    def tree_part_fwd(self, query_states, key_states, value_states, tree_mask, cache_lens, prefix_lse, bsz, q_len):
        # update kv cache
        _, current_kv_len, all_kv_len = tree_mask.size()
        # print(f"all_kv_len - current_kv_len: {all_kv_len - current_kv_len}")
        range_indices = cache_lens.unsqueeze(-1) + self.range_indices[all_kv_len - current_kv_len : all_kv_len].unsqueeze(0)
        bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
        self.K_Cache[bsz_indices, range_indices] = key_states
        self.V_Cache[bsz_indices, range_indices] = value_states

        all_cache_indices = cache_lens.unsqueeze(-1) + self.range_indices[0 :all_kv_len].unsqueeze(0)
        key_states = self.K_Cache[bsz_indices, all_cache_indices]
        value_states = self.V_Cache[bsz_indices, all_cache_indices].to(torch.float32)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute(0, 2, 3, 1)
        value_states = value_states.transpose(1, 2)
        attn_score = torch.matmul(query_states, key_states) * self.softmax_scale
        attn_score = attn_score.masked_fill(tree_mask.unsqueeze(1) == 0, -float('inf')).to(torch.float32)
        attn_weight = torch.softmax(attn_score, dim=-1)
        current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
        current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
        if torch._dynamo.is_compiling():
            prefix_lse = prefix_lse.reshape(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        else:
            prefix_lse = prefix_lse.view(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse)
        return current_out, weight

    def vanilla_training(
            self,
            hidden_states,
            position_embeddings,
            ):
    
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2GlideDecoderLayer(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlideAttention", "Qwen2MLP", "Qwen2RMSNorm"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.layer_idx = 0
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.self_attn = GlideAttention(config, self.layer_idx)
        self.cross_attn = GlideAttention(config, self.layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._init_weights
        self.config = config
    
    def set_max_gen_len(self, max_gen_len):
        self.self_attn.max_len = max_gen_len
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        position_embeddings,
        llm_kv,
        cache_lens=None,
        exec_type=None,
        llm_kv_len=None,
        tree_mask=None,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            exec_type="sa_" + exec_type,
            tree_mask=tree_mask,
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        residual = hidden_states
        hidden_states = self.post_self_attention_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states, 
            position_embeddings=position_embeddings, 
            cache_lens=cache_lens, 
            exec_type="ca_" +exec_type, 
            k_cache=llm_kv[0], 
            v_cache=llm_kv[1], 
            llm_kv_len=llm_kv_len,
            tree_mask=tree_mask,
        )
        hidden_states += residual

        # FFN
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states

class Qwen2Glide(PretrainedModelParallelPreSplitMixin, Qwen2ForCausalLM):
    def __init__(self, config, target_model_path, glide_path=None):
        super().__init__(config)
        model = Qwen2ForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, device_map="auto")
        self.model = model.model
        self.lm_head = model.lm_head

        if glide_path is None:
            self.draft_model = Qwen2GlideDecoderLayer(config)
        else:
            self.draft_model = Qwen2GlideDecoderLayer.from_pretrained(glide_path, torch_dtype=torch.float16, device_map="auto")
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        for param in self.draft_model.parameters():
            param.requires_grad = True
        
        self.post_init()
        
    def compute_fused_loss(self, hidden_states, labels):

        shift_hidden_states = hidden_states[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_head_weight = self.lm_head.weight.float().contiguous()

        shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        loss = lce(lm_head_weight, shift_hidden_states, shift_labels)
        return loss
    
    def compute_loss(self, hidden_states, labels):
        logits = self.lm_head(hidden_states).float()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return loss

    def forward(
        self,
        input_ids,
        labels,
        position_ids=None,
        cache_lens=None,
        **kwargs,
    ):
        if position_ids is None:
            if input_ids.size(1) > 1200:
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            else:
                sink = random.randint(0, 4)
                random_offset = max(min(30000, self.config.model_max_length - 1000) - input_ids.size(1), 0)
                random_offset = random.randint(0, random_offset)
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
                position_ids[:, sink:] = random_offset + position_ids[:, sink:]
        if cache_lens is not None:
            position_ids = position_ids + cache_lens

        labels[labels.eq(self.config.pad_token_id)] = -100
        with torch.inference_mode():
            llm_outputs = self.model(
                input_ids=input_ids,
                exec_type="glide_training",
                position_ids=position_ids,
                inputs_embeds=None,
                cache_lens=cache_lens,
            )
            llm_loss = self.compute_fused_loss(llm_outputs.last_hidden_state, labels)
        llm_last_kv = llm_outputs.past_key_values
        position_embeddings = self.model.rotary_emb(llm_last_kv[0], position_ids)
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = self.draft_model(hidden_states=hidden_states, position_embeddings=position_embeddings, llm_kv=llm_last_kv, exec_type="training")
        loss = self.compute_fused_loss(hidden_states, labels)

        return CausalLMOutputWithPast(
            llm_loss=llm_loss,
            loss=loss,
        )
    
    def vanilla_generate(self, input_ids, prompt_length, max_gen_len=64, eos_id=151645):
        assert input_ids != None, "please give the input"
        bsz = input_ids.size(0)
        output_ids = input_ids.new_zeros((bsz, max_gen_len))
        
        self.set_max_gen_len(max_gen_len)
        
        cache_lens = input_ids.new_zeros((bsz)).int()
        hidden_states = self.model.forward(input_ids, exec_type="prefill").last_hidden_state
        input_len = prompt_length
        output_ids[:, 0] = self.lm_head(hidden_states[range(bsz), input_len-1, :]).argmax(dim=-1)
        cache_lens += input_len
        num = 0

        torch.cuda.synchronize()
        start_time = time.time()

        # autoregressive decoding
        for _ in range(1, max_gen_len):
            input_ids = output_ids[range(bsz), cache_lens - input_len].view(bsz, -1)
            hidden_states = self.model.forward(input_ids, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
            llm_output = self.lm_head(hidden_states[:, -1, :]).argmax(dim=-1)
            cache_lens += 1
            num += bsz
            output_ids[range(bsz), cache_lens - input_len] = llm_output.view(-1)
            if (output_ids.eq(eos_id)).any():
                break

        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time

        return output_ids, num, elapsed_time

    def spec_generate(self, input_ids, prompt_length, gamma=4, max_gen_len=64, eos_id=151645, temperature=0.0):
        assert input_ids != None, "please give the input"
        bsz = input_ids.size(0)
        output_ids = input_ids.new_zeros((bsz, max_gen_len + gamma))
        spec_mask = input_ids.new_zeros((bsz, max_gen_len + gamma))
        
        self.set_max_gen_len(max_gen_len + 128)
        self.draft_model.set_max_gen_len(max_gen_len + 128)
        
        cache_lens = input_ids.new_zeros((bsz)).int()
        hidden_states = self.model.forward(input_ids, exec_type="prefill")["last_hidden_state"]
        input_len = prompt_length
        logits = self.lm_head(hidden_states[range(bsz), input_len-1, :])
        output_ids[:, 0] = logits.argmax(dim=-1)
        cache_lens += input_len
        draft_cache_lens = cache_lens.clone()
        spec_buffer = output_ids.new_zeros((bsz, gamma + 1))
        spec_buffer[:, 0] = output_ids[:, 0]
        spec_logits = output_ids.new_zeros((bsz, gamma + 1, self.vocab_size), dtype=torch.float32)
        spec_logits[:, 0] = logits

        # Glide prefill
        hidden_states = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        self.draft_model(
            hidden_states=hidden_states, 
            position_embeddings=position_embeddings, 
            llm_kv=(self.model.layers[-1].self_attn.K_Cache, self.model.layers[-1].self_attn.V_Cache), 
            cache_lens=draft_cache_lens.clone(), 
            llm_kv_len=cache_lens.clone(),
            exec_type="prefill",
        )

        # spec tokens
        double_flag = False
        next_spec_start_token = output_ids.new_zeros((bsz, 2))
        count = 0
        num = 0
        next_spec_start_token[:, 0] = output_ids[:, 0]

        torch.cuda.synchronize()
        start_time = time.time()
        # record_time = time.time()
        # autoregressive decoding
        for out_index in range(1, max_gen_len):
            # speculative decoding
            for spec_steps in range(0, gamma):
                
                # we should use draft_cache_lens for indexing hidden_states and output_ids, even position_ids.
                # spec_steps is only used for checking different conditions.
                # to avoid any confusion.

                # cache lens is exactly the length of kv cache, so, it will also be the first one of position_ids.
                if spec_steps == 0:
                    if double_flag:
                        hidden_states = self.model.embed_tokens(next_spec_start_token[:, 0:2])
                        position_ids = torch.arange(0, 2)[None, :].to(input_ids.device) + draft_cache_lens[:, None]
                        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
                    else:
                        hidden_states = self.model.embed_tokens(next_spec_start_token[:, 0, None])
                        position_ids = draft_cache_lens[:, None]
                        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)            

                else:
                    hidden_states = self.model.embed_tokens(spec_buffer[:, spec_steps, None])
                    position_ids = draft_cache_lens[:, None]
                    position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

                hidden_states = self.draft_model(
                    hidden_states=hidden_states, 
                    position_embeddings=position_embeddings, 
                    llm_kv=(self.model.layers[-1].self_attn.K_Cache, self.model.layers[-1].self_attn.V_Cache), 
                    cache_lens=draft_cache_lens.clone(), 
                    llm_kv_len=cache_lens.clone(), 
                    exec_type="decoding"
                )
                
                if double_flag and (spec_steps == 0):
                    # double batch id, if double accept, then gather the -1 token, else, gather the -2 token.
                    draft_cache_lens += 1 + double_input
                    current_logp = self.lm_head(hidden_states[:, -2:, :])
                    spec_buffer[:, spec_steps + 1] = current_logp.argmax(dim=-1)[range(bsz), double_input]
                    spec_logits[:, spec_steps + 1, :] = current_logp[range(bsz), double_input, :]
                else:
                    draft_cache_lens += 1
                    current_logp = self.lm_head(hidden_states[:, -1, :])
                    spec_buffer[:, spec_steps + 1] = current_logp.argmax(dim=-1).view(-1,)
                    spec_logits[:, spec_steps + 1, :] = current_logp

            hidden_states = self.model.forward(spec_buffer, cache_lens=cache_lens.clone(), exec_type="decoding").last_hidden_state
            llm_verify_logits = self.lm_head(hidden_states[:, -gamma - 1:, :])
            llm_verify_output = llm_verify_logits.argmax(dim=-1)

            if temperature > 0:
                q_probs = F.softmax(spec_logits[:, 1:, :], dim=-1)
                p_probs = F.softmax(llm_verify_logits[:, :-1, :], dim=-1)
                gather_index = spec_buffer[:, 1:].unsqueeze(-1)
                q_token_prob = torch.gather(q_probs, dim=-1, index=gather_index).squeeze(-1)
                p_token_prob = torch.gather(p_probs, dim=-1, index=gather_index).squeeze(-1)
                eps = 1e-9
                ratio = (p_token_prob + eps) / (q_token_prob + eps)
                alpha = torch.clip(ratio, 0.0, 1.0) # equal to min(ratio, 1)
                random_vals = torch.rand_like(alpha)
                accept_mask = random_vals.lt(alpha)
                p_distribution = torch.distributions.Categorical(p_probs.reshape(-1, p_probs.size(-1)))
                p_resample_tokens = p_distribution.sample().reshape(bsz, gamma)
                llm_verify_output[:, :-1] = torch.where(
                    accept_mask,
                    spec_buffer[:, 1:],
                    p_resample_tokens
                )
                verification = accept_mask.cumprod(dim=-1)
                correct_len = verification.sum(dim=-1) + 1
                llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

            else:
                verification = llm_verify_output[:, :-1].eq(spec_buffer[:, 1:]).cumprod(dim=-1)
                correct_len = verification.sum(dim=-1) + 1
                llm_verify_output[:, 1:] = llm_verify_output[:, 1:] * verification

            row_indices = torch.arange(bsz, device=cache_lens.device).unsqueeze(1)
            col_indices = (cache_lens - input_len).unsqueeze(1) + torch.arange(1, gamma + 1, device=cache_lens.device)
            output_ids[row_indices, col_indices] = llm_verify_output[:, :gamma]

            bonus_token = llm_verify_output[range(bsz), correct_len - 1]
            output_ids[range(bsz), cache_lens - input_len + correct_len] = bonus_token
            cache_lens += correct_len
            double_input = correct_len.eq(gamma + 1).to(torch.int)
            double_flag = double_input.eq(1).any()

            if double_flag:
                next_spec_start_token[:, 0] = llm_verify_output[range(bsz), correct_len - 2]
                next_spec_start_token[:, 1] = llm_verify_output[range(bsz), correct_len - 1]
                next_spec_start_token[:, 0] = (1 - double_input.int()) * next_spec_start_token[:, 1] + double_input.int() * next_spec_start_token[:, 0]  
            else:
                next_spec_start_token[:, 0] = bonus_token
            spec_buffer[:, 0] = bonus_token
            
            count += (correct_len - 1).sum()
            num += bsz
            correct_len = correct_len.clamp(max=gamma)
            draft_cache_lens = cache_lens - double_input

            if (cache_lens - input_len).max() + gamma + 2 > output_ids.size(1):
                break
            if (output_ids.eq(self.config.eos_token_id)).any():
                # print("double buffer spec end")
                break

        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return output_ids, count, num, elapsed_time, spec_mask

    def tree_spec_generate(self, input_ids, prompt_length, tree_shape=None, max_gen_len=64, eos_id=151645, temperature=0.0):
        assert input_ids != None, "please give the input"
        if not hasattr(self, "range_tensor"):
            self.range_tensor = torch.arange(0, 1024)[None, :].to(input_ids.device)  # init
            self.reverse_range_tensor = torch.arange(-1024, -32 + 1).unsqueeze(0).to(input_ids.device)
            self.oned_range_tensor = torch.arange(0, 1024).to(input_ids.device)
            self.diag_matrix = input_ids.new_zeros((1024, 1024))[None, :, :]
            self.diag_matrix[:, range(1024), range(1024)] = 1
        
        self.set_max_gen_len(max_gen_len + 256)
        self.draft_model.set_max_gen_len(max_gen_len + 256)

        # spec tokens
        if tree_shape is None:
            cand_num_per_step = [4, 16, 16, 16, 16]
        else:
            cand_num_per_step = tree_shape
        acc_num_per_step = [1] + [0 for _ in range(len(cand_num_per_step))]
        for i in range(1, len(cand_num_per_step) + 1):
            acc_num_per_step[i] = acc_num_per_step[i - 1] + cand_num_per_step[i - 1]
        
        bsz = input_ids.size(0)
        output_ids = input_ids.new_zeros((bsz, max_gen_len))
        spec_mask = input_ids.new_zeros((bsz, max_gen_len))

        # input_len : the length of input ids for each instance in the batch
        # cache_lens : the length of large llm kv cache for each instance in the batch
        # draft_cache_lens : the length of small llm kv cache for each instance in the batch
        input_len = prompt_length
        cache_lens = input_ids.new_zeros((bsz)).int()
        target_cache_lens_for_draft = input_ids.new_zeros((bsz)).int()
        draft_cache_lens = input_ids.new_zeros((bsz)).int()
        
        # count : the output token number of small model
        # num : the output token number of large model
        count = 0
        num = 0

        # prefill LLM
        hidden_states = self.model.forward(input_ids, exec_type="prefill")["last_hidden_state"]
        output_prob = self.lm_head(hidden_states[range(bsz), input_len - 1, ...])
        output_ids[:, 0] = output_prob.argmax(dim=-1)
        num += bsz
        cache_lens += input_len
        target_cache_lens_for_draft += input_len
        draft_cache_lens += input_len
        vocab_size = output_prob.size(-1)
        all_spec = output_ids.new_zeros((bsz, sum(cand_num_per_step) + 1))
        all_spec[:, 0] = output_ids[:, 0]
        spec_logits = output_ids.new_zeros((bsz, sum(cand_num_per_step) + 1, vocab_size), dtype=torch.float32)
        spec_logits[:, 0] = output_prob

        # prefill glide
        position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
        hidden_states = self.model.embed_tokens(input_ids)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids) 
        self.draft_model(
            hidden_states=hidden_states, position_embeddings=position_embeddings, 
            llm_kv=(self.model.layers[-1].self_attn.K_Cache, self.model.layers[-1].self_attn.V_Cache), 
            cache_lens=draft_cache_lens.clone(), llm_kv_len=target_cache_lens_for_draft.clone(), exec_type="prefill"
        )
        
        gamma = len(cand_num_per_step)

        # acc_ids : accepted token ids after each round of verification
        # Initially, it contains the output of the large llm in the prefilling stage
        acc_ids = output_ids[:, 0].unsqueeze(-1)
        acc_num = acc_ids.ne(self.config.pad_token_id).sum(dim=-1)

        tree_mask = input_ids.new_zeros(bsz, sum(cand_num_per_step) + 1, sum(cand_num_per_step) + 1)
        tree_mask[:, :, 0] = 1

        diag_one = input_ids.new_zeros(bsz, sum(cand_num_per_step) + 1, sum(cand_num_per_step) + 1)
        diag_one[:, range(tree_mask.size(1)), range(tree_mask.size(1))] = 1

        father_index = tree_mask.new_zeros((bsz, sum(cand_num_per_step) + 1))
        history_logp_sum = tree_mask.new_zeros((bsz, sum(cand_num_per_step) + 1), dtype=torch.float32)

        torch.cuda.synchronize()
        start_time = time.time()
        
        # autoregressive decoding
        for out_index in range(1, max_gen_len):
            # print("draft_cache_lens: ", draft_cache_lens)
            temp_input_len = acc_num
            father_index.zero_()
            history_logp_sum.zero_()

            pred_num = cand_num_per_step[0]
            hidden_states = self.model.embed_tokens(acc_ids)
            position_ids = torch.arange(0, acc_ids.size(1), device=input_ids.device)[None, :] + draft_cache_lens[:, None]
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            hidden_states = self.draft_model(
                hidden_states=hidden_states, 
                position_embeddings=position_embeddings, 
                llm_kv=(self.model.layers[-1].self_attn.K_Cache,
                        self.model.layers[-1].self_attn.V_Cache), 
                cache_lens=draft_cache_lens.clone(), 
                llm_kv_len=target_cache_lens_for_draft.clone(), 
                exec_type="decoding", 
            )

            draft_cache_lens += temp_input_len - 1
            current_logp = self.lm_head(hidden_states[range(bsz), temp_input_len - 1, :]).view(bsz, -1).float().log_softmax(dim=-1)
            topk_logp, pred_ids = current_logp.topk(dim=-1, k=pred_num, largest=True, sorted=True)

            tree_mask[:, 1:acc_num_per_step[1]] += diag_one[:, 1:acc_num_per_step[1]]
            current_tree_mask = tree_mask[:, 1:acc_num_per_step[1], :acc_num_per_step[1]]
            all_spec[:, 1:acc_num_per_step[1]] = pred_ids
            spec_logits[:, 0] = current_logp
            father_index[:, 1:acc_num_per_step[1]] = 0
            history_logp_sum[:, 1:acc_num_per_step[1]] = topk_logp
            
            for micro_step in range(1, gamma):
                pred_num = cand_num_per_step[micro_step]
                hidden_states = self.model.embed_tokens(all_spec[:, acc_num_per_step[micro_step-1]:acc_num_per_step[micro_step]])
                position_ids = draft_cache_lens[:, None] + current_tree_mask.sum(dim=-1) - 1
                position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

                hidden_states = self.draft_model(
                    hidden_states=hidden_states, 
                    position_embeddings=position_embeddings,
                    llm_kv=(self.model.layers[-1].self_attn.K_Cache, 
                            self.model.layers[-1].self_attn.V_Cache),
                    cache_lens=draft_cache_lens.clone(), 
                    llm_kv_len=target_cache_lens_for_draft.clone(), 
                    exec_type="tree_decoding", 
                    tree_mask=current_tree_mask
                )

                current_logp = self.lm_head(hidden_states).float().log_softmax(dim=-1)
                current_logp_sum = current_logp + history_logp_sum[:, acc_num_per_step[micro_step-1]:acc_num_per_step[micro_step], None]

                # # eagle tree
                # topk_p, topk_index = current_logp.topk(dim=-1, k=pred_num)
                # cu_scores = topk_p + history_logp_sum[:, acc_num_per_step[micro_step-1]:acc_num_per_step[micro_step], None]
                # topk_logp_sum, topk_cs_index = torch.topk(cu_scores.view(bsz, -1), pred_num, dim=-1)
                # father_ids = topk_cs_index // pred_num + acc_num_per_step[micro_step - 1]
                # pred_ids = topk_index.view(bsz, -1).gather(-1, topk_cs_index)
                
                # # cape tree
                # current_logp_sum[:, 1:, :] = float('-inf')
                # topk_logp_sum, topk_indices_flat = current_logp_sum.view(bsz, -1).topk(dim=-1, k=pred_num)
                # topk_logp_sum, topk_indices = topk_logp_sum.view(bsz, pred_num), topk_indices_flat.view(bsz, pred_num)
                # father_ids = topk_indices // vocab_size + acc_num_per_step[micro_step - 1]
                # print("father_ids", father_ids)
                # pred_ids = topk_indices % vocab_size
                
                # beam tree
                topk_logp_sum, topk_indices_flat = current_logp_sum.view(bsz, -1).topk(dim=-1, k=pred_num)
                topk_logp_sum, topk_indices = topk_logp_sum.view(bsz, pred_num), topk_indices_flat.view(bsz, pred_num)
                father_ids = topk_indices // vocab_size + acc_num_per_step[micro_step - 1]
                pred_ids = topk_indices % vocab_size
                
                for i in range(bsz):
                    tree_mask[i, acc_num_per_step[micro_step] : acc_num_per_step[micro_step + 1]] = \
                        tree_mask[i, father_ids[i]] + diag_one[i, acc_num_per_step[micro_step] : acc_num_per_step[micro_step + 1]]
                current_tree_mask = tree_mask[:, acc_num_per_step[micro_step]:acc_num_per_step[micro_step + 1], :acc_num_per_step[micro_step + 1]]
                all_spec[:, acc_num_per_step[micro_step] : acc_num_per_step[micro_step+1]] = pred_ids
                spec_logits[:, acc_num_per_step[micro_step-1] : acc_num_per_step[micro_step]] = current_logp
                history_logp_sum[:, acc_num_per_step[micro_step] : acc_num_per_step[micro_step + 1]] = topk_logp_sum
            draft_cache_lens += 1
            veri_spec = tree_mask.new_zeros((bsz, sum(cand_num_per_step) + gamma + 1))
            veri_spec[:, :acc_ids.size(1)] = acc_ids
            for i in range(bsz):
                veri_spec[i, temp_input_len[i]:temp_input_len[i] + sum(cand_num_per_step)] = all_spec[i, 1:sum(cand_num_per_step)+1]
            new_tree_mask = tree_mask.new_ones((bsz, veri_spec.size(1), veri_spec.size(1)))
            for i in range(bsz):
                new_tree_mask[i, temp_input_len[i]:temp_input_len[i] + sum(cand_num_per_step), temp_input_len[i]:temp_input_len[i] + sum(cand_num_per_step)] = tree_mask[i, 1:, 1:]
            new_tree_mask = torch.tril(new_tree_mask)
            
            hidden_states = self.model.forward(veri_spec, cache_lens=cache_lens.clone(), exec_type="tree_decoding", tree_mask=new_tree_mask)["last_hidden_state"]
            new_hidden_states = hidden_states.new_zeros((bsz, sum(cand_num_per_step)+1, hidden_states.shape[-1]))
            for i in range(bsz):
                new_hidden_states[i, :sum(cand_num_per_step)+1] = hidden_states[i, temp_input_len[i]-1:temp_input_len[i] + sum(cand_num_per_step)]
            hidden_states = new_hidden_states
            llm_logits = self.lm_head(hidden_states)

            if temperature > 0:
                cache_lens += temp_input_len - 1
                acc_ids, acc_num = self.verify_stochastic(
                    input_ids=all_spec,       # bsz * f_seq
                    tree_mask=tree_mask,      # bsz * f_seq * f_seq
                    p_llm=llm_logits,         # bsz * f_seq, target llm logits
                    p_ssm=spec_logits,        # bsz * f_seq, draft llm logits
                    temperature=temperature,
                )
                output_ids[self.oned_range_tensor[:bsz, None], (cache_lens - input_len).unsqueeze(1) + self.oned_range_tensor[:acc_ids.size(-1)]] = acc_ids
            else:
                cache_lens += temp_input_len - 1  # to help kv moving in verification
                all_llm_pred = llm_logits.argmax(dim=-1)
                acc_ids, acc_num, double_input = self.tree_verification(all_spec, all_llm_pred, tree_mask, cache_lens, non_leaf_len=acc_num_per_step[-2])
                cache_lens += 1
                output_ids[self.oned_range_tensor[:bsz, None], (cache_lens - input_len).unsqueeze(1) + self.oned_range_tensor[:acc_ids.size(-1)]] = acc_ids
            target_cache_lens_for_draft += acc_num
            count += (acc_num - 1).sum()
            num += bsz
            tree_mask.fill_(0)
            tree_mask[:, :, 0] = 1
            all_spec.fill_(0)
            all_spec[:, 0] = acc_ids[range(bsz), acc_num - 1]

            if (cache_lens + acc_num - input_len).max() + gamma + 2 > output_ids.size(1):
                break
            if (output_ids.eq(eos_id)).any():
                break
        
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return output_ids, count, num, elapsed_time, spec_mask

    @torch.compile
    def tree_verification(self, input_ids, output_ids, tree_mask, cache_lens, non_leaf_len):
        '''
        input_ids: bsz * flatten_seqlen (f_seq)
        output_ids: bsz * f_seq
        tree_mask: bsz * f_seq * f_seq
        '''
        bsz, fseqlen, _ = tree_mask.size()
        father_index = ((tree_mask - self.diag_matrix[:, :fseqlen, :fseqlen] )* self.range_tensor[:, :fseqlen].unsqueeze(1)).argmax(dim=-1)

        verify = output_ids.gather(1, father_index).eq(input_ids)
        verify[:, 0] = True
        masked_verify = tree_mask * verify[:, None, :]  # bsz * fseqlen * fseqlen
        final_verify = masked_verify.sum(dim=-1).eq(tree_mask.sum(dim=-1))  # bsz * fseqlen

        # last_select_index: the last chosen child node, then search for ancestors on the tree based on the child node
        last_select_index = (final_verify * self.range_tensor[:, :final_verify.size(1)]).argmax(dim=-1) # bsz
        double_input = (last_select_index >= non_leaf_len)
        # The mask for the last chosen token
        select_mask = tree_mask[self.range_tensor[0, :bsz], last_select_index, :]
        acc_num = select_mask.sum(dim=-1)
        acc_max_num = acc_num.max()

        # index_mapping represents the index of all selected nodes, and places all 0 on the right while keeping the order on the left
        # e.g., change [1, 0, 0, 0, 1, 0] to [-6, 0, 0, 0, 5, 0] and sort it
        index_mapping = select_mask * self.reverse_range_tensor[:, :final_verify.size(1)]
        index_mapping = torch.argsort(index_mapping, dim=-1)[:, :acc_max_num]
        acc_ids = output_ids.gather(dim=-1, index=index_mapping)

        # When selecting KV, add cached or index it out
        # After verification, move it on site
        bsz_index = self.oned_range_tensor[:bsz, None]
        seq_offset = self.oned_range_tensor[:fseqlen]
        seq_index = cache_lens.unsqueeze(1) + seq_offset
        change_len = index_mapping.size(-1)
        change_index = cache_lens.unsqueeze(1) + self.oned_range_tensor[:change_len]

        # It is very tricky here. We only move the last layer's kv cache.
        # This is because the last layer's kv cache is the only one that would be used by the draft model in the next speculation step.
        # The other layers' kv cache would be rewritten in the next large model forward pass, because only up to five tokens would not bring 
        # too much computation overhead, but moving all layers' kv cache would be too expensive.
        layer = self.model.layers[-1]
        last_turn_key_cache = layer.self_attn.K_Cache[bsz_index, seq_index].view(bsz, fseqlen, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)
        last_turn_value_cache = layer.self_attn.V_Cache[bsz_index, seq_index].view(bsz, fseqlen, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim)
        layer.self_attn.K_Cache[bsz_index, change_index] = last_turn_key_cache[bsz_index, index_mapping]
        layer.self_attn.V_Cache[bsz_index, change_index] = last_turn_value_cache[bsz_index, index_mapping]

        return acc_ids, acc_num, double_input.int()

    @torch.compile
    def verify_stochastic(
        self, 
        input_ids,      # bsz * f_seq
        tree_mask,      # bsz * f_seq * f_seq
        p_llm,          # bsz * f_seq, target llm logits
        p_ssm,          # bsz * f_seq, draft llm logits
        temperature,
    ):
        bsz, fseqlen, _ = tree_mask.size()
        p_llm = F.softmax(p_llm / temperature, dim=-1)
        p_ssm = F.softmax(p_ssm / temperature, dim=-1)

        father_index = (
            (tree_mask - self.diag_matrix[:, :fseqlen, :fseqlen]) 
            * self.range_tensor[:, :fseqlen].unsqueeze(1)
        ).argmax(dim=-1)

        acc_ids = input_ids.new_zeros((bsz, tree_mask.sum(-1).max()+1))
        acc_num = input_ids.new_zeros(bsz)

        for b in range(bsz):
            verified_tokens = []
            current_node = 0  
            verified_tokens.append(input_ids[b, current_node])

            while True:
                
                children = []
                for u in range(fseqlen):
                    if u != current_node and father_index[b, u] == current_node:
                        children.append(u)
                
                if len(children) == 0:
                    break

                candidate_children = children[:]  # duplicate one for the convenience of removing elements
                chosen_child = None

                while len(candidate_children) > 0:
                    s = random.choice(candidate_children)
                    r = random.random()
                    eps = 1e-9
                    ratio = (p_llm[b, current_node, s] + eps) / (p_ssm[b, current_node, s] + eps)

                    if r <= ratio:
                        verified_tokens.append(input_ids[b, s])
                        current_node = s
                        chosen_child = s
                        break
                    else:
                        candidate_children.remove(s)
                        p_llm[b, current_node, :] = p_llm[b, current_node, :] - p_ssm[b, current_node, :]
                        p_llm[b, current_node, :] = torch.clamp(p_llm[b, current_node, :], min=0)
                        denom_llm = p_llm[b, current_node, :].sum()
                        if denom_llm > 0:
                            p_llm[b, current_node, :] = p_llm[b, current_node, :] / denom_llm

                if chosen_child is None:
                    break

            new_node = torch.multinomial(p_llm[b, current_node, :], num_samples=1).item()
            verified_tokens.append(new_node)

            acc_num[b]= len(verified_tokens)
            for i in range(len(verified_tokens)):
                acc_ids[b, i] = verified_tokens[i]

        return acc_ids, acc_num
