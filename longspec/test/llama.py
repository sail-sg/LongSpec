import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)

from flash_attn import flash_attn_func, flash_attn_with_kvcache

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Modification here to support tree decoding
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.K_Cache = None
        self.V_Cache = None
        self.answer_K_Cache = None
        self.answer_V_Cache = None
        self.max_len = 512
        self.log_ratio = math.log(0.7)
        self.prefix_lens = None
        self.layer_idx = layer_idx
        self.last_layer = (config.num_hidden_layers == self.layer_idx + 1)
        self.softmax_scale = 1 / (128 ** 0.5)
        self.range_indices = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        tree_mask=None,
        exec_type="training",
        induction_head=False,
    ):
        
        kv_cache = None
        if exec_type == "prefill":
            y = self.prefill(hidden_states, position_embeddings)
        elif exec_type == "decoding":
            y = self.decoding(hidden_states, position_embeddings, cache_lens)
        elif exec_type == "training":
            y = self.training(hidden_states, position_embeddings)
        elif exec_type == "glide_training":
            y, kv_cache = self.glide_training(hidden_states, position_embeddings, induction_head)
        elif exec_type == "tree_decoding":
            y = self.tree_decoding(hidden_states, position_embeddings, cache_lens, tree_mask)
        elif exec_type == "prefill_torch":
            y = self.prefill_torch(hidden_states, position_embeddings)
        elif exec_type == "decoding_torch":
            y = self.decoding_torch(hidden_states, position_embeddings, cache_lens)
        elif exec_type == "magicdec_prefill":
            y = self.magicdec_prefill(hidden_states, position_embeddings)
        elif exec_type == "magicdec_decoding":
            y = self.fix_stream_spec(hidden_states, position_embeddings, cache_lens)
        else:
            raise ValueError(f"Unknown inference_type: {exec_type}")
        return y, kv_cache
     
    def prefill_torch(
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
        self.K_Cache = query_states.new_zeros((bsz, self.num_key_value_heads, q_len + self.max_len, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, self.num_key_value_heads, q_len + self.max_len, self.head_dim))
        self.K_Cache[:, :, :q_len, :] = key_states.transpose(1, 2)
        self.V_Cache[:, :, :q_len, :] = value_states.transpose(1, 2)
        self.range_indices = torch.arange(1024, device=self.K_Cache.device)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))

        return attn_output

    def decoding_torch(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        self.K_Cache[:, :, cache_lens : cache_lens + q_len, :] = key_states
        self.V_Cache[:, :, cache_lens : cache_lens + q_len, :] = value_states
        K_total = torch.cat((self.K_Cache[:, :, :cache_lens, :], key_states), dim=2)
        V_total = torch.cat((self.V_Cache[:, :, :cache_lens, :], value_states), dim=2)
        K_total = K_total.repeat_interleave(self.num_key_value_groups, dim=1).transpose(3, 2)
        V_total = V_total.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = torch.triu(torch.ones(q_len, q_len, device=query_states.device), diagonal=1).bool()
        total_mask = torch.cat((torch.zeros((q_len, cache_lens), device=query_states.device, dtype=bool), mask), dim=1)
        scores = torch.matmul(query_states, K_total) * self.softmax_scale
        scores = scores.masked_fill(total_mask, float('-inf'))
        attn_weights = F.softmax(scores.float(), dim=-1)
        attn_output = torch.matmul(attn_weights.to(V_total.dtype), V_total)
        attn_output = attn_output.transpose(1, 2).contiguous()
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

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        self.K_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.K_Cache[:, :q_len] = key_states
        self.V_Cache[:, :q_len] = value_states
        self.range_indices = torch.arange(1024, device=self.K_Cache.device)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))

        return attn_output

    def magicdec_prefill(
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
        self.K_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.V_Cache = query_states.new_zeros((bsz, q_len + self.max_len, self.num_key_value_heads, self.head_dim))
        self.K_Cache[:, :q_len] = key_states
        self.V_Cache[:, :q_len] = value_states
        self.range_indices = torch.arange(1024, device=self.K_Cache.device)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))

        self.buffer_size = 1024
        buffer_size = self.buffer_size
        self.stream_k_cache = query_states.new_zeros(bsz, buffer_size + 32 + self.max_len, self.num_key_value_heads, self.head_dim)
        self.stream_v_cache = query_states.new_zeros(bsz, buffer_size + 32 + self.max_len, self.num_key_value_heads, self.head_dim)
        self.stream_k_cache[:, 0: 32] = self.K_Cache[:, 0: 32]
        self.stream_v_cache[:, 0: 32] = self.V_Cache[:, 0: 32]
        self.stream_k_cache[:, 32: buffer_size + 32] = self.K_Cache[:, max(q_len-buffer_size, 0): q_len]
        self.stream_v_cache[:, 32: buffer_size + 32] = self.V_Cache[:, max(q_len-buffer_size, 0): q_len]

        return attn_output

    def glide_training(
            self,
            hidden_states,
            position_embeddings,
            induction_head=False
            ):
    
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if induction_head:
            k = key_states.clone()
            v = value_states.clone()
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if self.last_layer:
            if induction_head:
                kv_cache = (k, v)
            else:
                kv_cache = (key_states, value_states)
        else:
            kv_cache = None
        return attn_output, kv_cache

    def decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
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

        attn_output = flash_attn_with_kvcache(query_states, self.K_Cache, self.V_Cache, key_states, value_states, 
                                             causal=True, cache_seqlens=cache_lens)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def fix_stream_spec(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
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
        attn_output = flash_attn_with_kvcache(query_states, self.stream_k_cache, self.stream_v_cache, key_states, value_states, 
                                             causal=True, cache_seqlens=cache_lens)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def tree_decoding(
            self,
            hidden_states,
            position_embeddings,
            cache_lens,
            tree_mask=None,
            ):
        '''
        tree_mask: bsz fseq fseq (flatten_seqlen)
        '''
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if tree_mask is None:
            assert q_len == 1, "You are in the first step of tree decoding, thus you should not input qlen > 2 without tree mask"
            attn_output = flash_attn_with_kvcache(query_states, self.K_Cache, self.V_Cache, key_states, value_states, cache_seqlens=cache_lens, causal=True)

        else:
            prefix_o, prefix_lse = flash_attn_with_kvcache(query_states, self.K_Cache, self.V_Cache, cache_seqlens=cache_lens, return_softmax_lse=True)
            current_out, weight = self.tree_part_fwd(query_states, key_states, value_states, tree_mask, cache_lens, prefix_lse, bsz, q_len)
            attn_output = prefix_o * weight + current_out * (1 - weight)

        attn_output = attn_output.view(bsz, q_len, self.hidden_size).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output

    @torch.compile
    def tree_part_fwd(self, query_states, key_states, value_states, tree_mask, cache_lens, prefix_lse, bsz, q_len):
        range_indices = cache_lens.unsqueeze(-1) + self.range_indices[:tree_mask.size(-1)].unsqueeze(0)  # 计算范围
        bsz_indices = self.range_indices[:bsz].unsqueeze(-1)
        self.K_Cache[bsz_indices, range_indices] = key_states
        self.V_Cache[bsz_indices, range_indices] = value_states
        
        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.permute(0, 2, 3, 1)
        value_states = value_states.transpose(1, 2)
        if self.last_layer:
            attn_score = torch.matmul(query_states * self.softmax_scale, key_states)
        else:
            attn_score = torch.matmul(query_states, key_states) * self.softmax_scale
        attn_score = attn_score.to(torch.float32)
        attn_score_tree_mask = tree_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
        attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
        current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
        current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)
        if torch._dynamo.is_compiling():
            prefix_lse = prefix_lse.reshape(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        else:
            prefix_lse = prefix_lse.view(bsz, self.num_heads, q_len, -1).transpose(1, 2)
        weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
        return current_out, weight

    def training(
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

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True).view(bsz, q_len, -1)
        
        attn_output = self.o_proj(attn_output)

        return attn_output


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaAttention,
    "sdpa": LlamaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:


        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            cache_lens=cache_lens,
            flex_attn=flex_attn,
            exec_type=exec_type,
            tree_mask=tree_mask,
            induction_head=induction_head,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, kv_cache)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids,
        position_ids=None,
        position_embeddings=None,
        inputs_embeds=None,
        cache_lens=None,
        flex_attn=None,
        exec_type=None,
        tree_mask=None,
        induction_head=False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if position_ids is None:
            if tree_mask is None:
                position_ids = torch.arange(0, input_ids.size(1))[None, :].to(input_ids.device)
            else:
                position_ids = tree_mask.sum(dim=-1) - 1
            if cache_lens is not None:
                position_ids = position_ids + cache_lens[:, None]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings,
                    cache_lens,
                    flex_attn,
                    exec_type,
                    tree_mask,
                    induction_head,
                )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if exec_type == "glide_training":
            kv_cache = layer_outputs[1]
        else:
            kv_cache = None
        # add hidden states from the last decoder layer

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache,
            hidden_states=None,
            attentions=None,
        )

class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_max_gen_len(self, max_gen_len):
        for layer in self.model.layers:
            layer.self_attn.max_len = max_gen_len
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids,
        position_ids=None,
        position_embeddings=None,
        inputs_embeds=None,
        labels=None,
        cache_lens=None,
        exec_type="training",
        induction_head=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds,
            cache_lens=cache_lens,
            exec_type=exec_type,
            induction_head=induction_head,
        )

        hidden_states = outputs[0]
        last_kv = outputs[1]

        loss = None
        if labels is not None:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            loss_fn = LigerFusedLinearCrossEntropyLoss()
            hidden_dim = hidden_states.size(-1)
            loss = loss_fn(self.lm_head.weight, hidden_states[:, 1:].reshape(-1, hidden_dim), labels[:, :-1].view(-1))
        else:
            logits = self.lm_head(hidden_states[:, -128:, :]).float()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=last_kv,
            hidden_states=None,
            attentions=None,
        )
