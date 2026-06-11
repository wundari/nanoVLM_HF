# %%
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from torch.nn.attention import SDPBackend, sdpa_kernel


from jaxtyping import Float, Int
from configs.config import VLMConfig


# %%
class RMSNorm(nn.Module):
    """
    Root mean square layer normalization.

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    source:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69

    Args:
        cfg: VLMConfig containing lm_hidden_dim and lm_rms_eps.

    Forward input:
        x: Float[Tensor, "B T lm_hidden_dim"] input tensor with shape
            (B, T, lm_hidden_dim) and floating point dtype.

    Returns:
        Float[Tensor, "B T lm_hidden_dim"] normalized output tensor with the
            same shape and dtype as the input.
    """

    def __init__(self, cfg: VLMConfig):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(
        self, x: Float[Tensor, "B T lm_hidden_dim"]
    ) -> Float[Tensor, "B T lm_hidden_dim"]:

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_rms = x * self.weight * torch.rsqrt(var + self.eps)

        return x_rms


class RotaryEmbedding(nn.Module):
    """
    Compute Rotary Embedding to introduce positional dependency to input
    sequence without additional training parameters and relative distance
    of token position ids through angle rotation.

    source:
    https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): Hidden dimension size.
            - lm_n_heads (int): Number of attention heads.
            - lm_re_base (float): Base for rotary embedding frequencies.
            - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
            - lm_attn_scaling (float): Attention scaling factor.
    """

    def __init__(self, cfg):
        super().__init__()
        assert (
            cfg.lm_hidden_dim % cfg.lm_n_heads == 0
        ), "Hidden dimension must be divisible by number of heads"

        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads  # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(
        self, position_ids: Float[Tensor, "B T"]
    ) -> tuple[Float[Tensor, "B T"], Float[Tensor, "B T"]]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)
                containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq

        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()

        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)

        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)

        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`),
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]

    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LM_GroupedQueryAttention(nn.Module):

    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.n_embd = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert (
            self.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        assert self.n_embd % self.n_heads == 0, "n_embd must be divisible by n_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.n_embd // self.n_heads

        self.q_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            in_features=self.n_embd,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            in_features=self.n_embd, out_features=self.n_embd, bias=False
        )

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: Float[Tensor, "B T C"],
        cos: Float[Tensor, "B T head_dim"],
        sin: Float[Tensor, "B T head_dim"],
        attention_mask: Float[Tensor, "B total_lv_len"] | None = None,
        block_kv_cache: dict = None,
    ):

        is_prefill = block_kv_cache is None
        B, T_curr, C = x.size()

        q_curr = (
            self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)
        )  # [B, n_heads, T_curr, head_dim]
        k_curr = (
            self.k_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, n_kv_heads, T_curr, head_dim]
        v_curr = (
            self.v_proj(x)
            .view(B, T_curr, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, n_kv_heads, T_curr, head_dim]

        # apply rotation embedding to q and k
        q, k_rot = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # check if we can use cached keys and values
        if not is_prefill and block_kv_cache["key"] is not None:
            # concatenate with cached K, V
            # k_rot and v_curr are for the new tokens
            k = block_kv_cache["key"]
            v = block_kv_cache["value"]
            k = torch.cat([k, k_rot], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache["key"] = k
            block_kv_cache["value"] = v

        else:
            # no cache, this is the first pass (prefill)
            k = k_rot
            v = v_curr
            block_kv_cache = {"key": k, "value": v}

        # repeat K, V for Grouped Query Attention
        k_expanded = k.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # [B, n_heads, T_kv, head_dim]
        v_expanded = v.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # [B, n_heads, T_kv, head_dim]

        T_kv = k_expanded.size(2)  # total sequence len of keys/values

        # prepare attention mask for SDPA
        # attention mask is [B, T_kv_total_len], 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # the current attention_mask parameter is assumed to be [B, T_kv_total_len]
            # convert to [B, 1, 1, T_kv] for SDPA
            mask_for_keys = attention_mask[
                :, :T_kv
            ]  # ensure mask matches key len [B, T_kv]
            additive_attn_mask = (
                1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()
            ) * torch.finfo(
                q.dtype
            ).min  # [B, 1, 1, T_kv]

        # compute attention using scaled dot product attention
        # during decode, no additional masking needed as [1, T_kv] is naturally causal
        is_causal = T_curr == T_kv and T_curr > 1
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=additive_attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # [B, n_heads, T_kv, head_dim]

        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)  # [B, T, C]
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache


class LM_MLP(nn.Module):
    """
    MLP layer for language model

    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    source:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.n_embd = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(
            in_features=self.n_embd, out_features=self.inter_dim, bias=False
        )
        self.up_proj = nn.Linear(
            in_features=self.n_embd, out_features=self.inter_dim, bias=False
        )
        self.down_proj = nn.Linear(
            in_features=self.inter_dim, out_features=self.n_embd, bias=False
        )

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:

        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x


class LM_Block(nn.Module):
    """
    source: https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222

    """

    def __init__(self, cfg: VLMConfig):

        super().__init__()
        self.mlp = LM_MLP(cfg)
        self.attn = LM_GroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)  # Norm layer for input
        self.norm2 = RMSNorm(cfg)  # Norm layer post attention

    def forward(
        self,
        x: Float[Tensor, "B T n_embd"],
        cos: Float[Tensor, "B T head_dim"],
        sin: Float[Tensor, "B T head_dim"],
        attention_mask: Float[Tensor, "B total_lv_len"] | None = None,
        block_kv_cache: dict | None = None,
    ) -> tuple[Float[Tensor, "B T C"], dict]:
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """

        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache


class LanguageModel(nn.Module):

    def __init__(self, cfg: VLMConfig):

        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embedding = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([LM_Block(cfg) for _ in range(cfg.lm_n_blocks)])
        self.norm = RMSNorm(cfg)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: Float[Tensor, "B T C"] | Int[Tensor, "B T"],
        attention_mask: Float[Tensor, "B total_lv_len"] | None = None,
        kv_cache: dict | None = None,
        start_pos: int = 0,
    ):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """

        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the len of the current input sequnce
        B, T_curr, _ = x.size()

        # create position_ids for the current sequence based on start_pos
        current_pos_ids = (
            torch.arange(start_pos, start_pos + T_curr, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        cos, sin = self.rotary_embedding(
            current_pos_ids
        )  # get rotarty pos embeddings for current tokens

        # init new kv_cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)

        # compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x = self.head(x)  # [B, T, vocab_size]

        return x, kv_cache

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, attention_mask=None, kv_cache=None, start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)

            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1:
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output,
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )
            last_output = decode_step_output[:, -1, :]

        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError

        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)

        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")

        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, "lm_vocab_size"):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(
                    f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})"
                )
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")

        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers

        # Create our model with potentially larger vocabulary
        model = cls(cfg)

        try:
            index_path = hf_hub_download(
                repo_id=cfg.lm_model_type, filename="model.safetensors.index.json"
            )
            with open(index_path, "r") as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index["weight_map"].values())))
            # Download all the sharded files
            safetensors_files = [
                hf_hub_download(repo_id=cfg.lm_model_type, filename=fn)
                for fn in safetensors_filenames
            ]
        except EntryNotFoundError:
            safetensors_files = [
                hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")
            ]

        sd = model.state_dict()

        mapping = {
            "model.embed_tokens.weight": "token_embedding.weight",
            "model.norm.weight": "norm.weight",
        }

        for i in range(cfg.lm_n_blocks):
            layer_prefix = f"model.layers.{i}."
            block_prefix = f"blocks.{i}."

            mapping.update(
                {
                    f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                    f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                    f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                    f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                    f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                    f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                    f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                    f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                    f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight",
                }
            )

        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()

        for safetensors_file in safetensors_files:
            with safetensors.safe_open(
                filename=safetensors_file, framework="pt", device="cpu"
            ) as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue

                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)

                        # Special handling for token embeddings if vocab sizes differ
                        if (
                            hf_key == "model.embed_tokens.weight"
                            and tensor.shape[0] != sd[our_key].shape[0]
                        ):
                            has_extended_embeddings = True
                            print(
                                f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}"
                            )

                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][: tensor.shape[0]].copy_(tensor)

                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(
                                sd[our_key][tensor.shape[0] :], mean=0.0, std=std
                            )

                            print(
                                f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings"
                            )
                            sd["head.weight"].copy_(
                                sd[our_key]
                            )  # Update the head weights as well
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )

                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(
                        f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})"
                    )

        # Load the state dict
        model.load_state_dict(sd)

        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, "head") and "head.weight" in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(
                    filename=safetensors_file, framework="pt", device="cpu"
                ) as f:
                    if "lm_head.weight" in f.keys():
                        lm_head = f.get_tensor("lm_head.weight")
                        if lm_head.shape[0] != sd["head.weight"].shape[0]:
                            print(
                                f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}"
                            )
                            # Copy existing weights
                            sd["head.weight"][: lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(
                                sd["head.weight"][lm_head.shape[0] :], mean=0.0, std=std
                            )
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break

        # Handle weight tying (if needed)
        if (
            cfg.lm_tie_weights
            and hasattr(model, "head")
            and hasattr(model, "token_embedding")
        ):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")

        print(
            f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model
