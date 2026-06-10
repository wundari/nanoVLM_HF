# %%
import torch
import torch.nn as nn
from torch import Tensor

from torch.nn.attention import sdpa_kernel, SDPBackend

from configs.config import VLMConfig
from jaxtyping import Float


# %%
class ViTPatchEmbeddings(nn.Module):

    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.embd_dim = cfg.vit_hidden_dim
        self.patch_size = cfg.vit_patch_size
        self.cls_flag = cfg.vit_cls_flag
        self.n_patches = int((self.img_size // self.patch_size) ** 2)

        # conv layer for converting img to tokens
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))

            # position encoder
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.n_patches + 1, self.embd_dim)
            )
        else:
            # position encoder
            self.position_embedding = nn.Parameter(
                torch.rand(1, self.n_patches, self.embd_dim)
            )

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B n_patches hidden_dim"]:

        x = self.conv(x)  # B, hidden_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, hidden_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, hidden_dim]

        # add cls token and position embedding
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)  # [B, n_patches + 1, hidden_dim]
        x = x + self.position_embedding  # [B, n_patches + 1, hidden_dim]

        return x


class ViTMultiHeadAttention(nn.Module):

    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.embd_dim = cfg.vit_hidden_dim
        self.n_heads = cfg.vit_n_heads
        self.dropout = cfg.vit_dropout

        assert (
            self.embd_dim % self.n_heads == 0
        ), "embd_dim must be divisible by n_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.qkv_proj = nn.Linear(
            in_features=self.embd_dim,
            out_features=3 * self.n_heads * self.head_dim,
            bias=True,
        )

        self.out_proj = nn.Linear(
            in_features=self.embd_dim, out_features=self.embd_dim, bias=True
        )

        # dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self, x: Float[Tensor, "B T hidden_dim"]
    ) -> Float[Tensor, "B T hidden_dim"]:

        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # [B, T, 3*embd_dim]
        q, k, v = qkv.split(C, dim=2)  # each [B, T, embd_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # [B, n_heads, T, head_dim]

        # compute attention scores
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # ViT attention is bidirectional
            )  # [B, n_heads, T, head_dim]

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class ViTMLP(nn.Module):

    def __init__(self, cfg: VLMConfig):

        super().__init__()

        self.emdb_dim = cfg.vit_hidden_dim
        self.inter_dim = cfg.vit_inter_dim

        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(in_features=self.emdb_dim, out_features=self.inter_dim)
        self.fc2 = nn.Linear(in_features=self.inter_dim, out_features=self.emdb_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(
        self, x: Float[Tensor, "B T hidden_dim"]
    ) -> Float[Tensor, "B T hidden_dim"]:

        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class ViTBlock(nn.Module):

    def __init__(self, cfg: VLMConfig):

        super().__init__()

        self.embd_dim = cfg.vit_hidden_dim
        self.eps = cfg.vit_ln_eps

        self.attn = ViTMultiHeadAttention(cfg)
        self.mlp = ViTMLP(cfg)
        self.ln1 = nn.LayerNorm(normalized_shape=self.embd_dim, eps=self.eps)
        self.ln2 = nn.LayerNorm(normalized_shape=self.embd_dim, eps=self.eps)

    def forward(
        self, x: Float[Tensor, "B T hidden_dim"]
    ) -> Float[Tensor, "B T hidden_dim"]:

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class ViT(nn.Module):

    def __init__(self, cfg: VLMConfig):

        super().__init__()

        self.cfg = cfg
        self.embd_dim = cfg.vit_hidden_dim
        self.eps = cfg.vit_ln_eps
        self.cls_flag = cfg.vit_cls_flag

        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embd_dim, eps=self.eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: Float[Tensor, "B C H W"]):

        # convert image to tokens
        x = self.patch_embedding(x)  # [B, T, hidden_dim]

        x = self.dropout(x)  # [B, T, hidden_dim]

        # attention blocks
        for block in self.blocks:
            x = block(x)  # [B, T, hidden_dim]

        if self.cls_flag:
            x = self.layer_norm(
                x[:, 0]
            )  # if there is a class token the model only needs to process CLS token
        else:
            x = self.layer_norm(x)

        return x

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Vision Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import SiglipVisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout = hf_config.attention_dropout
        cfg.vit_hidden_dim = hf_config.hidden_size
        cfg.vit_img_size = hf_config.image_size
        cfg.vit_inter_dim = hf_config.intermediate_size
        cfg.vit_ln_eps = hf_config.layer_norm_eps
        cfg.vit_n_heads = hf_config.num_attention_heads
        cfg.vit_n_blocks = hf_config.num_hidden_layers
        cfg.vit_patch_size = hf_config.patch_size
        model = cls(cfg)
        safetensors_file = hf_hub_download(
            repo_id=cfg.vit_model_type, filename="model.safetensors"
        )

        sd = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(cfg.vit_n_blocks):
            # Layer norms
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = (
                f"blocks.{i}.ln1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = (
                f"blocks.{i}.ln1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = (
                f"blocks.{i}.ln2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = (
                f"blocks.{i}.ln2.bias"
            )

            # MLP
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (
                f"blocks.{i}.mlp.fc1.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (
                f"blocks.{i}.mlp.fc1.bias"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (
                f"blocks.{i}.mlp.fc2.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (
                f"blocks.{i}.mlp.fc2.bias"
            )

            # Output projection
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                f"blocks.{i}.attn.out_proj.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                f"blocks.{i}.attn.out_proj.bias"
            )

        with safetensors.safe_open(
            filename=safetensors_file, framework="pt", device="cpu"
        ) as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if "position_embedding" in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.cfg.vit_n_blocks):
                q_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
                )
                k_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
                )
                v_weight = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
                )

                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                q_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
                )
                k_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
                )
                v_bias = f.get_tensor(
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
                )

                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)

        model.load_state_dict(sd)
        print(
            f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model


# %%
