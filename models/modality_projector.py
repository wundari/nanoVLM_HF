# %%
import torch
import torch.nn as nn
from torch import Tensor
from configs.config import VLMConfig

from jaxtyping import Float


# %%
class ModalityProjector(nn.Module):

    def __init__(self, cfg: VLMConfig):
        super().__init__()

        self.cfg = cfg
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        self.input_dim = cfg.vit_hidden_dim * (self.scale_factor**2)
        self.output_dim = cfg.lm_hidden_dim

        self.proj = nn.Linear(
            in_features=self.input_dim, out_features=self.output_dim, bias=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(
        self, x: Float[Tensor, "B n_patches vit_embd_dim"]
    ) -> Float[Tensor, "B subPatch_x_subPatch vit_embd_dim_expanded"]:
        """
        input: x: output from ViT module
            shape: [B, n_patches, vit_embd_dim]
                n_patches = T (sequence length)

        Returns: x [B, subPatch * subPatch, vit_embd_dim_expanded]
            subPatch = (n_patches ** 0.5) / scale_factor
            vit_embd_dim_expanded: scale_factor**2 * vit_embd_dim
        """

        B, T, embd_dim = x.size()
        seq_root = int(T**0.5)  # 32
        assert (
            seq_root**2 == T
        ), "Sequence len must be a perfect square for pixel shuffle"
        assert (
            seq_root % self.scale_factor == 0
        ), f"The square root of seq_len (T) must be divisible by MP scale factor: {self.scale_factor}"

        h = w = seq_root  # 32
        x = x.view(B, h, w, embd_dim)
        h_out = h // self.scale_factor  # 8
        w_out = w // self.scale_factor  # 8

        x = x.reshape(
            B, h_out, self.scale_factor, w_out, self.scale_factor, embd_dim
        )  # [B, h_out, scale_factor, w_out, scale_factor, vit_embd_dim]
        x = x.permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()  # [B, h_out, w_out, scale_factor, scale_factor, vit_embd_dim]
        x = x.reshape(
            B, h_out * w_out, (self.scale_factor**2) * embd_dim
        )  # [B, h_out * w_out, scale_factor**2 * vit_embd_dim]

        return x

    def forward(
        self, x: Float[Tensor, "B n_patches vit_embd_dim"]
    ) -> Float[Tensor, "B subPatch_x_subPatch lm_embd_dim"]:

        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x
