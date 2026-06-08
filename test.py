# %%
from configs.config import TrainConfig, VLMConfig
from data.processors import get_tokenizer, get_image_string

# %%

cfg = VLMConfig()

tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

print("pad_token: ", tokenizer.pad_token)
print("image_token: ", tokenizer.image_token)
print("image_token_id: ", tokenizer.image_token_id)

image_string = get_image_string(
    tokenizer=tokenizer,
    splitted_image_counts=[(1, 1)],
    mp_image_token_length=cfg.mp_image_token_length,
)

image_token_count = image_string.count(tokenizer.image_token)

print("image_string preview: ", image_string[:200])
print("image token count: ", image_token_count)

assert tokenizer.image_token_id is not None
assert image_token_count == cfg.mp_image_token_length

print("Tokenizer/data skeleton check passed")

# %% test CausalSelfAttention
import torch
from models.language_model import CausalSelfAttention

B, T, C = 2, 5, cfg.lm_hidden_dim
n_heads = cfg.lm_n_heads

cfg = VLMConfig()
# cfg.lm_hidden_dim = C
# cfg.lm_n_heads = n_heads
print(cfg.lm_hidden_dim, cfg.lm_n_heads)

attn = CausalSelfAttention(cfg)
print(attn.n_embd)
x = torch.randn(B, T, C, requires_grad=True)
y = attn(x)
print("input shape: ", x.shape)
print("output shape: ", y.shape)
assert y.shape == x.shape

# causal test
x1 = torch.randn(B, T, C, requires_grad=True)
x2 = x1.clone()
x2[:, 3:] = torch.randn(B, T - 3, C)  # change future tokens only

y1 = attn(x1)
y2 = attn(x2)

# position 0, 1, 2 should be identical because they cannot see position 3, 4
assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5), "causal mask failed"

loss = y.sum()
loss.backward()
# %% test ViT
import torch
from models.vision_transformer import ViT
from configs.config import VLMConfig

cfg = VLMConfig()
vit = ViT(cfg)

x = torch.randn(2, 3, cfg.vit_img_size, cfg.vit_img_size)
y = vit(x)  # [B, n_patches or T, hidden_dim or embd_dim]

print("output: ", y.shape)

loss = y.sum()
loss.backward()

# %% sanity check modality projector
import torch
from models.modality_projector import ModalityProjector
from configs.config import VLMConfig

cfg = VLMConfig()
mp = ModalityProjector(cfg)

# calculate n_patches or sequence length T
n_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2
x = torch.randn(2, n_patches, cfg.vit_hidden_dim)
y = mp(x)

T_mp = int(n_patches // (mp.scale_factor**2))
output_shape_gt = (2, T_mp, cfg.lm_hidden_dim)
print("output shape: ", y.shape)
print("expected output shape: ", output_shape_gt)
assert y.shape == output_shape_gt, "MP sanity check passed"
