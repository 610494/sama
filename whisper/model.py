import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .resnet import ResEncoder
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class GatedXAttnSubBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)   # cross-attn
        self.attn_ln = LayerNorm(n_state)                # layernorm for cross-attn
        self.attn_gate = nn.Parameter(torch.tensor([0.]))    # scalar gate

    def forward(self, x: Tensor, xt: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Cross-attn
        x_ln = self.attn_ln(x)
        
        # 將 mask 傳入 self.attn
        # 當 xt 為 None (Baseline) 且 mask 有值時，這會變成 Causal Self-Attention
        attn_out, _ = self.attn(x_ln, xt, mask=mask)  

        # 使用一個 scalar gate 來控制 cross-attn 的影響力
        x_i = attn_out * self.attn_gate.tanh()

        return x_i

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, n_state: int, n_head: int, cross_attention: bool = False, 
            add_gated_x_attn: int = 0, num_langs: int = 0,
        ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (MultiHeadAttention(n_state, n_head) if cross_attention else None)
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        
        # --- Teacher (TG-ASR) 模組 ---
        self.add_gated_x_attn = add_gated_x_attn
        self.num_langs = num_langs
        if self.add_gated_x_attn != 0:
            print("add gated x attn layer")
            self.gated_x_attn_layers = nn.ModuleList()
            for i in range(num_langs):
                subblock = GatedXAttnSubBlock(n_state, n_head)
                self.gated_x_attn_layers.append(subblock)

            self.ff_ln = LayerNorm(n_state)
            self.ff = nn.Sequential(
                Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
            )
            self.ff_gate = nn.Parameter(torch.tensor([0.])) 

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        xt_list: Optional[List[Tensor]] = None,
    ):
        # 1. Teacher Block (TG-ASR)
        if self.add_gated_x_attn != 0: 
            # 把 mask 傳進去
            x, gated_feats = self.apply_gated_x_attn_multi(x, xt_list, mask=mask)

        # 2. Self-Attention Block
        attn_residual = x
        x = attn_residual + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]

        # 3. Cross-Attention Block
        if self.cross_attn:
            cross_attn_residual = x
            x = cross_attn_residual + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        
        # 4. MLP Block
        mlp_residual = x
        x = mlp_residual + self.mlp(self.mlp_ln(x))

        return x

    def apply_gated_x_attn_multi(self, x: Tensor, xt_list: List[Tensor], mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        if len(xt_list) > self.num_langs:
            raise ValueError(
                f"Got {len(xt_list)} translations but only support up to {self.num_langs}"
            )

        x_origin = x
        total_delta = 0

        # print(f'len(xt_list): {len(xt_list)}, self.num_langs: {self.num_langs}')
        for i, xt in enumerate(xt_list):
            # [關鍵邏輯]
            # 如果 xt 是 None (Baseline模式)，這層是 Self-Attn，必須給 Causal Mask 防止偷看答案。
            # 如果 xt 有值 (Proposed模式)，這層是 Cross-Attn，不應該給 Causal Mask (因為 BERT text 是全見的)。
            current_mask = mask if xt is None else None
            
            # 傳入 current_mask
            delta_i = self.gated_x_attn_layers[i](x_origin, xt, mask=current_mask)
            total_delta += delta_i

        x_after = x_origin + total_delta
        x_after = x_after + self.ff(self.ff_ln(x_after)) * self.ff_gate.tanh()
        
        return x_after, total_delta

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, False) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)      

    def forward(self, x: Tensor, track_norm=False, padding_mask=None):  
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        if track_norm:
            x_norm = torch.linalg.norm(x, dim=-1).mean()

        # NOTE: pos embedding has max length of 1500 (30s after conv downsample from 3000 mel frames)
        if x.shape[1] > 1500:
            x = x[ :, :1500, :]
    
        # NOTE: if max_len is 30s, then the cropping doesn't do anything.
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype) # trim pos embedding
        
        for layer, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_post(x)

        if track_norm:
            return x, x_norm
        return x

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout_rate: float,
        add_gated_x_attn: int, bert_hidden_size: int, num_langs: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, 
                                        cross_attention=True, 
                                        add_gated_x_attn=add_gated_x_attn,
                                        num_langs=num_langs,
                                        )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(dropout_rate)     

        # 儲存 bert_dim 以便在 forward 中做安全檢查
        self.bert_hidden_size = bert_hidden_size

        # 1. Text Projection
        if bert_hidden_size != n_state:
            self.text_projection = nn.Linear(bert_hidden_size, n_state)
        else:
            self.text_projection = nn.Identity()

        # 2. Audio Projection
        self.audio_projection = nn.Linear(n_state, n_state)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, 
                xt_list: Optional[List[Tensor]] = None, return_hidden: bool = False
        ):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0

        # Decoder Input (x) 的 Positional Embedding
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        # --- [處理 xt_list 優化版 (v3)] ---
        processed_xt_list = []
        if xt_list is not None:
            for xt in xt_list:
                if xt is None:
                    processed_xt_list.append(None)
                    continue

                # 1. 如果維度等於 BERT 設定的維度 (768)
                if xt.shape[-1] == self.bert_hidden_size:
                    # Case A: Text / Pseudo Text
                    xt = self.text_projection(xt)

                    # 只有當長度在 Decoder 支援的範圍內時，才加 Positional Embedding
                    if xt.shape[1] <= self.positional_embedding.shape[0]:
                        xt = xt + self.positional_embedding[: xt.shape[1]]

                    else:
                        pass 

                # 2. 如果維度等於 Whisper Model 維度 (1024)
                elif xt.shape[-1] == self.ln.normalized_shape[0]: 
                    # Case B: Audio
                    xt = self.audio_projection(xt)
                    # Audio 本來就不加 PE

                else:
                    raise ValueError(f"Unknown input dimension {xt.shape[-1]}.")

                xt = xt.to(xa.dtype)
                processed_xt_list.append(xt)
        # --- [結束] ---
        
        hidden_states = []

        for block in self.blocks:
            x = block(
                x, xa, mask=self.mask, kv_cache=kv_cache, 
                xt_list=processed_xt_list, 
            )

            if return_hidden:
                hidden_states.append(x)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        if return_hidden:
            return hidden_states, x, logits
        else:
            return logits

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout_rate: float,
                add_gated_x_attn: int, bert_dim: int, num_langs: int,
                ):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout_rate,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dropout_rate,
            add_gated_x_attn,
            bert_dim,
            num_langs,
        )


    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function