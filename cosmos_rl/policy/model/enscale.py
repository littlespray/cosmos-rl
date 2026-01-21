# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Iterable, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from cosmos_rl.policy.kernel.modeling_utils import FlashAttnMeta


class EnscaleEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, scale_idx: Iterable[int]) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.scale_idx = list(scale_idx)

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def extract(self, images: Any) -> list[torch.Tensor]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.encoder.device)
        hidden_states = self.encoder(**inputs).hidden_states
        return [hidden_states[i] for i in self.scale_idx]


class EnscaleHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        enscale_dim: int,
        num_heads: int,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        self.q_norm = nn.RMSNorm(model_dim)
        self.k_norm = nn.RMSNorm(model_dim)
        self.k_proj = nn.Linear(enscale_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(enscale_dim, model_dim, bias=False)
        self.o_proj = nn.Linear(model_dim, model_dim, bias=False)
        assert model_dim % num_heads == 0, (
            f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.attn_func = FlashAttnMeta().flash_attn_func
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * ffn_multiplier),
            nn.GELU(),
            nn.Linear(model_dim * ffn_multiplier, model_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, hidden_states: torch.Tensor, scale_embeddings: torch.Tensor
    ) -> torch.Tensor:
        scale_embeddings = scale_embeddings.to(device=hidden_states.device, dtype=hidden_states.dtype)

        xq = self.q_norm(hidden_states)
        xk = self.k_norm(self.k_proj(scale_embeddings))
        xv = self.v_proj(scale_embeddings)

        b, qlen, _ = xq.shape
        klen = xk.shape[1]
        xq = xq.view(b, qlen, self.num_heads, self.head_dim)
        xk = xk.view(b, klen, self.num_heads, self.head_dim)
        xv = xv.view(b, klen, self.num_heads, self.head_dim)

        input_dtype = xq.dtype
        # flash_attn only support bfloat16/float16
        # Cast qkv to torch.bfloat16 if dtype for forward is torch.float32
        if input_dtype == torch.float32:
            xq = xq.to(torch.bfloat16)
            xk = xk.to(torch.bfloat16)
            xv = xv.to(torch.bfloat16)

        attn_out = self.attn_func(xq, xk, xv, causal=False).reshape(b, qlen, -1).to(input_dtype)
        attn_out = self.o_proj(attn_out)
        out = self.gate * self.ffn(attn_out)

        return out
