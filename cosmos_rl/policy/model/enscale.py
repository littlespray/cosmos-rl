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
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from cosmos_rl.policy.kernel.modeling_utils import FlashAttnMeta


class EnscaleDinoEncoder(nn.Module):
    def __init__(self, dino_model_name: str, dino_feature_layers: Iterable[int]) -> None:
        super().__init__()
        self.dino_model_name = dino_model_name
        self.dino_feature_layers = list(dino_feature_layers)

        with torch.device("cpu"):
            dino_config = AutoConfig.from_pretrained(dino_model_name)
            dino_config.output_hidden_states = True
            self.dino = AutoModel.from_pretrained(dino_model_name, config=dino_config)
            self.processor = AutoImageProcessor.from_pretrained(dino_model_name)

        self.dino.requires_grad_(False)
        self.dino.eval()

    @torch.no_grad()
    def extract(self, images: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(images=images, return_tensors="pt")
        dino_device = next(self.dino.parameters()).device
        dino_dtype = next(self.dino.parameters()).dtype
        inputs = {k: v.to(device=dino_device, dtype=dino_dtype) for k, v in inputs.items()}
        outputs = self.dino(**inputs)
        hs = outputs.hidden_states
        i1, i2 = self.dino_feature_layers
        n = len(hs)
        if i1 < 0:
            i1 = n + i1
        if i2 < 0:
            i2 = n + i2
        return hs[i1], hs[i2]


class EnscaleHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        enscale_dim: int,
        dino_hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.feature_proj_1 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.feature_proj_2 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.q_proj = nn.Linear(model_dim, enscale_dim, bias=False)
        self.k_proj = nn.Linear(enscale_dim, enscale_dim, bias=False)
        self.v_proj = nn.Linear(enscale_dim, enscale_dim, bias=False)
        self.o_proj = nn.Linear(enscale_dim, enscale_dim, bias=False)
        assert enscale_dim % num_heads == 0, (
            f"enscale_dim ({enscale_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = enscale_dim // num_heads
        self.attn_func = FlashAttnMeta().flash_attn_func
        self.ffn = nn.Sequential(
            nn.Linear(enscale_dim, enscale_dim * ffn_multiplier),
            nn.GELU(),
            nn.Linear(enscale_dim * ffn_multiplier, model_dim),
        )
        # Learnable gate (init=0) for stability: start as no-op, learn contribution.
        # Note: FSDP fully_shard does not support 0-dim scalar params, so use (1,).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, hidden_states: torch.Tensor, feat_1: torch.Tensor, feat_2: torch.Tensor
    ) -> torch.Tensor:
        # Handle DTensor: work on local tensor to avoid TP layout issues
        is_dtensor = hasattr(hidden_states, "device_mesh")
        if is_dtensor:
            _device_mesh = hidden_states.device_mesh
            _placements = hidden_states.placements
            hidden_states = hidden_states.to_local()

        feat_1 = feat_1.to(device=hidden_states.device, dtype=hidden_states.dtype)
        feat_2 = feat_2.to(device=hidden_states.device, dtype=hidden_states.dtype)
        kv = torch.cat(
            [self.feature_proj_1(feat_1), self.feature_proj_2(feat_2)], dim=1
        )
        xq = self.q_proj(self.q_norm(hidden_states))
        xk = self.k_proj(kv)
        xv = self.v_proj(kv)

        b, qlen, _ = xq.shape
        klen = xk.shape[1]
        xq = xq.view(b, qlen, self.num_heads, self.head_dim)
        xk = xk.view(b, klen, self.num_heads, self.head_dim)
        xv = xv.view(b, klen, self.num_heads, self.head_dim)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            xq = xq.to(torch.bfloat16)
            xk = xk.to(torch.bfloat16)
            xv = xv.to(torch.bfloat16)

        attn_out = self.attn_func(xq, xk, xv, causal=False).reshape(b, qlen, -1)
        attn_out = self.o_proj(attn_out.to(input_dtype))
        out = self.gate * self.ffn(attn_out)

        # Wrap back to DTensor with same placement
        if is_dtensor:
            out = torch.distributed.tensor.DTensor.from_local(out, _device_mesh, _placements)
        return out
