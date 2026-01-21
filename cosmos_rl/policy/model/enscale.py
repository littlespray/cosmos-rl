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
        self.feature_proj_1 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.feature_proj_2 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.query_proj = nn.Linear(model_dim, enscale_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            enscale_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(enscale_dim, enscale_dim * ffn_multiplier),
            nn.GELU(),
            nn.Linear(enscale_dim * ffn_multiplier, model_dim),
        )

    def forward(
        self, hidden_states: torch.Tensor, feat_1: torch.Tensor, feat_2: torch.Tensor
    ) -> torch.Tensor:
        feat_1 = feat_1.to(device=hidden_states.device, dtype=hidden_states.dtype)
        feat_2 = feat_2.to(device=hidden_states.device, dtype=hidden_states.dtype)
        kv = torch.cat(
            [self.feature_proj_1(feat_1), self.feature_proj_2(feat_2)], dim=1
        )
        query = self.query_proj(hidden_states)
        attn_out, _ = self.cross_attn(query, kv, kv, need_weights=False)
        return self.ffn(attn_out)
