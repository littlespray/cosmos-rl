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

from typing import Iterable, List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel


class Enscale(nn.Module):
    def __init__(
        self,
        model_dim: int,
        enscale_dim: int,
        dino_model_name: str,
        dino_feature_layers: Iterable[int],
        num_heads: int,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.enscale_dim = enscale_dim
        self.dino_model_name = dino_model_name
        self.dino_feature_layers = list(dino_feature_layers)

        if enscale_dim % num_heads != 0:
            raise ValueError(
                f"enscale_dim({enscale_dim}) must be divisible by num_heads({num_heads})"
            )

        with torch.device("cpu"):
            dino_config = AutoConfig.from_pretrained(dino_model_name)
            dino_config.output_hidden_states = True
            self.dino = AutoModel.from_pretrained(
                dino_model_name, config=dino_config
            )
            self.processor = AutoImageProcessor.from_pretrained(dino_model_name)

        self.dino.requires_grad_(False)
        self.dino.eval()

        dino_hidden_dim = self.dino.config.hidden_size
        self.feature_proj_1 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.feature_proj_2 = nn.Linear(dino_hidden_dim, enscale_dim, bias=False)
        self.query_proj = nn.Linear(model_dim, enscale_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            enscale_dim, num_heads=num_heads, batch_first=True
        )

        ffn_dim = enscale_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(enscale_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, model_dim),
        )

    def _normalize_feature_layers(self, num_layers: int) -> List[int]:
        layers: List[int] = []
        for idx in self.dino_feature_layers:
            normalized = idx + num_layers if idx < 0 else idx
            if 0 <= normalized < num_layers:
                layers.append(normalized)
        if len(layers) < 2:
            layers = [num_layers - 2, num_layers - 1]
        if layers[0] == layers[1]:
            layers = [layers[0], min(layers[0] + 1, num_layers - 1)]
        return layers[:2]

    def forward(
        self, hidden_states: torch.Tensor, images: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if images is None:
            raise ValueError("enscale expects image inputs but got None")

        batch_size = hidden_states.shape[0]
        if not torch.is_tensor(images):
            if isinstance(images, (list, tuple)):
                images_list = list(images)
                if len(images_list) != batch_size:
                    raise ValueError(
                        "enscale expects one image per sample. "
                        f"Got {len(images_list)} images for batch_size={batch_size}."
                    )
                inputs = self.processor(images=images_list, return_tensors="pt")
            else:
                if batch_size != 1:
                    raise ValueError(
                        "enscale expects a list of images matching batch size. "
                        f"Got a single image for batch_size={batch_size}."
                    )
                inputs = self.processor(images=images, return_tensors="pt")
        else:
            if images.ndim != 4:
                raise ValueError(
                    "enscale expects images with shape (batch, channels, height, width); "
                    f"got ndim={images.ndim}, shape={tuple(images.shape)}"
                )

            if images.shape[1] != 3:
                raise ValueError(
                    "enscale expects images with 3 channels in dim=1; "
                    f"got shape={tuple(images.shape)}"
                )

            images = images.detach()
            if not torch.is_floating_point(images):
                images = images.to(dtype=torch.float32)

            image_min = float(images.min().item())
            image_max = float(images.max().item())
            if image_min >= -1.0 and image_max <= 1.0:
                # Qwen/SigLIP-style normalized pixel_values: (x - 0.5) / 0.5 -> [-1, 1]
                images = images.mul(0.5).add(0.5)
            elif image_min >= 0.0 and image_max <= 1.0:
                # Raw [0, 1] is fine.
                pass
            elif image_min >= 0.0 and image_max <= 255.0:
                images = images / 255.0
            else:
                raise ValueError(
                    "enscale expects pixel_values in [-1,1] (Qwen/SigLIP) or raw images "
                    "in [0,1] or [0,255] before DINO preprocessing; "
                    f"got min={image_min:.4f}, max={image_max:.4f}."
                )

            inputs = self.processor(images=images, return_tensors="pt")

        dino_device = next(self.dino.parameters()).device
        dino_dtype = next(self.dino.parameters()).dtype
        inputs = {k: v.to(device=dino_device, dtype=dino_dtype) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.dino(**inputs, output_hidden_states=True)

        hidden_states_list = outputs.hidden_states
        feature_layers = self._normalize_feature_layers(len(hidden_states_list))
        feat_1 = hidden_states_list[feature_layers[0]]
        feat_2 = hidden_states_list[feature_layers[1]]

        kv_1 = self.feature_proj_1(feat_1)
        kv_2 = self.feature_proj_2(feat_2)
        kv = torch.cat([kv_1, kv_2], dim=1)

        query = self.query_proj(hidden_states)
        attn_out, _ = self.cross_attn(query, kv, kv, need_weights=False)
        return self.ffn(attn_out)
