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

"""
BabyVision visual reasoning dataset for GRPO/DAPO training.

BabyVision is a visual reasoning benchmark with tasks like fine-grained
discrimination, visual tracking, spatial perception, and pattern recognition.
Each sample contains an image and a question; the model should respond with
chain-of-thought reasoning and a final answer in \\boxed{Answer} format.

Usage:
    cosmos-rl --config configs/qwen3/qwen3-8b-babyvision-grpo.toml \\
        cosmos_rl/tools/dataset/babyvision_grpo.py
"""

import re
import base64
import io
from typing import Any, List, Dict, Optional, Union

from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from PIL import Image as PILImage

from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer import HFVLMDataPacker
from cosmos_rl.dispatcher.algo.reward import boxed_math_reward_fn
from cosmos_rl.utils.logging import logger


SYSTEM_PROMPT = (
    "You are a visual reasoning expert. Carefully observe the image and "
    "answer the question. Think step by step and provide your final answer "
    "in \\boxed{Answer} format."
)

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{.*\})*\})*)\}")


def format_choices(choices: list) -> str:
    """Format multiple choice options as (A), (B), (C), etc."""
    return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


def babyvision_reward_fn(
    to_be_evaluated: str, reference: Optional[str] = None, **kwargs
) -> float:
    if not reference:
        return 0.0

    try:
        score = boxed_math_reward_fn(to_be_evaluated, reference, **kwargs)
        if score > 0:
            return score
    except Exception:
        pass

    matches = BOXED_RE.findall(to_be_evaluated or "")
    if not matches:
        return 0.0
    model_answer = matches[-1].strip().lower()
    ref_norm = str(reference).strip().lower()

    if model_answer == ref_norm:
        return 1.0
    if ref_norm in model_answer or model_answer in ref_norm:
        return 1.0

    return 0.0


def _encode_image(img) -> Optional[str]:
    """Encode a PIL Image to base64 string."""
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    if isinstance(img, str):
        return img
    return None


class BabyvisionDataset(Dataset):
    """BabyVision visual reasoning dataset for GRPO/DAPO training."""

    def setup(self, config: CosmosConfig, *args, **kwargs):
        self.config = config
        grpo_config = config.train.train_policy

        self.dataset = load_dataset(
            grpo_config.dataset.name,
            grpo_config.dataset.subset or None,
        )
        if grpo_config.dataset.split:
            if isinstance(grpo_config.dataset.split, list):
                self.dataset = ConcatDataset(
                    [self.dataset[s] for s in grpo_config.dataset.split]
                )
            else:
                self.dataset = self.dataset[grpo_config.dataset.split]

        logger.info(f"[BabyvisionDataset] Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        item = self.dataset[idx]
        question = item["question"]

        ans_type = item.get("ansType", "blank")
        if ans_type == "choice" and item.get("options"):
            question += "\nChoices:\n" + format_choices(item["options"])

        question += (
            "\nThink about the question and give your final answer "
            "in \\boxed{Answer} format."
        )

        user_conv: List[Dict[str, Any]] = [{"type": "text", "text": question}]

        if "image" in item and item["image"] is not None:
            img_b64 = _encode_image(item["image"])
            if img_b64:
                user_conv.insert(0, {"type": "image", "image": img_b64})

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_conv},
        ]

    def get_reference_answer(self, idx: int) -> str:
        item = self.dataset[idx]
        ans_type = item.get("ansType", "blank")

        if ans_type == "choice" and item.get("choiceAns") is not None:
            try:
                return chr(65 + int(item["choiceAns"]))
            except (ValueError, TypeError):
                pass

        if "blankAns" in item and item["blankAns"]:
            return str(item["blankAns"])

        return str(item.get("answer", ""))


class BabyvisionValDataset(BabyvisionDataset):
    """BabyVision validation dataset. Loads from [validation] config section."""

    def setup(self, config: CosmosConfig, *args, **kwargs):
        if not config.validation.enable:
            logger.warning(
                "Validation not enabled, skipping BabyvisionValDataset setup."
            )
            return

        self.config = config
        self.dataset = load_dataset(
            config.validation.dataset.name,
            config.validation.dataset.subset or None,
        )
        if config.validation.dataset.split:
            if isinstance(config.validation.dataset.split, list):
                self.dataset = ConcatDataset(
                    [self.dataset[s] for s in config.validation.dataset.split]
                )
            else:
                self.dataset = self.dataset[config.validation.dataset.split]

        logger.info(f"[BabyvisionValDataset] Loaded {len(self.dataset)} samples")


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return BabyvisionDataset()

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        if not config.validation.enable:
            return None
        return BabyvisionValDataset()

    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
        reward_fns=[babyvision_reward_fn],
        filter_reward_fns=[babyvision_reward_fn],
        data_packer=HFVLMDataPacker(),
        val_data_packer=HFVLMDataPacker(),
    )
