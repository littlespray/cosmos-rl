#!/usr/bin/env python3
"""
Inspect token-length distribution of HuggingFaceH4/llava-instruct-mix-vsft
using the same Qwen3-VL processing logic as Cosmos-RL SFT training.

Key point: for Qwen3-VL, the processor will insert vision patch placeholders
into input_ids (often using pad_token_id), so plain text tokenization is not
accurate for context-length planning.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import toml
from datasets import load_dataset

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.packer.qwen3_vl_data_packer import Qwen3_VL_DataPacker
from cosmos_rl.utils import util


try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _parse_percentiles(s: str) -> List[float]:
    vals: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("percentiles cannot be empty")
    for p in vals:
        if p < 0 or p > 100:
            raise ValueError(f"percentile out of range: {p}")
    return vals


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 0:
        return x
    return ((x + multiple - 1) // multiple) * multiple


def _get_dataset_iter(
    dataset_name: str,
    split: str,
    subset: str = "",
    streaming: bool = False,
    seed: int = 0,
    max_samples: int = 0,
) -> Tuple[Iterable[Dict[str, Any]], Optional[int]]:
    ds_kwargs: Dict[str, Any] = {}
    if subset:
        ds_kwargs["name"] = subset

    ds = load_dataset(dataset_name, split=split, streaming=streaming, **ds_kwargs)

    if streaming:
        # Streaming datasets don't support shuffle/select consistently across versions.
        def _iter() -> Iterable[Dict[str, Any]]:
            n = 0
            for row in ds:
                yield row
                n += 1
                if max_samples and max_samples > 0 and n >= max_samples:
                    break

        return _iter(), None

    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        return ds, int(max_samples)
    return ds, int(len(ds))


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="/tmp/cosmos-rl/config.toml",
        help="Path to Cosmos-RL TOML config (used to pick model_name_or_path and model_max_length).",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/llava-instruct-mix-vsft",
        help="HF dataset name.",
    )
    ap.add_argument("--subset", type=str, default="", help="HF subset/config name.")
    ap.add_argument("--split", type=str, default="train", help="HF split.")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (takes first --max-samples samples).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to scan. Use 0 (default) to scan the full split.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed (non-streaming only).")
    ap.add_argument(
        "--percentiles",
        type=str,
        default="50,90,95,99,100",
        help="Comma-separated percentiles to report.",
    )
    ap.add_argument(
        "--recommend-mode",
        type=str,
        default="p99",
        choices=["p95", "p99"],
        help="Which percentile to base the recommended max context on.",
    )
    ap.add_argument(
        "--round-to",
        type=int,
        default=256,
        help="Round recommended max context up to this multiple.",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write JSON results.",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print a short progress line every N processed samples (0 disables).",
    )
    ap.add_argument(
        "--intermediate-every",
        type=int,
        default=20000,
        help="Write an intermediate JSON snapshot every N processed samples (0 disables).",
    )
    ap.add_argument(
        "--intermediate-json",
        type=str,
        default="",
        help="Path for intermediate JSON snapshots. Defaults to --out-json + '.partial.json' if --out-json is set.",
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config_dict = toml.load(f)
    cosmos_config = CosmosConfig.from_dict(config_dict)

    # Initialize the same packer used for training Qwen3-VL.
    packer = Qwen3_VL_DataPacker()
    util.call_setup(packer, cosmos_config)

    ps = _parse_percentiles(args.percentiles)

    seq_lens: List[int] = []
    vision_pads: List[int] = []
    failed = 0

    rows_iter, total = _get_dataset_iter(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        streaming=args.streaming,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    if tqdm is not None:
        rows_iter = tqdm(
            rows_iter,
            desc="Scanning samples",
            total=total,
        )

    start_t = time.time()
    last_t = start_t

    def _compute_quantiles_from_lists() -> Tuple[Dict[str, int], Dict[str, int]]:
        lens = np.asarray(seq_lens, dtype=np.int64)
        vpads = np.asarray(vision_pads, dtype=np.int64)

        def q(arr: np.ndarray, p: float) -> int:
            return int(np.ceil(np.percentile(arr, p)))

        return ({str(p): q(lens, p) for p in ps}, {str(p): q(vpads, p) for p in ps})

    intermediate_path = args.intermediate_json
    if not intermediate_path and args.out_json:
        intermediate_path = args.out_json + ".partial.json"

    for i, row in enumerate(rows_iter, start=1):
        try:
            out = packer.sft_process_sample(row)
            input_ids = out["input_ids"]
            seq_lens.append(len(input_ids))
            vision_pads.append(
                int(np.sum(np.asarray(input_ids) == packer.tokenizer.pad_token_id))
            )
        except Exception:
            failed += 1

        if args.progress_every and i % args.progress_every == 0:
            now = time.time()
            dt = max(now - last_t, 1e-9)
            total_dt = max(now - start_t, 1e-9)
            rate = args.progress_every / dt
            avg_rate = i / total_dt
            eta_s = None
            if total is not None and avg_rate > 0:
                eta_s = int((total - i) / avg_rate)
            msg = f"[progress] {i}"
            if total is not None:
                msg += f"/{total}"
            msg += f" processed, failed={failed}, rate={rate:.2f} it/s, avg={avg_rate:.2f} it/s"
            if eta_s is not None:
                msg += f", eta~{eta_s}s"
            print(msg, flush=True)
            last_t = now

        if args.intermediate_every and i % args.intermediate_every == 0 and intermediate_path:
            length_q, vpad_q = _compute_quantiles_from_lists()
            _write_json(
                intermediate_path,
                {
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "split": args.split,
                    "streaming": args.streaming,
                    "max_samples": args.max_samples,
                    "processed_so_far": i,
                    "total": total,
                    "failed": failed,
                    "model_name_or_path": cosmos_config.policy.model_name_or_path,
                    "seq_len_percentiles_so_far": length_q,
                    "vision_pad_token_percentiles_so_far": vpad_q,
                },
            )

    if not seq_lens:
        raise RuntimeError(
            f"No samples processed successfully (failed={failed}). "
            "Likely processor/tokenizer mismatch or dataset schema changed."
        )

    lens = np.asarray(seq_lens, dtype=np.int64)
    vpads = np.asarray(vision_pads, dtype=np.int64)

    def q(arr: np.ndarray, p: float) -> int:
        # numpy percentile returns float; we want an int threshold that's safe.
        return int(np.ceil(np.percentile(arr, p)))

    length_q = {str(p): q(lens, p) for p in ps}
    vpad_q = {str(p): q(vpads, p) for p in ps}

    model_max_len = getattr(cosmos_config.policy, "model_max_length", None)
    n_over = int(np.sum(lens > model_max_len)) if model_max_len else 0
    over_rate = float(n_over / lens.size) if model_max_len else 0.0

    target_p = 95.0 if args.recommend_mode == "p95" else 99.0
    rec = _round_up(q(lens, target_p), args.round_to)

    result = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "streaming": args.streaming,
        "max_samples": args.max_samples,
        "processed": int(lens.size),
        "failed": int(failed),
        "model_name_or_path": cosmos_config.policy.model_name_or_path,
        "model_max_length_in_config": int(model_max_len) if model_max_len else None,
        "over_max_length_count": int(n_over) if model_max_len else None,
        "over_max_length_rate": over_rate if model_max_len else None,
        "seq_len_percentiles": length_q,
        "vision_pad_token_percentiles": vpad_q,
        "recommend_mode": args.recommend_mode,
        "recommend_max_context": int(rec),
        "round_to": int(args.round_to),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    _write_json(args.out_json, result)


if __name__ == "__main__":
    main()

