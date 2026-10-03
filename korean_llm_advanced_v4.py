import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset as HFDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
import random
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, replace
import requests
import logging
import json
from pathlib import Path
import hashlib
import time
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import threading
import queue
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Frame, Label, StringVar
import copy
import traceback
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

# ==========================================
# 로깅 설정
# ==========================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

LOSS_HISTORY_FILE = LOG_DIR / "loss_history.json"
loss_history = []

def save_loss_history():
    try:
        with open(LOSS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Loss history save failed: {e}")

def load_loss_history():
    global loss_history
    if LOSS_HISTORY_FILE.exists():
        try:
            with open(LOSS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                loss_history = json.load(f)
            logger.info(f"✅ Loaded {len(loss_history)} previous loss records")
        except Exception as e:
            logger.warning(f"Failed to load loss history: {e}")
            loss_history = []

# ==========================================
# 데이터셋 기본 설정
# ==========================================
DATASETS_DIR = Path("./datasets")
DATASETS_CACHE_DIR = DATASETS_DIR / "cache"
DATASETS_MANIFEST_FILE = DATASETS_DIR / "datasets_manifest.json"

def ensure_datasets_dir():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Datasets directory ready: {DATASETS_DIR.absolute()}")

# ==========================================
# 데이터셋 다운로드 및 관리
# ==========================================
class DatasetManager:
    DEFAULT_SFT_DATASETS = [
        {
            "name": "nlpai-lab/kullm-v2",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        },
        {
            "name": "beomi/KoAlpaca-v1.1a",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "input", "output"]
        }
    ]

    def __init__(self, dataset_configs: Optional[List[Dict]] = None, cache_dir: Path = DATASETS_CACHE_DIR):
        self.cache_dir = cache_dir
        self.dataset_configs = dataset_configs or self.DEFAULT_SFT_DATASETS
        self.manifest = self._load_manifest()
        ensure_datasets_dir()

    def _load_manifest(self) -> Dict:
        if DATASETS_MANIFEST_FILE.exists():
            with open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(DATASETS_MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _get_dataset_hash(self, config: Dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def download_dataset(self, config: Dict, force: bool = False) -> Optional[str]:
        dataset_hash = self._get_dataset_hash(config)
        dataset_name = config["name"]

        if dataset_hash in self.manifest and not force:
            cached_path = self.manifest[dataset_hash].get("path")
            if cached_path and Path(cached_path).exists():
                logger.info(f"✅ Using cached dataset: {dataset_name}")
                return cached_path

        logger.info(f"📥 Downloading {dataset_name}...")

        last_error = None
        error_details = []

        if config.get("streaming"):
            # Explicit streaming is for large corpora: cache only the requested
            # finite shard instead of downloading the whole source dataset first.
            strategies = [{"name": "streaming", "streaming": True, "force_redownload": False}]
        else:
            strategies = [
                {"name": "standard", "streaming": False, "force_redownload": False},
                {"name": "streaming", "streaming": True, "force_redownload": False},
                {"name": "force_redownload", "streaming": False, "force_redownload": True},
            ]

        for attempt, strategy in enumerate(strategies, 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/{len(strategies)} - Strategy: {strategy['name']}")

                load_kwargs = {
                    "path": config["name"],
                    "split": config["split"],
                    "cache_dir": str(self.cache_dir),
                }

                if config.get("config"):
                    load_kwargs["name"] = config["config"]

                if strategy["streaming"]:
                    load_kwargs["streaming"] = True

                if strategy["force_redownload"]:
                    load_kwargs["download_mode"] = "force_redownload"

                ds = load_dataset(**load_kwargs)

                if strategy["streaming"]:
                    logger.info("📥 Converting streaming dataset to regular dataset...")
                    # Never materialize an unbounded streaming iterator in memory.
                    max_examples = config.get("max_examples")
                    if max_examples is None:
                        raise ValueError("Streaming fallback requires max_examples")
                    ds = HFDataset.from_list(list(itertools.islice(ds, max_examples)))

                if config.get("max_examples") and not strategy["streaming"]:
                    ds = ds.select(range(min(len(ds), config["max_examples"])))

                local_path = self.cache_dir / f"{dataset_hash}"
                local_path.mkdir(exist_ok=True, parents=True)

                ds.to_parquet(str(local_path / "data.parquet"))

                self.manifest[dataset_hash] = {
                    "name": dataset_name,
                    "config": config,
                    "path": str(local_path),
                    "num_examples": len(ds),
                    "download_strategy": strategy["name"]
                }
                self._save_manifest()

                logger.info(f"✅ Dataset saved: {local_path} ({len(ds)} examples) via {strategy['name']}")
                return str(local_path)

            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()

                error_details.append({
                    "attempt": attempt,
                    "strategy": strategy["name"],
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "traceback": tb_str
                })

                logger.warning(f"⚠️ Attempt {attempt} failed ({strategy['name']})")
                logger.warning(f"   ↳ Error Type : {error_type}")
                logger.warning(f"   ↳ Error Msg  : {error_msg}")

                if attempt < len(strategies):
                    wait_time = attempt * 2
                    logger.info(f"⏳ Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)

        logger.error("=" * 80)
        logger.error(f"❌ Failed to download {dataset_name} after {len(strategies)} attempts")
        logger.error("=" * 80)

        for detail in error_details:
            logger.error(f"[Attempt {detail['attempt']}] Strategy: {detail['strategy']}")
            logger.error(f"  - Type   : {detail['error_type']}")
            logger.error(f"  - Message: {detail['error_msg']}")
            logger.error(f"  - Traceback:\n{detail['traceback']}")
            logger.error("-" * 60)

        logger.error(f"📌 Last error summary: {type(last_error).__name__}: {last_error}")
        logger.error("=" * 80)
        return None

    def get_or_download_all(self, force: bool = False) -> List[str]:
        paths = []
        failed = []

        for config in self.dataset_configs:
            path = self.download_dataset(config, force=force)
            if path:
                paths.append(path)
            else:
                failed.append(config["name"])

        logger.info(f"✅ Ready with {len(paths)} datasets")

        if failed:
            logger.warning("=" * 60)
            logger.warning(f"⚠️ 다음 데이터셋 다운로드 실패 ({len(failed)}개):")
            for name in failed:
                logger.warning(f"   - {name}")
            logger.warning("=" * 60)

        return paths

# ==========================================
# 로컬 데이터셋 클래스
# ==========================================
class LocalKoreanDataset(Dataset):
    def __init__(self, dataset_paths: List[str], tokenizer, max_len: int = 256,
                 data_samples_per_dataset: Optional[int] = None,
                 train_on_response_only: bool = True,
                 cache_dir: Path = DATASETS_CACHE_DIR):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_on_response_only = train_on_response_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = []
        self.offsets = []
        self.total_samples = 0

        logger.info("📚 Loading local datasets (RAM-efficient Parquet mode)...")
        logger.info(f"📂 Dataset cache directory: {self.cache_dir.absolute()}")

        for dataset_path in dataset_paths:
            try:
                parquet_file = Path(dataset_path) / "data.parquet"
                if not parquet_file.exists():
                    logger.warning(f"⚠️ Parquet file not found: {parquet_file}")
                    continue

                logger.info(f"📖 Opening parquet directly: {parquet_file}")

                # 여기서는 메타데이터 확인용으로만 열고,
                # ParquetFile 객체 자체는 self.datasets에 저장하지 않음
                pf = pq.ParquetFile(str(parquet_file))
                num_rows = pf.metadata.num_rows
                num_row_groups = pf.num_row_groups

                if data_samples_per_dataset is not None:
                    num_rows = min(num_rows, data_samples_per_dataset)

                if num_rows <= 0:
                    logger.warning(f"⚠️ Dataset is empty: {parquet_file}")
                    continue

                row_groups = []
                accumulated = 0

                for group_idx in range(num_row_groups):
                    group_rows = pf.metadata.row_group(group_idx).num_rows

                    if accumulated >= num_rows:
                        break

                    rows_to_use = min(group_rows, num_rows - accumulated)
                    row_groups.append((group_idx, accumulated, rows_to_use))
                    accumulated += rows_to_use

                # ★ pickle 가능한 값만 저장
                dataset_info = {
                    "path": str(parquet_file),
                    "num_rows": num_rows,
                    "row_groups": row_groups
                }

                self.offsets.append(self.total_samples)
                self.datasets.append(dataset_info)
                self.total_samples += num_rows

                logger.info(
                    f"✅ Registered {num_rows:,} samples from {Path(dataset_path).name}"
                )
                logger.info(f"   Row groups: {num_row_groups:,}")

            except Exception as e:
                logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")

        logger.info(f"✅ Total samples available: {self.total_samples:,}")

    def _clean_text(self, value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _build_instruction_sample(
        self,
        instruction: str,
        input_text: str,
        output: str
    ) -> str:
        instruction, input_text, output = map(
            self._clean_text,
            (instruction, input_text, output)
        )

        parts = []
        if instruction:
            parts.append(f"### 질문: {instruction}")
        if input_text:
            parts.append(f"### 입력: {input_text}")

        parts.append(f"### 응답: {output}")
        return "\n".join(parts)

    def _extract_text(self, item: Dict) -> Optional[str]:
        if item.get("text"):
            text = self._clean_text(item["text"])

        elif item.get("content"):
            text = self._clean_text(item["content"])

        elif item.get("document"):
            text = self._clean_text(item["document"])

        elif item.get("body"):
            text = self._clean_text(item["body"])

        elif "instruction" in item and "output" in item:
            text = self._build_instruction_sample(
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", "")
            )

        elif "question" in item and "response" in item:
            q, r, s = map(
                self._clean_text,
                (
                    item.get("question", ""),
                    item.get("response", ""),
                    item.get("system_prompt", "")
                )
            )

            if s:
                text = f"### 시스템: {s}\n### 질문: {q}\n### 응답: {r}"
            else:
                text = f"### 질문: {q}\n### 응답: {r}"

        elif "question" in item and "answer" in item:
            text = (
                f"### 질문: {self._clean_text(item.get('question', ''))}\n"
                f"### 응답: {self._clean_text(item.get('answer', ''))}"
            )

        elif "prompt" in item and "response" in item:
            text = (
                f"### 질문: {self._clean_text(item.get('prompt', ''))}\n"
                f"### 응답: {self._clean_text(item.get('response', ''))}"
            )

        else:
            text = None

        return text if text and len(text) > 5 else None

    def __len__(self):
        return self.total_samples

    def _find_dataset_and_local_index(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        for i, offset in enumerate(self.offsets):
            end = (
                self.offsets[i + 1]
                if i + 1 < len(self.offsets)
                else self.total_samples
            )

            if offset <= idx < end:
                return i, idx - offset

        raise IndexError(idx)

    def _get_item(self, idx: int) -> Dict:
        dataset_idx, local_idx = self._find_dataset_and_local_index(idx)
        info = self.datasets[dataset_idx]

        # ★ worker 프로세스 안에서 직접 ParquetFile을 생성
        # Dataset 자체에는 ParquetReader가 저장되지 않음
        pf = pq.ParquetFile(info["path"])

        for group_idx, group_start, group_rows in info["row_groups"]:
            if group_start <= local_idx < group_start + group_rows:
                row_in_group = local_idx - group_start

                table = pf.read_row_group(group_idx)
                row = table.slice(row_in_group, 1).to_pylist()[0]

                return row

        raise IndexError(idx)

    def _encode_sample(self, text: str) -> Tuple[List[int], List[int]]:
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        if eos_id is None or pad_id is None:
            raise ValueError(
                "Tokenizer must define separate eos_token_id and pad_token_id"
            )

        marker = "### 응답:"

        if self.train_on_response_only and marker in text:
            prompt, response = text.split(marker, 1)

            prompt_ids = self.tokenizer.encode(
                prompt + marker + "\n",
                add_special_tokens=False
            )

            response_ids = self.tokenizer.encode(
                response,
                add_special_tokens=False
            )

            prompt_budget = min(
                len(prompt_ids),
                max(32, self.max_len // 3)
            )

            prompt_ids = prompt_ids[:prompt_budget]

            response_budget = self.max_len - len(prompt_ids) - 1
            response_ids = response_ids[:max(0, response_budget)]

            ids = prompt_ids + response_ids + [eos_id]
            labels = [-100] * len(prompt_ids) + response_ids + [eos_id]

        else:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_len - 1
            )

            ids.append(eos_id)
            labels = list(ids)

        ids = ids[:self.max_len]
        labels = labels[:self.max_len]

        padding_length = self.max_len - len(ids)

        if padding_length > 0:
            ids += [pad_id] * padding_length
            labels += [-100] * padding_length

        return ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            item = self._get_item(idx)
            text = self._extract_text(item)

            if not text:
                raise ValueError("Empty/unsupported sample")

            encoded, labels = self._encode_sample(text)

            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        except Exception as e:
            logger.warning(
                f"Dataset sample error at index {idx}: {e}"
            )

            pad_id = self.tokenizer.pad_token_id or 0

            return {
                "input_ids": torch.full(
                    (self.max_len,),
                    pad_id,
                    dtype=torch.long
                ),
                "labels": torch.full(
                    (self.max_len,),
                    -100,
                    dtype=torch.long
                )
            }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack(
            [sample["input_ids"] for sample in batch]
        ),
        "labels": torch.stack(
            [sample["labels"] for sample in batch]
        )
    }

# ==========================================
# 1. 아키텍처
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq, head_dim_2 = cos.shape
    head_dim = head_dim_2 * 2

    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, d = x.shape

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k.detach(), v.detach())

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and s > 1)
        )

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        pad_token_id: int = 128004,
        dim: int = 1920,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight

        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        return f[start:start + length]

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s = tokens.shape
        x = self.embed(tokens)

        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]

        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                x, kv = checkpoint(
                    layer, x, f_cos, f_sin, None,
                    use_reentrant=False
                )
            else:
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)

            new_kv_caches.append(kv)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if labels is not None:
            # ``reduction='mean'`` returns NaN when a malformed batch has no
            # target tokens (every label is -100).  Preserve mean-loss
            # semantics for valid batches while making that case a zero-loss
            # batch with zero gradients.
            token_loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            target_count = (labels[..., 1:] != -100).sum().clamp_min(1)
            loss = token_loss / target_count

        return logits, loss, new_kv_caches

# ==========================================
# 3. 생성 함수
# ==========================================
@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    device: torch.device = None
) -> str:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    prompt_text = f"### 질문: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    kv_caches = None
    output_tokens = tokens

    for step in range(max_tokens):
        input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens

        with torch.no_grad():
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)

        next_logits = logits[:, -1, :] / temperature

        if repetition_penalty != 1.0:
            for token_id in set(output_tokens[0].tolist()):
                if next_logits[0, token_id] < 0:
                    next_logits[0, token_id] *= repetition_penalty
                else:
                    next_logits[0, token_id] /= repetition_penalty

        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        next_token = torch.multinomial(probs, num_samples=1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if output_tokens.shape[1] > 512:
            break

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text

    model.train()
    return response

# ==========================================
# 4. 체크포인트 저장/로드
# ==========================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_path: str
):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    load_training_state: bool = True,
) -> int:
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Model state loaded")

        if not load_training_state:
            logger.info("✅ Model-only resume: optimizer and scheduler were reset")
            return 0

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✅ Optimizer state loaded")

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("✅ Scheduler state loaded")

        start_step = checkpoint['step']
        logger.info(f"✅ Checkpoint loaded from step {start_step}")
        return start_step

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest = checkpoints[-1]
    latest_path = os.path.join(checkpoint_dir, latest)
    logger.info(f"Found latest checkpoint: {latest}")
    return latest_path

# ==========================================
# GUI: Loss 그래프 + 채팅 창
# ==========================================
class TrainingMonitorGUI:
    def __init__(self, tokenizer, device, model_config: dict, checkpoint_dir: str):
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_config = model_config
        self.checkpoint_dir = checkpoint_dir
        self.chat_model = None
        self.current_ckpt_path = None
        self.running = True
        self.msg_queue = queue.Queue()

        self.root = tk.Tk()
        self.root.title("KoreanLLM Training Monitor 📊 + Chat")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        left_frame = Frame(self.root, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        Label(left_frame, text="📉 Loss Curve (실시간)", font=("Arial", 12, "bold")).pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label="Loss")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = Frame(self.root, width=550)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ckpt_label = Label(right_frame, text="현재 체크포인트: (아직 없음)", font=("Arial", 10), fg="blue")
        self.ckpt_label.pack(pady=5)

        Label(right_frame, text="💬 모델과 대화하기 (CPU 구동)", font=("Arial", 12, "bold")).pack()

        self.chat_display = scrolledtext.ScrolledText(right_frame, height=25, width=60, state='disabled', wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)

        input_frame = Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = Entry(input_frame, font=("Arial", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)

        send_btn = Button(input_frame, text="전송", command=self.send_message, width=8)
        send_btn.pack(side=tk.RIGHT)

        refresh_btn = Button(right_frame, text="🔄 최신 체크포인트 로드", command=self.load_latest_checkpoint)
        refresh_btn.pack(pady=5)

        self.root.after(1000, self.update_gui)

    def on_close(self):
        self.running = False
        self.root.destroy()

    def update_gui(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg["type"] == "loss":
                    self._update_plot()
                elif msg["type"] == "ckpt":
                    self.current_ckpt_path = msg["path"]
                    self.ckpt_label.config(text=f"현재 체크포인트: {Path(msg['path']).name}")
                elif msg["type"] == "log":
                    self._append_chat(f"[시스템] {msg['text']}\n", "system")
        except queue.Empty:
            pass

        if self.running:
            self.root.after(1000, self.update_gui)

    def _update_plot(self):
        if not loss_history:
            return
        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]
        self.line.set_data(steps, losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _append_chat(self, text: str, tag: str = "user"):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def load_latest_checkpoint(self):
        ckpt = find_latest_checkpoint(self.checkpoint_dir)
        if not ckpt:
            self._append_chat("[시스템] 체크포인트가 아직 없습니다.\n", "system")
            return

        try:
            self._append_chat(f"[시스템] 체크포인트 로딩 중(CPU): {Path(ckpt).name} ...\n", "system")
            self.root.update()

            if self.chat_model is None:
                self.chat_model = KoreanLLM(**self.model_config).to(self.device)

            checkpoint = torch.load(ckpt, map_location=self.device)
            self.chat_model.load_state_dict(checkpoint['model_state_dict'])
            self.chat_model.eval()

            self.current_ckpt_path = ckpt
            self.ckpt_label.config(text=f"현재 체크포인트: {Path(ckpt).name}")
            self._append_chat(f"[시스템] 로드 완료(CPU)! 이제 대화할 수 있어요.\n", "system")
        except Exception as e:
            self._append_chat(f"[시스템] 로드 실패: {e}\n", "system")

    def send_message(self, event=None):
        prompt = self.user_input.get().strip()
        if not prompt:
            return

        self.user_input.delete(0, tk.END)
        self._append_chat(f"나: {prompt}\n", "user")

        if self.chat_model is None:
            self._append_chat("[시스템] 먼저 '최신 체크포인트 로드' 버튼을 눌러주세요.\n", "system")
            return

        try:
            self._append_chat("모델(CPU): 생각 중...\n", "model")
            self.root.update()

            response = generate(
                self.chat_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=512,
                temperature=0.6,
                top_p=0.95,
                device=self.device
            )

            self.chat_display.config(state='normal')
            self.chat_display.delete("end-2l", "end-1l")
            self.chat_display.config(state='disabled')

            self._append_chat(f"모델: {response}\n\n", "model")
        except Exception as e:
            self._append_chat(f"[시스템] 생성 오류: {e}\n", "system")

    def notify_loss(self):
        self.msg_queue.put({"type": "loss"})

    def notify_checkpoint(self, path: str):
        self.msg_queue.put({"type": "ckpt", "path": path})

    def notify_log(self, text: str):
        self.msg_queue.put({"type": "log", "text": text})

    def run(self):
        self.root.mainloop()

gui_monitor: Optional[TrainingMonitorGUI] = None

# ==========================================
# 5. 메인 학습 루프
# ==========================================
@dataclass
class TrainingConfig:
    stage: str = "auto"
    dataset_specs: Optional[List[str]] = None
    batch_size: int = 2
    max_steps: int = 50000
    pretrain_steps: int = 100000
    sft_steps: int = 10000
    accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    eval_interval: int = 5000
    max_seq_len: int = 512
    num_workers: int = 4
    use_bfloat16: bool = True
    use_8bit_optimizer: bool = True
    weight_decay: float = 0.1
    validation_split: float = 0.02
    validation_batches: int = 32
    enable_gui: bool = True # 기본 값
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    force_redownload: bool = False
    stream_datasets: bool = False
    samples_per_dataset: Optional[int] = None

def setup_distributed(seed: int, rank: int = 0, world_size: int = 1):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

def build_dataset_configs(config: TrainingConfig) -> List[Dict]:
    """Turn CLI dataset names into DatasetManager configurations.

    For SFT, the original two instruction datasets remain the default.  For
    pretraining, require an explicit dataset so users intentionally choose a
    licensed Korean text corpus instead of accidentally training on SFT data.
    Dataset syntax: `repository_id` or `repository_id:configuration`.
    """
    if not config.dataset_specs:
        if config.stage == "sft":
            return DatasetManager.DEFAULT_SFT_DATASETS
        if config.stage == "pretrain":
            # Quality-filtered Korean corpus with cross-source MinHash deduplication.
            return [{
                "name": "AdaMLLab/KorMix",
                "config": "minhash_deduped",
                "split": "train",
                "text_keys": ["text", "content", "document", "body"],
                "max_examples": config.samples_per_dataset,
                "streaming": config.stream_datasets,
            }]
        raise ValueError("Pretraining requires --dataset REPOSITORY[:CONFIG]")

    configs = []
    for spec in config.dataset_specs:
        name, separator, dataset_config = spec.partition(":")
        configs.append({
            "name": name,
            "config": dataset_config if separator else None,
            "split": "train",
            "text_keys": ["text", "content", "document", "body"],
            "max_examples": config.samples_per_dataset,
            "streaming": config.stream_datasets,
        })
    return configs

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_bfloat16: bool, max_batches: int) -> Optional[float]:
    if len(loader) == 0:
        return None
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if device.type == "cuda" and use_bfloat16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(input_ids, labels=labels)
        else:
            _, loss, _ = model(input_ids, labels=labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else None

def main(config: TrainingConfig = TrainingConfig(), load_training_state: bool = True):
    global gui_monitor

    if config.stage == "auto":
        logger.info("🚀 Auto mode: pretraining followed by SFT")
        pretrain_dir = "checkpoints/pretrain"
        pretrain_resume = config.resume_from_checkpoint
        if pretrain_resume is None and find_latest_checkpoint(pretrain_dir):
            pretrain_resume = "latest"
        pretrain_config = replace(
            config,
            stage="pretrain",
            max_steps=config.pretrain_steps,
            checkpoint_dir=pretrain_dir,
            resume_from_checkpoint=pretrain_resume,
            enable_gui=False,
        )
        main(pretrain_config)

        pretrain_checkpoint = find_latest_checkpoint(pretrain_dir)
        if pretrain_checkpoint is None:
            logger.error("Auto mode stopped: no pretraining checkpoint was produced, so SFT cannot start.")
            return

        logger.info(f"✅ Pretraining complete. Starting SFT from {pretrain_checkpoint}")
        sft_config = replace(
            config,
            stage="sft",
            dataset_specs=None,
            max_steps=config.sft_steps,
            checkpoint_dir="checkpoints/sft",
            resume_from_checkpoint=pretrain_checkpoint,
        )
        # SFT starts from pretrained weights, but requires a fresh optimizer
        # and learning-rate schedule for its own training horizon.
        main(sft_config, load_training_state=False)
        return

    setup_distributed(config.seed)
    ensure_datasets_dir()
    load_loss_history()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"📂 Datasets directory: {DATASETS_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")

    # ============================================
    # 1. 토크나이저 로드
    # ============================================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )

    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        logger.info("✅ Added separate pad token: <|pad|>")

    logger.info(f"Tokenizer: vocab_size={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}")

    # ============================================
    # 2. 데이터셋 다운로드 및 로드
    # ============================================
    logger.info("Setting up datasets...")
    try:
        dataset_configs = build_dataset_configs(config)
    except ValueError as e:
        logger.error(str(e))
        return
    manager = DatasetManager(dataset_configs=dataset_configs)

    dataset_paths = manager.get_or_download_all(force=config.force_redownload)

    if not dataset_paths:
        logger.error("❌ No datasets available!")
        return

    dataset = LocalKoreanDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        data_samples_per_dataset=config.samples_per_dataset,
        train_on_response_only=(config.stage == "sft")
    )

    if len(dataset) == 0:
        logger.error("❌ Dataset is empty!")
        return

    validation_size = max(1, int(len(dataset) * config.validation_split)) if len(dataset) > 1 and config.validation_split > 0 else 0
    train_size = len(dataset) - validation_size
    if train_size == 0:
        logger.error("Validation split leaves no training examples")
        return
    split_generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=split_generator)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   shuffle=False, collate_fn=collate_fn, pin_memory=(device.type == "cuda")) if validation_size else None

    # ============================================
    # 3. 모델 생성 (VRAM 최적화: BF16 변환)
    # ============================================
    logger.info("Creating model...")
    model_config = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1920,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    )
    model = KoreanLLM(**model_config).to(device)
    if device.type == "cuda" and config.use_bfloat16:
        model = model.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")

    # ============================================
    # GUI 시작
    # ============================================
    def start_gui():
        global gui_monitor
        gui_monitor = TrainingMonitorGUI(tokenizer, device, model_config, config.checkpoint_dir)
        gui_monitor.run()

    if config.enable_gui:
        gui_thread = threading.Thread(target=start_gui, daemon=True)
        gui_thread.start()
        time.sleep(1.5)
        logger.info("🖥️  Monitoring GUI started (Loss graph + Chat)")

    # ============================================
    # 4. 옵티마이저와 스케줄러 (VRAM 최적화: 8-bit Optimizer)
    # ============================================
    if config.use_8bit_optimizer and bnb is not None and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        logger.info("Using bitsandbytes AdamW8bit")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        logger.info("Using torch AdamW")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    # BF16 사용 시 Scaler는 기본적으로 필요 없으나, 하위 호환성을 위해 유지
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and not config.use_bfloat16) else None

    # ============================================
    # 5. 체크포인트 로드
    # ============================================
    start_step = 0

    if config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint

        if checkpoint_path.lower() == 'latest':
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
            if checkpoint_path is None:
                logger.warning("No checkpoint found, starting from scratch")

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
            start_step = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device,
                load_training_state=load_training_state,
            )
            if gui_monitor:
                gui_monitor.notify_checkpoint(checkpoint_path)

    # ============================================
    # 6. 학습 루프
    # ============================================
    logger.info(f"🚀 Starting training from step {start_step}...")
    logger.info(f"📊 Dataset size: train={train_size}, validation={validation_size}")
    logger.info(f"📊 Total batches per epoch: {len(loader)}")

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    step = 0

    try:
        epoch = 0
        while True:
            epoch += 1
            logger.info(f"\n📍 Epoch {epoch}")

            for batch_idx, batch in enumerate(loader):
                actual_step = (step // config.accumulation_steps) + start_step

                if actual_step >= config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                    break

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if device.type == 'cuda' and config.use_bfloat16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, loss, _ = model(input_ids, labels=labels)
                        loss_scaled = loss / config.accumulation_steps

                    # bfloat16은 scaler.scale이 필요 없으므로 바로 backward
                    loss_scaled.backward()
                else:
                    _, loss, _ = model(input_ids, labels=labels)
                    loss_scaled = loss / config.accumulation_steps
                    loss_scaled.backward()

                running_loss += loss.item()

                if step % 4 == 0:
                    print(".", end="", flush=True)

                if (step + 1) % config.accumulation_steps == 0:
                    if device.type == 'cuda' and config.use_bfloat16:
                        # bfloat16은 unscale이 필요 없음
                        pass
                    elif scaler:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if scaler and not config.use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                    actual_step = (step + 1) // config.accumulation_steps + start_step
                    avg_loss = running_loss / config.accumulation_steps
                    lr = scheduler.get_last_lr()[0]

                    validation_loss = (
                        evaluate(model, validation_loader, device, config.use_bfloat16, config.validation_batches)
                        if validation_loader and actual_step % config.eval_interval == 0 else None
                    )
                    validation_text = f" | Val: {validation_loss:.4f}" if validation_loss is not None else ""
                    log_msg = f"[Step {actual_step:5d}] Train: {avg_loss:.4f}{validation_text} | LR: {lr:.2e} | Tokens/step: {config.batch_size * config.max_seq_len * config.accumulation_steps}"
                    print(f"\n{log_msg}")
                    logger.info(log_msg)

                    loss_history.append({
                        "step": actual_step,
                        "loss": float(avg_loss),
                        "validation_loss": float(validation_loss) if validation_loss is not None else None,
                        "lr": float(lr),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    save_loss_history()

                    if gui_monitor:
                        gui_monitor.notify_loss()

                    running_loss = 0.0

                    if actual_step % config.eval_interval == 0:
                        logger.info("\n📝 Generating samples...")
                        prompts = [
                            "한국의 수도는",
                            "인공지능이란",
                            "안녕?"
                        ]
                        for prompt in prompts:
                            response = generate(
                                model, tokenizer, prompt=prompt,
                                max_tokens=50, temperature=0.7, top_p=0.95, device=device
                            )
                            logger.info(f"  Q: {prompt}\n  A: {response}")

                        checkpoint_path = f"{config.checkpoint_dir}/korean_llm_{actual_step:05d}.pth"
                        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)

                        if gui_monitor:
                            gui_monitor.notify_checkpoint(checkpoint_path)
                            gui_monitor.notify_log(f"체크포인트 저장됨: {Path(checkpoint_path).name}")

                step += 1

            if actual_step >= config.max_steps:
                break

    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        actual_step = (step // config.accumulation_steps) + start_step
        checkpoint_path = f"{config.checkpoint_dir}/korean_llm_interrupted_{actual_step:05d}.pth"
        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
        if gui_monitor:
            gui_monitor.notify_checkpoint(checkpoint_path)

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

    logger.info("🎉 Training completed!")
    save_loss_history()

    if gui_monitor and gui_monitor.running:
        logger.info("GUI가 열려 있습니다. 창을 닫으면 종료됩니다.")
        while gui_monitor.running:
            time.sleep(1)

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train KoreanLLM from scratch: pretraining or SFT")
    parser.add_argument("--stage", choices=["auto", "pretrain", "sft"], default="auto",
                        help="Default auto runs pretraining and then SFT. Select a stage only for manual control.")
    parser.add_argument("--dataset", action="append", dest="dataset_specs",
                        help="Hugging Face dataset as repository_id or repository_id:config. Repeat for multiple datasets.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50_000, help="Steps for a manually selected stage.")
    parser.add_argument("--pretrain-steps", type=int, default=100_000)
    parser.add_argument("--sft-steps", type=int, default=10_000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--validation-split", type=float, default=0.02)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-dataset", type=int)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--stream-datasets", action="store_true",
                        help="Stream and cache only --samples-per-dataset records; required for huge corpora.")
    parser.add_argument("--no-bfloat16", action="store_false", dest="use_bfloat16")
    parser.add_argument("--no-8bit-optimizer", action="store_false", dest="use_8bit_optimizer")
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", dest="enable_gui", default=True,
                           help="Enable the training monitor GUI (enabled by default).")
    gui_group.add_argument("--no-gui", action="store_false", dest="enable_gui",
                           help="Disable the training monitor GUI.")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))

if __name__ == "__main__":
    main(parse_args())
