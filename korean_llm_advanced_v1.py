import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset as HFDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
import random
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import hashlib
import time
import traceback

# ==========================================
# 로깅 설정
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 데이터셋 기본 설정
# ==========================================

DATASETS_DIR = Path("./datasets")
DATASETS_CACHE_DIR = DATASETS_DIR / "cache"
DATASETS_MANIFEST_FILE = DATASETS_DIR / "datasets_manifest.json"

def ensure_datasets_dir():
    """데이터셋 디렉토리 생성"""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Datasets directory ready: {DATASETS_DIR.absolute()}")

# ==========================================
# 데이터셋 다운로드 및 관리
# ==========================================

class DatasetManager:
    """로컬 데이터셋 관리 클래스"""

    DATASETS_CONFIG = [
        {
            "name": "maywell/korean_textbooks",
            "config": "tiny-textbooks",
            "split": "train",
            "text_key": "text"
        },
        {
            "name": "beomi/KoAlpaca-v1.1a",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "output"]
        }
    ]

    def __init__(self, cache_dir: Path = DATASETS_CACHE_DIR):
        self.cache_dir = cache_dir
        self.manifest = self._load_manifest()
        ensure_datasets_dir()

    def _load_manifest(self) -> Dict:
        """매니페스트 파일 로드"""
        if DATASETS_MANIFEST_FILE.exists():
            with open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """매니페스트 파일 저장"""
        with open(DATASETS_MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _get_dataset_hash(self, config: Dict) -> str:
        """데이터셋 config의 해시값 생성"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def download_dataset(self, config: Dict, force: bool = False) -> Optional[str]:
        """
        데이터셋 다운로드
        - 3가지 전략으로 재시도: standard, streaming, force_redownload
        - 실패 시 자세한 에러 로그 출력
        """
        dataset_hash = self._get_dataset_hash(config)
        dataset_name = config['name']

        # 캐시 확인
        if dataset_hash in self.manifest and not force:
            cached_path = self.manifest[dataset_hash].get('path')
            if cached_path and Path(cached_path).exists():
                logger.info(f"✅ Using cached dataset: {dataset_name}")
                return cached_path

        logger.info(f"📥 Downloading {dataset_name}...")

        last_error = None
        error_details = []

        # ============================================================
        # 일반 데이터셋 처리 (표준 전략)
        # ============================================================
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
                    ds = HFDataset.from_list(list(ds))

                local_path = self.cache_dir / f"{dataset_hash}"
                local_path.mkdir(exist_ok=True, parents=True)

                ds.to_parquet(str(local_path / "data.parquet"))

                self.manifest[dataset_hash] = {
                    'name': dataset_name,
                    'config': config,
                    'path': str(local_path),
                    'num_examples': len(ds),
                    'download_strategy': strategy['name']
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
                    "strategy": strategy['name'],
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

        # 모든 시도 실패
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
        """모든 데이터셋 다운로드 또는 캐시 로드"""
        paths = []
        failed = []

        for config in self.DATASETS_CONFIG:
            path = self.download_dataset(config, force=force)
            if path:
                paths.append(path)
            else:
                failed.append(config['name'])

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
    """로컬 파일에서 로드하는 한국어 데이터셋"""

    def __init__(
        self,
        dataset_paths: List[str],
        tokenizer,
        max_len: int = 256,
        data_samples_per_dataset: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info("📚 Loading local datasets...")

        for dataset_path in dataset_paths:
            try:
                parquet_file = Path(dataset_path) / "data.parquet"
                if not parquet_file.exists():
                    logger.warning(f"Parquet file not found: {parquet_file}")
                    continue

                ds = HFDataset.from_parquet(str(parquet_file))

                if data_samples_per_dataset:
                    ds = ds.select(range(min(len(ds), data_samples_per_dataset)))

                texts = self._extract_texts(ds)
                self.samples.extend(texts)

                logger.info(f"✅ Loaded {len(texts)} samples from {Path(dataset_path).name}")

            except Exception as e:
                logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                continue

        logger.info(f"✅ Total samples loaded: {len(self.samples)}")

    def _extract_texts(self, ds) -> List[str]:
        """데이터셋에서 텍스트 추출"""
        texts = []

        for item in ds:
            text = None

            if "text" in item and item["text"]:
                text = item["text"]
            elif "instruction" in item and "output" in item:
                text = f"### 질문: {item['instruction']}\n### 응답: {item['output']}"
            elif "question" in item and "response" in item:  # OpenOrca-gugugo-ko
                system = item.get("system_prompt", "")
                if system:
                    text = f"### 질문: {item['question']}\n### 응답: {item['response']}"
                else:
                    text = f"### 질문: {item['question']}\n### 응답: {item['response']}"
            elif "question" in item and "answer" in item:
                text = f"### 질문: {item['question']}\n### 응답: {item['answer']}"
            elif "prompt" in item and "response" in item:
                text = f"### 질문: {item['prompt']}\n### 응답: {item['response']}"

            if text and len(text) > 5:
                texts.append(text)

        return texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.samples[idx]

        try:
            # [MINIMAL FIX] EOS를 명시적으로 추가 (텍스트 끝에 한 번만)
            # 기존에는 text + self.tokenizer.eos_token을 사용했는데,
            # 이제는 tokenize 후 eos_token_id를 추가
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_len - 1  # EOS 공간 남기기
            )

            # [MINIMAL FIX] EOS 토큰을 명시적으로 한 개 추가
            encoded.append(self.tokenizer.eos_token_id)

            # [MINIMAL FIX] 남은 공간을 PAD로 채우기 (PAD와 EOS 분리)
            if len(encoded) < self.max_len:
                encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
            else:
                # 이미 max_len보다 크면 그냥 max_len으로 자르기
                encoded = encoded[:self.max_len]

            return torch.tensor(encoded, dtype=torch.long)

        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            # [MINIMAL FIX] PAD 토큰으로 실패한 배치 채우기
            return torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """배치 콜레이션"""
    return torch.stack(batch)

# ==========================================
# 1. 아키텍처 (개선된 버전 - 기존 코드 유지)
# ==========================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE(Rotary Position Embedding) 주파수 사전계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE 적용"""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    """SwiGLU 활성화 함수"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Multi-Head Attention with KV-Cache"""
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
    """Transformer 블록"""
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
    """한국어 LLM 모델"""
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        dim: int = 1280,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        
        f_cos, f_sin = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
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
        return f[start:start + length].unsqueeze(0).unsqueeze(0)
    
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
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss, new_kv_caches

# ==========================================
# 3. 생성 함수
# ==========================================

@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    device: torch.device = None
) -> str:
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    prompt_text = f"### 지시: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    kv_caches = None
    output_tokens = tokens
    
    for step in range(max_tokens):
        input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens
        
        with torch.no_grad():
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)
        
        next_logits = logits[:, -1, :] / temperature
        
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        if output_tokens.shape[1] > 512:
            logger.warning("Generated sequence too long, truncating")
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
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
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
    device: torch.device
) -> int:
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Model state loaded")
        
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
# 5. 메인 학습 루프
# ==========================================

@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 2
    max_steps: int = 50000
    accumulation_steps: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    checkpoint_interval: int = 100
    eval_interval: int = 500
    max_seq_len: int = 256
    num_workers: int = 4
    use_bfloat16: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    # 새로운 옵션들
    download_datasets: bool = False  # True면 데이터셋 다시 다운로드
    samples_per_dataset: Optional[int] = None  # None이면 전체 사용

def setup_distributed(rank: int = 0, world_size: int = 1):
    """분산학습 설정"""
    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)

def main(config: TrainingConfig = TrainingConfig()):
    """메인 학습 함수 (로컬 데이터셋 기반)"""
    setup_distributed()
    ensure_datasets_dir()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"📂 Datasets directory: {DATASETS_DIR.absolute()}")
    
    # ============================================
    # 1. 토크나이저 로드
    # ============================================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # ============================================
    # 2. 데이터셋 다운로드 및 로드
    # ============================================
    logger.info("Setting up datasets...")
    manager = DatasetManager()
    
    # 데이터셋 다운로드 (또는 캐시 로드)
    dataset_paths = manager.get_or_download_all(force=config.download_datasets)
    
    if not dataset_paths:
        logger.error("❌ No datasets available!")
        return
    
    # 로컬 데이터셋 생성
    dataset = LocalKoreanDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        data_samples_per_dataset=config.samples_per_dataset
    )
    
    if len(dataset) == 0:
        logger.error("❌ Dataset is empty!")
        return
    
    # 데이터로더
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # ============================================
    # 3. 모델 생성
    # ============================================
    logger.info("Creating model...")
    model = KoreanLLM(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1280,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")
    
    # ============================================
    # 4. 옵티마이저와 스케줄러
    # ============================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # ============================================
    # 5. 체크포인트 로드
    # ============================================
    start_step = 0
    
    if config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint
        
        if checkpoint_path.lower() == 'latest':
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No checkpoint found, starting from scratch")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
            start_step = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device
            )
    
    # ============================================
    # 6. 학습 루프
    # ============================================
    logger.info(f"🚀 Starting training from step {start_step}...")
    logger.info(f"📊 Dataset size: {len(dataset)} samples")
    logger.info(f"📊 Total batches per epoch: {len(loader)}")
    
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    step = start_step * config.accumulation_steps
    
    try:
        epoch = 0
        while step // config.accumulation_steps < config.max_steps:
            epoch += 1
            logger.info(f"\n📍 Epoch {epoch}")
            
            for batch_idx, batch in enumerate(loader):
                if step // config.accumulation_steps >= config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                    break
                
                batch = batch.to(device)
                
                # Forward pass with AMP
                if device.type == 'cuda' and config.use_bfloat16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, loss, _ = model(batch, labels=batch)
                        loss = loss / config.accumulation_steps
                    
                    scaler.scale(loss).backward()
                else:
                    _, loss_val, _ = model(batch, labels=batch)
                    loss = loss_val / config.accumulation_steps
                    loss.backward()
                
                running_loss += loss.item() * config.accumulation_steps
                
                # Progress indicator
                if step % 4 == 0:
                    print(".", end="", flush=True)
                
                # Gradient accumulation
                if (step + 1) % config.accumulation_steps == 0:
                    if device.type == 'cuda' and config.use_bfloat16:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    if device.type == 'cuda' and config.use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # 로깅
                    actual_step = (step + 1) // config.accumulation_steps
                    avg_loss = running_loss / config.accumulation_steps
                    lr = scheduler.get_last_lr()[0]
                    
                    print(f"\n[Step {actual_step:5d}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/step: {config.batch_size * config.max_seq_len}")
                    
                    running_loss = 0.0
                    
                    # 평가 및 저장
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
                                max_tokens=50, temperature=0.7, device=device
                            )
                            logger.info(f"  Q: {prompt}\n  A: {response}")
                        
                        # 체크포인트 저장
                        checkpoint_path = f"checkpoints/korean_llm_{actual_step:05d}.pth"
                        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
                
                step += 1
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        actual_step = step // config.accumulation_steps
        checkpoint_path = f"checkpoints/korean_llm_interrupted_{actual_step:05d}.pth"
        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
    
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    
    logger.info("🎉 Training completed!")

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=32,
        max_steps=50000,
        warmup_steps=200,
        learning_rate=5e-5,
        eval_interval=100,
        resume_from_checkpoint='latest' if find_latest_checkpoint() else None,
        download_datasets=False,  # True로 바꾸면 강제로 다시 다운로드
        samples_per_dataset=None
    )
    main(config)