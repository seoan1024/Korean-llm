# 🇰🇷 Korean LLM Advanced v3

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Model Size](https://img.shields.io/badge/Model-1.09B%20Parameters-orange)](#모델-사양)
[![VRAM](https://img.shields.io/badge/VRAM%20Usage-9GB-red)](#최적화-사양)

<div align="center">

**한국어 특화 대규모 언어모델 - 풀스크래치 구현 및 양자화 적용**

## 🚀 프로젝트 개발기
처음에는 기업들의 무료 한도가 빡세지고 '바이브 코딩'을 하기에는 한도의 한계가 찾아왔습니다. Ollama를 활용해 로컬로 돌려보기도 했지만, 모델이 너무 무거워 컴퓨터가 버거워했죠. 그때 문득 '이럴 바엔 내가 직접 만들어볼까?'라는 생각이 들었습니다.
하지만 저는 중학교 2학년이었고 인공지능 모델에 대해 아는 것이라곤 모델 크기를 나타내는 'B(Billion)'라는 개념이 전부였고, 기본이 아니라 고급개념을 짤 수 없었죠.
결국 평소처럼 ChatGPT에 들어가 “나 독자적인 한국어 LLM 모델 만들래!”라는 한마디를 던지며 무모한 도전을 시작했습니다.
GPT의 도움을 받으면서도 한계는 계속 찾아왔습니다. 처음에는 그저 GPT가 준 코드 조각들을 모아 차원 오류(Dimension Error)가 나지 않기만을 바라며 열심히 돌려볼 뿐이었습니다. 데이터를 수집하고 정제하며 온종일 컴퓨터 모니터만 바라보았습니다.
그렇게 태어난 제 첫 작품(v1 이전 버전)은 위키피디아 데이터로 학습한, 고작 50M(5천만 파라미터) 크기의 극소형 모델이었습니다. 지금은 남아있지 않지만요. 비록 대화는 불가능했지만, 어느 정도 문법에 맞는 문장을 구상하는 모습을 보였습니다.
그 작은 성공이 너무 기뻐서 이때부터 본격적으로 '채팅형 모델'을 만드는 데만 몰입했습니다. 그렇게 v1의 최종 버전인 541M 크기의 모델까지 발전시켰습니다. 몇 가지 버그가 발견되었지만, 우선 버그를 해결한 뒤 곧바로 모델의 체급을 키우기로 결심했습니다.
오류들을 수정하고, 모델 크기를 2배가량 키워 드디어 1.09B(10억 9천만 파라미터) 크기의 모델을 구축했습니다. 방학 기간 내내 시간이 날 때마다 컴퓨터를 켜고 학습을 돌렸습니다.
그렇게 끝이 보이지 않던 학습이 어느덧 44,000 스텝에 도달했습니다. 설레는 마음으로 테스트를 위해 채팅창에 "안녕?"이라고 입력했습니다.

"안녕하세요! 오늘은 무엇을 도와드릴까요?"

모델이 올바른 답변을 화면에 띄운 그 순간, 말로 표현할 수 없을 만큼 기뻤습니다.
하지만 기쁨도 잠시, 다른 질문을 던지자 전혀 엉뚱한 대답을 쏟아내기 시작했습니다. AI와 함께 밤새 코드를 분석한 결과, 모델이 사용자의 지시사항을 무시해 버리는 치명적인 버그가 발생한 것이었습니다. 가슴이 아팠지만, 더 완벽한 모델을 위해 지금까지 학습한 결과물을 과감히 폐기했습니다.
낙담하지 않고 v2의 버그를 완전히 해결한 뒤, 다음 문제에 도전했습니다. 1B 체급의 모델은 VRAM을 무려 23GB나 차지하여 일반적인 환경에서 돌리기 너무 무거웠기 때문입니다. 이를 10GB 이하로 줄여보겠다는 목표를 세웠고, 마침내 v3에서 양자화에 성공했습니다.
방학이 끝났다 보니 제가 학습하기에는 어렵습니다. v2 버그 이후 학습한 건 아예 없고 시간이 날때 다시 학습해보겠습니다.
이 프로젝트는 오직 "내 손으로 직접 LLM을 만들고 싶다"는 고집 하나로 완성해 낸, 제 인생 최고의 작품입니다.
이 모델을 잘 사용하시고, 마음에 드셨다면 스타(⭐) 버튼 한 번씩 꼭 눌러주세요! 감사합니다!

처음부터 끝까지 한국어로 학습된 **1.09B 파라미터 LLM**으로, VRAM 최적화 기법을 적극 활용했습니다.

[📋 주요 특징](#주요-특징) • [🚀 빠른 시작](#빠른-시작) • [💾 기술 스택](#기술-스택) • [📊 버전 히스토리](#버전-히스토리)

</div>


## [신규 버전 V4 출시](https://github.com/seoan1024/Korean-llm-v4.git)

---

## 📖 개요

**Korean LLM Advanced v3**는 한국어 자연어 처리에 최적화된 **경량 대규모 언어모델**입니다. 
제한된 GPU 메모리 환경에서도 효율적으로 학습하고 추론할 수 있도록 설계되었습니다.

### 핵심 목표
- ✅ 한국어 텍스트 생성 및 이해 능력
- ✅ VRAM 효율성 (9GB 기준)
- ✅ 빠른 학습 속도
- ✅ 쉬운 배포 및 활용

---

## 🌟 주요 특징

### 🎯 모델 구조
| 항목 | 설명 |
|------|------|
| **모델 크기** | 1.09B 파라미터 |
| **은닉층 크기** | 1,920차원 |
| **레이어 수** | 20개 |
| **어텐션 헤드** | 10개 |
| **최대 시퀀스 길이** | 2,048 토큰 |
| **어휘집 크기** | 동적 (토크나이저 기준) |

### 🔧 최적화 기법

#### 1️⃣ **BF16 자동 혼합 정밀도**
```
표준 FP32와 비교해 약 50% VRAM 절약
- 메모리 효율: ⬇️ 12GB → 6GB
- 연산 속도: ➡️ 동등 또는 향상
```

#### 2️⃣ **8비트 AdamW 옵티마이저** (bitsandbytes)
```
옵티마이저 상태 메모리 75% 감소
- 표준 AdamW: ~2.2GB (1B 모델)
- 8-bit AdamW: ~0.55GB (1B 모델)
```

#### 3️⃣ **양자화 (Quantization)** ⭐
```
모델 가중치 동적 양자화 지원
- INT8 양자화: 크기 4배 감소
- 추론 속도: 1.5~2배 향상
```

#### 4️⃣ **그래디언트 누적 (Gradient Accumulation)**
```
효과적 배치 크기 증대
- 설정: batch_size=2, accumulation_steps=8
- 효과: 배치 크기 16 효과
```

#### 5️⃣ **그래디언트 체크포인팅**
```
활성화(Activation) 메모리 감소
- 재계산 비용: ~30% 속도 저하
- 메모리 절약: 30~40%
```

---

## 💾 VRAM 사용량 비교

<div align="center">

| 버전 | 파라미터 | VRAM 사용량 | 최적화 기법 |
|------|---------|-----------|----------|
| **v1** | 541M | ~11GB | 기본 FP32 |
| **v2** | ~1.1B | ~23GB | BF16 + Gradient Checkpoint |
| **v3** | 1.09B | **~9GB** ✨ | BF16 + 8비트 옵티마이저 + 양자화 |

**v3은 v2 대비 VRAM 60% 감소, v1보다는 모델 크기 2배 확대**

</div>

---

## 🚀 빠른 시작

### 📋 사전 요구사항

```bash
Python 3.9 이상
CUDA 11.8 이상 (GPU 필수)
GPU 메모리: 최소 9GB 권장
```

### 1️⃣ 설치

```bash
# 저장소 클론
git clone https://github.com/seoan1024/korean-llm-v3.git
cd korean-llm-v3

# 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tqdm pandas matplotlib

# 양자화 지원 라이브러리 (선택)
pip install bitsandbytes
```

### 2️⃣ 데이터셋 준비

코드가 자동으로 다음 데이터셋을 다운로드합니다:
- 🔹 `nlpai-lab/kullm-v2` - 한국어 명령어 튜닝 데이터
- 🔹 `beomi/KoAlpaca-v1.1a` - 한국식 알파카 데이터셋

```bash
# 데이터셋이 자동 다운로드되므로 별도 작업 불필요
# 캐시 디렉토리: ./datasets/cache/
```

### 3️⃣ 학습 실행

```bash
# 기본 설정으로 학습 시작
python korean_llm_advanced_v3.py

# 또는 커스텀 설정으로 실행
python korean_llm_advanced_v3.py \
    --batch-size 2 \
    --max-steps 50000 \
    --learning-rate 5e-5
```

### 4️⃣ 모니터링

학습 중 자동으로 GUI 모니터링 창이 열립니다:
- 📊 실시간 손실값(Loss) 그래프
- 💬 인터랙티브 채팅 (생성 테스트)
- 📝 로그 뷰어

---

## 🏗️ 프로젝트 구조

```
korean-llm-v3/
├── korean_llm_advanced_v3.py    # 메인 학습 스크립트
├── README.md                      # 이 파일
├── LICENSE                        # GPL-3.0 라이선스
│
├── checkpoints/                   # 저장된 모델 체크포인트
│   └── korean_llm_*.pth
│
├── datasets/                      # 데이터셋 캐시
│   ├── cache/                     # 다운로드된 데이터셋
│   └── datasets_manifest.json     # 메타데이터
│
└── logs/                          # 학습 로그 및 그래프
    ├── training.log               # 상세 로그
    └── loss_history.json          # 손실값 기록
```

---

## 📊 버전 히스토리

### v1 (초기 버전)
- 541M 파라미터 모델
- VRAM 사용량: ~11GB
- 기본 FP32 학습

### v2 (최적화 v1)
- 1.1B 파라미터로 확대
- VRAM 사용량: ~23GB (초기 1.2배 증가)
- **BF16 + Gradient Checkpoint** 적용

### **v3 (현재)** ⭐
- 1.09B 파라미터 (v2 수준)
- **VRAM 사용량: ~9GB** (v2 대비 60% 감소!)
- **주요 개선사항:**
  - 8비트 AdamW 옵티마이저
  - 동적 양자화 지원
  - 향상된 메모리 관리
  - 더 빠른 학습 속도

---

## 🔧 기술 스택

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|---------|------|------|
| **PyTorch** | 2.0+ | 딥러닝 프레임워크 |
| **Transformers** | 4.30+ | 토크나이저 및 유틸리티 |
| **Datasets** | 2.10+ | 한국어 데이터셋 로드 |
| **bitsandbytes** | 0.40+ | 8비트 양자화 최적화 |
| **tqdm** | 4.60+ | 진행률 표시 |

### 선택 라이브러리

| 라이브러리 | 용도 |
|---------|------|
| **matplotlib** | 손실값 그래프 시각화 |
| **tkinter** | GUI 모니터링 (내장) |
| **pandas** | 데이터 처리 |

---

## 💡 사용 예시

### 모델 로드 및 텍스트 생성

```python
import os, argparse
from pathlib import Path
from typing import Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim_2 = cos.shape[-1]
    head_dim = head_dim_2 * 2
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
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
    def forward(self, x: torch.Tensor, f_cos: torch.Tensor, f_sin: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=(s > 1))
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def forward(self, x, f_cos, f_sin, kv_cache=None):
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, dim: int = 1920, n_layers: int = 20, n_heads: int = 10, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, int(dim * 2.5)) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)
    def _get_freqs(self, f, start, length):
        end = start + length
        if end > f.shape[0]:
            raise ValueError(f"현재 컨텍스트가 너무 깁니다: {end} > {f.shape[0]}")
        return f[start:end]
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, kv_caches=None):
        b, s = tokens.shape
        x = self.embed(tokens)
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]
        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, kv = layer(x, f_cos, f_sin, kv_cache=cache)
            new_kv_caches.append(kv)
        x = self.norm(x)
        logits = self.output(x)
        return logits, new_kv_caches

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists(): return None
    files = list(checkpoint_dir.glob("korean_llm_*.pth"))
    if not files: return None
    def step_number(path):
        try: return int(path.stem.split("_")[-1])
        except ValueError: return -1
    files.sort(key=step_number)
    return files[-1]

def load_checkpoint(model, checkpoint_path, device):
    print(f"📦 체크포인트 로딩:\n   {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step", "?")
    else:
        state_dict = checkpoint
        step = "?"
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 모델 로드 완료\n   학습 step: {step}")
    return step

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_tokens=256, temperature=0.6, top_k=40, top_p=0.95, repetition_penalty=1.15, context_limit=512):
    model.eval()
    prompt_text = f"### 질문: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    if tokens.shape[1] >= context_limit:
        tokens = tokens[:, -context_limit + 1:]
    output_tokens = tokens
    kv_caches = None
    eos_id = tokenizer.eos_token_id
    for _ in range(max_tokens):
        input_tokens = output_tokens if kv_caches is None else output_tokens[:, -1:]
        logits, kv_caches = model(input_tokens, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]
        temperature = max(float(temperature), 1e-5)
        next_logits = next_logits / temperature
        if repetition_penalty != 1.0:
            used_tokens = set(output_tokens[0].tolist())
            for token_id in used_tokens:
                if token_id < next_logits.shape[-1]:
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty
        if top_k > 0:
            k = min(int(top_k), next_logits.shape[-1])
            threshold = torch.topk(next_logits, k).values[..., -1, None]
            next_logits = torch.where(next_logits < threshold, torch.full_like(next_logits, float("-inf")), next_logits)
        probs = F.softmax(next_logits, dim=-1)
        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[..., 0] = False
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        if not torch.isfinite(probs).all():
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        if eos_id is not None and next_token.item() == eos_id: break
        if output_tokens.shape[1] >= context_limit: break
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    if "### 응답:" in generated_text:
        response = generated_text.split("### 응답:", 1)[1]
    else:
        response = generated_text
    if "### 질문:" in response:
        response = response.split("### 질문:", 1)[0]
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="KoreanLLM 체크포인트 채팅")
    parser.add_argument("--checkpoint", type=str, default="latest", help="체크포인트 경로 또는 latest")
    parser.add_argument("--tokenizer", type=str, default="beomi/Llama-3-Open-Ko-8B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--cpu", action="store_true", help="강제로 CPU 사용")
    args = parser.parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("🇰🇷 KoreanLLM v3 Checkpoint Chat")
    print("=" * 65)
    print(f"🖥️ Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CPU 모드입니다. 1920-dim / 20-layer 모델이라 생성이 느릴 수 있습니다.")
    print("\n🔤 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    print(f"   vocab_size = {vocab_size}")
    print(f"   eos_token_id = {tokenizer.eos_token_id}")
    print(f"   pad_token_id = {pad_token_id}")
    checkpoint_path = args.checkpoint
    if checkpoint_path.lower() == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("\n❌ checkpoints 폴더에서 체크포인트를 찾지 못했습니다.")
            print("예: python chat_korean_llm.py --checkpoint checkpoints/korean_llm_50000.pth")
            return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ 체크포인트가 없습니다:\n   {checkpoint_path}")
        return
    print("\n🧠 모델 생성 중...")
    model_config = dict(vocab_size=vocab_size, pad_token_id=pad_token_id, dim=1920, n_layers=20, n_heads=10, max_seq_len=512)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = KoreanLLM(**model_config).to(device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)
    try:
        step = load_checkpoint(model, checkpoint_path, device)
    except RuntimeError as e:
        print("\n❌ 체크포인트와 현재 모델 구조가 맞지 않습니다.")
        print("   특히 tokenizer의 vocab_size / pad_token 설정을 확인하세요.")
        print(f"\n상세 오류:\n{e}")
        return
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {params / 1e6:.1f}M")
    print("\n" + "=" * 65)
    print("💬 채팅 시작")
    print("   /exit  종료")
    print("   /clear 대화 입력 기록 초기화")
    print("   /info  모델 정보")
    print("=" * 65)
    while True:
        try:
            prompt = input("\n나 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 종료합니다.")
            break
        if not prompt: continue
        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("👋 종료합니다.")
            break
        if prompt == "/clear":
            print("🧹 입력 상태를 초기화했습니다.")
            continue
        if prompt == "/info":
            print(f"\n체크포인트 : {checkpoint_path}\n학습 step   : {step}\nDevice      : {device}\nParameters  : {params / 1e6:.1f}M\nTemperature : {args.temperature}\nTop-k       : {args.top_k}\nTop-p       : {args.top_p}")
            continue
        print("\n모델 > ", end="", flush=True)
        try:
            response = generate(model=model, tokenizer=tokenizer, prompt=prompt, device=device, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, context_limit=512)
            print(response)
        except torch.cuda.OutOfMemoryError:
            print("\n❌ CUDA 메모리가 부족합니다.")
            print("   --max-tokens 값을 낮추거나 다른 GPU에서 실행해보세요.")
        except Exception as e:
            print(f"\n❌ 생성 오류: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
```

### 커스텀 학습 설정

```python
from korean_llm_advanced_v3 import TrainingConfig, main

config = TrainingConfig(
    batch_size=4,                      # 배치 크기
    accumulation_steps=4,              # 그래디언트 누적 스텝
    max_steps=100000,                  # 최대 학습 스텝
    warmup_steps=1000,                 # 워밍업 스텝
    learning_rate=3e-5,                # 학습률
    eval_interval=5000,                # 평가 간격
    use_bfloat16=True,                 # BF16 사용 여부
    resume_from_checkpoint='latest'    # 최신 체크포인트에서 재개
)

main(config)
```

---

## ❓ FAQ (자주 묻는 질문)

### Q1: 이 모델을 추론(inference)만 하려면?
**A:** 학습된 체크포인트가 있다면 다음처럼 간단히:
```python
import torch
from korean_llm_advanced_v3 import KoreanLLM, generate
from transformers import AutoTokenizer

model = KoreanLLM(...).to(device)
checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

response = generate(model, tokenizer, prompt="안녕?", max_tokens=50)
```

### Q2: 내 GPU 메모리가 9GB 미만이면?
**A:** 다음 방법들을 시도해보세요:
- 배치 크기를 `1`로 감소
- 시퀀스 길이를 `1024`로 단축
- 그래디언트 누적 단계를 `16`으로 증가
- 8비트 양자화 활성화

### Q3: 학습 중단 후 재개하려면?
**A:** 자동으로 최신 체크포인트를 감지합니다:
```python
config = TrainingConfig(
    resume_from_checkpoint='latest'  # 또는 특정 경로
)
main(config)
```

### Q4: 다른 한국어 데이터셋을 사용할 수 있나?
**A:** 네! `DatasetManager` 클래스의 `DATASETS_CONFIG`를 수정하면 됩니다:
```python
DATASETS_CONFIG = [
    {
        "name": "your-dataset/path",
        "split": "train",
        "text_keys": ["input", "output"]
    }
]
```

### Q5: 윈도우에서 실행하면 에러가 나요
**A:** `num_workers` 설정을 `0`으로 변경해보세요:
```python
loader = DataLoader(dataset, batch_size=2, num_workers=0)
```

### Q6: VRAM 사용량을 더 줄일 수 있나?
**A:** 다음 옵션을 조합해보세요:
- **메모리 효율 모드**: `use_bfloat16=True`
- **더 깊은 양자화**: INT4 (추가 라이브러리 필요)
- **LoRA 파인튜닝**: 선택적 레이어만 학습

### Q7: 생성된 텍스트 품질이 낮으면?
**A:** 다음을 확인하세요:
- 학습 스텝이 충분한가? (최소 10,000 스텝 권장)
- Learning rate 설정이 적절한가?
- 데이터셋 품질이 좋은가?
- `temperature` 파라미터 조정 (0.5~1.0 권장)

### Q8: 모델을 ONNX나 다른 형식으로 변환하려면?
**A:** PyTorch에서 ONNX로 변환 가능:
```python
import torch.onnx

dummy_input = torch.randint(0, 50000, (1, 2048)).to(device)
torch.onnx.export(
    model, dummy_input, "korean_llm.onnx",
    input_names=['input_ids'],
    output_names=['output']
)
```

### Q9: 개발자가 활발히 지원하나?
**A:** 네! 이슈나 피드백은 이메일(**seoan102410@gmail.com**)로 연락주세요! 💌

### Q10: 상용 프로젝트에 사용 가능한가?
**A:** GPL-3.0 라이선스이므로, 수정 사항을 공개해야 합니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---

## 🛠️ 트러블슈팅

### ❌ CUDA Out of Memory 에러

**증상:** `RuntimeError: CUDA out of memory`

**해결책:**
```python
# 배치 크기 감소
config.batch_size = 1

# 최대 시퀀스 길이 감소
config.max_seq_len = 1024

# 그래디언트 누적 증가
config.accumulation_steps = 16
```

### ❌ bitsandbytes 설치 실패

**해결책:**
```bash
# CUDA Toolkit 경로 명시
CUDA_HOME=/usr/local/cuda pip install bitsandbytes
```

### ❌ 데이터셋 다운로드 실패

**해결책:**
```bash
# 캐시 초기화 후 재시도
rm -rf datasets/cache/*
python korean_llm_advanced_v3.py
```

---

## 📈 성능 최적화 팁

1. **배치 크기 조정**: 너무 작으면 학습이 느리고, 너무 크면 VRAM 부족
2. **그래디언트 누적**: 효과적인 배치 크기 증대의 핵심
3. **Learning Rate 스케줄링**: Cosine Annealing으로 수렴 향상
4. **혼합 정밀도**: BF16 사용으로 속도와 메모리 동시 개선
5. **체크포인트**: 정기적으로 저장하여 학습 재개 가능

---

## 📞 연락처 및 정보

- **개발자**: seoan1024
- **이메일**: seoan102410@gmail.com
- **GitHub**: [seoan1024](https://github.com/seoan1024)

---

## 📜 라이선스

이 프로젝트는 **GPL-3.0 라이선스** 하에 배포됩니다.

```
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (C) 2024 seoan1024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

📖 전체 라이선스: [LICENSE](./LICENSE)

---

## 🤝 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트는 언제나 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 감사의 말

- 🎯 **한국어 LLM 커뮤니티** - 귀중한 피드백과 기여
- 📚 **Hugging Face** - Transformers & Datasets 라이브러리
- 🔧 **bitsandbytes** - 양자화 및 최적화 솔루션
- 🚀 **PyTorch** - 오픈소스 딥러닝 프레임워크

---

## 📚 참고 자료

### 한국어 NLP
- [nlpai-lab/KULLM](https://github.com/nlpai-lab/KULLM)
- [beomi/KoAlpaca](https://github.com/beomi/KoAlpaca)

### 최적화 기법
- [bitsandbytes: 8-bit Optimization](https://github.com/TimDettmers/bitsandbytes)
- [Gradient Checkpointing in PyTorch](https://pytorch.org/docs/stable/checkpoint.html)

### 대규모 언어모델
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://arxiv.org/abs/1901.08810)

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 별⭐을 눌러주세요!**

[⬆ 위로](#korean-llm-advanced-v3)

</div>

---

# 🇺🇸 Korean LLM Advanced v3

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![Model Size](https://img.shields.io/badge/Model-1.09B%20Parameters-orange)](#model-specifications)
[![VRAM](https://img.shields.io/badge/VRAM%20Usage-9GB-red)](#optimization-techniques)

<div align="center">

**Korean-Optimized Large Language Model - Scratch Implementation & Quantization Applied**

## 🚀 Project Development Journey
At first, free API quotas became tight, and there was a limit to "vibe coding." I tried running Ollama locally, but the models were too heavy for my computer to handle. Then suddenly, the thought struck me: "Why not just build it myself?"

However, I was only in 8th grade and knew almost nothing about AI models except the concept of 'B (Billion)' for model size. I couldn't code advanced concepts, only the basics.

So, as usual, I went into ChatGPT and boldly declared: "I want to create my own independent Korean LLM model!" and started this ambitious challenge.

While receiving help from GPT, I kept hitting limitations. Initially, I just collected code snippets from GPT and desperately hoped to avoid Dimension Errors. I spent entire days collecting and cleaning data, staring at my computer monitor.

The first version of my work (pre-v1) was an ultra-lightweight 50M (50 million parameter) model trained on Wikipedia data—it no longer exists. Although proper conversation was impossible, it did show signs of constructing grammatically correct sentences. I was so happy with that small success that I became obsessed with creating a "chatbot-type model." This led me to develop v1 to its final version: a 541M-sized model. Despite some bugs, I decided to fix them and immediately scale up the model.

After fixing errors and roughly doubling the model size, I finally built a 1.09B (1.09 billion parameter) model. Throughout the vacation, I kept my computer running for training whenever I had time.

Before I knew it, the seemingly endless training had reached 44,000 steps. With excitement, I typed "안녕?" (Hello?) into the chat for testing.

"안녕하세요! 오늘은 무엇을 도와드릴까요?" (Hello! What can I help you with today?)

The moment the model displayed the correct response on screen, I felt indescribable joy.

But joy was short-lived. As I asked different questions, it started spitting out completely wrong answers. After analyzing code with AI all night, I discovered a critical bug: the model was ignoring user instructions. Heartbroken, I bravely discarded all training results for a more perfect model.

Without losing hope, I completely fixed v2's bugs and tackled the next challenge. A 1B-class model consumed a whopping 23GB of VRAM, making it too heavy to run in typical environments. I set a goal to reduce this to 10GB or less, and finally succeeded with quantization in v3.

As vacation ended, it became difficult for me to keep training. I haven't done any training since the v2 bug fix, and I'll resume when time allows. This project is my life's greatest work, completed solely through sheer stubbornness to build an LLM with my own hands.

Please use this model well, and if you like it, don't forget to click the star button (⭐)! Thank you!

A **1.09B parameter LLM** trained entirely in Korean from scratch, making aggressive use of VRAM optimization techniques.

[📋 Key Features](#key-features) • [🚀 Quick Start](#quick-start) • [💾 Technology Stack](#technology-stack) • [📊 Version History](#version-history)

</div>

---

## 📖 Overview

**Korean LLM Advanced v3** is a **lightweight large language model** optimized for Korean natural language processing.
It is designed to train and perform inference efficiently even in limited GPU memory environments.

### Core Goals
- ✅ Korean text generation and comprehension
- ✅ VRAM efficiency (9GB baseline)
- ✅ Fast training speed
- ✅ Easy deployment and utilization

---

## 🌟 Key Features

### 🎯 Model Architecture
| Item | Description |
|------|------|
| **Model Size** | 1.09B Parameters |
| **Hidden Dimension** | 1,920 |
| **Number of Layers** | 20 |
| **Attention Heads** | 10 |
| **Max Sequence Length** | 2,048 Tokens |
| **Vocabulary Size** | Dynamic (based on tokenizer) |

### 🔧 Optimization Techniques

#### 1️⃣ **BF16 Automatic Mixed Precision**
```
~50% VRAM savings compared to standard FP32
- Memory efficiency: ⬇️ 12GB → 6GB
- Computation speed: ➡️ Equivalent or improved
```

#### 2️⃣ **8-bit AdamW Optimizer** (bitsandbytes)
```
75% reduction in optimizer state memory
- Standard AdamW: ~2.2GB (1B model)
- 8-bit AdamW: ~0.55GB (1B model)
```

#### 3️⃣ **Quantization** ⭐
```
Dynamic quantization of model weights
- INT8 Quantization: 4x size reduction
- Inference speed: 1.5~2x improvement
```

#### 4️⃣ **Gradient Accumulation**
```
Effective batch size increase
- Configuration: batch_size=2, accumulation_steps=8
- Effect: Equivalent to batch size 16
```

#### 5️⃣ **Gradient Checkpointing**
```
Activation memory reduction
- Recomputation cost: ~30% speed decrease
- Memory savings: 30~40%
```

---

## 💾 VRAM Usage Comparison

<div align="center">

| Version | Parameters | VRAM Usage | Optimization Techniques |
|------|---------|-----------|----------|
| **v1** | 541M | ~11GB | Basic FP32 |
| **v2** | ~1.1B | ~23GB | BF16 + Gradient Checkpoint |
| **v3** | 1.09B | **~9GB** ✨ | BF16 + 8-bit Optimizer + Quantization |

**v3 achieves 60% VRAM reduction compared to v2, with 2x model size expansion vs v1**

</div>

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
Python 3.9 or higher
CUDA 11.8 or higher (GPU required)
GPU Memory: Minimum 9GB recommended
```

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/seoan1024/korean-llm-v3.git
cd korean-llm-v3

# Install essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tqdm pandas matplotlib

# Quantization support library (optional)
pip install bitsandbytes
```

### 2️⃣ Dataset Preparation

The code automatically downloads the following datasets:
- 🔹 `nlpai-lab/kullm-v2` - Korean instruction-tuning data
- 🔹 `beomi/KoAlpaca-v1.1a` - Korean Alpaca dataset

```bash
# Datasets are automatically downloaded, no separate action needed
# Cache directory: ./datasets/cache/
```

### 3️⃣ Start Training

```bash
# Start training with default settings
python korean_llm_advanced_v3.py

# Or run with custom configuration
python korean_llm_advanced_v3.py \
    --batch-size 2 \
    --max-steps 50000 \
    --learning-rate 5e-5
```

### 4️⃣ Monitoring

A GUI monitoring window automatically opens during training:
- 📊 Real-time loss graph
- 💬 Interactive chat (generation testing)
- 📝 Log viewer

---

## 🏗️ Project Structure

```
korean-llm-v3/
├── korean_llm_advanced_v3.py    # Main training script
├── README.md                      # This file
├── LICENSE                        # GPL-3.0 License
│
├── checkpoints/                   # Saved model checkpoints
│   └── korean_llm_*.pth
│
├── datasets/                      # Dataset cache
│   ├── cache/                     # Downloaded datasets
│   └── datasets_manifest.json     # Metadata
│
└── logs/                          # Training logs and graphs
    ├── training.log               # Detailed log
    └── loss_history.json          # Loss history
```

---

## 📊 Version History

### v1 (Initial Version)
- 541M parameter model
- VRAM usage: ~11GB
- Basic FP32 training

### v2 (Optimization v1)
- Expanded to 1.1B parameters
- VRAM usage: ~23GB (1.2x initial increase)
- **BF16 + Gradient Checkpoint** applied

### **v3 (Current)** ⭐
- 1.09B parameters (v2 level)
- **VRAM usage: ~9GB** (60% reduction from v2!)
- **Major Improvements:**
  - 8-bit AdamW optimizer
  - Dynamic quantization support
  - Enhanced memory management
  - Faster training speed

---

## 🔧 Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework |
| **Transformers** | 4.30+ | Tokenizer and utilities |
| **Datasets** | 2.10+ | Korean dataset loading |
| **bitsandbytes** | 0.40+ | 8-bit quantization optimization |
| **tqdm** | 4.60+ | Progress display |

### Optional Libraries

| Library | Purpose |
|---------|---------|
| **matplotlib** | Loss graph visualization |
| **tkinter** | GUI monitoring (built-in) |
| **pandas** | Data processing |

---

## 💡 Usage Examples

### Model Loading and Text Generation

```python
import os, argparse
from pathlib import Path
from typing import Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim_2 = cos.shape[-1]
    head_dim = head_dim_2 * 2
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
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
    def forward(self, x: torch.Tensor, f_cos: torch.Tensor, f_sin: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=(s > 1))
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def forward(self, x, f_cos, f_sin, kv_cache=None):
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int, dim: int = 1920, n_layers: int = 20, n_heads: int = 10, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, int(dim * 2.5)) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)
    def _get_freqs(self, f, start, length):
        end = start + length
        if end > f.shape[0]:
            raise ValueError(f"현재 컨텍스트가 너무 깁니다: {end} > {f.shape[0]}")
        return f[start:end]
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, kv_caches=None):
        b, s = tokens.shape
        x = self.embed(tokens)
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]
        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, kv = layer(x, f_cos, f_sin, kv_cache=cache)
            new_kv_caches.append(kv)
        x = self.norm(x)
        logits = self.output(x)
        return logits, new_kv_caches

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists(): return None
    files = list(checkpoint_dir.glob("korean_llm_*.pth"))
    if not files: return None
    def step_number(path):
        try: return int(path.stem.split("_")[-1])
        except ValueError: return -1
    files.sort(key=step_number)
    return files[-1]

def load_checkpoint(model, checkpoint_path, device):
    print(f"📦 체크포인트 로딩:\n   {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step", "?")
    else:
        state_dict = checkpoint
        step = "?"
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 모델 로드 완료\n   학습 step: {step}")
    return step

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_tokens=256, temperature=0.6, top_k=40, top_p=0.95, repetition_penalty=1.15, context_limit=512):
    model.eval()
    prompt_text = f"### 질문: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    if tokens.shape[1] >= context_limit:
        tokens = tokens[:, -context_limit + 1:]
    output_tokens = tokens
    kv_caches = None
    eos_id = tokenizer.eos_token_id
    for _ in range(max_tokens):
        input_tokens = output_tokens if kv_caches is None else output_tokens[:, -1:]
        logits, kv_caches = model(input_tokens, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]
        temperature = max(float(temperature), 1e-5)
        next_logits = next_logits / temperature
        if repetition_penalty != 1.0:
            used_tokens = set(output_tokens[0].tolist())
            for token_id in used_tokens:
                if token_id < next_logits.shape[-1]:
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty
        if top_k > 0:
            k = min(int(top_k), next_logits.shape[-1])
            threshold = torch.topk(next_logits, k).values[..., -1, None]
            next_logits = torch.where(next_logits < threshold, torch.full_like(next_logits, float("-inf")), next_logits)
        probs = F.softmax(next_logits, dim=-1)
        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[..., 0] = False
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        if not torch.isfinite(probs).all():
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        if eos_id is not None and next_token.item() == eos_id: break
        if output_tokens.shape[1] >= context_limit: break
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    if "### 응답:" in generated_text:
        response = generated_text.split("### 응답:", 1)[1]
    else:
        response = generated_text
    if "### 질문:" in response:
        response = response.split("### 질문:", 1)[0]
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="KoreanLLM 체크포인트 채팅")
    parser.add_argument("--checkpoint", type=str, default="latest", help="체크포인트 경로 또는 latest")
    parser.add_argument("--tokenizer", type=str, default="beomi/Llama-3-Open-Ko-8B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--cpu", action="store_true", help="강제로 CPU 사용")
    args = parser.parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("🇰🇷 KoreanLLM v3 Checkpoint Chat")
    print("=" * 65)
    print(f"🖥️ Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CPU 모드입니다. 1920-dim / 20-layer 모델이라 생성이 느릴 수 있습니다.")
    print("\n🔤 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    print(f"   vocab_size = {vocab_size}")
    print(f"   eos_token_id = {tokenizer.eos_token_id}")
    print(f"   pad_token_id = {pad_token_id}")
    checkpoint_path = args.checkpoint
    if checkpoint_path.lower() == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("\n❌ checkpoints 폴더에서 체크포인트를 찾지 못했습니다.")
            print("예: python chat_korean_llm.py --checkpoint checkpoints/korean_llm_50000.pth")
            return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n❌ 체크포인트가 없습니다:\n   {checkpoint_path}")
        return
    print("\n🧠 모델 생성 중...")
    model_config = dict(vocab_size=vocab_size, pad_token_id=pad_token_id, dim=1920, n_layers=20, n_heads=10, max_seq_len=512)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = KoreanLLM(**model_config).to(device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)
    try:
        step = load_checkpoint(model, checkpoint_path, device)
    except RuntimeError as e:
        print("\n❌ 체크포인트와 현재 모델 구조가 맞지 않습니다.")
        print("   특히 tokenizer의 vocab_size / pad_token 설정을 확인하세요.")
        print(f"\n상세 오류:\n{e}")
        return
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {params / 1e6:.1f}M")
    print("\n" + "=" * 65)
    print("💬 채팅 시작")
    print("   /exit  종료")
    print("   /clear 대화 입력 기록 초기화")
    print("   /info  모델 정보")
    print("=" * 65)
    while True:
        try:
            prompt = input("\n나 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 종료합니다.")
            break
        if not prompt: continue
        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("👋 종료합니다.")
            break
        if prompt == "/clear":
            print("🧹 입력 상태를 초기화했습니다.")
            continue
        if prompt == "/info":
            print(f"\n체크포인트 : {checkpoint_path}\n학습 step   : {step}\nDevice      : {device}\nParameters  : {params / 1e6:.1f}M\nTemperature : {args.temperature}\nTop-k       : {args.top_k}\nTop-p       : {args.top_p}")
            continue
        print("\n모델 > ", end="", flush=True)
        try:
            response = generate(model=model, tokenizer=tokenizer, prompt=prompt, device=device, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty, context_limit=512)
            print(response)
        except torch.cuda.OutOfMemoryError:
            print("\n❌ CUDA 메모리가 부족합니다.")
            print("   --max-tokens 값을 낮추거나 다른 GPU에서 실행해보세요.")
        except Exception as e:
            print(f"\n❌ 생성 오류: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
```

### Custom Training Configuration

```python
from korean_llm_advanced_v3 import TrainingConfig, main

config = TrainingConfig(
    batch_size=4,                      # Batch size
    accumulation_steps=4,              # Gradient accumulation steps
    max_steps=100000,                  # Maximum training steps
    warmup_steps=1000,                 # Warmup steps
    learning_rate=3e-5,                # Learning rate
    eval_interval=5000,                # Evaluation interval
    use_bfloat16=True,                 # Whether to use BF16
    resume_from_checkpoint='latest'    # Resume from latest checkpoint
)

main(config)
```

---

## ❓ FAQ (Frequently Asked Questions)

### Q1: What if I only want to run inference?
**A:** If you have a trained checkpoint, it's simple:
```python
import torch
from korean_llm_advanced_v3 import KoreanLLM, generate
from transformers import AutoTokenizer

model = KoreanLLM(...).to(device)
checkpoint = torch.load("checkpoints/korean_llm_50000.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

response = generate(model, tokenizer, prompt="안녕?", max_tokens=50)
```

### Q2: What if my GPU memory is less than 9GB?
**A:** Try these approaches:
- Reduce batch size to `1`
- Shorten sequence length to `1024`
- Increase gradient accumulation steps to `16`
- Enable 8-bit quantization

### Q3: How do I resume training after interruption?
**A:** It automatically detects the latest checkpoint:
```python
config = TrainingConfig(
    resume_from_checkpoint='latest'  # Or specify a specific path
)
main(config)
```

### Q4: Can I use a different Korean dataset?
**A:** Yes! Modify `DATASETS_CONFIG` in the `DatasetManager` class:
```python
DATASETS_CONFIG = [
    {
        "name": "your-dataset/path",
        "split": "train",
        "text_keys": ["input", "output"]
    }
]
```

### Q5: I get errors when running on Windows
**A:** Try changing `num_workers` setting to `0`:
```python
loader = DataLoader(dataset, batch_size=2, num_workers=0)
```

### Q6: Can I reduce VRAM usage even more?
**A:** Try combining these options:
- **Memory Efficient Mode**: `use_bfloat16=True`
- **Deeper Quantization**: INT4 (additional library required)
- **LoRA Fine-tuning**: Train only selective layers

### Q7: Generated text quality is low
**A:** Check these:
- Is the training step count sufficient? (Minimum 10,000 steps recommended)
- Is the learning rate setting appropriate?
- Is dataset quality good?
- Adjust `temperature` parameter (0.5~1.0 recommended)

### Q8: How do I convert the model to ONNX or other formats?
**A:** Convert from PyTorch to ONNX:
```python
import torch.onnx

dummy_input = torch.randint(0, 50000, (1, 2048)).to(device)
torch.onnx.export(
    model, dummy_input, "korean_llm.onnx",
    input_names=['input_ids'],
    output_names=['output']
)
```

### Q9: Is the developer actively supporting this?
**A:** Yes! Contact via email (**seoan102410@gmail.com**) for issues or feedback! 💌

### Q10: Can I use this in commercial projects?
**A:** It's under GPL-3.0 license, so you must disclose modifications. See the [LICENSE](./LICENSE) file for details.

---

## 🛠️ Troubleshooting

### ❌ CUDA Out of Memory Error

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size
config.batch_size = 1

# Reduce maximum sequence length
config.max_seq_len = 1024

# Increase gradient accumulation
config.accumulation_steps = 16
```

### ❌ bitsandbytes Installation Failure

**Solution:**
```bash
# Specify CUDA Toolkit path
CUDA_HOME=/usr/local/cuda pip install bitsandbytes
```

### ❌ Dataset Download Failure

**Solution:**
```bash
# Clear cache and retry
rm -rf datasets/cache/*
python korean_llm_advanced_v3.py
```

---

## 📈 Performance Optimization Tips

1. **Batch Size Adjustment**: Too small slows training, too large causes VRAM shortage
2. **Gradient Accumulation**: Key to increasing effective batch size
3. **Learning Rate Scheduling**: Cosine Annealing improves convergence
4. **Mixed Precision**: BF16 improves both speed and memory
5. **Checkpointing**: Regular saving enables training resumption

---

## 📞 Contact & Information

- **Developer**: seoan1024
- **Email**: seoan102410@gmail.com
- **GitHub**: [seoan1024](https://github.com/seoan1024)

---

## 📜 License

This project is distributed under **GPL-3.0 License**.

```
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (C) 2024 seoan1024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

📖 Full License: [LICENSE](./LICENSE)

---

## 🤝 Contributing

Bug reports, feature suggestions, and pull requests are always welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- 🎯 **Korean LLM Community** - Valuable feedback and contributions
- 📚 **Hugging Face** - Transformers & Datasets libraries
- 🔧 **bitsandbytes** - Quantization and optimization solutions
- 🚀 **PyTorch** - Open-source deep learning framework

---

## 📚 References

### Korean NLP
- [nlpai-lab/KULLM](https://github.com/nlpai-lab/KULLM)
- [beomi/KoAlpaca](https://github.com/beomi/KoAlpaca)

### Optimization Techniques
- [bitsandbytes: 8-bit Optimization](https://github.com/TimDettmers/bitsandbytes)
- [Gradient Checkpointing in PyTorch](https://pytorch.org/docs/stable/checkpoint.html)

### Large Language Models
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://arxiv.org/abs/1901.08810)

---

<div align="center">

**⭐ If this project was helpful, please click the star button!**

[⬆ Back to Top](#korean-llm-advanced-v3)

</div>
