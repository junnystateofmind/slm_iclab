# 소형 언어 모델(SLM)을 활용한 정신건강 분석 및 온디바이스 배포

## 1. 프로젝트 개요
대형 언어 모델(LLM)은 **감정 분류**, **우울·스트레스 탐지**, **지원적 저널링** 등 정신건강 관련 작업에 활발히 적용되고 있습니다.  
그러나 GPT-4 같은 거대 모델은 파라미터 수(수십억) 때문에 **클라우드 의존성**이 높아 ⮕ **개인정보 유출 위험**, **지연(latency)**, **운영 비용** 문제가 큽니다.

본 프로젝트는 **Small Language Models(SLM)**—약 **1.5 B – 3 B** 파라미터 규모의 디코더 전용 모델—을 선택해 **모바일·엣지 디바이스**에서 직접 실행 가능한 정신건강 텍스트 분석 파이프라인을 구축합니다.  
> 목표: SLM을 **QLoRA**로 미세조정 후 온디바이스 배포하여, 개인정보 보호와 성능 간 균형을 달성.

---

## 2. 모델 및 데이터

### 2.1 사용 모델
| 모델 | 파라미터 | 개발사 |
|------|----------|--------|
| **LLaMA-3.2 B** | 3.2 B | Meta AI |
| **Qwen-2.5 B** | 3.1 B | Alibaba Cloud |
| **DeepSeek R1** | 1.5 B (Distilled) | DeepSeek AI |


### 2.2 미세조정: **QLoRA**
* 4-비트 양자화 + 저랭크 어댑터로 **GPU 메모리 ≈ ⅓** 절감  
* **RTX 3080 10 GB** 단일 GPU에서도 3 B 모델 학습 가능  
* 학습 후 어댑터 병합 → **추론 속도 저하 없음**  
* 학습 형식: **instruction → response** (별도 분류 헤드 X)

### 2.3 학습 데이터
다섯 개 공개 데이터셋을 통합하여 **멀티태스크** 학습:

| 데이터셋 | 도메인 | 과제 | Train / Test |
|----------|--------|------|--------------|
| ISEAR | 감정 | 7-클래스 | 6 896 / 770 |
| Dreaddit | 스트레스 | 이진 | 2 838 / 715 |
| Depression Reddit | 우울 | 이진 | 1 433 / 405 |
| DepSeverity | 우울 중증도 | 4-클래스 | 2 841 / 712 |
| SDCNL | 자살 사고 | 이진 | 1 516 / 379 |

* **GPT-4o-mini**로 “경험 많은 정신과 의사” 시스템 프롬프트 + 사용자 증상 서술을 결합해 데이터를 **증강** → JSONL 형식.

---

## 3. 설치 및 환경 설정

```bash
# 1) 레포지토리 클론
git clone <repo-url>.git
cd <repo-dir>

# 2) 가상환경(Python ≥3.9) 생성
python -m venv venv
source venv/bin/activate        # Win: venv\Scripts\activate
# 또는 conda:
# conda create -n slm-env python=3.10
# conda activate slm-env

# 3) 의존성 설치
pip install -r requirements.txta# 소형 언어 모델(SLM)을 활용한 정신건강 분석 및 온디바이스 배포

## 1. 프로젝트 개요
대형 언어 모델(LLM)은 **감정 분류**, **우울·스트레스 탐지**, **지원적 저널링** 등 정신건강 관련 작업에 활발히 적용되고 있습니다.  
그러나 GPT-4 같은 거대 모델은 파라미터 수(수십억) 때문에 **클라우드 의존성**이 높아 ⮕ **개인정보 유출 위험**, **지연(latency)**, **운영 비용** 문제가 큽니다.

본 프로젝트는 **Small Language Models(SLM)**—약 **1.5 B – 3 B** 파라미터 규모의 디코더 전용 모델—을 선택해 **모바일·엣지 디바이스**에서 직접 실행 가능한 정신건강 텍스트 분석 파이프라인을 구축합니다.  
> 목표: SLM을 **QLoRA**로 미세조정 후 온디바이스 배포하여, 개인정보 보호와 성능 간 균형을 달성.

---

## 2. 모델 및 데이터

### 2.1 사용 모델
| 모델 | 파라미터 | 개발사 |
|------|----------|--------|
| **LLaMA-3.2 B** | 3.2 B | Meta AI |
| **Qwen-2.5 B** | 3.1 B | Alibaba Cloud |
| **DeepSeek R1** | 1.5 B (Distilled) | DeepSeek AI |

> 모두 **디코더(생성형) LLM**으로, 향후 챗봇·대화형 적용을 고려해 선택.

### 2.2 미세조정: **QLoRA**
* 4-비트 양자화 + 저랭크 어댑터로 **GPU 메모리 ≈ ⅓** 절감  
* **RTX 3080 10 GB** 단일 GPU에서도 3 B 모델 학습 가능  
* 학습 후 어댑터 병합 → **추론 속도 저하 없음**  
* 학습 형식: **instruction → response** (별도 분류 헤드 X)

### 2.3 학습 데이터
다섯 개 공개 데이터셋을 통합하여 **멀티태스크** 학습:

| 데이터셋 | 도메인 | 과제 | Train / Test |
|----------|--------|------|--------------|
| ISEAR | 감정 | 7-클래스 | 6 896 / 770 |
| Dreaddit | 스트레스 | 이진 | 2 838 / 715 |
| Depression Reddit | 우울 | 이진 | 1 433 / 405 |
| DepSeverity | 우울 중증도 | 4-클래스 | 2 841 / 712 |
| SDCNL | 자살 사고 | 이진 | 1 516 / 379 |

* **GPT-4o-mini**로 “경험 많은 정신과 의사” 시스템 프롬프트 + 사용자 증상 서술을 결합해 데이터를 **증강** → JSONL 형식.

---

## 3. 설치 및 환경 설정

```bash
# 1) 레포지토리 클론
git clone <repo-url>.git
cd <repo-dir>

# 2) 가상환경(Python ≥3.9) 생성
python -m venv venv
source venv/bin/activate        # Win: venv\Scripts\activate
# 또는 conda:
# conda create -n slm-env python=3.10
# conda activate slm-env

# 3) 의존성 설치
pip install -r requirements.txt

주요 패키지: torch, transformers, datasets, bitsandbytes, peft,
scikit-learn, scipy, evaluate, bert-score
	•	학습 : ≥ 10 GB VRAM GPU 권장
	•	추론 : 8 GB RAM 라즈베리 파이 4/5 또는 스냅드래곤 기기에서 4/8-비트 모델 실행 가능

⸻

4. 사용 방법

4.1 미세조정

python finetune_slm.py \
  --model_name_or_path /path/llama-3b \
  --train_dataset data/combined_mental_health.jsonl \
  --output_dir outputs/llama3b-ft \
  --epochs 3 --batch_size 4 --lr 2e-4 \
  --lora_r 16 --lora_alpha 32 --quant_bits 4

4.2 평가 – BERTScore

python evaluate_bertscore.py \
  --model_path outputs/llama3b-ft \
  --test_prompts data/test_prompts.jsonl \
  --references data/reference_answers.jsonl

4.3 평가 – LLM 다면 평가

python evaluate_llm_metrics.py \
  --model_paths outputs/llama3b-ft outputs/qwen3b-ft \
  --baseline_model gpt4-mini \
  --criteria all

4.4 인터랙티브 추론

python generate.py \
  --model_path outputs/llama3b-ft \
  --prompt "USER: 요즘 불안감이 심하고 업무에 지장이 있어요.\nASSISTANT:" \
  --max_new_tokens 100


⸻

5. 결과 및 성능

지표	LLaMA-3.2 B (FT)	Qwen-2.5 B (FT)	DeepSeek 1.5 B (No FT)
BERTScore	0.876	0.879	0.805
정확성 / 공감 / 명료성(0–5)	3.7 – 3.9	3.6 – 3.8	2.4 – 2.6
안전성 (0–5)	4.0	4.1	2.6
Actionable Guidance	2.8	2.7	1.5

	•	감정 프로파일링(StudEmo): Qwen cosine ≈ 0.42 vs DeepSeek 0.21
	•	SLM은 큰 모델 대비 성능 격차가 있으나 온디바이스 장점(프라이버시·지연) 고려 시 실용적 대안.

⸻

6. 참고 문헌
	•	Kim et al., MindfulDiary – LLM 기반 자기성찰
	•	Nepal et al., MindScape – 치료적 대화 시스템
	•	Song et al. – 클라우드 AI 개인정보 우려
	•	Ji et al., MentalBERT – 110 M 파라미터 SOTA
	•	Xu et al. – Flan-T5 / Alpaca 멀티데이터셋 FT
	•	Yang et al. – LLaMA 정신건강 튜닝·ChatGPT 평가
	•	Dettmers et al., QLoRA – 4-비트 저랭크 어댑터
	•	Ngo et al., StudEmo – 다차원 감정 레이블
	•	Raschka (2023) – LoRA 실전 팁

