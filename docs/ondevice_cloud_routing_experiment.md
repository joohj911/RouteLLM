# On-device/Cloud LLM Routing — 사전 Router 기반 접근 초안 실험 보고서

## 목적

- On-device/Cloud LLM routing 관점에서, **accuracy를 최대한 유지하면서 weak(on-device) model 사용 비율을 얼마나 높일 수 있는지** 확인
- 핵심 질문: full strong(cloud) model 대비 성능 하락을 작게 유지하는 조건에서, 어느 정도의 요청을 weak model로 처리할 수 있는가
- 본 실험은 특정 routing 방법론의 **최종 성능 검증이 아님**
  - 사전 router(prompt 입력만으로 실행 전 모델을 선택하는 router) 기반 접근이 strong/weak routing 문제에서 어느 정도 가능성이 있는지 **초안 수준으로 확인**하는 것이 목적
  - 결과는 최종 성능이 아니라 추가 실험 필요성을 판단하기 위한 기준으로 활용

---

## 1. 관련 방법론 소개

### 1.1 RouteLLM

- Strong model과 weak model 사이에서 **prompt별로 어떤 모델을 사용할지 결정**하는 binary routing framework
- Query에 대해 strong model이 weak model보다 더 나은 응답을 낼 가능성(`P(strong wins | query)`)을 예측하고, 이 score를 기준으로 routing
- Threshold를 조절해 **strong model 호출 비율과 성능 사이의 trade-off**를 제어
- 논문에서는 여러 router 구조(Similarity-weighted ranking, Matrix Factorization, BERT classifier, Causal LLM classifier)를 비교

### 1.2 UniRoute

- 여러 LLM 후보 중 **비용과 성능을 함께 고려해 적절한 모델을 선택**하는 routing 관점에서 참고한 방법론
- 각 LLM을 validation set에서의 error 패턴으로 표현하고, prompt feature와 결합해 예상 error를 추정
- 본 실험에서는 cluster 기반 router(K-means cluster별 평균 error를 이용)를 비교 대상으로 사용

### 1.3 본 실험에서 사용한 Router (MF router)

- 본 실험의 핵심 router로 **RouteLLM의 Matrix Factorization(MF) router**를 사용
- MF router를 사용한 이유
  - RouteLLM 논문에서 **성능과 routing overhead 측면에서 가장 안정적인 대표 방법**으로 제시됨
  - Query embedding과 model embedding의 latent interaction만 학습하므로 **가벼우며, 사전 router로 적합**
  - 연속적인 routing score를 출력하므로 **threshold 조절로 weak/strong 비율을 유연하게 제어** 가능
- 비교를 위해 Random routing, UniRoute(cluster 기반)도 함께 평가

---

## 2. 실험 세팅

### 2.1 데이터 (BFCL)

- 평가 데이터로 **BFCL(Berkeley Function Calling Leaderboard) v4**의 single-turn 카테고리(non-live + live)를 사용
- BFCL은 query와 함께 **호출 가능한 function 정의**가 주어지고, 모델이 적절한 function call을 생성하는지를 채점하는 **function calling / tool calling 중심** 데이터
- 각 sample은 query(question)와 function 목록으로 구성되며, 모델의 출력이 정답 function call과 일치하는지로 pass/fail 판정

**BFCL 데이터 예시**

```json
{
  "id": "simple_python_0",
  "question": [[{"role": "user",
    "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]],
  "function": [{
    "name": "calculate_triangle_area",
    "description": "Calculate the area of a triangle given its base and height.",
    "parameters": {
      "type": "dict",
      "properties": {
        "base":   {"type": "integer", "description": "The base of the triangle."},
        "height": {"type": "integer", "description": "The height of the triangle."},
        "unit":   {"type": "string",  "description": "The unit of measure (defaults to 'units')."}
      },
      "required": ["base", "height"]
    }
  }]
}
```

- 위 예시처럼 모델은 query를 보고 **올바른 function(`calculate_triangle_area`) 선택 + argument(`base=10`, `height=5`) 추출**을 모두 정확히 수행해야 pass

### 2.2 모델 / 임베딩 / Router 설정

- **모델 쌍 (strong vs weak)** — 두 조합을 평가
  - Pair A: `Qwen3.5-0.8B` (weak) vs `Qwen3.5-9B` (strong)
  - Pair B: `Qwen3.5-2B` (weak) vs `Qwen3.5-9B` (strong)
- **Embedding model**: `multilingual-e5-small`
  - 기존에 사용하던 embedding(STE)과 **동일한 크기**이면서 **MTEB 성능이 유사**
  - RouteLLM MF router가 사용하는 query embedding을 **OpenAI embedding 대신 multilingual-e5-small로 대체**하여 실험 (로컬에서 동작, 별도 API 불필요)
- **Router**: MF router(주), Random / UniRoute(비교)
- **실험 방식**: **threshold를 조절**하면서 weak model 사용 비율과 전체 성능 변화를 함께 확인

---

## 3. 평가 방식

- 각 sample에 대해 router가 **strong 또는 weak 중 하나의 모델을 선택**하고, 선택된 모델의 pass/fail 결과로 전체 성능(pass rate)을 계산
- **full strong model(모든 요청을 strong에 보냄)의 성능을 기준선**으로 삼고, router 적용 후 성능이 얼마나 유지되는지 확인
  - 본 실험 기준선: full strong = **81.62%**
- **핵심 평가 포인트**: full strong 대비 성능 하락이 **1% 이내(pass rate ≥ 80.62%)**인 조건에서, weak model을 **몇 %까지 사용**할 수 있었는지
- threshold별 모든 세부 수치를 나열하기보다, graph와 핵심 결과(1% 이내 weak 사용 비율) 중심으로 해석

---

## 4. 실험 결과

### 4.1 Threshold 및 graph 해석 가이드

- Graph는 **x축 = strong model 호출 비율(%)**, **y축 = 전체 pass rate(%)**로, threshold를 바꿔가며 측정한 곡선
- threshold는 **weak 사용 비율과 전체 성능 사이의 trade-off를 확인하기 위한 기준**으로 사용
  - threshold가 높을수록 weak로 많이 보내고(=strong 호출 비율 낮음), 낮을수록 strong으로 많이 보냄
  - 곡선을 따라 오른쪽(strong 비율 ↑)으로 갈수록 pass rate가 full strong(81.62%)에 가까워짐
- 곡선이 **왼쪽(weak 비율이 높은 구간)에서도 높은 pass rate를 유지**할수록, 성능 손실 없이 weak를 더 많이 쓸 수 있다는 의미
- 좋은 router일수록 동일한 strong 호출 비율에서 **Random 곡선보다 위**에 위치

```
[Graph 삽입 위치]
routing_curves.png — Pair A / Pair B 각각에 대해
x축: Strong Model Calls (%), y축: Pass Rate (%)
곡선: MF / UniRoute (+ Random), 기준선: weak-only / strong-only(81.62%)
```

### 4.2 주요 결과

- 곡선 전반에서 **MF와 UniRoute는 Random 대비 위쪽**에 위치 → 사전 router가 무작위보다 의미 있는 routing 신호를 학습함을 확인
- 다만 **full strong 대비 1% 이내(≥ 80.62%)**라는 엄격한 조건에서 weak로 보낼 수 있는 비율은 아래와 같이 제한적

| 모델 쌍 | Random | MF router | UniRoute |
|---|---|---|---|
| Pair A (0.8B vs 9B) | ~0% | **약 20%** | 약 19% |
| Pair B (2B vs 9B) | ~10% | **약 10%** | 약 39% |

- **Pair A**: MF router는 weak를 **약 20%**까지 사용해도 성능 하락 1% 이내(weak 20% 지점 pass 80.66%), Random은 사실상 0%
- **Pair B**: MF router는 **약 10%**(weak 10% 지점 pass 80.8%)로 Random과 큰 차이가 없었고, UniRoute는 **약 39%**까지 사용 가능(weak 38.82% 지점 pass 80.8%)
- 종합하면, 사전 router로 **무작위보다 분명히 나은 routing은 가능**하지만, 1% 이내라는 조건에서 MF router의 weak 사용 비율은 **약 10–20% 수준으로 낮은 편**

### 4.3 결과 분석

- weak model 사용 가능 비율이 낮게 나온 데에는 다음과 같은 원인이 작용했을 가능성이 있음 (단정이 아닌 추정)

- **데이터 특성: function calling 중심의 난이도**
  - BFCL은 단순 질의응답이 아니라 **function calling / tool calling** 중심이라, weak model이 안정적으로 처리하기 어려운 경향
  - **tool selection + argument extraction + value grounding**이 모두 정확해야 pass로 인정 → weak model의 작은 오류 하나도 전체 성능 하락으로 직결
  - 즉 "weak가 안전하게 처리 가능한 쉬운 sample"의 비중 자체가 일반 QA보다 작을 수 있음

- **데이터 수 부족: safe subset 학습의 한계**
  - BFCL 데이터 수가 충분하지 않아, router가 **weak model이 성공 가능한 safe subset을 충분히 학습**하지 못했을 가능성
  - 특히 소형 카테고리의 sample 수가 적어, weak가 잘 처리하는 영역과 그렇지 않은 영역의 경계를 세밀하게 구분하기 어려웠을 수 있음

- **방법론 간 차이**
  - Pair B에서 UniRoute가 MF보다 높은 weak 사용 비율을 보인 점은, 동일 데이터에서도 **router 구조에 따라 weak safe subset을 구분하는 정도가 다를 수 있음**을 시사

- **해석 범위**
  - 본 실험은 사전 router 기반 접근의 **초안 test 성격**
  - 위 수치를 최종 성능으로 해석하기보다는, **추가 실험(데이터 보강, router 구조/세팅 비교)의 필요성을 확인한 결과**로 정리하는 것이 적절

---

## 5. 추가 진행 계획

### 5.1 NVIDIA 공개 proxy-based router 확인

- `NVIDIA-AI-Blueprints/llm-router` 구조 확인
- 해당 router가 **prompt classification 기반으로 LLM을 선택하는 proxy 구조**인지 확인
- 현재 on-device/cloud routing task에 **바로 적용 가능한지**, 또는 **custom policy 학습이 필요한지** 검토

### 5.2 BFCL 데이터 augmentation 후 router 성능 재확인

- 현재 BFCL 데이터 수가 충분하지 않아 router 학습에 한계가 있을 수 있음
- BFCL 데이터를 **augmentation하여 학습 데이터를 늘린 뒤** MF router 성능을 다시 확인
- 특히 **weak model이 안정적으로 처리 가능한 safe subset**을 더 잘 구분할 수 있는지 확인
