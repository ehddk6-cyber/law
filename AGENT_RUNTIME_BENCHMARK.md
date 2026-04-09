# Agent Runtime Benchmark

`Q00/ouroboros`를 벤치마킹해서, 현재 Legal QA 프로젝트에서 `Codex CLI`, `Claude Code`, `Qwen CLI`, `Gemini CLI` 같은 외부 에이전트 CLI를 내부 대화형 법률시험문제 답변 레이어로 쓰는 방식을 정리한 문서다.

## 1. Ouroboros에서 배울 핵심

### 1. 런타임과 워크플로우를 분리한다

Ouroboros의 핵심은 `Claude Code`, `Codex CLI` 자체가 아니라 그 위에 있는 공통 하네스다.

- 공통 단계: `interview -> seed -> execute -> evaluate -> evolve`
- 런타임은 교체 가능
- 동일한 명세와 평가 계약을 다른 런타임에 태울 수 있음

우리 프로젝트에도 그대로 필요하다.

- 검색/근거/시험문제 판정은 공통 엔진
- 최종 자연어 정리나 반대 검토는 외부 CLI 런타임
- 어떤 에이전트를 써도 입출력 계약은 동일해야 함

### 2. 세션형 런타임을 표준화한다

Ouroboros는 `Codex CLI`나 `Claude Code`를 단순 subprocess가 아니라 세션형 runtime처럼 다룬다.

우리도 단순 `run_agent(prompt)`를 넘어서 아래 단위를 갖는 게 맞다.

- `prepare_prompt()`
- `run_once()`
- `resume_if_supported()`
- `probe_health()`
- `normalize_output()`

### 3. 사람용 인터페이스와 에이전트용 인터페이스를 분리하되 결과 객체는 하나로 유지한다

Ouroboros는 `ooo` 스킬과 터미널 CLI가 분리되어도 내부 결과 계약은 유지한다.

우리도 같은 방향이 맞다.

- 사람: `./legalqa answer "..."`
- 에이전트: `./legalqa --format json answer "..."`
- 내부 결과는 항상 `AnswerPacket.to_dict()`

### 4. 런타임별 강점을 역할로 분리한다

Ouroboros 문서는 runtime 차이를 숨기지 않고 드러낸다.

- Codex: 강한 코딩/다단계 작업
- Claude Code: 풍부한 도구와 세션 연속성

우리도 모델별 역할을 분리해야 한다.

- `grounded retrieval`: 로컬 공통 엔진
- `explanation`: OpenRouter / Gemini / Qwen
- `contrarian review`: Codex / Claude Code
- `tie-break`: 다른 provider 한 번 더

### 5. 평가를 런타임 밖에 둔다

Ouroboros의 가장 좋은 점 중 하나는 평가 게이트가 런타임 외부에 있다는 점이다.

우리도 답변 품질 판단을 모델에게 맡기지 말고, 외부 엔진이 먼저 잡아야 한다.

- exact grounding 여부
- 시험문제 선택지별 supported / unsupported / indeterminate
- final answer source
- warnings

### 6. “명세 우선”을 법률 QA에 맞게 바꾸면 “근거 우선”이다

Ouroboros는 `seed`가 중심이다. 우리 프로젝트에서는 그 자리를 `grounded packet`이 대신한다.

즉 외부 에이전트는 빈 프롬프트에서 시작하면 안 된다.

반드시 아래를 먼저 받아야 한다.

- 원질문
- route
- exact/fts/vector 결과
- top citations
- option_reviews
- exam_assessment
- 경고 문구

이렇게 해야 CLI 에이전트가 “모델 환각”이 아니라 “근거를 정리하는 후처리기”가 된다.

## 2. 현재 프로젝트에 맞는 내부 대화형 구조

지금 코드 기준 핵심은 이미 있다.

- grounded retrieval: `qa/unified.py`
- grounded answer/evaluator: `qa/answering.py`
- provider abstraction: `providers/`
- agent execution: `qa/agent_exec.py`
- human/agent CLI: `qa/legalqa_cli.py`, `legalqa`

여기서 외부 CLI 에이전트가 내부적으로 대화하듯 답하게 하려면, 단일 호출보다 `2단계 내부 대화`가 맞다.

### 권장 2단계 구조

#### 단계 1. Solver

입력:

- 사용자 질문
- grounded evidence
- citations
- option reviews
- exam assessment

출력:

- 임시 답변
- 각 보기별 판단 이유
- 남는 불확실성

#### 단계 2. Critic

입력:

- 같은 grounded evidence
- solver 답변

출력:

- 근거 이탈 여부
- 과도한 추론 여부
- unsupported를 supported처럼 말했는지
- 최종 보정 답변

즉 “내부 대화”는 자유 채팅이 아니라 아래처럼 고정된 구조가 되어야 한다.

```text
grounded engine
  -> solver agent
  -> critic agent
  -> final normalizer
```

## 3. 각 CLI를 어디에 쓰는 게 맞는가

### Codex CLI

가장 적합한 역할:

- solver
- 구조화된 최종 답변기
- 긴 시험문제 보기 정리

강점:

- 긴 컨텍스트 정리
- 도구 호출과 파일 참조에 강함
- 현재 환경에서 이미 실동작 검증됨

약점:

- 속도/비용이 항상 최저는 아님

### Claude Code

가장 적합한 역할:

- critic
- 반대 검토
- 근거 누락 탐지

강점:

- 세션형 작업에 적합
- 도구 연동이 좋음

약점:

- 현재 환경은 로그인 필요

### Qwen CLI

가장 적합한 역할:

- 저비용 초안 생성
- fallback solver

강점:

- 로컬/클라우드 운용 자유도
- 비용 효율

약점:

- 법률 조문 미세표현 정밀도 편차
- 지금은 `ollama-qwen` 결과 품질 편차가 있어 후순위가 맞음

### Gemini CLI

가장 적합한 역할:

- 빠른 재서술
- 보조 critic

강점:

- 빠른 응답

약점:

- quota 제약
- 현재 환경에서 안정 운영 경로 아님

## 4. 우리 프로젝트에 권장하는 실제 오케스트레이션

### 운영 기본

```text
question
  -> grounded retrieval
  -> exam evaluator
  -> if grounded exact:
       answer directly
     else:
       solver runtime
       -> critic runtime
       -> final answer
```

### 추천 런타임 체인

#### 안전 우선

```text
grounded direct
  -> openrouter stepfun
  -> codex
  -> ollama-minimax
```

#### 대화형 2에이전트 우선

```text
solver: codex
critic: claude-code
fallback critic: openrouter
fallback solver: ollama-minimax
```

#### 비용 우선

```text
solver: openrouter stepfun
critic: ollama-minimax
fallback: codex
```

## 5. 구현 원칙

### 1. 외부 CLI는 “생성기”이지 “사실 판정기”가 아니다

사실 판정은 항상 `GroundedAnswerer`가 먼저 한다.

### 2. 에이전트는 raw question만 받으면 안 된다

반드시 `grounded packet`을 넣어야 한다.

### 3. unsupported / indeterminate를 억지로 정답화하면 안 된다

비정확한 자동 채점보다 보수적 보류가 낫다.

### 4. 에이전트별 출력은 공통 JSON으로 정규화해야 한다

최종적으로는 아래 형태로 모아야 한다.

```json
{
  "solver": "...",
  "critic": "...",
  "final": "...",
  "grounded": true,
  "citations": ["..."],
  "status": "candidate_answer"
}
```

## 6. 우리 프로젝트에서 바로 추가할 다음 단계

우선순위는 이 순서가 맞다.

1. `agent conversation mode` 추가
2. `solver/critic` 2단계 실행기 추가
3. `legalqa answer --mode debate` 지원
4. debate 결과도 기존 `AnswerPacket` 안으로 수용
5. 회귀 평가에 `debate mode` 케이스 추가

## 7. 결론

Ouroboros를 벤치마킹해서 가져와야 하는 것은 “복잡한 철학”이 아니라 아래 3개다.

- 런타임 분리
- 공통 명세/평가 계약
- 세션형 에이전트 오케스트레이션

우리 프로젝트에선 이것이 곧 아래 구조다.

```text
LAW OPEN DATA grounding
  -> 시험문제 판정 엔진
  -> solver CLI agent
  -> critic CLI agent
  -> 최종 답변
```

즉 `Codex CLI`, `Claude Code`, `Qwen CLI`, `Gemini CLI`는 정답의 근거를 만드는 엔진이 아니라, 이미 확보된 근거를 바탕으로 더 잘 설명하고 검토하는 “내부 대화형 실행기”로 쓰는 것이 맞다.
