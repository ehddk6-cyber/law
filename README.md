# Legal QA

LAW OPEN DATA 코퍼스를 기반으로 동작하는 법률 검색·시험문제 평가·근거 기반 답변 시스템이다.

현재 핵심 기능:
- exact/hybrid retrieval
- 전 법영역 grounded answer
- 시험문제형 evaluator
  - OX
  - 4지선다
  - `ㄱ/ㄴ/ㄷ` 조합형
  - 개수형
- 멀티 provider fallback
  - `openrouter`
  - `ollama-minimax`
  - `groq`
  - `glm`
  - `codex`
  - `claude-code`
  - `ollama-qwen`
  - `claude-stepfree`

## Runtime

기본 런타임:
- provider: `openrouter`
- model: `stepfun/step-3.5-flash:free`
- fallback: `openrouter -> ollama-minimax -> groq -> glm -> codex -> claude-code -> ollama-qwen -> claude-stepfree`

관련 설정 파일:
- [settings/provider.json](/home/ehddk/ai/law_open_data/law_open_data/settings/provider.json)

## Quick Start

```bash
cd /home/ehddk/ai/law_open_data/law_open_data
python3 qa/answer_cli.py --query "행정심판법 제18조"
```

권장 CLI:

```bash
cd /home/ehddk/ai/law_open_data/law_open_data
./legalqa answer "행정심판법 제18조"
```

코딩 에이전트용 JSON 출력:

```bash
./legalqa --format json answer "행정심판법 제18조"
./legalqa --format json search "2004헌마275"
./legalqa --format json health
./legalqa --format json schema
./legalqa --format json eval
```

LAW OPEN DATA 근거만 쓰는 judge layer:

```bash
./legalqa --format json answer "행정심판법 제18조" \
  --response-mode grounded \
  --judge-agent codex \
  --judge-mode single

./legalqa --format json answer "다음 중 틀린 것의 개수는?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
ㄹ. 84누180
1) 1개
2) 2개
3) 3개
4) 4개" \
  --response-mode grounded \
  --judge-agent codex \
  --judge-critic-agent qwen \
  --judge-mode debate
```

grounded만 강제:

```bash
python3 qa/answer_cli.py --query "행정심판법 제18조" --response-mode grounded
```

provider를 직접 지정:

```bash
python3 qa/answer_cli.py --query "행정심판법 제18조" --llm-provider openrouter --llm-model "stepfun/step-3.5-flash:free"
python3 qa/answer_cli.py --query "행정심판법 제18조" --llm-provider ollama-minimax
python3 qa/answer_cli.py --query "행정심판법 제18조" --llm-provider ollama-qwen
./legalqa answer "행정심판법 제18조" --llm-provider openrouter --llm-model "stepfun/step-3.5-flash:free"
```

## CLI

`legalqa`는 사람용 텍스트 출력과 에이전트용 JSON 출력을 둘 다 지원한다.

서브커맨드:
- `answer`: grounded answer 생성
- `search`: retrieval만 수행
- `health`: 런타임/에이전트 상태 확인
- `schema`: 안정된 결과 스키마 출력
- `eval`: 회귀 평가 실행

judge 관련:
- `--judge-agent`: grounded packet을 검토할 solver agent
- `--judge-critic-agent`: solver 결과를 재검토할 critic agent
- `--judge-mode single|debate`: 단일 judge 또는 solver/critic 2단계
- 지원 judge agent: `codex`, `claude-code`, `claude-stepfree`, `qwen`, `gemini`

예시:

```bash
./legalqa answer "다음 중 틀린 것의 개수는?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
ㄹ. 84누180
1) 1개
2) 2개
3) 3개
4) 4개"

./legalqa search "05-0096"
./legalqa health
./legalqa schema
./legalqa eval --strict
```

## HTTP API

서버 실행:

```bash
cd /home/ehddk/ai/law_open_data/law_open_data
python3 qa/http_api.py
```

브라우저 UI:

```bash
http://127.0.0.1:8765/app
```

헬스체크:

```bash
curl http://127.0.0.1:8765/health
```

응답 스키마:

```bash
curl http://127.0.0.1:8765/schema
```

질의:

```bash
curl "http://127.0.0.1:8765/answer?query=행정심판법%20제18조"
curl -X POST "http://127.0.0.1:8765/answer" \
  -H "Content-Type: application/json" \
  -d '{"query":"다음 중 틀린 것의 개수는?\nㄱ. 행정심판법 제18조\nㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.\nㄷ. 2004헌마275\nㄹ. 84누180\n1) 1개\n2) 2개\n3) 3개\n4) 4개"}'
```

## Response Shape

`/answer`의 `result`는 다음 핵심 필드를 가진다.

- `schema_version`
- `query`
- `scope`
- `question_type`
- `route`
- `status`
- `grounded`
- `answer`
- `evidence`
- `citations`
- `warnings`
- `source_type`
- `doc_id`
- `response_mode`
- `final_answer_source`

상세 스키마는 [qa/response_schema.py](/home/ehddk/ai/law_open_data/law_open_data/qa/response_schema.py)와 `/schema`를 보면 된다.

## Evaluation

grounded 회귀 평가:

```bash
cd /home/ehddk/ai/law_open_data/law_open_data
python3 qa/eval_report.py --response-mode grounded --out .artifacts/eval/grounded.json
```

현재 기본 회귀셋은 다음을 포함한다.
- exact source lookup
- review-only fallback
- OX
- 부정문
- 틀린 것의 개수

관련 파일:
- [qa/eval_report.py](/home/ehddk/ai/law_open_data/law_open_data/qa/eval_report.py)
- [qa/self_check_answer.py](/home/ehddk/ai/law_open_data/law_open_data/qa/self_check_answer.py)
- [qa/self_check_providers.py](/home/ehddk/ai/law_open_data/law_open_data/qa/self_check_providers.py)

## Provider Notes

- `openrouter`: 현재 주력 경로
- `ollama-minimax`: 현재 가장 안정적인 로컬/클라우드 fallback
- `ollama-qwen`: 느리고 품질 편차가 있어 후순위
- `groq`: 현재 403
- `glm`: 현재 모델/요청 조건 이슈로 400
- `gemini`: 현재 quota 제한
- `claude-stepfree`: 연결은 됐지만 `empty_result`가 발생할 수 있음

## Artifacts

주요 산출물:
- [qa](/home/ehddk/ai/law_open_data/law_open_data/qa)
- [providers](/home/ehddk/ai/law_open_data/law_open_data/providers)
- [AGENT_RUNTIME_BENCHMARK.md](/home/ehddk/ai/law_open_data/law_open_data/AGENT_RUNTIME_BENCHMARK.md)
- [.artifacts](/home/ehddk/ai/law_open_data/law_open_data/.artifacts)

## Current Status

완료:
- 코퍼스 정규화
- 통합 검색
- grounded evaluator
- 시험문제 규칙층
- 멀티 provider fallback
- 사람용/에이전트용 공용 CLI
- HTTP API
- 기본 UI
- 응답 스키마
- 회귀 평가 리포트

남은 일:
- provider별 세부 장애 대응
- `llm_preferred` 회귀셋 확장
