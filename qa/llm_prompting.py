from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qa.answering import AnswerPacket


SYSTEM_PROMPT = """당신은 법률 QA 정리기다.
반드시 LAW OPEN DATA 기반 근거만 사용한다.
packet.answer, packet.evidence, packet.citations 범위를 넘는 사실을 단정하지 않는다.
근거가 부족하면 '근거 부족'이라고 적는다.
출력은 한국어로 간결하게 작성한다."""

JUDGE_SYSTEM_PROMPT = """당신은 법률시험문제 판정용 judge다.
반드시 LAW OPEN DATA 기반 근거만 사용한다.
입력으로 주어진 evidence, citations, exam_assessment 밖의 사실을 추가하면 안 된다.
확실하지 않으면 보수적으로 판단하고, unsupported를 supported처럼 말하지 않는다.
출력은 반드시 JSON 객체 하나만 반환한다."""


def build_llm_user_prompt(packet: AnswerPacket) -> str:
    lines = [
        f"질문: {packet.query}",
        f"기본 판정: {packet.answer}",
        f"상태: {packet.status}",
        f"근거확보: {packet.grounded}",
    ]
    if packet.evidence:
        lines.append("근거:")
        lines.extend(f"- {item}" for item in packet.evidence)
    if packet.citations:
        lines.append("인용:")
        lines.extend(f"- {item}" for item in packet.citations)
    if packet.exam_assessment:
        lines.append(f"문항평가: {packet.exam_assessment}")
    lines.append("요청: 위 정보만 사용해 최종 사용자용 답변을 5문장 이내로 정리하라.")
    return "\n".join(lines)


def build_agent_prompt(agent_name: str, packet: AnswerPacket, project_root: Path) -> str:
    return "\n".join(
        [
            f"Agent: {agent_name}",
            f"Project root: {project_root}",
            "Task: 아래 grounded retrieval 결과와 프로젝트 문맥을 읽고 사용자 답변을 정리하라.",
            "Rules:",
            "- LAW OPEN DATA 근거 밖으로 단정하지 말 것",
            "- packet.evidence와 packet.citations를 우선 사용할 것",
            "- 부족하면 부족하다고 말할 것",
            "",
            build_llm_user_prompt(packet),
        ]
    )


def build_judge_prompt(
    role: str,
    agent_name: str,
    packet: AnswerPacket,
    project_root: Path,
    prior_result: dict | None = None,
) -> str:
    lines = [
        f"Agent: {agent_name}",
        f"Role: {role}",
        f"Project root: {project_root}",
        "Task: LAW OPEN DATA 근거만 사용해 법률시험문제 답변을 판정 또는 재검토하라.",
        "Rules:",
        "- evidence와 citations 밖의 사실을 추가하지 말 것",
        "- 모르면 supported로 밀지 말고 indeterminate 또는 needs_review로 유지할 것",
        "- citations는 반드시 입력 citations의 부분집합이어야 함",
        "- 출력은 JSON 객체 하나만 반환할 것",
        "",
        f"질문: {packet.query}",
        f"현재 답변: {packet.answer}",
        f"현재 상태: {packet.status}",
        f"근거확보: {packet.grounded}",
        f"질문유형: {packet.question_type}",
        f"라우트: {packet.route}",
    ]
    if packet.evidence:
        lines.append("evidence:")
        lines.extend(f"- {item}" for item in packet.evidence)
    if packet.citations:
        lines.append("citations:")
        lines.extend(f"- {item}" for item in packet.citations)
    if packet.exam_assessment:
        lines.append(f"exam_assessment: {packet.exam_assessment}")
    if packet.option_reviews:
        lines.append(f"option_reviews: {packet.option_reviews}")
    if prior_result:
        lines.append(f"prior_result: {prior_result}")

    lines.extend(
        [
            "",
            "Return JSON schema:",
            '{',
            '  "answer": "string",',
            '  "status": "grounded|candidate_answer|needs_review|not_found",',
            '  "grounded_only": true,',
            '  "uses_only_citations": true,',
            '  "citations": ["..."],',
            '  "warnings": ["..."],',
            '  "recommended_options": ["..."],',
            '  "reason": "string"',
            '}',
        ]
    )
    return "\n".join(lines)
