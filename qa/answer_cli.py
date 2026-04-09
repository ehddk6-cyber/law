from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.answering import GroundedAnswerer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render grounded legal answers from unified retrieval."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-model")
    parser.add_argument("--include-agent-prompts", action="store_true")
    parser.add_argument(
        "--response-mode", choices=["grounded", "llm_preferred", "llm_only"]
    )
    parser.add_argument("--explain-style", choices=["admin_exam"])
    parser.add_argument(
        "--run-agent",
        choices=["codex", "claude-code", "claude-stepfree", "qwen", "gemini"],
    )
    parser.add_argument("--run-agent-model")
    args = parser.parse_args()

    answerer = GroundedAnswerer(args.root)
    packet = answerer.answer(
        args.query,
        limit=args.limit,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        include_agent_prompts=args.include_agent_prompts,
        response_mode=args.response_mode,
        run_agent_name=args.run_agent,
        run_agent_model=args.run_agent_model,
        explain_style=args.explain_style,
    )
    print(f"scope={packet.scope}")
    print(f"question_type={packet.question_type}")
    print(f"route={packet.route}")
    print(f"status={packet.status}")
    print(f"grounded={packet.grounded}")
    print(f"source_type={packet.source_type}")
    print(f"doc_id={packet.doc_id}")
    print(f"explain_style={packet.explain_style}")
    print(f"final_answer_source={packet.final_answer_source}")
    print("answer=" + packet.answer)
    if packet.evidence:
        print("evidence:")
        for item in packet.evidence:
            print(item)
    if packet.citations:
        print("citations:")
        for citation in packet.citations:
            print(citation)
    if packet.warnings:
        print("warnings:")
        for warning in packet.warnings:
            print(warning)
    if packet.option_reviews:
        print("option_reviews:")
        for review in packet.option_reviews:
            print(
                f"{review['label']} status={review['status']} grounded={review['grounded']} "
                f"support_level={review['support_level']} source_type={review['source_type']} doc_id={review['doc_id']}"
            )
            print(f"text={review['text']}")
            if review.get("anchor_query"):
                print(f"anchor_query={review['anchor_query']}")
            if review.get("referenced_labels"):
                print("referenced_labels=" + ", ".join(review["referenced_labels"]))
            if review.get("count_value") is not None:
                print(f"count_value={review['count_value']}")
            print(f"answer={review['answer']}")
            print(f"support_reason={review['support_reason']}")
            if review.get("effective_support_level"):
                print(f"effective_support_level={review['effective_support_level']}")
                print(f"effective_support_reason={review['effective_support_reason']}")
            if review["citations"]:
                print("citations=" + " | ".join(review["citations"]))
            if review.get("sub_reviews"):
                print("sub_reviews:")
                for sub in review["sub_reviews"]:
                    print(
                        f"{sub['label']} status={sub['status']} grounded={sub['grounded']} "
                        f"support_level={sub['support_level']}"
                    )
                    print(f"text={sub['text']}")
                    print(f"support_reason={sub['support_reason']}")
            if review.get("sub_assessment"):
                print("sub_assessment:")
                print(
                    "supported_labels="
                    + ", ".join(review["sub_assessment"]["supported_labels"])
                )
                print(
                    "unsupported_labels="
                    + ", ".join(review["sub_assessment"]["unsupported_labels"])
                )
                print(
                    "indeterminate_labels="
                    + ", ".join(review["sub_assessment"]["indeterminate_labels"])
                )
    if packet.exam_assessment:
        print("exam_assessment:")
        for key in ("status", "count_answer", "recommended_options", "reason"):
            if key not in packet.exam_assessment:
                continue
            value = packet.exam_assessment[key]
            if isinstance(value, list):
                print(f"{key}=" + ", ".join(value))
            else:
                print(f"{key}={value}")
    if packet.teaching_explanation:
        print("teaching_explanation=" + str(packet.teaching_explanation))
        extra_keys = [
            key
            for key in packet.exam_assessment.keys()
            if key not in {"status", "count_answer", "recommended_options", "reason"}
        ]
        for key in extra_keys:
            value = packet.exam_assessment[key]
            if isinstance(value, list):
                print(f"{key}=" + ", ".join(value))
            else:
                print(f"{key}={value}")
    if packet.llm_provider or packet.llm_error:
        print("llm:")
        print(f"provider={packet.llm_provider}")
        print(f"model={packet.llm_model}")
        print(f"response_mode={packet.response_mode}")
        if packet.llm_answer:
            print("llm_answer=" + packet.llm_answer)
        if packet.llm_error:
            print("llm_error=" + packet.llm_error)
    if packet.fallback_trace:
        print("fallback_trace=" + " | ".join(packet.fallback_trace))
    if packet.agent_prompts:
        print("agent_prompts:")
        for key, value in packet.agent_prompts.items():
            print(f"[{key}]")
            print(value)
    if packet.agent_runs:
        print("agent_runs:")
        for key, value in packet.agent_runs.items():
            print(f"[{key}] ok={value.get('ok')}")
            if value.get("output"):
                print(value["output"])
            if value.get("error"):
                print("error=" + str(value["error"]))


if __name__ == "__main__":
    main()
