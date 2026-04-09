from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import shutil
import subprocess
from typing import Sequence

from normalizers.text import normalize_lookup_text
from qa.cli import format_article_no
from qa.unified import UnifiedResult


CASE_NO_LINE_RE = re.compile(r"사건번호:\s*(.+)")
ISSUE_NO_LINE_RE = re.compile(r"해석례번호:\s*(.+)")
ENTRY_HEADER_RE = re.compile(r"^\[(\d+)\]\s+(.+)$")
DATE_LINE_RE = re.compile(r"(선고일|종국일|의결일|회신일자):\s*(.+)")


@dataclass
class ExactVerification:
    status: str
    grounded: bool
    warnings: list[str]
    official_text: str | None = None
    official_ref: str | None = None
    mismatch_reason: str | None = None


class KoreanLawExactVerifier:
    def __init__(
        self,
        api_key: str = "da",
        command: str = "korean-law",
        timeout: int = 20,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.command = shutil.which(command) or command
        self.available = shutil.which(command) is not None

    def verify(
        self,
        decision_strategy: str,
        result: UnifiedResult,
    ) -> ExactVerification | None:
        if not decision_strategy.startswith("exact_"):
            return None
        if not self.available:
            return ExactVerification(
                status="needs_review",
                grounded=False,
                warnings=["주의: korean-law 검증기를 찾지 못해 exact 결과를 공식 조회로 재검증하지 못했습니다."],
            )

        if result.source_type == "law":
            return self._verify_law_article(result)
        if result.source_type == "prec":
            return self._verify_precedent(result)
        if result.source_type == "detc":
            return self._verify_constitutional_decision(result)
        if result.source_type == "decc":
            return self._verify_admin_appeal(result)
        if result.source_type == "expc":
            return self._verify_interpretation(result)
        return None

    def _run(self, args: Sequence[str]) -> tuple[bool, str]:
        cmd = [self.command, *args, "--apiKey", self.api_key]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                check=False,
            )
        except Exception as exc:
            return False, str(exc)
        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        if completed.returncode != 0:
            return False, error or output or f"exit={completed.returncode}"
        return True, output

    def _verify_law_article(self, result: UnifiedResult) -> ExactVerification:
        payload = result.payload
        article_no = format_article_no(payload["article_no"])
        ok, output = self._run(
            ["get_law_text", "--lawId", str(payload["law_id"]), "--jo", article_no]
        )
        official_ref = (
            f"law.go.kr / lawId={payload['law_id']} / {article_no} / verified={date.today().isoformat()}"
        )
        if not ok:
            return ExactVerification(
                status="needs_review",
                grounded=False,
                warnings=[f"주의: 공식 조문 검증 호출에 실패했습니다: {output}"],
                official_ref=official_ref,
            )

        article_lines = []
        capture = False
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("제") and article_no.replace(" ", "")[:3] in line.replace(" ", ""):
                capture = True
            if capture:
                article_lines.append(line)
        official_text = "\n".join(article_lines).strip()
        official_norm = normalize_lookup_text(official_text)
        local_norm = normalize_lookup_text(payload.get("article_text", ""))

        if not official_norm:
            return ExactVerification(
                status="verification_failed",
                grounded=False,
                warnings=["주의: 공식 조회 응답에서 조문 본문을 추출하지 못했습니다."],
                official_ref=official_ref,
            )

        if local_norm and (local_norm in official_norm or official_norm in local_norm):
            return ExactVerification(
                status="grounded",
                grounded=True,
                warnings=["주의: 로컬 exact hit를 공식 조문 조회로 교차검증했습니다."],
                official_text=official_text,
                official_ref=official_ref,
            )

        return ExactVerification(
            status="verification_failed",
            grounded=False,
            warnings=["주의: 로컬 조문 본문과 공식 조문 본문이 일치하지 않습니다."],
            official_text=official_text,
            official_ref=official_ref,
            mismatch_reason="local_exact_mismatch",
        )

    def _verify_identifier_lookup(
        self,
        result: UnifiedResult,
        command_args: Sequence[str],
        identifier: str,
        label: str,
        identifier_pattern: re.Pattern[str] = CASE_NO_LINE_RE,
    ) -> ExactVerification:
        ok, output = self._run(command_args)
        official_ref = f"law.go.kr / {label} / verified={date.today().isoformat()}"
        if not ok:
            return ExactVerification(
                status="needs_review",
                grounded=False,
                warnings=[f"주의: 공식 사건 검증 호출에 실패했습니다: {output}"],
                official_ref=official_ref,
            )

        normalized_output = normalize_lookup_text(output)
        normalized_identifier = normalize_lookup_text(identifier)
        identifier_match = identifier_pattern.search(output)
        extracted_identifier = (
            identifier_match.group(1).strip() if identifier_match else identifier
        )
        normalized_extracted = normalize_lookup_text(extracted_identifier)
        identifier_found = normalized_identifier in normalized_output

        if normalized_extracted == normalized_identifier or identifier_found:
            return ExactVerification(
                status="grounded",
                grounded=True,
                warnings=["주의: 로컬 exact hit를 공식 사건 전문 조회로 교차검증했습니다."],
                official_ref=official_ref,
            )

        return ExactVerification(
            status="verification_failed",
            grounded=False,
            warnings=["주의: 로컬 사건 식별자와 공식 조회 식별자가 일치하지 않습니다."],
            official_ref=official_ref,
            mismatch_reason="local_exact_identifier_mismatch",
        )

    def _verify_precedent(self, result: UnifiedResult) -> ExactVerification:
        return self._verify_identifier_lookup(
            result,
            ["get_precedent_text", "--id", str(result.payload["serial_no"])],
            result.payload["case_no"],
            f"판례 {result.payload['case_no']} / get_precedent_text",
        )

    def _verify_constitutional_decision(self, result: UnifiedResult) -> ExactVerification:
        return self._verify_identifier_lookup(
            result,
            ["get_constitutional_decision_text", "--id", str(result.payload["serial_no"])],
            result.payload["case_no"],
            f"헌재결정례 {result.payload['case_no']} / get_constitutional_decision_text",
        )

    def _verify_interpretation(self, result: UnifiedResult) -> ExactVerification:
        payload = result.payload
        serial_no = str(payload["serial_no"])
        issue_no = str(payload["issue_no"])
        title = (payload.get("title") or "").strip()
        decision_date = normalize_lookup_text(payload.get("decision_date") or "")

        search_ok, search_output = self._run(
            ["search_interpretations", "--query", issue_no, "--display", "10"]
        )
        search_ref = (
            f"law.go.kr / 법령해석례 {issue_no} / search_interpretations / "
            f"verified={date.today().isoformat()}"
        )
        if not search_ok:
            return ExactVerification(
                status="needs_review",
                grounded=False,
                warnings=[f"주의: 법령해석례 검색 검증 호출에 실패했습니다: {search_output}"],
                official_ref=search_ref,
            )

        search_match = self._parse_search_entry(
            search_output,
            expected_id=serial_no,
            expected_title=title,
            expected_date=decision_date,
            identifier_pattern=ISSUE_NO_LINE_RE,
        )
        if not search_match:
            return ExactVerification(
                status="verification_failed",
                grounded=False,
                warnings=["주의: 법령해석례 검색 결과에서 로컬 exact와 일치하는 항목을 찾지 못했습니다."],
                official_ref=search_ref,
                mismatch_reason="local_exact_identifier_mismatch",
            )

        get_ok, get_output = self._run(["get_interpretation_text", "--id", serial_no])
        warnings = [
            "주의: 로컬 exact hit를 법령해석례 검색 결과 ID/제목/회신일자로 교차검증했습니다."
        ]
        official_ref = search_ref
        if get_ok:
            official_ref = (
                f"law.go.kr / 법령해석례 {issue_no} / "
                f"search_interpretations+get_interpretation_text / verified={date.today().isoformat()}"
            )
            warnings.append("주의: 법령해석례 전문 조회도 함께 확인했습니다.")
        else:
            warnings.append("주의: 법령해석례 전문 조회는 실패했지만 검색 결과 기준으로 exact 검증을 통과했습니다.")

        return ExactVerification(
            status="grounded",
            grounded=True,
            warnings=warnings,
            official_text=get_output if get_ok else None,
            official_ref=official_ref,
        )

    def _verify_admin_appeal(self, result: UnifiedResult) -> ExactVerification:
        payload = result.payload
        serial_no = str(payload["serial_no"])
        case_name = (payload.get("case_name") or "").strip()
        decision_date = normalize_lookup_text(payload.get("decision_date") or "")

        search_ok, search_output = self._run(
            ["search_admin_appeals", "--query", str(payload["case_no"]), "--display", "10"]
        )
        search_ref = (
            f"law.go.kr / 행정심판례 {payload['case_no']} / search_admin_appeals / "
            f"verified={date.today().isoformat()}"
        )
        if not search_ok:
            return ExactVerification(
                status="needs_review",
                grounded=False,
                warnings=[f"주의: 행정심판례 검색 검증 호출에 실패했습니다: {search_output}"],
                official_ref=search_ref,
            )

        search_match = self._parse_search_entry(
            search_output,
            expected_id=serial_no,
            expected_title=case_name,
            expected_date=decision_date,
        )
        if not search_match:
            return ExactVerification(
                status="verification_failed",
                grounded=False,
                warnings=["주의: 행정심판례 검색 결과에서 로컬 exact와 일치하는 항목을 찾지 못했습니다."],
                official_ref=search_ref,
                mismatch_reason="local_exact_identifier_mismatch",
            )

        get_ok, get_output = self._run(["get_admin_appeal_text", "--id", serial_no])
        warnings = [
            "주의: 로컬 exact hit를 행정심판례 검색 결과 ID/제목/의결일로 교차검증했습니다."
        ]
        official_ref = search_ref
        if get_ok:
            official_ref = (
                f"law.go.kr / 행정심판례 {payload['case_no']} / "
                f"search_admin_appeals+get_admin_appeal_text / verified={date.today().isoformat()}"
            )
            warnings.append("주의: 행정심판례 전문 조회도 함께 확인했습니다.")
        else:
            warnings.append("주의: 행정심판례 전문 조회는 실패했지만 검색 결과 기준으로 exact 검증을 통과했습니다.")

        return ExactVerification(
            status="grounded",
            grounded=True,
            warnings=warnings,
            official_text=get_output if get_ok else None,
            official_ref=official_ref,
        )

    def _parse_search_entry(
        self,
        output: str,
        expected_id: str,
        expected_title: str,
        expected_date: str,
        identifier_pattern: re.Pattern[str] = CASE_NO_LINE_RE,
    ) -> dict[str, str] | None:
        current: dict[str, str] | None = None
        entries: list[dict[str, str]] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            header_match = ENTRY_HEADER_RE.match(line)
            if header_match:
                if current:
                    entries.append(current)
                current = {"id": header_match.group(1), "title": header_match.group(2).strip()}
                continue
            if current is None:
                continue
            date_match = DATE_LINE_RE.search(line)
            if date_match:
                current["date"] = normalize_lookup_text(date_match.group(2).strip())
                continue
            identifier_match = identifier_pattern.search(line)
            if identifier_match:
                current["identifier"] = identifier_match.group(1).strip()
                continue
        if current:
            entries.append(current)

        expected_title_norm = normalize_lookup_text(expected_title)
        for entry in entries:
            if entry.get("id") != expected_id:
                continue
            entry_title_norm = normalize_lookup_text(entry.get("title", ""))
            entry_date_norm = normalize_lookup_text(entry.get("date", ""))
            title_matches = not expected_title_norm or (
                expected_title_norm in entry_title_norm or entry_title_norm in expected_title_norm
            )
            date_matches = not expected_date or not entry_date_norm or entry_date_norm == expected_date
            if title_matches and date_matches:
                return entry
        return None
