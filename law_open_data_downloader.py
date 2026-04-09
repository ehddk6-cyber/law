# -*- coding: utf-8 -*-
"""
국가법령정보 공동활용 LAW OPEN DATA 전체 수집기 (병렬 수집 버전)
"""

from __future__ import annotations

import json
import os
import re
import time
import pathlib
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import requests

# =========================
# 기본 설정
# =========================
OC = "da"

BASE_SEARCH = "https://www.law.go.kr/DRF/lawSearch.do"
BASE_SERVICE = "https://www.law.go.kr/DRF/lawService.do"

ROOT_DIR = pathlib.Path("law_open_data")
DISPLAY = 100
TIMEOUT = 60
RETRIES = 3
RATE_LIMIT_SECONDS = 0.15
MAX_WORKERS = 5  # 병렬 수집용 스레드 수

# 기본 저장 정책
DOWNLOAD_ALL_FORMATS = False
DOWNLOAD_MOBILE = True
DOWNLOAD_BODY = True
DOWNLOAD_KB_BASE = True
DOWNLOAD_SEED_APIS = True

# =========================
# 도우미
# =========================

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; law-open-data-bulk-downloader/1.0)",
        "Accept": "*/*",
    }
)

def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def safe_name(text: str, max_len: int = 160) -> str:
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", text).strip(" ._")
    return text[:max_len] if len(text) > max_len else text

def write_text(path: pathlib.Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def write_json(path: pathlib.Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: pathlib.Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def request_text(url: str, params: Dict[str, Any]) -> str:
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = SESSION.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(1.2 * attempt)
    raise last_error

def request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    text = request_text(url, params)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"JSON 디코딩 실패: {url} params={params} 응답앞부분={text[:500]}")

def recursive_find_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return obj
        out: List[Dict[str, Any]] = []
        for x in obj:
            out.extend(recursive_find_items(x))
        return out

    if isinstance(obj, dict):
        preferred_keys = [
            "law", "admrul", "ordin", "prec", "detc", "expc", "decc", "trty", "lstrm",
            "elaw", "school", "public", "pi", "ppc", "eiac", "ftc", "acr", "fsc",
            "kcc", "nlrc", "iaciac", "oclt", "ecc", "sfc", "nhrck", "specialDecc",
            "deccList", "result", "results", "items", "list"
        ]
        for k in preferred_keys:
            if k in obj:
                found = recursive_find_items(obj[k])
                if found:
                    return found
        for v in obj.values():
            found = recursive_find_items(v)
            if found:
                return found
    return []

def recursive_find_total(obj: Any) -> Optional[int]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == "totalcnt":
                try:
                    return int(v)
                except Exception:
                    pass
            r = recursive_find_total(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = recursive_find_total(v)
            if r is not None:
                return r
    return None

def first_nonempty(item: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in item and str(item[k]).strip():
            return str(item[k]).strip()
    return None

def dedupe_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for k, v in pairs:
        key = (k, v)
        if key not in seen and v:
            seen.add(key)
            out.append((k, v))
    return out

@dataclass
class Category:
    name: str
    list_target: str
    body_target: Optional[str]
    note: str = ""
    mobile: bool = False
    list_only: bool = False
    query_default: Optional[str] = None
    confidence: str = "verified"
    list_formats: Tuple[str, ...] = ("JSON",)
    body_formats: Tuple[str, ...] = ("XML",)

    def effective_list_formats(self) -> Tuple[str, ...]:
        if DOWNLOAD_ALL_FORMATS:
            return ("HTML", "XML", "JSON")
        return self.list_formats

    def effective_body_formats(self) -> Tuple[str, ...]:
        if self.list_only or not self.body_target:
            return tuple()
        if self.mobile:
            return ("HTML",)
        if DOWNLOAD_ALL_FORMATS:
            return ("HTML", "XML", "JSON")
        return self.body_formats

def build_request_identifiers(cat: Category, item: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if cat.name in {"law", "eflaw", "m_law", "m_eflaw"}:
        v = first_nonempty(item, ["법령ID", "ID", "id"])
        if v: pairs.append(("ID", v))
        v = first_nonempty(item, ["법령일련번호", "MST", "mst"])
        if v: pairs.append(("MST", v))
        return dedupe_pairs(pairs)

    serial_like_keys = [
        "판례일련번호", "헌재결정례일련번호", "법령해석례일련번호", "행정심판례일련번호",
        "조약일련번호", "법령용어일련번호", "특별행정심판재결례일련번호",
        "결정문일련번호", "행정규칙일련번호", "자치법규일련번호", "일련번호",
        "법령일련번호", "MST"
    ]
    direct_id_keys = ["행정규칙ID", "자치법규ID", "법령용어ID", "법령ID", "ID", "id"]
    for k in serial_like_keys:
        if k in item and str(item[k]).strip():
            if k == "법령일련번호": pairs.append(("MST", str(item[k]).strip()))
            else: pairs.append(("ID", str(item[k]).strip()))
    for k in direct_id_keys:
        if k in item and str(item[k]).strip():
            pairs.append(("ID", str(item[k]).strip()))
    return dedupe_pairs(pairs)

# 카테고리 설정 함수들 (생략하지 않고 모두 포함)
def core_categories() -> List[Category]:
    return [
        Category("law", "law", "law", note="현행법령(공포일)"),
        Category("eflaw", "eflaw", "eflaw", note="현행법령(시행일)"),
        Category("admrul", "admrul", "admrul", note="현행 행정규칙"),
        Category("ordin", "ordin", "ordin", note="현행 자치법규"),
        Category("prec", "prec", "prec", note="판례"),
        Category("detc", "detc", "detc", note="헌재결정례"),
        Category("expc", "expc", "expc", note="법령해석례"),
        Category("decc", "decc", "decc", note="행정심판례"),
        Category("trty", "trty", "trty", note="조약"),
        Category("lstrm", "lstrm", "lstrm", note="법령용어"),
        Category("elaw", "elaw", "elaw", note="영문법령"),
        Category("licbyl", "licbyl", None, note="법령 별표·서식", list_only=True, query_default="*"),
        Category("admbyl", "admbyl", None, note="행정규칙 별표·서식", list_only=True, query_default="*"),
        Category("ordinbyl", "ordinbyl", None, note="자치법규 별표·서식", list_only=True, query_default="*"),
        Category("school", "school", "school", note="대학/학칙"),
        Category("public", "public", "public", note="지방공사공단"),
        Category("pi", "pi", "pi", note="공공기관"),
    ]

def committee_categories() -> List[Category]:
    return [
        Category("ppc", "ppc", "ppc", note="개인정보보호위원회 결정문"),
        Category("eiac", "eiac", "eiac", note="고용보험심사위원회 결정문"),
        Category("ftc", "ftc", "ftc", note="공정거래위원회 결정문"),
        Category("acr", "acr", "acr", note="국민권익위원회 결정문"),
        Category("fsc", "fsc", "fsc", note="금융위원회 결정문"),
        Category("nlrc", "nlrc", "nlrc", note="노동위원회 결정문"),
        Category("kcc", "kcc", "kcc", note="방송미디어통신위원회 결정문"),
        Category("iaciac", "iaciac", "iaciac", note="산업재해보상보험재심사위원회 결정문"),
        Category("oclt", "oclt", "oclt", note="중앙토지수용위원회 결정문"),
        Category("ecc", "ecc", "ecc", note="중앙환경분쟁조정위원회 결정문"),
        Category("sfc", "sfc", "sfc", note="증권선물위원회 결정문"),
        Category("nhrck", "nhrck", "nhrck", note="국가인권위원회 결정문"),
    ]

def ministry_interpretation_categories() -> List[Category]:
    return [
        Category("moelCgmExpc", "moelCgmExpc", "moelCgmExpc", note="고용노동부 법령해석"),
        Category("molitCgmExpc", "molitCgmExpc", "molitCgmExpc", note="국토교통부 법령해석"),
        Category("moisCgmExpc", "moisCgmExpc", "moisCgmExpc", note="행정안전부 법령해석"),
        Category("meCgmExpc", "meCgmExpc", "meCgmExpc", note="기후에너지환경부 법령해석"),
        Category("kcsCgmExpc", "kcsCgmExpc", "kcsCgmExpc", note="관세청 법령해석"),
        Category("moeCgmExpc", "moeCgmExpc", "moeCgmExpc", note="교육부 법령해석"),
        Category("msitCgmExpc", "msitCgmExpc", "msitCgmExpc", note="과학기술정보통신부 법령해석"),
        Category("mpvaCgmExpc", "mpvaCgmExpc", "mpvaCgmExpc", note="국가보훈부 법령해석"),
        Category("mndCgmExpc", "mndCgmExpc", "mndCgmExpc", note="국방부 법령해석"),
        Category("mafraCgmExpc", "mafraCgmExpc", "mafraCgmExpc", note="농림축산식품부 법령해석"),
        Category("mcstCgmExpc", "mcstCgmExpc", "mcstCgmExpc", note="문화체육관광부 법령해석"),
        Category("mojCgmExpc", "mojCgmExpc", "mojCgmExpc", note="법무부 법령해석"),
        Category("mohwCgmExpc", "mohwCgmExpc", "mohwCgmExpc", note="보건복지부 법령해석"),
        Category("motieCgmExpc", "motieCgmExpc", "motieCgmExpc", note="산업통상부 법령해석"),
        Category("mogefCgmExpc", "mogefCgmExpc", "mogefCgmExpc", note="성평등가족부 법령해석"),
        Category("mofaCgmExpc", "mofaCgmExpc", "mofaCgmExpc", note="외교부 법령해석"),
        Category("mssCgmExpc", "mssCgmExpc", "mssCgmExpc", note="중소벤처기업부 법령해석"),
        Category("mouCgmExpc", "mouCgmExpc", "mouCgmExpc", note="통일부 법령해석"),
        Category("molegCgmExpc", "molegCgmExpc", "molegCgmExpc", note="법제처 법령해석"),
        Category("mfdsCgmExpc", "mfdsCgmExpc", "mfdsCgmExpc", note="식품의약품안전처 법령해석"),
        Category("mpmCgmExpc", "mpmCgmExpc", "mpmCgmExpc", note="인사혁신처 법령해석"),
        Category("kmaCgmExpc", "kmaCgmExpc", "kmaCgmExpc", note="기상청 법령해석"),
        Category("khsCgmExpc", "khsCgmExpc", "khsCgmExpc", note="국가유산청 법령해석"),
        Category("rdaCgmExpc", "rdaCgmExpc", "rdaCgmExpc", note="농촌진흥청 법령해석"),
        Category("npaCgmExpc", "npaCgmExpc", "npaCgmExpc", note="경찰청 법령해석"),
        Category("dapaCgmExpc", "dapaCgmExpc", "dapaCgmExpc", note="방위사업청 법령해석"),
        Category("mmaCgmExpc", "mmaCgmExpc", "mmaCgmExpc", note="병무청 법령해석"),
        Category("kfsCgmExpc", "kfsCgmExpc", "kfsCgmExpc", note="산림청 법령해석"),
        Category("nfaCgmExpc", "nfaCgmExpc", "nfaCgmExpc", note="소방청 법령해석"),
        Category("okaCgmExpc", "okaCgmExpc", "okaCgmExpc", note="재외동포청 법령해석"),
        Category("ppsCgmExpc", "ppsCgmExpc", "ppsCgmExpc", note="조달청 법령해석"),
        Category("kdcaCgmExpc", "kdcaCgmExpc", "kdcaCgmExpc", note="질병관리청 법령해석"),
        Category("kostatCgmExpc", "kostatCgmExpc", "kostatCgmExpc", note="국가데이터처 법령해석"),
        Category("kipoCgmExpc", "kipoCgmExpc", "kipoCgmExpc", note="지식재산처 법령해석"),
        Category("kcgCgmExpc", "kcgCgmExpc", "kcgCgmExpc", note="해양경찰청 법령해석"),
        Category("naaccCgmExpc", "naaccCgmExpc", "naaccCgmExpc", note="행정중심복합도시건설청 법령해석"),
    ]

def special_admin_appeal_categories() -> List[Category]:
    return [
        Category("ttSpecialDecc", "ttSpecialDecc", "ttSpecialDecc", note="조세심판원 특별행정심판례"),
        Category("kmstSpecialDecc", "kmstSpecialDecc", "kmstSpecialDecc", note="해양안전심판원 특별행정심판례"),
        Category("acrSpecialDecc", "acrSpecialDecc", "acrSpecialDecc", note="국민권익위원회 특별행정심판례"),
        Category("adapSpecialDecc", "adapSpecialDecc", "adapSpecialDecc", note="인사혁신처 소청심사위원회 특별행정심판재결례"),
    ]

def advisory_categories() -> List[Category]:
    return [Category("baiConsult", "baiConsult", "baiConsult", note="감사원 사전컨설팅 의견서", confidence="inferred")]

def kb_base_categories() -> List[Category]:
    return [
        Category("lstrmAI", "lstrmAI", None, note="법령정보지식베이스 법령용어", list_only=True, query_default="*"),
        Category("dlytrm", "dlytrm", None, note="법령정보지식베이스 일상용어", list_only=True, query_default="*"),
    ]

def mobile_categories() -> List[Category]:
    return [
        Category("m_law", "law", "law", note="모바일 법령", mobile=True, body_formats=("HTML",)),
        Category("m_admrul", "admrul", "admrul", note="모바일 행정규칙", mobile=True, body_formats=("HTML",)),
        Category("m_ordin", "ordin", "ordin", note="모바일 자치법규", mobile=True, body_formats=("HTML",)),
        Category("m_prec", "prec", "prec", note="모바일 판례", mobile=True, body_formats=("HTML",)),
        Category("m_detc", "detc", "detc", note="모바일 헌재결정례", mobile=True, body_formats=("HTML",)),
        Category("m_expc", "expc", "expc", note="모바일 법령해석례", mobile=True, body_formats=("HTML",)),
        Category("m_decc", "decc", "decc", note="모바일 행정심판례", mobile=True, body_formats=("HTML",)),
        Category("m_trty", "trty", "trty", note="모바일 조약", mobile=True, body_formats=("HTML",)),
        Category("m_ordinbyl", "ordinbyl", None, note="모바일 자치법규 별표·서식", mobile=True, list_only=True, query_default="*"),
        Category("m_lstrm", "lstrm", None, note="모바일 법령용어", mobile=True, list_only=True),
    ]

def category_dir(cat: Category) -> pathlib.Path: return ROOT_DIR / cat.name
def list_dir(cat: Category) -> pathlib.Path: return category_dir(cat) / "list"
def body_dir(cat: Category) -> pathlib.Path: return category_dir(cat) / "body"
def logs_dir() -> pathlib.Path: return ROOT_DIR / "logs"

def base_params_for_list(cat: Category, page: int, fmt: str) -> Dict[str, Any]:
    params = {"OC": OC, "target": cat.list_target, "type": fmt, "display": DISPLAY, "page": page}
    if cat.query_default: params["query"] = cat.query_default
    if cat.mobile: params["mobileYn"] = "Y"
    return params

def base_params_for_body(cat: Category, fmt: str, key: str, value: str) -> Dict[str, Any]:
    params = {"OC": OC, "target": cat.body_target, "type": fmt, key: value}
    if cat.mobile: params["mobileYn"] = "Y"
    return params

def save_list_page_raw(cat: Category, page: int, fmt: str, text: str) -> None:
    write_text(list_dir(cat) / fmt.lower() / f"page_{page}.{fmt.lower()}", text)

def save_body_raw(cat: Category, identifier_key: str, identifier_value: str, fmt: str, text: str) -> None:
    filename = f"{identifier_key}_{safe_name(identifier_value)}.{fmt.lower()}"
    write_text(body_dir(cat) / fmt.lower() / filename, text)

def log_error(cat_name: str, stage: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    append_jsonl(logs_dir() / "errors.jsonl", {"ts": now_ts(), "category": cat_name, "stage": stage, "message": message, "extra": extra or {}})

def log_progress(message: str) -> None:
    print(f"[{now_ts()}] {message}", flush=True)

def fetch_list_json_page(cat: Category, page: int) -> Dict[str, Any]:
    data = request_json(BASE_SEARCH, base_params_for_list(cat, page, "JSON"))
    save_list_page_raw(cat, page, "JSON", json.dumps(data, ensure_ascii=False, indent=2))
    time.sleep(RATE_LIMIT_SECONDS)
    return data

def save_extra_list_formats(cat: Category, page: int) -> None:
    for fmt in cat.effective_list_formats():
        if fmt == "JSON": continue
        try:
            save_list_page_raw(cat, page, fmt, request_text(BASE_SEARCH, base_params_for_list(cat, page, fmt)))
        except Exception as e:
            log_error(cat.name, "save_extra_list_formats", str(e), {"page": page, "format": fmt})
        time.sleep(RATE_LIMIT_SECONDS)

def download_body_for_item(cat: Category, item: Dict[str, Any]) -> None:
    if not DOWNLOAD_BODY or cat.list_only or not cat.body_target: return
    identifiers = build_request_identifiers(cat, item)
    if not identifiers:
        log_error(cat.name, "extract_identifier", "본문 식별자 추출 실패", {"item": item})
        return
    for key, value in identifiers:
        success = False
        for fmt in cat.effective_body_formats():
            try:
                save_body_raw(cat, key, value, fmt, request_text(BASE_SERVICE, base_params_for_body(cat, fmt, key, value)))
                success = True
            except Exception as e:
                log_error(cat.name, "download_body", str(e), {"key": key, "val": value, "fmt": fmt})
            time.sleep(RATE_LIMIT_SECONDS)
        if success: break

def crawl_category(cat: Category) -> None:
    ensure_dir(category_dir(cat))
    write_json(category_dir(cat) / "meta.json", asdict(cat))
    if (category_dir(cat) / "summary.json").exists():
        log_progress(f"SKIP {cat.name}")
        return
    
    page = 1
    index_path = category_dir(cat) / "index.jsonl"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            current_items = sum(1 for _ in f)
        if current_items > 0:
            page = (current_items // DISPLAY) + 1
            log_progress(f"RESUME {cat.name} from page={page}")

    total_saved = 0
    total_items = None
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while True:
            try:
                data = fetch_list_json_page(cat, page)
                items = recursive_find_items(data)
                total_items = recursive_find_total(data)
                save_extra_list_formats(cat, page)
            except Exception as e:
                log_error(cat.name, "fetch_list_page", str(e), {"page": page})
                break
            
            log_progress(f"{cat.name} page={page} items={len(items)} total={total_items}")
            if not items: break
            
            for item in items: append_jsonl(category_dir(cat) / "index.jsonl", item)
            
            if not cat.list_only:
                # 병렬 본문 다운로드
                list(executor.map(lambda it: download_body_for_item(cat, it), items))
            
            total_saved += (page - 1) * DISPLAY + len(items) # 단순 누적용이 아닌 전체 카운트 반영 필요 시 수정
            if len(items) < DISPLAY: break
            page += 1
            
    write_json(category_dir(cat) / "summary.json", {"category": cat.name, "completed_at": now_ts()})

def main() -> None:
    ensure_dir(ROOT_DIR)
    ensure_dir(logs_dir())
    categories = []
    categories.extend(core_categories())
    categories.extend(committee_categories())
    categories.extend(ministry_interpretation_categories())
    categories.extend(special_admin_appeal_categories())
    categories.extend(advisory_categories())
    if DOWNLOAD_KB_BASE: categories.extend(kb_base_categories())
    if DOWNLOAD_MOBILE: categories.extend(mobile_categories())
    
    for cat in categories:
        try:
            log_progress(f"START {cat.name}")
            crawl_category(cat)
            log_progress(f"END {cat.name}")
        except Exception as e:
            log_error(cat.name, "fatal", str(e), {"traceback": traceback.format_exc()})

if __name__ == "__main__":
    main()
