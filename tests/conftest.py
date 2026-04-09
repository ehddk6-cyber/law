from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / ".artifacts"

SAMPLE_LAW_TEXT = (
    "제14조 취소소송의 제기 취소소송은 처분등이 있음을 안 날부터 90일 이내에 제기하여야 한다. "
    "제27조 보조참가 이해관계가 있는 제3자는 재판소의 허가를 얻어 당사자의 보조참가를 할 수 있다. "
    "제28조 사정판결 원고의 청구에 이유 없음에도 불구하고 처분등을 취소하지 아니하면 "
    "공공복리에 적합하지 아니하다고 인정한 때에는 재판소는 처분등의 취소를 기각하는 판결을 선고할 수 있다. "
    "제38조 준용 제14조부터 제27조까지의 규정은 무효확인소송에 준용한다."
)


@pytest.fixture
def artifacts_path() -> Path:
    return ARTIFACTS


@pytest.fixture
def sample_law_text() -> str:
    return SAMPLE_LAW_TEXT
