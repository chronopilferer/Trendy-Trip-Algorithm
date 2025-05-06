import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_record(json_path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    주어진 JSON 파일을 로드하거나, 파일이 없거나 파싱 실패 시 defaults의 복사본을 반환합니다.

    Args:
        json_path: 로드할 JSON 파일 경로
        defaults: 파일이 없을 때 사용할 기본 dict

    Returns:
        JSON 데이터를 dict로 반환하거나, defaults의 복사본
    """
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_path.name}: {e}")
    return defaults.copy()


def load_json_records(json_dir: Path) -> List[Dict[str, Any]]:
    """
    주어진 디렉토리 내 모든 JSON 파일을 로드하여 dict 리스트로 반환합니다.
    100개 단위로 진행 상황을 로깅하고, 파싱 실패 파일은 에러로 기록합니다.

    Args:
        json_dir: JSON 파일들이 위치한 디렉토리 경로

    Returns:
        JSON 레코드의 리스트
    """
    records: List[Dict[str, Any]] = []
    json_dir = Path(json_dir)

    if not json_dir.exists() or not json_dir.is_dir():
        logger.error(f"Invalid JSON directory: {json_dir}")
        return records

    json_paths = list(json_dir.glob("*.json"))
    if not json_paths:
        logger.warning(f"No JSON files found in: {json_dir}")
        return records

    total = len(json_paths)
    for idx, path in enumerate(json_paths, start=1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {path.name}: {e}")
        if idx % 100 == 0 or idx == total:
            logger.info(f"Loaded {idx}/{total} JSON files")

    return records